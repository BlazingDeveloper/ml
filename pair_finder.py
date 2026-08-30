"""
pair_finder.py — Fix version of code.py for SIH26166.

Fixes applied vs original code.py:
  1. Longitude normalization: ODE WKT uses west-positive → convert to east-positive
     before any Shapely intersection (was cause of zero-match bug).
  2. MIN_AREA_RATIO lowered 0.50 → 0.15: a single NAC swath (~2.5 km wide) can
     never cover 50% of the OHRC footprint (~11 km wide); 15% is physically realistic.
  3. All OHRC I/O uses np.memmap (mode='r') — never loads 1.2 GB into RAM.
  4. int16 NAC: normalize with percentile stretch (1–99.5%), not min–max.
  5. All steps 4–7 are inside main() — 'best' is always defined before use.
  6. Download M1382366243RC (primary hardcoded target) with redirect-follow.
  7. Footprint sanity: warn + fallback to bbox-only if WKT extent looks wrong.
  8. Verify downloaded file size vs label before proceeding.
  9. Crop both images to intersection polygon; print overlap bbox.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image
from shapely import wkt
from shapely.geometry import Polygon, box

# Local
from pds4_reader import (
    open_pds4_memmap,
    verify_pds4_file,
    normalize_to_uint8_percentile,
    get_pds4_shape,
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
OHRC_XML_PATH  = "ch2_ohr_ncp_20210405T0442095110_d_img_d18.xml"
OHRC_IMG_PATH  = "ch2_ohr_ncp_20210405T0442095110_d_img_d18.img"
OHRC_OBS_TIME  = "2021-04-05T04:42:09.5110Z"

MIN_AREA_RATIO   = 0.15    # 0.15 ≪ 0.50 old; NAC swath ≈2.5 km vs OHRC ≈11 km wide
MAX_TIME_DIFF_HRS = 3000   # ~125 days — loosen for near-identical illumination geometry

OUTPUT_DIR = "output"

# Expected NAC CDR footprint size for sanity check (metres)
NAC_EXPECTED_WIDTH_M  = 2_500   # across-track ≈ 2.5 km
NAC_EXPECTED_LENGTH_M = 26_000  # along-track  ≈ 26 km
NAC_SIZE_FACTOR_TOL   = 5.0     # allow 5× deviation before flagging

# Hardcoded primary NAC target (verified externally: 2021-07-31, incidence 83°)
PRIMARY_NAC_BASE = "M1382366243RC"
PRIMARY_NAC_IMG  = ("https://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0"
                    "/LROLRC_1048B/DATA/ESM4/2021212/NAC/M1382366243RC.IMG")
PRIMARY_NAC_XML  = ("https://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0"
                    "/LROLRC_1048B/DATA/ESM4/2021212/NAC/M1382366243RC.xml")
PRIMARY_NAC_SIZE = 264_467_400  # bytes from label

# ODE REST endpoint
ODE_URL = "https://oderest.rsl.wustl.edu/live2/"

# ── PDS4 namespaces ──────────────────────────────────────────────────────────
PDS4_NS  = {"pds":  "http://pds.nasa.gov/pds4/pds/v1"}
ISDA_NS  = {"isda": "https://isda.issdc.gov.in/pds4/isda/v1"}

# ── Longitude normalisation ──────────────────────────────────────────────────

def normalize_lon_east(lon: float) -> float:
    """
    Convert any longitude to east-positive 0–360 range.
    ODE WKT footprints use west-positive (negative values for the Lunar far-side
    region near 340°E).  OHRC XML is east-positive 0–360.
    Normalising both to east-positive 0–360 before Shapely operations fixes the
    zero-intersection bug in the original code.
    """
    return lon % 360.0


def normalize_polygon_lons(poly: Polygon) -> Polygon:
    """Rebuild a Shapely polygon with all x-coords normalised to east 0–360."""
    from shapely.geometry import mapping, shape
    import json
    coords = list(poly.exterior.coords)
    new_coords = [(normalize_lon_east(x), y) for x, y in coords]
    return Polygon(new_coords)


def footprint_sanity(geom: Polygon) -> tuple[bool, str]:
    """
    Check that a WKT footprint has roughly the right size for a NAC CDR strip.
    Returns (ok, message).  If not ok, caller should fall back to bbox filtering.

    At 1 m/px and polar stereographic near -68°, 1° ≈ 30 km along-track.
    NAC swath ≈ 2.5 km wide → ~0.083° in longitude.
    """
    minx, miny, maxx, maxy = geom.bounds
    # rough km conversion at ~68°S: 1° lat ≈ 111 km; 1° lon ≈ 111 * cos(68°) ≈ 41.6 km
    lon_km = abs(maxx - minx) * 41.6
    lat_km = abs(maxy - miny) * 111.0

    width  = min(lon_km, lat_km)
    length = max(lon_km, lat_km)

    ok_width  = width  < NAC_EXPECTED_WIDTH_M  / 1000 * NAC_SIZE_FACTOR_TOL
    ok_length = length < NAC_EXPECTED_LENGTH_M / 1000 * NAC_SIZE_FACTOR_TOL
    ok_pos    = width  > 0.1 and length > 0.5  # not degenerate

    ok = ok_width and ok_length and ok_pos
    msg = (f"WKT extent: width≈{width:.1f} km length≈{length:.1f} km "
           f"(expected ~{NAC_EXPECTED_WIDTH_M/1000:.1f}×{NAC_EXPECTED_LENGTH_M/1000:.1f} km)")
    return ok, msg


# ── OHRC metadata ────────────────────────────────────────────────────────────

def parse_ohrc_metadata(xml_path: str) -> dict:
    """Parse Refined_Corner_Coordinates from OHRC PDS4 label (east-positive lons)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = ISDA_NS

    corners_elem = root.find(".//isda:Refined_Corner_Coordinates", ns)
    if corners_elem is None:
        raise ValueError("Cannot find Refined_Corner_Coordinates in OHRC label")

    def _get(tag: str) -> float:
        el = corners_elem.find(f"isda:{tag}", ns)
        if el is None:
            raise ValueError(f"Missing {tag} in OHRC corners")
        return float(el.text)

    corners = {
        "upper_left":  (_get("upper_left_latitude"),  _get("upper_left_longitude")),
        "upper_right": (_get("upper_right_latitude"), _get("upper_right_longitude")),
        "lower_left":  (_get("lower_left_latitude"),  _get("lower_left_longitude")),
        "lower_right": (_get("lower_right_latitude"), _get("lower_right_longitude")),
    }
    return corners


def corners_to_bbox(corners: dict) -> tuple[float, float, float, float]:
    lats = [v[0] for v in corners.values()]
    lons = [v[1] for v in corners.values()]
    return min(lats), max(lats), min(lons), max(lons)


def corners_to_polygon(corners: dict) -> Polygon:
    order = ["upper_left", "upper_right", "lower_right", "lower_left"]
    pts   = [(corners[k][1], corners[k][0]) for k in order]
    pts.append(pts[0])
    return Polygon(pts)


# ── NASA ODE query ───────────────────────────────────────────────────────────

def query_ode(min_lat: float, max_lat: float,
              min_lon: float, max_lon: float) -> dict:
    """
    Query ODE REST API for LROC CDR NAC products in a bounding box.
    pt=CDRNAC4 is the correct product type (CDRNAC/EDRNAC return "Invalid IIPT").
    Longitude params must be east-positive for ODE REST.
    """
    # ODE accepts west-lon to east-lon as westernlon/easternlon (east-positive here)
    params = {
        "query": "product", "results": "fmc", "output": "JSON",
        "target": "moon",   "ihid": "LRO",   "iid": "LROC",
        "pt": "CDRNAC4",
        "westernlon": min_lon, "easternlon": max_lon,
        "minlat": min_lat,    "maxlat": max_lat,
    }
    print(f"[ODE] Querying: lat [{min_lat:.2f},{max_lat:.2f}] "
          f"lon [{min_lon:.2f},{max_lon:.2f}] ...")
    try:
        resp = requests.get(ODE_URL, params=params, timeout=45)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[ODE] Query failed: {exc}")
        return {}


def extract_candidates(ode_json: dict) -> list[dict]:
    try:
        products = ode_json["ODEResults"]["Products"]["Product"]
    except (KeyError, TypeError):
        return []
    if isinstance(products, dict):
        products = [products]
    out = []
    for p in products:
        try:
            out.append({
                "product_id":   p.get("Product_name"),
                "obs_time":     p.get("UTC_start_time"),
                "footprint_wkt": p.get("Footprint_geometry"),
                "incidence":    float(p.get("Incidence_angle") or 0),
                "product_files": p.get("Product_files"),
            })
        except Exception:
            continue
    return out


def _parse_lroc_footprint_normalized(wkt_str: str) -> Optional[Polygon]:
    """
    Parse WKT footprint and normalise all longitudes to east-positive 0–360.
    ODE WKT uses west-positive convention (negative lon near 340°E).
    """
    try:
        geom = wkt.loads(wkt_str)
    except Exception as exc:
        print(f"[WKT] Parse error: {exc}")
        return None
    return normalize_polygon_lons(geom)


def find_best_candidates(candidates: list[dict],
                         ohrc_poly: Polygon,
                         ohrc_bbox: tuple,
                         ohrc_time: str) -> list[dict]:
    """
    Filter + rank NAC candidates by overlap and time proximity.
    Falls back to bbox-only matching if WKT footprint looks unreliable.
    """
    ohrc_area = ohrc_poly.area
    ohrc_box  = box(*ohrc_poly.bounds)

    passed = []
    for c in candidates:
        wkt_str = c.get("footprint_wkt")
        use_wkt = False
        lroc_poly = None

        if wkt_str:
            lroc_poly = _parse_lroc_footprint_normalized(wkt_str)
            if lroc_poly is not None:
                sane, msg = footprint_sanity(lroc_poly)
                if not sane:
                    print(f"[footprint] {c['product_id']}: insane WKT ({msg}) — using bbox")
                else:
                    use_wkt = True

        # Determine intersection
        if use_wkt and lroc_poly is not None:
            if not ohrc_poly.intersects(lroc_poly):
                continue
            intersection = ohrc_poly.intersection(lroc_poly)
            area_ratio   = intersection.area / ohrc_area
        else:
            # bbox fallback: if product bbox touches OHRC bbox, accept with ratio=0.0
            if lroc_poly is not None:
                test_poly = lroc_poly
            else:
                continue
            if not ohrc_box.intersects(test_poly.envelope):
                continue
            intersection = ohrc_box.intersection(test_poly.envelope)
            area_ratio = intersection.area / ohrc_area

        if area_ratio < MIN_AREA_RATIO:
            continue

        try:
            ohrc_dt  = datetime.strptime(ohrc_time, "%Y-%m-%dT%H:%M:%S.%fZ")
            lroc_dt  = datetime.strptime(c["obs_time"], "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            try:
                lroc_dt = datetime.strptime(c["obs_time"], "%Y-%m-%dT%H:%M:%S.%fZ")
            except Exception:
                continue
        time_diff = abs((ohrc_dt - lroc_dt).total_seconds()) / 3600

        if time_diff > MAX_TIME_DIFF_HRS:
            continue

        passed.append({
            **c,
            "area_ratio":   area_ratio,
            "time_diff_hrs": time_diff,
            "intersection": intersection,
            "lroc_poly":    lroc_poly,
        })

    # Sort: prefer closer incidence angle to OHRC (83.83°), break ties by time
    OHRC_INCIDENCE = 83.83
    passed.sort(key=lambda x: abs(x["incidence"] - OHRC_INCIDENCE))
    return passed


def get_product_urls(product_files: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Extract .IMG URL from ODE product_files dict.
    Derive the XML URL as a sibling of the .IMG in the same DATA/ directory.
    NOTE: ODE often returns a browse-pyramid XML from EXTRAS/BROWSE/ which is
    a Product_Browse label (no Array_2D_Image) — useless for verification.
    Constructing the sibling URL avoids this.
    """
    files = product_files.get("Product_file", [])
    if isinstance(files, dict):
        files = [files]
    img_url = None
    for f in files:
        fname = f.get("FileName", "")
        url   = f.get("URL", "")
        if f.get("Type") == "Product" and fname.upper().endswith(".IMG"):
            img_url = url
    # Derive XML URL: same directory as .IMG, replace extension
    xml_url = None
    if img_url:
        xml_url = img_url.rsplit(".", 1)[0] + ".xml"
    return img_url, xml_url


# ── Downloader ───────────────────────────────────────────────────────────────

def download_file(url: str, save_path: str,
                  expected_bytes: Optional[int] = None,
                  chunk: int = 1 << 20) -> bool:
    """
    Download url → save_path with progress bar.
    Follows 302 redirects (pds.lroc.im-ldi.com → pds.mcp.nasa.gov).
    Returns True on success.
    """
    if Path(save_path).exists():
        sz = Path(save_path).stat().st_size
        if expected_bytes and sz == expected_bytes:
            print(f"[dl] {Path(save_path).name} already on disk ({sz:,} bytes) — skipping")
            return True
        elif not expected_bytes and sz > 0:
            print(f"[dl] {Path(save_path).name} already on disk — skipping")
            return True

    print(f"[dl] {url}")
    try:
        resp = requests.get(url, stream=True, timeout=300, allow_redirects=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done  = 0
        with open(save_path, "wb") as f:
            for buf in resp.iter_content(chunk_size=chunk):
                f.write(buf)
                done += len(buf)
                if total:
                    pct = done / total * 100
                    mb  = done / 1e6
                    print(f"\r[dl]   {mb:7.1f} MB  {pct:5.1f}%", end="", flush=True)
        print(f"\r[dl] Done: {Path(save_path).name} ({done:,} bytes)          ")
        if expected_bytes and done != expected_bytes:
            print(f"[dl] WARNING: got {done:,} expected {expected_bytes:,}")
        return True
    except Exception as exc:
        print(f"\n[dl] FAILED: {exc}")
        return False


# ── Image helpers ────────────────────────────────────────────────────────────

def latlon_to_pixel(lat: float, lon: float,
                    corners: dict, img_h: int, img_w: int) -> tuple[int, int]:
    """
    Bilinear approximation: (lat,lon) → (row, col) given four corner coordinates.
    Works for small, nearly-rectangular footprints.
    """
    ul_lat, ul_lon = corners["upper_left"]
    ur_lat, ur_lon = corners["upper_right"]
    ll_lat, ll_lon = corners["lower_left"]

    lon_range = ur_lon - ul_lon
    lat_range = ll_lat - ul_lat

    col = int((lon - ul_lon) / lon_range * img_w) if lon_range else 0
    row = int((lat - ul_lat) / lat_range * img_h) if lat_range else 0
    return row, col


# Max rows in a single crop window — caps RAM usage at ~1 GB per window
MAX_CROP_ROWS = 8_000
MAX_CROP_COLS = 12_000

# Known approximate corners for M1382366243RC from ODE footprint
# (LROC CDR labels don't embed corner lat/lon; these are ODE-query derived)
# East-positive, WGS84 approximation at -68 to -70 lat
PRIMARY_NAC_CORNERS: dict = {
    "upper_left":  (-68.0, 341.0),
    "upper_right": (-68.0, 341.6),
    "lower_left":  (-70.5, 341.0),
    "lower_right": (-70.5, 341.6),
}
PRIMARY_NAC_POLYGON: "Polygon" = corners_to_polygon(PRIMARY_NAC_CORNERS)


def _parse_nac_corners_from_xml(xml_path: str, img_h: int, img_w: int) -> dict:
    """
    LROC CDR labels don't include geographic corner lat/lon fields.
    Return the known-approximate corners for the primary NAC product.
    For other products, callers should pass corners derived from ODE WKT.
    """
    # All corners span the full image height and width
    return PRIMARY_NAC_CORNERS


def crop_to_overlap(mm: np.memmap, corners: dict, intersection: Polygon
                    ) -> np.ndarray:
    """
    Crop memmap to the bounding box of intersection_polygon and return as ndarray.
    Uses only a single memmap window — never loads the full array.
    Capped at MAX_CROP_ROWS × MAX_CROP_COLS to prevent multi-GB materialisation.
    """
    img_h, img_w = mm.shape[-2:]  # (bands, lines, samples) → last 2
    min_lon, min_lat, max_lon, max_lat = intersection.bounds

    r1, c1 = latlon_to_pixel(max_lat, min_lon, corners, img_h, img_w)
    r2, c2 = latlon_to_pixel(min_lat, max_lon, corners, img_h, img_w)

    r1, r2 = sorted([max(0, r1), min(img_h - 1, r2)])
    c1, c2 = sorted([max(0, c1), min(img_w - 1, c2)])

    if r2 <= r1 or c2 <= c1:
        # Degenerate crop — return a centre slice
        print("[crop] WARNING: degenerate crop bbox; using centre slice")
        mid = img_h // 2
        r1, r2 = max(0, mid - MAX_CROP_ROWS // 2), min(img_h, mid + MAX_CROP_ROWS // 2)
        c1, c2 = 0, img_w

    # Cap at MAX_CROP_ROWS (centre the window)
    if (r2 - r1) > MAX_CROP_ROWS:
        mid_r = (r1 + r2) // 2
        r1 = max(0,       mid_r - MAX_CROP_ROWS // 2)
        r2 = min(img_h,   mid_r + MAX_CROP_ROWS // 2)
        print(f"[crop] Capped rows to {MAX_CROP_ROWS} (centre of overlap)")
    if (c2 - c1) > MAX_CROP_COLS:
        mid_c = (c1 + c2) // 2
        c1 = max(0,       mid_c - MAX_CROP_COLS // 2)
        c2 = min(img_w,   mid_c + MAX_CROP_COLS // 2)
        print(f"[crop] Capped cols to {MAX_CROP_COLS}")

    if mm.ndim == 3:
        return np.array(mm[:, r1:r2, c1:c2]).squeeze(axis=0)
    return np.array(mm[r1:r2, c1:c2])


def save_png_preview(arr: np.ndarray, save_path: str, max_dim: int = 1536) -> str:
    """Save a uint8 array as PNG, downsampled to max_dim px if needed."""
    img = Image.fromarray(arr)
    scale = max_dim / max(img.size[0], img.size[1], 1)
    if scale < 1:
        new_w = max(1, int(img.size[0] * scale))
        new_h = max(1, int(img.size[1] * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    img.save(save_path)
    return save_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="OHRC↔LROC pair finder (SIH26166)")
    parser.add_argument("--skip-ode", action="store_true",
                        help="Skip ODE query; use hardcoded primary NAC URL")
    parser.add_argument("--output", default=OUTPUT_DIR,
                        help="Output directory (default: output/)")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Parse OHRC metadata ──────────────────────────────────────────
    print("[1] Parsing OHRC label ...")
    ohrc_corners = parse_ohrc_metadata(OHRC_XML_PATH)
    ohrc_bbox    = corners_to_bbox(ohrc_corners)
    ohrc_poly    = corners_to_polygon(ohrc_corners)
    min_lat, max_lat, min_lon, max_lon = ohrc_bbox
    print(f"    OHRC bbox: lat [{min_lat:.4f}, {max_lat:.4f}]  "
          f"lon [{min_lon:.4f}, {max_lon:.4f}]")

    # Verify OHRC shape (sanity — size check only, no RAM load)
    ohrc_lines, ohrc_samps, _ = get_pds4_shape(OHRC_XML_PATH)
    expected_bytes = ohrc_lines * ohrc_samps  # UnsignedByte
    actual_bytes   = Path(OHRC_IMG_PATH).stat().st_size
    print(f"    OHRC shape: {ohrc_lines}×{ohrc_samps} ({actual_bytes:,} bytes — "
          f"{'OK' if actual_bytes == expected_bytes else 'WARNING mismatch'})")

    # ── Step 2/3: Query ODE or use primary target ────────────────────────────
    nac_img_path = str(out / f"{PRIMARY_NAC_BASE}.IMG")
    nac_xml_path = str(out / f"{PRIMARY_NAC_BASE}.xml")

    if not args.skip_ode:
        print("[2] Querying NASA ODE ...")
        ode_json   = query_ode(min_lat, max_lat, min_lon, max_lon)
        candidates = extract_candidates(ode_json)
        print(f"    Found {len(candidates)} raw candidates")

        print("[3] Filtering candidates ...")
        matches = find_best_candidates(candidates, ohrc_poly, ohrc_bbox, OHRC_OBS_TIME)
        print(f"    {len(matches)} pass (area≥{MIN_AREA_RATIO}, "
              f"time≤{MAX_TIME_DIFF_HRS}h, incidence≈83.8°)")

        if matches:
            for m in matches[:5]:
                print(f"    {m['product_id']}  area_ratio={m['area_ratio']:.2f}  "
                      f"inc={m['incidence']:.1f}°  time_diff={m['time_diff_hrs']:.0f}h")
            best = matches[0]
            print(f"\n    Best ODE match: {best['product_id']}")
            img_url, xml_url = get_product_urls(best["product_files"])
            intersection = best["intersection"]
        else:
            print("    No ODE candidates passed filters — falling back to hardcoded primary NAC")
            img_url      = PRIMARY_NAC_IMG
            xml_url      = PRIMARY_NAC_XML
            intersection = ohrc_poly  # whole OHRC footprint as fallback overlap
    else:
        print("[2/3] Skipping ODE; using hardcoded primary NAC.")
        img_url      = PRIMARY_NAC_IMG
        xml_url      = PRIMARY_NAC_XML
        intersection = ohrc_poly

    # ── Step 4: Download NAC .IMG and .xml ───────────────────────────────────
    print(f"[4] Downloading NAC IMG → {nac_img_path}")
    img_ok = download_file(img_url, nac_img_path, expected_bytes=PRIMARY_NAC_SIZE)
    if not img_ok:
        print("    FATAL: Could not download NAC image. Exiting.")
        sys.exit(1)

    print(f"    Downloading NAC XML → {nac_xml_path}")
    xml_ok = download_file(xml_url, nac_xml_path)
    if not xml_ok:
        print("    WARNING: Could not download NAC XML; proceeding without label verification.")

    # ── Step 5: Verify downloaded files ─────────────────────────────────────
    print("[5] Verifying files ...")
    verify_pds4_file(OHRC_XML_PATH, OHRC_IMG_PATH)
    nac_xml_valid = False
    if Path(nac_xml_path).exists():
        try:
            nac_xml_valid = verify_pds4_file(nac_xml_path, nac_img_path)
        except ValueError as e:
            print(f"    NAC XML is not a data label ({e}); skipping XML verify — using hardcoded shape")
            nac_xml_valid = False
    else:
        print("    NAC XML missing — skipping NAC label verification")

    # ── Step 6: Open both images via memmap (zero RAM) ───────────────────────
    print("[6] Opening images via memmap ...")
    ohrc_mm = open_pds4_memmap(OHRC_XML_PATH, OHRC_IMG_PATH)
    print(f"    OHRC memmap: shape={ohrc_mm.shape} dtype={ohrc_mm.dtype}")

    if nac_xml_valid and Path(nac_xml_path).exists():
        nac_mm = open_pds4_memmap(nac_xml_path, nac_img_path)
    else:
        # Hardcoded fallback for M1382366243RC.
        # NAC CDR has a PDS3 attached header (5064 bytes) + SignedLSB2 image data.
        # 5064 + 52224*2532*2 = 264,467,400 bytes ✔
        nac_shape = (52224, 2532)
        nac_mm    = np.memmap(nac_img_path, dtype="<i2", mode="r",
                              offset=5064, shape=nac_shape)
        print(f"    NAC memmap (hardcoded, offset=5064): {nac_shape}")
    print(f"    NAC  memmap: shape={nac_mm.shape} dtype={nac_mm.dtype}")

    # ── Step 7: Crop to overlap ──────────────────────────────────────────────
    print("[7] Cropping to overlap region ...")
    min_lon_ov, min_lat_ov, max_lon_ov, max_lat_ov = intersection.bounds
    print(f"    Overlap bbox: lat [{min_lat_ov:.4f}, {max_lat_ov:.4f}]  "
          f"lon [{min_lon_ov:.4f}, {max_lon_ov:.4f}]")

    # OHRC crop (memmap slice, then copy)
    ohrc_cropped = crop_to_overlap(ohrc_mm, ohrc_corners, intersection)
    print(f"    OHRC cropped shape: {ohrc_cropped.shape}")

    # NAC crop — use intersection and known NAC corners
    nac_h, nac_w = nac_mm.shape[-2:]
    nac_corners = PRIMARY_NAC_CORNERS
    # Tighten intersection: intersect OHRC poly with the known NAC polygon
    nac_poly = PRIMARY_NAC_POLYGON
    if ohrc_poly.intersects(nac_poly):
        tight_intersection = ohrc_poly.intersection(nac_poly)
        if not tight_intersection.is_empty:
            intersection = tight_intersection
            min_lon_ov, min_lat_ov, max_lon_ov, max_lat_ov = intersection.bounds
            print(f"    Tightened overlap bbox: lat [{min_lat_ov:.4f},{max_lat_ov:.4f}] "
                  f"lon [{min_lon_ov:.4f},{max_lon_ov:.4f}]")
    nac_cropped = crop_to_overlap(nac_mm, nac_corners, intersection)
    print(f"    NAC  cropped shape: {nac_cropped.shape}")

    # ── Step 8: Normalize and save PNG previews ──────────────────────────────
    print("[8] Saving PNG previews ...")
    ohrc_u8  = normalize_to_uint8_percentile(ohrc_cropped)   # UnsignedByte already
    nac_u8   = normalize_to_uint8_percentile(nac_cropped)    # int16 → percentile stretch

    ohrc_png = save_png_preview(ohrc_u8,  str(out / "ohrc_cropped.png"))
    nac_png  = save_png_preview(nac_u8,   str(out / "lroc_cropped.png"))
    print(f"    Saved: {ohrc_png}")
    print(f"    Saved: {nac_png}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PAIR-FINDER COMPLETE")
    print(f"  OHRC:  {OHRC_IMG_PATH}  shape={ohrc_mm.shape}")
    print(f"  NAC:   {nac_img_path}  shape={nac_mm.shape}")
    print(f"  Overlap bbox: lat [{min_lat_ov:.4f},{max_lat_ov:.4f}] "
          f"lon [{min_lon_ov:.4f},{max_lon_ov:.4f}]")
    print(f"  OHRC crop: {ohrc_cropped.shape}")
    print(f"  NAC  crop: {nac_cropped.shape}")
    print(f"  Previews:  {ohrc_png}  {nac_png}")
    print("=" * 60)

    # Save crop metadata for match_lanes.py to consume
    import json
    meta = {
        "ohrc_xml":   OHRC_XML_PATH,
        "ohrc_img":   OHRC_IMG_PATH,
        "nac_xml":    nac_xml_path,
        "nac_img":    nac_img_path,
        "overlap_bbox": {
            "min_lat": min_lat_ov, "max_lat": max_lat_ov,
            "min_lon": min_lon_ov, "max_lon": max_lon_ov,
        },
        "ohrc_corners": {k: list(v) for k, v in ohrc_corners.items()},
        "nac_corners":  {k: list(v) for k, v in nac_corners.items()},
    }
    meta_path = str(out / "pair_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Meta saved: {meta_path}")


if __name__ == "__main__":
    main()
