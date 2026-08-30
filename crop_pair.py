"""
crop_pair.py — crop the verified OHRC/LROC pair down to their overlap
region and save both as normalized PNGs for SuperPoint/LightGlue.

Uses pds4_reader's memmap-based read_pds4_window() — neither full image
is ever loaded into RAM, only the small overlap window from each.

FIX (this version): normalization now excludes fill/no-data pixels
(e.g. LROC's -32768 sentinel) from the percentile calculation. Without
this, if fill pixels make up more than ~1% of a crop, np.percentile()
at the low end lands inside the fill population and returns something
close to -32768 itself — which massively inflates the stretch range
and crushes all real pixel values into a narrow band near 255 (a
blank-looking white image), even though the underlying data is fine.
"""

import numpy as np
from PIL import Image
from shapely import wkt
from shapely.geometry import Polygon

from pds4_reader import read_pds4_window, get_pds4_shape

# ── CONFIG — the verified pair from this session ────────────────────────────
OHRC_XML_PATH = "ch2_ohr_ncp_20210405T0442095110_d_img_d18.xml"   # <- confirm this matches your real filename
OHRC_IMG_PATH = "ch2_ohr_ncp_20210405T0442095110_d_img_d18.img"  # <- confirm real filename

LROC_XML_PATH = "output/lroc/M1365934552LC.xml"
LROC_IMG_PATH = "output/lroc/M1365934552LC.IMG"

# OHRC refined corners (lat, lon) — from the label, known orientation
OHRC_CORNERS = {
    'upper_left':  (-68.217976, 341.273442),
    'upper_right': (-68.218462, 341.542943),
    'lower_left':  (-69.061928, 341.204145),
    'lower_right': (-69.056313, 341.484142),
}

# LROC footprint polygon from ODE (lon, lat order, matches shapely convention)
# NOTE: unlike OHRC, we don't know which vertex corresponds to which image
# corner/orientation — ODE's footprint doesn't carry that. We fall back to
# treating the image as north-up (higher lat = row 0) over its own bbox.
# This is a known simplification — fine for a first real test, but not
# geometrically exact.
LROC_FOOTPRINT_WKT = "POLYGON ((340.77 -69.31, 341.04 -68.85, 341.36 -68.87, 341.1 -69.33, 340.77 -69.31))"

# Fill/no-data sentinel value used by LROC CDR products at strip edges.
# OHRC (UnsignedByte) has no such sentinel, so fill_value=None for it.
LROC_FILL_VALUE = -32768

OUTPUT_OHRC_PNG = "output/ohrc_cropped.png"
OUTPUT_LROC_PNG = "output/lroc_cropped.png"
# ─────────────────────────────────────────────────────────────────────────────


def normalize_to_uint8_percentile_masked(data, lo=1.0, hi=99.5, fill_value=None):
    """
    Percentile-clip stretch to uint8, excluding fill/no-data pixels from
    the percentile calculation itself. See module docstring for why this
    matters.

    fill_value=None means "no fill value to mask" — behaves like a plain
    percentile stretch (safe default for data with no sentinel, e.g. OHRC).
    """
    data = data.astype(np.float32)

    if fill_value is not None:
        valid_mask = data != fill_value
    else:
        valid_mask = np.ones_like(data, dtype=bool)

    if not valid_mask.any():
        return np.zeros_like(data, dtype=np.uint8)

    valid_data = data[valid_mask]
    p_lo = float(np.percentile(valid_data, lo))
    p_hi = float(np.percentile(valid_data, hi))

    if p_hi == p_lo:
        out = np.zeros_like(data, dtype=np.uint8)
    else:
        stretched = np.clip((data - p_lo) / (p_hi - p_lo) * 255.0, 0, 255)
        out = stretched.astype(np.uint8)

    if fill_value is not None:
        out[~valid_mask] = 0  # fill pixels rendered as black, not squashed into the stretch

    return out


def ohrc_polygon_from_corners(corners):
    ring_order = ['upper_left', 'upper_right', 'lower_right', 'lower_left']
    points = [(corners[n][1], corners[n][0]) for n in ring_order]  # (lon, lat)
    points.append(points[0])
    return Polygon(points)


def latlon_to_ohrc_pixel(lat, lon, corners, img_lines, img_samples):
    """
    Affine approx using upper_left/upper_right/lower_left corners
    (parallelogram assumption — ignores rotation-induced skew from
    lower_right; a known simplification, fine for the resolution here).
    """
    ul_lat, ul_lon = corners['upper_left']
    ur_lat, ur_lon = corners['upper_right']
    ll_lat, ll_lon = corners['lower_left']
    lon_range = ur_lon - ul_lon
    lat_range = ll_lat - ul_lat
    col = int((lon - ul_lon) / lon_range * img_samples)
    row = int((lat - ul_lat) / lat_range * img_lines)
    return row, col


def latlon_to_lroc_pixel(lat, lon, min_lat, max_lat, min_lon, max_lon, img_lines, img_samples):
    """
    Simple bbox-based affine mapping for LROC (north-up assumption).
    See module docstring note on LROC_FOOTPRINT_WKT above.
    """
    row = int((max_lat - lat) / (max_lat - min_lat) * img_lines)
    col = int((lon - min_lon) / (max_lon - min_lon) * img_samples)
    return row, col


def main():
    # ── OHRC geometry ──
    ohrc_lines, ohrc_samples, _ = get_pds4_shape(OHRC_XML_PATH)
    ohrc_poly = ohrc_polygon_from_corners(OHRC_CORNERS)
    print(f"OHRC shape: lines={ohrc_lines}, samples={ohrc_samples}")

    # ── LROC geometry ──
    lroc_lines, lroc_samples, _ = get_pds4_shape(LROC_XML_PATH)
    lroc_poly = wkt.loads(LROC_FOOTPRINT_WKT)
    lroc_min_lon, lroc_min_lat, lroc_max_lon, lroc_max_lat = lroc_poly.bounds
    print(f"LROC shape: lines={lroc_lines}, samples={lroc_samples}")

    # ── Real overlap polygon (precise, not just bbox) ──
    if not ohrc_poly.intersects(lroc_poly):
        raise RuntimeError("Polygons don't actually intersect — check inputs.")
    intersection = ohrc_poly.intersection(lroc_poly)
    min_lon, min_lat, max_lon, max_lat = intersection.bounds
    print(f"Overlap bounds: lat [{min_lat:.4f}, {max_lat:.4f}], "
          f"lon [{min_lon:.4f}, {max_lon:.4f}]")
    print(f"Overlap area ratio (of OHRC): {intersection.area / ohrc_poly.area:.3f}")

    # ── Convert overlap bounds to pixel windows on each image ──
    ohrc_row1, ohrc_col1 = latlon_to_ohrc_pixel(max_lat, min_lon, OHRC_CORNERS, ohrc_lines, ohrc_samples)
    ohrc_row2, ohrc_col2 = latlon_to_ohrc_pixel(min_lat, max_lon, OHRC_CORNERS, ohrc_lines, ohrc_samples)
    ohrc_row1, ohrc_row2 = sorted([max(0, ohrc_row1), min(ohrc_lines, ohrc_row2)])
    ohrc_col1, ohrc_col2 = sorted([max(0, ohrc_col1), min(ohrc_samples, ohrc_col2)])
    print(f"OHRC crop window: rows [{ohrc_row1}:{ohrc_row2}], cols [{ohrc_col1}:{ohrc_col2}] "
          f"({ohrc_row2-ohrc_row1} x {ohrc_col2-ohrc_col1} px)")

    lroc_row1, lroc_col1 = latlon_to_lroc_pixel(max_lat, min_lon, lroc_min_lat, lroc_max_lat,
                                                 lroc_min_lon, lroc_max_lon, lroc_lines, lroc_samples)
    lroc_row2, lroc_col2 = latlon_to_lroc_pixel(min_lat, max_lon, lroc_min_lat, lroc_max_lat,
                                                 lroc_min_lon, lroc_max_lon, lroc_lines, lroc_samples)
    lroc_row1, lroc_row2 = sorted([max(0, lroc_row1), min(lroc_lines, lroc_row2)])
    lroc_col1, lroc_col2 = sorted([max(0, lroc_col1), min(lroc_samples, lroc_col2)])
    print(f"LROC crop window: rows [{lroc_row1}:{lroc_row2}], cols [{lroc_col1}:{lroc_col2}] "
          f"({lroc_row2-lroc_row1} x {lroc_col2-lroc_col1} px)")

    if ohrc_row2 <= ohrc_row1 or ohrc_col2 <= ohrc_col1:
        raise RuntimeError("OHRC crop window is empty — check corner mapping / overlap bounds.")
    if lroc_row2 <= lroc_row1 or lroc_col2 <= lroc_col1:
        raise RuntimeError("LROC crop window is empty — check corner mapping / overlap bounds.")

    # ── Read only the overlap windows (memmap — never loads full arrays) ──
    print("Reading OHRC window...")
    ohrc_window = read_pds4_window(OHRC_XML_PATH, OHRC_IMG_PATH,
                                    ohrc_row1, ohrc_row2, ohrc_col1, ohrc_col2)
    print("Reading LROC window...")
    lroc_window = read_pds4_window(LROC_XML_PATH, LROC_IMG_PATH,
                                    lroc_row1, lroc_row2, lroc_col1, lroc_col2)

    # windows come back as (bands, rows, cols) — drop the band axis for single-band data
    ohrc_2d = ohrc_window[0] if ohrc_window.ndim == 3 else ohrc_window
    lroc_2d = lroc_window[0] if lroc_window.ndim == 3 else lroc_window

    # ── Report fill-pixel fraction before normalizing, so it's visible in the log ──
    lroc_fill_frac = np.mean(lroc_2d == LROC_FILL_VALUE)
    print(f"LROC crop fill-value fraction: {lroc_fill_frac:.3%}")

    # ── Normalize (fill-aware) + save ──
    ohrc_png = normalize_to_uint8_percentile_masked(ohrc_2d, fill_value=None)
    lroc_png = normalize_to_uint8_percentile_masked(lroc_2d, fill_value=LROC_FILL_VALUE)

    Image.fromarray(ohrc_png).save(OUTPUT_OHRC_PNG)
    Image.fromarray(lroc_png).save(OUTPUT_LROC_PNG)
    print(f"\nSaved: {OUTPUT_OHRC_PNG}  ({ohrc_png.shape})")
    print(f"Saved: {OUTPUT_LROC_PNG}  ({lroc_png.shape})")


if __name__ == "__main__":
    main()