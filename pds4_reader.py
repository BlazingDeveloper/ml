"""
pds4_reader.py — shared PDS4 (and minimal PDS3) image reader.

All OHRC and LROC CDR products relevant to SIH26166 are PDS4.
The .IMG is raw headerless binary; shape/dtype live in the paired .xml label.

KEY DESIGN: uses np.memmap (mode='r') so the 1.2 GB OHRC file is NEVER
fully loaded into RAM.  All callers must slice/tile, never index with [:].
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import re
import numpy as np

# ── PDS4 XML namespaces ──────────────────────────────────────────────────────
PDS4_NS: dict[str, str] = {
    "pds": "http://pds.nasa.gov/pds4/pds/v1",
}

# Map PDS4 data_type strings → numpy dtype strings
PDS4_DTYPE_MAP: dict[str, str] = {
    "UnsignedByte":     ">u1",
    "UnsignedLSB2":     "<u2",
    "UnsignedMSB2":     ">u2",
    "UnsignedLSB4":     "<u4",
    "UnsignedMSB4":     ">u4",
    "SignedLSB2":       "<i2",
    "SignedMSB2":       ">i2",
    "SignedLSB4":       "<i4",
    "SignedMSB4":       ">i4",
    "IEEE754LSBSingle": "<f4",
    "IEEE754MSBSingle": ">f4",
    "IEEE754LSBDouble": "<f8",
    "IEEE754MSBDouble": ">f8",
}


# ── Public API ───────────────────────────────────────────────────────────────

def open_pds4_memmap(xml_path, img_path):
    """
    Parse a PDS4 label and return a read-only memmap for the image array.
    Shape: (lines, samples) or (bands, lines, samples).
    NOT loaded into RAM — slicing triggers page-faults as needed.
    """
    lines, samples, bands, dtype, offset = _parse_pds4_label(xml_path)
    shape = (bands, lines, samples)  # always 3D — keeps all coordinates in output
    mm = np.memmap(img_path, dtype=dtype, mode="r", offset=offset, shape=shape)
    return mm


def read_pds4_window(xml_path, img_path, row_start, row_end, col_start, col_end):
    """
    Read a rectangular window without loading the whole file.
    Returns a plain ndarray (copied from the memmap window).
    """
    mm = open_pds4_memmap(xml_path, img_path)
    if mm.ndim == 2:
        window = mm[row_start:row_end, col_start:col_end]
    else:
        window = mm[:, row_start:row_end, col_start:col_end]
    return np.array(window)


def get_pds4_shape(xml_path):
    """Return (lines, samples, bands) — reads only the XML, never the .IMG."""
    lines, samples, bands, _, _ = _parse_pds4_label(xml_path)
    return lines, samples, bands


def verify_pds4_file(xml_path, img_path):
    """
    Verify that the .IMG file on disk matches the size declared in the label.
    Prints a one-line report. Returns True on success, False on mismatch.
    """
    lines, samples, bands, dtype, offset = _parse_pds4_label(xml_path)
    expected_bytes = lines * samples * bands * np.dtype(dtype).itemsize + offset
    actual_bytes = Path(img_path).stat().st_size
    declared = _declared_file_size(xml_path)

    ok = actual_bytes >= expected_bytes
    status = "OK" if ok else "MISMATCH"
    msg = (f"[pds4] {status}: {Path(img_path).name} "
           f"actual={actual_bytes:,} expected>={expected_bytes:,}")
    if declared:
        msg += f" declared={declared:,}"
    print(msg)
    if not ok:
        print(f"[pds4] WARNING: truncated? shape=({lines},{samples},{bands}) "
              f"dtype={dtype} offset={offset}")
    return ok


def normalize_to_uint8_percentile(data, lo=1.0, hi=99.5):
    """
    Percentile-clip stretch to uint8.
    Uses percentile not min/max: hot pixels in int16 NAC data would otherwise
    crush the visible range to near-zero with simple min-max.
    """
    data = data.astype(np.float32)
    p_lo = float(np.percentile(data, lo))
    p_hi = float(np.percentile(data, hi))
    if p_hi == p_lo:
        return np.zeros_like(data, dtype=np.uint8)
    data = np.clip((data - p_lo) / (p_hi - p_lo) * 255.0, 0, 255)
    return data.astype(np.uint8)


# ── PDS3 minimal fallback ────────────────────────────────────────────────────

def try_open_pds3(lbl_path, img_path):
    """
    Minimal PDS3 detached-label reader (no pvl/planetaryimage required).
    Returns a memmap (lines, samples) or None if parsing fails.
    """
    try:
        meta = _parse_pds3_label(lbl_path)
        lines    = int(meta["LINES"])
        samples  = int(meta["LINE_SAMPLES"])
        bits     = int(meta.get("SAMPLE_BITS", 8))
        sample_type = meta.get("SAMPLE_TYPE", "UNSIGNED_INTEGER").upper()

        dtype_map = {
            (8,  "UNSIGNED_INTEGER"): np.uint8,
            (16, "UNSIGNED_INTEGER"): ">u2",
            (16, "LSB_INTEGER"):      "<i2",
            (16, "MSB_INTEGER"):      ">i2",
            (32, "IEEE_REAL"):        ">f4",
        }
        dtype = dtype_map.get((bits, sample_type))
        if dtype is None:
            print(f"[pds3] Unknown SAMPLE_BITS={bits} SAMPLE_TYPE={sample_type}")
            return None

        label_recs   = int(meta.get("LABEL_RECORDS", 0))
        record_bytes = int(meta.get("RECORD_BYTES", 0))
        offset = label_recs * record_bytes

        shape = (lines, samples)
        mm = np.memmap(img_path, dtype=dtype, mode="r", offset=offset, shape=shape)
        print(f"[pds3] Opened {Path(img_path).name} shape={shape} dtype={dtype}")
        return mm

    except Exception as exc:
        print(f"[pds3] Fallback failed for {lbl_path}: {exc}")
        return None


# ── Internal helpers ─────────────────────────────────────────────────────────

def _parse_pds4_label(xml_path):
    """
    Returns (lines, samples, bands, dtype, byte_offset).
    Uses Clark notation {uri}tag for all searches — works regardless of
    whether the label declares a default namespace or uses explicit prefixes.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Detect the PDS4 namespace URI from the root tag (may vary by version)
    m = re.match(r'\{(.+?)\}', root.tag)
    ns_uri = m.group(1) if m else "http://pds.nasa.gov/pds4/pds/v1"
    N = lambda tag: f"{{{ns_uri}}}{tag}"   # Clark notation helper

    array_elem = (root.find(f".//{N('Array_2D_Image')}")
                  or root.find(f".//{N('Array_3D_Image')}"))
    if array_elem is None:
        raise ValueError(f"No Array_2D_Image / Array_3D_Image in {xml_path}")

    data_type_elem = array_elem.find(f"{N('Element_Array')}/{N('data_type')}")
    if data_type_elem is None:
        raise ValueError(f"Missing Element_Array/data_type in {xml_path}")
    data_type = data_type_elem.text.strip()

    offset_elem = array_elem.find(N("offset"))
    offset_bytes = int(offset_elem.text) if offset_elem is not None else 0

    axes = {}
    for axis in array_elem.findall(N("Axis_Array")):
        name_el  = axis.find(N("axis_name"))
        elems_el = axis.find(N("elements"))
        if name_el is not None and elems_el is not None:
            axes[name_el.text.strip()] = int(elems_el.text)

    lines   = axes.get("Line")   or axes.get("line")
    samples = axes.get("Sample") or axes.get("sample")
    bands   = int(axes.get("Band", 1) or 1)

    if lines is None or samples is None:
        raise ValueError(f"Cannot find Line/Sample axes in {xml_path}: {axes}")

    if data_type not in PDS4_DTYPE_MAP:
        raise ValueError(f"Unmapped PDS4 data_type '{data_type}'. Add to PDS4_DTYPE_MAP.")

    dtype = np.dtype(PDS4_DTYPE_MAP[data_type])
    return int(lines), int(samples), bands, dtype, int(offset_bytes)


def _declared_file_size(xml_path):
    """Return <file_size> from label if present, else None."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        m = re.match(r'\{(.+?)\}', root.tag)
        ns_uri = m.group(1) if m else "http://pds.nasa.gov/pds4/pds/v1"
        el = root.find(f".//{{{ns_uri}}}file_size")
        return int(el.text) if el is not None else None
    except Exception:
        return None


def _parse_pds3_label(lbl_path):
    """Bare-minimum PDS3 keyword=value parser."""
    meta = {}
    with open(lbl_path, "r", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("END") and "=" not in line:
                break
            if "=" in line and not line.startswith("/*"):
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').split("<")[0].strip()
                meta[key] = val
    return meta
