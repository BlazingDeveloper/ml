import os
import sys
import xml.etree.ElementTree as ET
import requests
import numpy as np
from datetime import datetime
from shapely import wkt
from shapely.geometry import Polygon
from PIL import Image


# ============================================================
# CONFIG — edit these for each run
# ============================================================
OHRC_XML_PATH = "ch2_ohr_ncp_20210405T0442095110_d_img_d18.xml"          # OHRC label file
OHRC_IMG_PATH = "ch2_ohr_ncp_20210405T0442095110_d_img_d18.img"          # OHRC actual image data file (set to your real filename)
OHRC_OBS_TIME = "2021-04-05T04:42:09.5110Z"  # from the XML's start_date_time
MIN_AREA_RATIO = 0.5      # candidate must cover at least 50% of OHRC footprint
MAX_TIME_DIFF_HRS = 2000  # candidate must be within ~83 days
OUTPUT_DIR = "output"
# ============================================================


# ------------------------------------------------------------
# PDS4 raw array reader (replaces planetaryimage.PDS3Image,
# which only understands PDS3-style embedded PVL labels —
# OHRC/LROC CDR products are PDS4: shape/dtype live in the XML,
# and the .IMG is raw headerless binary).
# ------------------------------------------------------------

PDS4_NS = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}

PDS4_DTYPE_MAP = {
    'UnsignedByte':        '>u1',
    'UnsignedLSB2':        '<u2',
    'UnsignedMSB2':        '>u2',
    'UnsignedLSB4':        '<u4',
    'UnsignedMSB4':        '>u4',
    'SignedLSB2':          '<i2',
    'SignedMSB2':          '>i2',
    'SignedLSB4':          '<i4',
    'SignedMSB4':          '>i4',
    'IEEE754LSBSingle':    '<f4',
    'IEEE754MSBSingle':    '>f4',
    'IEEE754LSBDouble':    '<f8',
    'IEEE754MSBDouble':    '>f8',
}


def read_pds4_image(xml_path, img_path):
    """
    Parse a PDS4 label's Array_2D_Image (or Array_3D_Image) element and load
    the corresponding raw binary data from img_path using numpy.

    Returns a numpy array shaped (lines, samples) or (bands, lines, samples).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    array_elem = root.find('.//pds:Array_2D_Image', PDS4_NS)
    if array_elem is None:
        array_elem = root.find('.//pds:Array_3D_Image', PDS4_NS)
    if array_elem is None:
        raise ValueError(
            f"No Array_2D_Image/Array_3D_Image element found in {xml_path}. "
            "Check the label's actual namespace/tag if this fires."
        )

    data_type = array_elem.find('pds:Element_Array/pds:data_type', PDS4_NS).text
    offset_elem = array_elem.find('pds:offset', PDS4_NS)
    offset_bytes = int(offset_elem.text) if offset_elem is not None else 0

    axes = {}
    for axis in array_elem.findall('pds:Axis_Array', PDS4_NS):
        name = axis.find('pds:axis_name', PDS4_NS).text
        elements = int(axis.find('pds:elements', PDS4_NS).text)
        axes[name] = elements

    lines = axes.get('Line')
    samples = axes.get('Sample')
    bands = axes.get('Band', 1)

    if data_type not in PDS4_DTYPE_MAP:
        raise ValueError(f"Unmapped PDS4 data_type '{data_type}' — add it to PDS4_DTYPE_MAP.")
    dtype = np.dtype(PDS4_DTYPE_MAP[data_type])

    shape = (bands, lines, samples) if bands > 1 else (lines, samples)
    count = int(np.prod(shape))

    with open(img_path, 'rb') as f:
        f.seek(offset_bytes)
        data = np.fromfile(f, dtype=dtype, count=count)

    data = data.reshape(shape)
    return data


# ------------------------------------------------------------
# OHRC metadata / geometry
# ------------------------------------------------------------

def parse_ohrc_metadata(xml_path):
    ns = {'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    corners_elem = root.find('.//isda:Refined_Corner_Coordinates', ns)
    corners = {
        'upper_left':  (float(corners_elem.find('isda:upper_left_latitude', ns).text),
                         float(corners_elem.find('isda:upper_left_longitude', ns).text)),
        'upper_right': (float(corners_elem.find('isda:upper_right_latitude', ns).text),
                         float(corners_elem.find('isda:upper_right_longitude', ns).text)),
        'lower_left':  (float(corners_elem.find('isda:lower_left_latitude', ns).text),
                         float(corners_elem.find('isda:lower_left_longitude', ns).text)),
        'lower_right': (float(corners_elem.find('isda:lower_right_latitude', ns).text),
                         float(corners_elem.find('isda:lower_right_longitude', ns).text)),
    }
    return corners


def corners_to_bbox(corners):
    lats = [lat for lat, lon in corners.values()]
    lons = [lon for lat, lon in corners.values()]
    return (min(lats), max(lats), min(lons), max(lons))


def ohrc_corners_to_polygon(corners):
    ring_order = ['upper_left', 'upper_right', 'lower_right', 'lower_left']
    points = [(corners[name][1], corners[name][0]) for name in ring_order]
    points.append(points[0])
    return Polygon(points)


# ------------------------------------------------------------
# NASA ODE (LROC) query
# ------------------------------------------------------------

def query_ode(min_lat, max_lat, min_lon, max_lon):
    url = "https://oderest.rsl.wustl.edu/live2/"
    params = {
        "query": "product",
        "results": "fmc",
        "output": "JSON",
        "target": "moon",
        "ihid": "LRO",
        "iid": "LROC",
        "pt": "CDRNAC4",
        "westernlon": min_lon,
        "easternlon": max_lon,
        "minlat": min_lat,
        "maxlat": max_lat,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_lroc_candidates(ode_json):
    candidates = []
    try:
        products = ode_json['ODEResults']['Products']['Product']
    except (KeyError, TypeError):
        return candidates
    if isinstance(products, dict):
        products = [products]
    for p in products:
        try:
            candidates.append({
                'product_id': p.get('Product_name'),
                'obs_time': p.get('UTC_start_time'),
                'footprint_wkt': p.get('Footprint_geometry'),
                'product_files': p.get('Product_files'),
            })
        except (TypeError, ValueError):
            continue
    return candidates


def parse_lroc_footprint(footprint_wkt_str):
    return wkt.loads(footprint_wkt_str)


def precise_overlap_area(poly1, poly2):
    if not poly1.intersects(poly2):
        return False, 0.0, None
    intersection = poly1.intersection(poly2)
    return True, intersection.area, intersection


def time_difference_hours(ohrc_time_str, lroc_time_str):
    ohrc_time = datetime.strptime(ohrc_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    lroc_time = datetime.strptime(lroc_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    return abs((ohrc_time - lroc_time).total_seconds()) / 3600


def find_best_match(candidates, ohrc_polygon, ohrc_time, min_area_ratio, max_time_diff_hrs):
    ohrc_area = ohrc_polygon.area
    passed = []
    for c in candidates:
        if not c['footprint_wkt']:
            continue
        try:
            lroc_polygon = parse_lroc_footprint(c['footprint_wkt'])
        except Exception:
            continue
        overlaps, area, intersection = precise_overlap_area(ohrc_polygon, lroc_polygon)
        if not overlaps:
            continue
        area_ratio = area / ohrc_area
        time_diff = time_difference_hours(ohrc_time, c['obs_time'])
        if area_ratio >= min_area_ratio and time_diff <= max_time_diff_hrs:
            passed.append({
                **c,
                'area_ratio': area_ratio,
                'time_diff_hrs': time_diff,
                'intersection': intersection,
                'lroc_polygon': lroc_polygon,
            })
    passed.sort(key=lambda x: x['time_diff_hrs'])
    return passed


def get_lroc_image_url(product_files):
    files = product_files.get('Product_file', [])
    if isinstance(files, dict):
        files = [files]
    for f in files:
        if f.get('Type') == 'Product' and f.get('FileName', '').upper().endswith('.IMG'):
            return f.get('URL')
    return None


def download_file(url, save_path):
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    downloaded = 0
    with open(save_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r{save_path}: {downloaded/total*100:.1f}%", end="")
    print()
    return save_path


# ------------------------------------------------------------
# Image processing: normalize, crop, save
# ------------------------------------------------------------

def normalize_to_uint8(data):
    data = data.astype(np.float32)
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-9) * 255
    return data_norm.astype(np.uint8)


def latlon_to_pixel_approx(lat, lon, corners, img_width, img_height):
    ul_lat, ul_lon = corners['upper_left']
    ur_lat, ur_lon = corners['upper_right']
    ll_lat, ll_lon = corners['lower_left']
    lon_range = ur_lon - ul_lon
    lat_range = ll_lat - ul_lat
    col = int((lon - ul_lon) / lon_range * img_width)
    row = int((lat - ul_lat) / lat_range * img_height)
    return row, col


def crop_to_overlap(image_array, corners, intersection_polygon):
    img_height, img_width = image_array.shape[:2]
    min_lon, min_lat, max_lon, max_lat = intersection_polygon.bounds
    row1, col1 = latlon_to_pixel_approx(max_lat, min_lon, corners, img_width, img_height)
    row2, col2 = latlon_to_pixel_approx(min_lat, max_lon, corners, img_width, img_height)
    row1, row2 = sorted([max(0, row1), min(img_height, row2)])
    col1, col2 = sorted([max(0, col1), min(img_width, col2)])
    return image_array[row1:row2, col1:col2]


def save_as_png(image_array, save_path, max_dim=1536):
    img = Image.fromarray(image_array)
    scale = max_dim / max(img.size)
    if scale < 1:
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    img.save(save_path)
    return save_path


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: parse OHRC metadata
    ohrc_corners = parse_ohrc_metadata(OHRC_XML_PATH)
    ohrc_bbox = corners_to_bbox(ohrc_corners)
    ohrc_polygon = ohrc_corners_to_polygon(ohrc_corners)
    print(f"OHRC bbox: {ohrc_bbox}")

    # Step 2: query NASA ODE for LROC candidates
    min_lat, max_lat, min_lon, max_lon = ohrc_bbox
    ode_json = query_ode(min_lat, max_lat, min_lon, max_lon)
    candidates = extract_lroc_candidates(ode_json)
    print(f"Found {len(candidates)} raw candidates")

    # Step 3: filter + rank by safety threshold, then time closeness
    matches = find_best_match(candidates, ohrc_polygon, OHRC_OBS_TIME,
                               MIN_AREA_RATIO, MAX_TIME_DIFF_HRS)
    print(f"{len(matches)} candidates pass safety threshold "
          f"(area>={MIN_AREA_RATIO}, time<={MAX_TIME_DIFF_HRS}h)\n")

    if not matches:
        print("No safe match found. Try loosening MIN_AREA_RATIO or MAX_TIME_DIFF_HRS.")
        sys.exit()

    for m in matches[:5]:
        print(f"{m['product_id']}  area_ratio={m['area_ratio']:.2f}  "
              f"time_diff_hrs={m['time_diff_hrs']:.1f}")

    best = matches[0]
    print(f"\nBest match: {best['product_id']}")

# Step 4: download the best LROC .IMG and XML files
img_url = get_lroc_image_url(best['product_files'])

if not img_url:
    print("No .IMG URL found for best match — stopping.")
    sys.exit()

# Find matching XML URL
xml_url = None
files = best['product_files'].get('Product_file', [])

if isinstance(files, dict):
    files = [files]

for f in files:
    filename = f.get('FileName', '')
    if filename.lower().endswith('.xml'):
        xml_url = f.get('URL')
        break

if not xml_url:
    print("No .xml URL found for best match — stopping.")
    sys.exit()

lroc_img_path = os.path.join(
    OUTPUT_DIR,
    best['product_id'] + ".IMG"
)

lroc_xml_path = os.path.join(
    OUTPUT_DIR,
    best['product_id'] + ".xml"
)

print(f"Downloading LROC IMG: {img_url}")
download_file(img_url, lroc_img_path)

print(f"Downloading LROC XML: {xml_url}")
download_file(xml_url, lroc_xml_path)
    # Step 5: load both images via the PDS4 reader
    print("Loading OHRC image...")
    ohrc_data = read_pds4_image(OHRC_XML_PATH, OHRC_IMG_PATH)
    print("Loading LROC image...")
    lroc_data = read_pds4_image(lroc_xml_path, lroc_img_path)

    # Step 6: crop both to the overlapping region
    print("Cropping to overlap region...")
    ohrc_cropped = crop_to_overlap(ohrc_data, ohrc_corners, best['intersection'])

    # NOTE: LROC crop corners are derived from the WKT bounding box (not true
    # rotated corners) — a known simplification, flagged for a later fix.
    lroc_corners_dict = {
        'upper_left': (best['lroc_polygon'].bounds[3], best['lroc_polygon'].bounds[0]),
        'upper_right': (best['lroc_polygon'].bounds[3], best['lroc_polygon'].bounds[2]),
        'lower_left': (best['lroc_polygon'].bounds[1], best['lroc_polygon'].bounds[0]),
    }
    lroc_cropped = crop_to_overlap(lroc_data, lroc_corners_dict, best['intersection'])

    # Step 7: normalize, resize, and save as PNG
    ohrc_out = save_as_png(normalize_to_uint8(ohrc_cropped),
                            os.path.join(OUTPUT_DIR, "ohrc_cropped.png"))
    lroc_out = save_as_png(normalize_to_uint8(lroc_cropped),
                            os.path.join(OUTPUT_DIR, "lroc_cropped.png"))

    print(f"\nSaved: {ohrc_out}")
    print(f"Saved: {lroc_out}")