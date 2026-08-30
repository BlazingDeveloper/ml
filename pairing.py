import xml.etree.ElementTree as ET
import requests
from datetime import datetime
from shapely import wkt
from shapely.geometry import Polygon


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

    params_elem = root.find('.//isda:Product_Parameters', ns)
    def get_val(tag):
        el = params_elem.find(f'isda:{tag}', ns)
        return el.text if el is not None else None

    return {
        'corners': corners,
        'pixel_resolution_m': get_val('pixel_resolution'),
        'spacecraft_altitude_km': get_val('spacecraft_altitude'),
        'roll_deg': get_val('roll'),
        'pitch_deg': get_val('pitch'),
        'yaw_deg': get_val('yaw'),
        'sun_azimuth_deg': get_val('sun_azimuth'),
        'sun_elevation_deg': get_val('sun_elevation'),
        'solar_incidence_deg': get_val('solar_incidence'),
        'projection': get_val('projection'),
    }


def corners_to_bbox(corners):
    lats = [lat for lat, lon in corners.values()]
    lons = [lon for lat, lon in corners.values()]
    return (min(lats), max(lats), min(lons), max(lons))


def ohrc_corners_to_polygon(corners):
    ring_order = ['upper_left', 'upper_right', 'lower_right', 'lower_left']
    points = [(corners[name][1], corners[name][0]) for name in ring_order]
    points.append(points[0])
    return Polygon(points)


def query_ode(min_lat, max_lat, min_lon, max_lon):
    
    url = "https://oderest.rsl.wustl.edu/live2/"
    params = {
        "numberproducts": 500,
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
                'center_lat': p.get('Center_latitude'),
                'center_lon': p.get('Center_longitude'),
                'incidence_angle': p.get('Incidence_angle'),
                'emission_angle': p.get('Emission_angle'),
                'phase_angle': p.get('Phase_angle'),
                'map_resolution': p.get('Map_resolution'),
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


def get_lroc_label_url(product_files):
    """Finds the .xml label URL (needed by pds4_reader.py to read the .IMG)."""
    files = product_files.get('Product_file', [])
    if isinstance(files, dict):
        files = [files]
    for f in files:
        if f.get('Type') == 'Product' and f.get('FileName', '').upper().endswith('.XML'):
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
