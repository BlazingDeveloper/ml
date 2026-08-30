import os
from pairing import (
    parse_ohrc_metadata, corners_to_bbox, ohrc_corners_to_polygon,
    query_ode, extract_lroc_candidates, find_best_match,
    get_lroc_image_url, get_lroc_label_url, download_file,
)

# ─── CONFIG — edit for each OHRC image ───
OHRC_XML_PATH = "data/ohrc/ch2_ohr_ncp_20210405T0442095110_d_img_d18.xml"
OHRC_OBS_TIME = "2021-04-05T04:42:09.5110Z"
MIN_AREA_RATIO = 0.05
MAX_TIME_DIFF_HRS = 2000
OUTPUT_DIR = "output/lroc"
# ──────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

ohrc_metadata = parse_ohrc_metadata(OHRC_XML_PATH)
ohrc_corners = ohrc_metadata['corners']
ohrc_bbox = corners_to_bbox(ohrc_corners)
ohrc_polygon = ohrc_corners_to_polygon(ohrc_corners)
print(f"OHRC bbox: {ohrc_bbox}")

min_lat, max_lat, min_lon, max_lon = ohrc_bbox
ode_json = query_ode(min_lat, max_lat, min_lon, max_lon)


candidates = extract_lroc_candidates(ode_json)
from pairing import precise_overlap_area, parse_lroc_footprint, time_difference_hours



print(f"Found {len(candidates)} raw candidates")

matches = find_best_match(candidates, ohrc_polygon, OHRC_OBS_TIME,
                           MIN_AREA_RATIO, MAX_TIME_DIFF_HRS)
print(f"{len(matches)} candidates pass the safety threshold\n")

if not matches:
    print("No safe match found. Try lowering MIN_AREA_RATIO or raising MAX_TIME_DIFF_HRS.")
    exit()

for m in matches[:5]:
    print(f"{m['product_id']}  area_ratio={m['area_ratio']:.2f}  time_diff_hrs={m['time_diff_hrs']:.1f}")

best = matches[0]
print(f"\nBest match: {best['product_id']}")

img_url = get_lroc_image_url(best['product_files'])
label_url = get_lroc_label_url(best['product_files'])

if not img_url or not label_url:
    print("Missing .IMG or .xml URL for best match.")
    exit()

img_path = os.path.join(OUTPUT_DIR, best['product_id'])
label_path = os.path.join(OUTPUT_DIR, best['product_id'].replace('.IMG', '.xml'))

print(f"Downloading image: {img_url}")
download_file(img_url, img_path)
print(f"Downloading label: {label_url}")
download_file(label_url, label_path)

print(f"\nDone.")
print(f"  Image: {img_path}")
print(f"  Label: {label_path}")
