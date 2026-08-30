"""
run_matching.py — Run SuperPoint + LightGlue on the cropped OHRC/LROC pair.

This is the first real cross-sensor matching test: genuine OHRC↔LROC keypoint
correspondence, not synthetic augmentation. Reports how many matches survive
RANSAC — that count is the first real data point for the project.
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

# ── CONFIG ──
OHRC_PNG = "output/ohrc_cropped.png"
LROC_PNG = "output/lroc_cropped.png"
OUTPUT_DIR = Path("output")
RANSAC_THRESH = 5.0   # px — reprojection threshold for RANSAC
# ─────────────

def load_gray(path: str) -> np.ndarray:
    """Load image as uint8 grayscale."""
    img = Image.open(path).convert("L")
    return np.array(img)


def np_to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HxW → float32 1x1xHxW in [0,1]."""
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0)


def main():
    print("=" * 60)
    print("SuperPoint + LightGlue  —  Real OHRC↔LROC Matching")
    print("=" * 60)

    # Load images
    ohrc = load_gray(OHRC_PNG)
    lroc = load_gray(LROC_PNG)
    print(f"OHRC crop: {ohrc.shape}  LROC crop: {lroc.shape}")

    # Resize OHRC to match LROC height for fair comparison
    # (they cover the same geographic extent but at different pixel scales)
    scale = lroc.shape[0] / ohrc.shape[0]
    ohrc_resized = cv2.resize(ohrc, (int(ohrc.shape[1] * scale), lroc.shape[0]),
                               interpolation=cv2.INTER_AREA)
    print(f"OHRC resized to: {ohrc_resized.shape} (scale={scale:.4f})")

    # Apply CLAHE for better contrast on both
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    ohrc_cl = clahe.apply(ohrc_resized)
    lroc_cl = clahe.apply(lroc)

    # Save CLAHE versions for inspection
    Image.fromarray(ohrc_cl).save(str(OUTPUT_DIR / "ohrc_clahe.png"))
    Image.fromarray(lroc_cl).save(str(OUTPUT_DIR / "lroc_clahe.png"))
    print("Saved CLAHE-enhanced versions for inspection")

    # Init SuperPoint + LightGlue
    device = "cpu"
    print(f"\nInitializing SuperPoint + LightGlue on {device}...")
    extractor = SuperPoint(max_num_keypoints=2048, nms_radius=3,
                            detection_threshold=0.005).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # Extract + match
    print("Extracting keypoints...")
    t0 = np_to_tensor(ohrc_cl).to(device)
    t1 = np_to_tensor(lroc_cl).to(device)

    with torch.no_grad():
        feats0 = extractor.extract(t0)
        feats1 = extractor.extract(t1)
        print(f"  OHRC keypoints: {feats0['keypoints'].shape[1]}")
        print(f"  LROC keypoints: {feats1['keypoints'].shape[1]}")

        print("Running LightGlue matching...")
        matches_data = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches_data = [rbd(x) for x in [feats0, feats1, matches_data]]

    m = matches_data["matches"]    # Nx2 indices
    scores = matches_data["scores"].cpu().numpy()
    print(f"\nRaw LightGlue matches: {m.shape[0]}")

    if m.shape[0] == 0:
        print("NO MATCHES FOUND. The crops likely don't show the same terrain,")
        print("or the LROC orientation assumption is wrong.")
        return

    pts0 = feats0["keypoints"][m[:, 0]].cpu().numpy()
    pts1 = feats1["keypoints"][m[:, 1]].cpu().numpy()

    print(f"Match scores — min: {scores.min():.3f}  mean: {scores.mean():.3f}  max: {scores.max():.3f}")

    # RANSAC verification (homography)
    if len(pts0) >= 4:
        H, mask = cv2.findHomography(
            pts0.reshape(-1, 1, 2), pts1.reshape(-1, 1, 2),
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=RANSAC_THRESH,
            confidence=0.999,
            maxIters=10000,
        )
        if mask is not None:
            inliers = int(mask.sum())
            ratio = inliers / len(pts0)
            print(f"\nRANSAC Results:")
            print(f"  Inliers: {inliers} / {len(pts0)} ({ratio:.1%})")
            print(f"  Homography found: {'YES' if H is not None else 'NO'}")

            if H is not None:
                # Compute reprojection errors for inliers
                pts0_h = pts0[mask.ravel().astype(bool)]
                pts1_h = pts1[mask.ravel().astype(bool)]
                proj = cv2.perspectiveTransform(
                    pts0_h.reshape(-1, 1, 2).astype(np.float32),
                    H.astype(np.float32)
                ).reshape(-1, 2)
                errors = np.linalg.norm(proj - pts1_h, axis=1)
                print(f"  Reprojection error — median: {np.median(errors):.2f}px  "
                      f"mean: {np.mean(errors):.2f}px  max: {np.max(errors):.2f}px")
        else:
            inliers = 0
            mask = np.zeros(len(pts0), dtype=np.uint8)
            print("RANSAC returned no result")
    else:
        print(f"Only {len(pts0)} matches — not enough for RANSAC (need ≥4)")
        inliers = 0
        mask = np.zeros(len(pts0), dtype=np.uint8)

    # Draw overlay
    mask_flat = mask.ravel().astype(bool) if mask is not None else np.zeros(len(pts0), dtype=bool)

    oh_rgb = cv2.cvtColor(ohrc_cl, cv2.COLOR_GRAY2BGR)
    lr_rgb = cv2.cvtColor(lroc_cl, cv2.COLOR_GRAY2BGR)
    h = max(oh_rgb.shape[0], lr_rgb.shape[0])
    oh_rgb = cv2.resize(oh_rgb, (int(oh_rgb.shape[1] * h / oh_rgb.shape[0]), h))
    lr_rgb = cv2.resize(lr_rgb, (int(lr_rgb.shape[1] * h / lr_rgb.shape[0]), h))
    w_off = oh_rgb.shape[1]
    canvas = np.hstack([oh_rgb, lr_rgb])

    # Red = rejected, Green = inlier
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(pts0, pts1)):
        p1 = (int(x0), int(y0))
        p2 = (int(x1) + w_off, int(y1))
        if mask_flat[i]:
            cv2.line(canvas, p1, p2, (0, 200, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, (0, 255, 0), -1)
            cv2.circle(canvas, p2, 3, (0, 255, 0), -1)
        else:
            cv2.line(canvas, p1, p2, (0, 0, 180), 1, cv2.LINE_AA)

    overlay_path = str(OUTPUT_DIR / "matches_overlay_real.png")
    cv2.imwrite(overlay_path, canvas)
    print(f"\nOverlay saved: {overlay_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — First Real Cross-Sensor Matching Result")
    print("=" * 60)
    print(f"  Raw LightGlue matches:  {m.shape[0]}")
    print(f"  RANSAC inliers:         {inliers}")
    print(f"  Inlier ratio:           {inliers/max(len(pts0),1):.1%}")
    print(f"  Verdict: {'GENUINE MATCHES FOUND ✓' if inliers >= 10 else 'TOO FEW — likely misaligned or different terrain'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
