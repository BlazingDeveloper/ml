"""
match_lanes.py — Two-lane OHRC↔LROC image matcher (SIH26166).

Usage:
  python match_lanes.py --selfpair   # self-pair ground-truth benchmark
  python match_lanes.py --real       # real OHRC↔NAC pair

Lanes:
  A  Classical: CLAHE → Sobel-mag → SIFT → FLANN → MAGSAC++
  B  Learned:   SuperPoint + LightGlue (auto-degrades to None if unavailable)

Arbiter merges/selects the best result.
Sub-pixel refinement: phase_cross_correlation on 64-px patches.
Uniformity: 8×8 grid, Shannon entropy, coverage %.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

# ── optional imports (degrade gracefully) ────────────────────────────────────
try:
    from skimage.registration import phase_cross_correlation
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False
    print("[warn] scikit-image not found — sub-pixel refinement disabled. "
          "pip install scikit-image")

try:
    import torch
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

_HAVE_LIGHTGLUE = False
_lightglue_extractor = None
_lightglue_matcher   = None

if _HAVE_TORCH:
    try:
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
        _HAVE_LIGHTGLUE = True
        print("[info] LightGlue found — Lane B will use SuperPoint+LightGlue")
    except ImportError:
        print("[warn] LightGlue not installed — Lane B disabled. "
              "pip install git+https://github.com/cvg/LightGlue.git")

# Local
from pds4_reader import open_pds4_memmap, normalize_to_uint8_percentile

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
OUTPUT_DIR     = Path("output")

# Preprocessing
WINDOW_SIZE    = 8_000     # px — OHRC window to sample (along-track)
CLAHE_CLIP     = 2.6       # empirically good for low-contrast crater scenes
CLAHE_TILE     = 8         # grid size for CLAHE

# Lane A
SIFT_NFEATURES   = 20_000
SIFT_CONTRAST_TH = 0.02    # low threshold → more keypoints in low-contrast flat areas
FLANN_RATIO      = 0.85    # Lowe ratio test
GEOM_PRIOR_PX    = 40      # search radius constraint at NAC scale (approx)
MAGSAC_THRESH    = 3.0     # px — MAGSAC++ inlier threshold
MAGSAC_CONF      = 0.999

# Lane B
TILE_SIZE        = 1024
TILE_OVERLAP     = 0.20    # 20% overlap between tiles
MAX_KP_PER_TILE  = 2048
SP_NMS_RADIUS    = 3
SP_CONF_THRESH   = 0.5

# Arbiter
MIN_ML_INLIERS   = 30
MIN_CL_INLIERS   = 30
ML_RATIO_THRESH  = 0.5     # inlier_ratio needed for ML to win
MERGE_REPROJ_TH  = 3.0     # px — threshold to call homographies "agreeing"

# Sub-pixel
SUBPIX_PATCH     = 64      # half-window for phase_cross_correlation
SUBPIX_UPSAMPLE  = 20      # sub-pixel resolution = 1/20 px

# Uniformity
GRID_N           = 8       # 8×8 = 64 cells
CELL_QUOTA       = 25      # max matches per cell

# Self-pair synthetic warp
SP_ROT_RANGE     = (5, 15)     # degrees
SP_SCALE_RANGE   = (1.5, 4.0)  # scale factor
SP_ASPECT_RANGE  = (0.9, 1.1)  # aspect ratio perturbation
SP_GAMMA_RANGE   = (0.6, 1.6)
SP_NOISE_SIGMA   = 5.0

# OHRC + NAC paths
OHRC_XML = "ch2_ohr_ncp_20210405T0442095110_d_img_d18.xml"
OHRC_IMG = "ch2_ohr_ncp_20210405T0442095110_d_img_d18.img"


# ════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

def clahe_enhance(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a uint8 grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                             tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    return clahe.apply(img)


def sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """
    Compute Sobel gradient magnitude (uint8).
    Polarity-blind: sun-azimuth flip inverts slope highlight/shadow polarity
    but leaves gradient magnitude unchanged → illumination-invariant structure.
    """
    img_f  = img.astype(np.float32)
    gx     = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy     = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mag    = np.sqrt(gx * gx + gy * gy)
    # stretch to uint8
    p99    = np.percentile(mag, 99)
    if p99 > 0:
        mag = np.clip(mag / p99 * 255, 0, 255)
    return mag.astype(np.uint8)


def downsample_to_match(src: np.ndarray, ref: np.ndarray,
                         src_gsd: float, ref_gsd: float) -> np.ndarray:
    """
    Gaussian pre-filter + area downsampling to resample src from src_gsd to ref_gsd.
    If GSD ratio ≤ 1 (src already coarser), returns src unchanged.
    """
    ratio = ref_gsd / src_gsd
    if ratio <= 1.0:
        return src
    # Gaussian blur before downsampling (anti-aliasing)
    sigma = ratio / 2.0
    k     = int(2 * math.ceil(3 * sigma) + 1)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(src, (k, k), sigma)
    new_w   = max(1, int(src.shape[1] / ratio))
    new_h   = max(1, int(src.shape[0] / ratio))
    return cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ════════════════════════════════════════════════════════════════════════════
# LANE A — CLASSICAL
# ════════════════════════════════════════════════════════════════════════════

def lane_a_classical(ohrc_u8: np.ndarray, nac_u8: np.ndarray
                     ) -> dict:
    """
    CLAHE → Sobel-magnitude → SIFT → FLANN ratio-test → MAGSAC++ homography.

    Returns dict with keys: kp_src, kp_ref, matches_all, inliers, H, mask, score, runtime_s
    """
    t0 = time.perf_counter()
    print("[Lane A] Starting classical pipeline ...")

    # Enhance
    ohrc_cl = clahe_enhance(ohrc_u8)
    nac_cl  = clahe_enhance(nac_u8)

    # Gradient magnitude (polarity-blind for sun-angle invariance)
    ohrc_grad = sobel_magnitude(ohrc_cl)
    nac_grad  = sobel_magnitude(nac_cl)

    # SIFT — low contrast threshold to find features in flat crater floors
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES,
                            contrastThreshold=SIFT_CONTRAST_TH)
    kp_src, des_src = sift.detectAndCompute(ohrc_grad, None)
    kp_ref, des_ref = sift.detectAndCompute(nac_grad,  None)
    print(f"[Lane A]   SIFT: {len(kp_src)} src kp, {len(kp_ref)} ref kp")

    if des_src is None or des_ref is None or len(kp_src) < 4 or len(kp_ref) < 4:
        print("[Lane A]   Too few keypoints — aborting lane A")
        return _empty_result("A")

    # FLANN matcher
    index_params  = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        raw_matches = flann.knnMatch(des_src, des_ref, k=2)
    except cv2.error as e:
        print(f"[Lane A]   FLANN error: {e}; falling back to BF matcher")
        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = bf.knnMatch(des_src, des_ref, k=2)

    # Ratio test + geometry-prior radius constraint
    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < FLANN_RATIO * n.distance:
            # Geometry prior: restrict search to ≈GEOM_PRIOR_PX radius
            pt_src = np.array(kp_src[m.queryIdx].pt)
            pt_ref = np.array(kp_ref[m.trainIdx].pt)
            if np.linalg.norm(pt_src - pt_ref) < GEOM_PRIOR_PX * 5:  # loose at matching stage
                good.append(m)

    print(f"[Lane A]   Ratio+geom filter: {len(good)} matches")
    if len(good) < 4:
        print("[Lane A]   Too few matches for homography")
        return _empty_result("A")

    pts_src = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # MAGSAC++ (cv2.USAC_MAGSAC): most robust against outlier-rich lunar data
    H, mask = cv2.findHomography(pts_src, pts_ref,
                                  method=cv2.USAC_MAGSAC,
                                  ransacReprojThreshold=MAGSAC_THRESH,
                                  confidence=MAGSAC_CONF,
                                  maxIters=10_000)

    if H is None or mask is None:
        print("[Lane A]   MAGSAC++ returned no homography")
        return _empty_result("A")

    mask_flat = mask.ravel().astype(bool)
    inlier_count = int(mask_flat.sum())
    inlier_ratio = inlier_count / max(len(good), 1)
    score = inlier_count * inlier_ratio
    rt = time.perf_counter() - t0

    print(f"[Lane A]   Inliers: {inlier_count}/{len(good)} "
          f"({inlier_ratio:.2%})  score={score:.1f}  t={rt:.1f}s")

    return {
        "lane": "A",
        "kp_src": kp_src, "kp_ref": kp_ref,
        "matches": good, "mask": mask_flat,
        "H": H,
        "inlier_count": inlier_count, "inlier_ratio": inlier_ratio,
        "score": score, "runtime_s": rt,
    }


# ════════════════════════════════════════════════════════════════════════════
# LANE B — LEARNED (SuperPoint + LightGlue)
# ════════════════════════════════════════════════════════════════════════════

def _init_lightglue():
    """Lazily initialise SuperPoint extractor and LightGlue matcher."""
    global _lightglue_extractor, _lightglue_matcher
    if _lightglue_extractor is not None:
        return True
    try:
        device = "cpu"
        _lightglue_extractor = SuperPoint(
            max_num_keypoints=MAX_KP_PER_TILE,
            nms_radius=SP_NMS_RADIUS,
            detection_threshold=SP_CONF_THRESH,
        ).eval().to(device)
        _lightglue_matcher = LightGlue(features="superpoint").eval().to(device)
        print("[Lane B] SuperPoint+LightGlue initialised (CPU)")
        return True
    except Exception as e:
        print(f"[Lane B] Init failed: {e}")
        return False


def _np_to_tensor(img: np.ndarray) -> "torch.Tensor":
    """uint8 HxW → float32 1x1xHxW tensor in [0,1]."""
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0)  # 1,1,H,W


def _run_lightglue_tile(img0: np.ndarray, img1: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run LightGlue on a single tile pair. Returns (pts0, pts1, scores)."""
    from lightglue.utils import rbd
    device = "cpu"
    t0 = _np_to_tensor(img0).to(device)
    t1 = _np_to_tensor(img1).to(device)
    with torch.no_grad():
        feats0 = _lightglue_extractor.extract(t0)
        feats1 = _lightglue_extractor.extract(t1)
        matches_data = _lightglue_matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches_data = [rbd(x) for x in [feats0, feats1, matches_data]]
    m0 = matches_data["matches"]  # N×2 int
    scores = matches_data["scores"].cpu().numpy()
    if m0.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    pts0 = feats0["keypoints"][m0[:, 0]].cpu().numpy()
    pts1 = feats1["keypoints"][m0[:, 1]].cpu().numpy()
    return pts0, pts1, scores


def lane_b_learned(ohrc_u8: np.ndarray, nac_u8: np.ndarray) -> dict:
    """
    SuperPoint + LightGlue matcher with tiling.
    Gracefully returns empty result if LightGlue unavailable.
    """
    if not _HAVE_LIGHTGLUE:
        print("[Lane B] LightGlue unavailable — skipping")
        return _empty_result("B")

    if not _init_lightglue():
        return _empty_result("B")

    t0 = time.perf_counter()
    print("[Lane B] Starting learned pipeline (tiled) ...")

    ohrc_cl = clahe_enhance(ohrc_u8)
    nac_cl  = clahe_enhance(nac_u8)

    H_oh, W_oh = ohrc_cl.shape
    H_na, W_na = nac_cl.shape

    stride = int(TILE_SIZE * (1 - TILE_OVERLAP))

    all_pts_src = []
    all_pts_ref = []
    all_scores  = []

    tile_count = 0
    for r0 in range(0, H_oh, stride):
        for c0 in range(0, W_oh, stride):
            r1 = min(r0 + TILE_SIZE, H_oh)
            c1 = min(c0 + TILE_SIZE, W_oh)
            tile_oh = ohrc_cl[r0:r1, c0:c1]

            # Corresponding tile in NAC (same pixel coordinates — assume pre-aligned)
            r0n = min(r0, max(0, H_na - (r1 - r0)))
            c0n = min(c0, max(0, W_na - (c1 - c0)))
            r1n = min(r0n + (r1 - r0), H_na)
            c1n = min(c0n + (c1 - c0), W_na)
            tile_na = nac_cl[r0n:r1n, c0n:c1n]

            if tile_oh.shape[0] < 64 or tile_oh.shape[1] < 64:
                continue
            if tile_na.shape[0] < 64 or tile_na.shape[1] < 64:
                continue

            try:
                pts0, pts1, scores = _run_lightglue_tile(tile_oh, tile_na)
            except Exception as e:
                print(f"[Lane B]   Tile ({r0},{c0}) error: {e}")
                continue

            if pts0.shape[0] == 0:
                continue

            # Translate tile-local coords to global image coords
            pts0[:, 0] += c0;  pts0[:, 1] += r0
            pts1[:, 0] += c0n; pts1[:, 1] += r0n

            mask_conf = scores >= SP_CONF_THRESH
            all_pts_src.append(pts0[mask_conf])
            all_pts_ref.append(pts1[mask_conf])
            all_scores.append(scores[mask_conf])
            tile_count += 1

    if not all_pts_src:
        print("[Lane B]   No matches found across all tiles")
        return _empty_result("B")

    pts_src = np.vstack(all_pts_src).astype(np.float32)
    pts_ref = np.vstack(all_pts_ref).astype(np.float32)
    scores_all = np.concatenate(all_scores)
    print(f"[Lane B]   {len(pts_src)} raw matches from {tile_count} tiles")

    if len(pts_src) < 4:
        print("[Lane B]   Too few matches for homography")
        return _empty_result("B")

    # MAGSAC++ verification (same as Lane A)
    H_mat, mask = cv2.findHomography(
        pts_src.reshape(-1, 1, 2), pts_ref.reshape(-1, 1, 2),
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=MAGSAC_THRESH,
        confidence=MAGSAC_CONF,
        maxIters=10_000,
    )

    if H_mat is None or mask is None:
        print("[Lane B]   MAGSAC++ returned no homography")
        return _empty_result("B")

    mask_flat = mask.ravel().astype(bool)
    inlier_count = int(mask_flat.sum())
    inlier_ratio = inlier_count / max(len(pts_src), 1)
    score = inlier_count * inlier_ratio
    rt = time.perf_counter() - t0

    print(f"[Lane B]   Inliers: {inlier_count}/{len(pts_src)} "
          f"({inlier_ratio:.2%})  score={score:.1f}  t={rt:.1f}s")

    return {
        "lane": "B",
        "pts_src": pts_src, "pts_ref": pts_ref,
        "scores_raw": scores_all, "mask": mask_flat,
        "H": H_mat,
        "inlier_count": inlier_count, "inlier_ratio": inlier_ratio,
        "score": score, "runtime_s": rt,
    }


def _empty_result(lane: str) -> dict:
    return {
        "lane": lane,
        "H": None, "mask": np.zeros(0, dtype=bool),
        "inlier_count": 0, "inlier_ratio": 0.0,
        "score": 0.0, "runtime_s": 0.0,
    }


# ════════════════════════════════════════════════════════════════════════════
# ARBITER
# ════════════════════════════════════════════════════════════════════════════

def _median_reproj_error(H: np.ndarray,
                          pts_src: np.ndarray,
                          pts_ref: np.ndarray) -> float:
    """Compute median reprojection error of pts_src→H→pts_ref."""
    if H is None or len(pts_src) == 0:
        return float("inf")
    pts_src_h = pts_src.reshape(-1, 1, 2).astype(np.float32)
    proj      = cv2.perspectiveTransform(pts_src_h, H).reshape(-1, 2)
    errs      = np.linalg.norm(proj - pts_ref.reshape(-1, 2), axis=1)
    return float(np.median(errs))


def _extract_inlier_pts(result: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract inlier point arrays from a lane result."""
    mask = result["mask"]
    if "kp_src" in result:
        # Lane A keypoint objects
        matches = result["matches"]
        kp_src  = result["kp_src"]
        kp_ref  = result["kp_ref"]
        pts_src = np.float32([kp_src[m.queryIdx].pt for m, ok in zip(matches, mask) if ok])
        pts_ref = np.float32([kp_ref[m.trainIdx].pt for m, ok in zip(matches, mask) if ok])
    elif "pts_src" in result:
        # Lane B raw arrays
        pts_src = result["pts_src"][mask]
        pts_ref = result["pts_ref"][mask]
    else:
        pts_src = pts_ref = np.zeros((0, 2), np.float32)
    return pts_src, pts_ref


def arbiter(res_a: dict, res_b: dict) -> tuple[dict, str, dict]:
    """
    Select winning result and optionally merge.

    Returns (winning_result, decision_reason, log_dict).
    """
    log = {
        "lane_A_inliers":     res_a["inlier_count"],
        "lane_A_ratio":       res_a["inlier_ratio"],
        "lane_A_score":       res_a["score"],
        "lane_A_runtime_s":   res_a["runtime_s"],
        "lane_B_inliers":     res_b["inlier_count"],
        "lane_B_ratio":       res_b["inlier_ratio"],
        "lane_B_score":       res_b["score"],
        "lane_B_runtime_s":   res_b["runtime_s"],
    }

    # Check if both have valid homographies
    both_valid = res_a["H"] is not None and res_b["H"] is not None

    if both_valid:
        pts_a_src, pts_a_ref = _extract_inlier_pts(res_a)
        if len(pts_a_src) > 0:
            reproj_b_on_a = _median_reproj_error(res_b["H"], pts_a_src, pts_a_ref)
        else:
            reproj_b_on_a = float("inf")
        log["homography_agreement_reproj_px"] = reproj_b_on_a
        homographies_agree = reproj_b_on_a < MERGE_REPROJ_TH
    else:
        homographies_agree = False
        log["homography_agreement_reproj_px"] = None

    # --- Decision tree ---
    # ML wins if: ratio ≥ 0.5 AND inliers ≥ 30
    if (res_b["inlier_ratio"] >= ML_RATIO_THRESH
            and res_b["inlier_count"] >= MIN_ML_INLIERS):
        winner = res_b
        reason = "ML (Lane B): ratio and inlier threshold met"

    elif res_a["inlier_count"] >= MIN_CL_INLIERS:
        winner = res_a
        reason = "Classical (Lane A): inlier threshold met"

    elif both_valid:
        # Merge: re-run MAGSAC on combined inlier sets
        pts_a_src, pts_a_ref = _extract_inlier_pts(res_a)
        pts_b_src, pts_b_ref = _extract_inlier_pts(res_b)
        if len(pts_a_src) > 0 or len(pts_b_src) > 0:
            all_src = np.vstack([pts_a_src, pts_b_src]).astype(np.float32)
            all_ref = np.vstack([pts_a_ref, pts_b_ref]).astype(np.float32)
            H_merged, mask_merged = cv2.findHomography(
                all_src.reshape(-1, 1, 2), all_ref.reshape(-1, 1, 2),
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=MAGSAC_THRESH,
                confidence=MAGSAC_CONF,
            )
            merged_inliers = int(mask_merged.sum()) if mask_merged is not None else 0
            merged_ratio   = merged_inliers / max(len(all_src), 1)
            winner = {
                "lane": "merged",
                "H": H_merged,
                "pts_src": all_src,
                "pts_ref": all_ref,
                "mask": mask_merged.ravel().astype(bool) if mask_merged is not None else np.zeros(0, bool),
                "inlier_count": merged_inliers,
                "inlier_ratio": merged_ratio,
                "score": merged_inliers * merged_ratio,
                "runtime_s": res_a["runtime_s"] + res_b["runtime_s"],
            }
            reason = "Merged (both lanes, re-verified)"
        else:
            winner = res_a
            reason = "Classical (Lane A): fallback (no merged pts)"

    else:
        # Low confidence — coarse geometry prior
        winner = res_a if res_a["score"] >= res_b["score"] else res_b
        reason = ("Low confidence: best-available result; "
                  "no lane met inlier threshold")
        log["warning"] = "LOW CONFIDENCE — geometry-prior coarse alignment only"

    log["decision"] = reason
    log["winner_lane"] = winner.get("lane", "unknown")
    log["winner_inliers"] = winner["inlier_count"]
    log["winner_ratio"]   = winner["inlier_ratio"]
    log["homographies_agree"] = homographies_agree

    print(f"[Arbiter] Decision: {reason}")
    return winner, reason, log


# ════════════════════════════════════════════════════════════════════════════
# SUB-PIXEL REFINEMENT
# ════════════════════════════════════════════════════════════════════════════

def subpixel_refine(winner: dict,
                    ohrc_u8: np.ndarray,
                    nac_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Refine each inlier match with phase_cross_correlation on 64-px patches.
    Returns (refined_src, refined_ref, residual_magnitudes).
    """
    if not _HAVE_SKIMAGE:
        print("[subpix] scikit-image unavailable — skipping refinement")
        pts_s, pts_r = _extract_inlier_pts(winner)
        return pts_s, pts_r, np.zeros(len(pts_s))

    pts_s, pts_r = _extract_inlier_pts(winner)
    if len(pts_s) == 0:
        return pts_s, pts_r, np.array([])

    H_oh, W_oh = ohrc_u8.shape
    H_na, W_na = nac_u8.shape
    half = SUBPIX_PATCH // 2

    refined_s  = []
    refined_r  = []
    residuals  = []

    for (xs, ys), (xr, yr) in zip(pts_s, pts_r):
        xs, ys, xr, yr = int(round(xs)), int(round(ys)), int(round(xr)), int(round(yr))

        # Bounds check
        if (ys - half < 0 or ys + half >= H_oh or
                xs - half < 0 or xs + half >= W_oh):
            refined_s.append([xs, ys]); refined_r.append([xr, yr])
            residuals.append(0.0)
            continue
        if (yr - half < 0 or yr + half >= H_na or
                xr - half < 0 or xr + half >= W_na):
            refined_s.append([xs, ys]); refined_r.append([xr, yr])
            residuals.append(0.0)
            continue

        patch_s = ohrc_u8[ys - half:ys + half, xs - half:xs + half].astype(np.float64)
        patch_r = nac_u8 [yr - half:yr + half, xr - half:xr + half].astype(np.float64)

        try:
            shift, _, _ = phase_cross_correlation(
                patch_r, patch_s, upsample_factor=SUBPIX_UPSAMPLE
            )
            dy, dx = shift
        except Exception:
            dy, dx = 0.0, 0.0

        refined_s.append([xs,      ys])
        refined_r.append([xr + dx, yr + dy])
        residuals.append(math.sqrt(dx * dx + dy * dy))

    refined_s = np.array(refined_s, np.float32)
    refined_r = np.array(refined_r, np.float32)
    residuals = np.array(residuals, np.float32)

    print(f"[subpix] Refined {len(residuals)} matches; "
          f"median residual={np.median(residuals):.3f}px "
          f"p95={np.percentile(residuals,95):.3f}px")
    return refined_s, refined_r, residuals


# ════════════════════════════════════════════════════════════════════════════
# UNIFORMITY
# ════════════════════════════════════════════════════════════════════════════

def uniformity_report(pts_ref: np.ndarray,
                       img_w: int, img_h: int) -> dict:
    """
    Compute uniformity of match distribution using an 8×8 grid.

    Returns dict with: cell_counts, coverage_pct, entropy, per_cell_table.
    """
    if len(pts_ref) == 0:
        return {"cell_counts": [], "coverage_pct": 0.0, "entropy": 0.0}

    cell_w = img_w / GRID_N
    cell_h = img_h / GRID_N

    counts = np.zeros((GRID_N, GRID_N), dtype=np.int32)
    for x, y in pts_ref:
        ci = min(int(x / cell_w), GRID_N - 1)
        ri = min(int(y / cell_h), GRID_N - 1)
        counts[ri, ci] += 1

    covered = int((counts > 0).sum())
    total   = GRID_N * GRID_N
    coverage_pct = covered / total * 100

    # Normalized Shannon entropy
    flat = counts.ravel()
    total_pts = flat.sum()
    if total_pts > 0:
        prob = flat / total_pts
        prob = prob[prob > 0]
        raw_entropy = -float(np.sum(prob * np.log(prob + 1e-12)))
        max_entropy = math.log(total)
        norm_entropy = raw_entropy / max_entropy
    else:
        norm_entropy = 0.0

    print(f"[uniformity] coverage={coverage_pct:.1f}%  "
          f"norm_entropy={norm_entropy:.3f}  cells={covered}/{total}")
    return {
        "cell_counts":   counts.tolist(),
        "coverage_pct":  round(coverage_pct, 2),
        "entropy":       round(norm_entropy, 4),
        "cells_covered": covered,
        "cells_total":   total,
    }


def apply_cell_quota(pts_s: np.ndarray, pts_r: np.ndarray,
                      residuals: np.ndarray,
                      img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cap matches per cell to CELL_QUOTA, keeping lowest-residual-first."""
    if len(pts_r) == 0:
        return pts_s, pts_r, residuals

    cell_w = img_w / GRID_N
    cell_h = img_h / GRID_N
    cells: dict[tuple, list] = {}

    for idx, (x, y) in enumerate(pts_r):
        ci = min(int(x / cell_w), GRID_N - 1)
        ri = min(int(y / cell_h), GRID_N - 1)
        cells.setdefault((ri, ci), []).append((residuals[idx], idx))

    keep_idx = []
    for cell_pts in cells.values():
        cell_pts.sort()  # ascending residual
        keep_idx.extend(i for _, i in cell_pts[:CELL_QUOTA])

    keep_idx = sorted(keep_idx)
    return pts_s[keep_idx], pts_r[keep_idx], residuals[keep_idx]


# ════════════════════════════════════════════════════════════════════════════
# OUTPUT WRITERS
# ════════════════════════════════════════════════════════════════════════════

def save_matches_csv(pts_s: np.ndarray, pts_r: np.ndarray,
                      residuals: np.ndarray, lane: str,
                      path: Path) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_x", "src_y", "ref_x", "ref_y", "residual_px", "lane"])
        for (xs, ys), (xr, yr), res in zip(pts_s, pts_r, residuals):
            w.writerow([f"{xs:.2f}", f"{ys:.2f}", f"{xr:.2f}", f"{yr:.2f}",
                        f"{res:.4f}", lane])
    print(f"[out] matches.csv saved → {path}")


def save_overlay_png(ohrc_u8: np.ndarray, nac_u8: np.ndarray,
                      pts_s_in: np.ndarray, pts_r_in: np.ndarray,
                      pts_s_all: Optional[np.ndarray], pts_r_all: Optional[np.ndarray],
                      path: Path, max_dim: int = 1536) -> None:
    """
    Side-by-side overlay image with green inlier lines and red outlier lines.
    """
    # Upscale to RGB
    oh_rgb  = cv2.cvtColor(ohrc_u8, cv2.COLOR_GRAY2BGR)
    nac_rgb = cv2.cvtColor(nac_u8,  cv2.COLOR_GRAY2BGR)

    # Resize both to same height for side-by-side
    h  = max(oh_rgb.shape[0], nac_rgb.shape[0])
    oh_rgb  = cv2.resize(oh_rgb,  (int(oh_rgb.shape[1]  * h / oh_rgb.shape[0]),  h))
    nac_rgb = cv2.resize(nac_rgb, (int(nac_rgb.shape[1] * h / nac_rgb.shape[0]), h))
    w_off   = oh_rgb.shape[1]

    canvas  = np.hstack([oh_rgb, nac_rgb])

    # Outliers (red)
    if pts_s_all is not None and pts_r_all is not None:
        for (xs, ys), (xr, yr) in zip(pts_s_all, pts_r_all):
            p1 = (int(xs), int(ys))
            p2 = (int(xr) + w_off, int(yr))
            cv2.line(canvas, p1, p2, (0, 0, 180), 1, cv2.LINE_AA)

    # Inliers (green)
    for (xs, ys), (xr, yr) in zip(pts_s_in, pts_r_in):
        p1 = (int(xs), int(ys))
        p2 = (int(xr) + w_off, int(yr))
        cv2.line(canvas, p1, p2, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, (0, 255, 0), -1)
        cv2.circle(canvas, p2, 3, (0, 255, 0), -1)

    # Downsample for saving
    scale = max_dim / max(canvas.shape[1], canvas.shape[0], 1)
    if scale < 1:
        canvas = cv2.resize(canvas,
                             (int(canvas.shape[1] * scale),
                              int(canvas.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), canvas)
    print(f"[out] overlay PNG saved → {path}")


def print_benchmark_table(res_a: dict, res_b: dict, winner: dict,
                           true_rmse: Optional[float] = None) -> str:
    lines = [
        "=" * 64,
        f"{'BENCHMARK TABLE':^64}",
        "=" * 64,
        f"{'Metric':<30} {'Lane A':>12} {'Lane B':>12}",
        "-" * 64,
        f"{'Inlier count':<30} {res_a['inlier_count']:>12} {res_b['inlier_count']:>12}",
        f"{'Inlier ratio':<30} {res_a['inlier_ratio']:>11.2%} {res_b['inlier_ratio']:>11.2%}",
        f"{'Score (cnt×ratio)':<30} {res_a['score']:>12.2f} {res_b['score']:>12.2f}",
        f"{'Runtime (s)':<30} {res_a['runtime_s']:>12.2f} {res_b['runtime_s']:>12.2f}",
        "-" * 64,
        f"{'Winner':<30} {'← A' if winner['lane'] == 'A' else '':>12} "
        f"{'← B' if winner['lane'] == 'B' else '':>12} "
        f"({'merged' if winner['lane'] == 'merged' else ''}) ",
    ]
    if true_rmse is not None:
        lines.append(f"{'TRUE RMSE (px)':<30} {true_rmse:>12.4f}")
    lines.append("=" * 64)
    table = "\n".join(lines)
    print(table)
    return table


# ════════════════════════════════════════════════════════════════════════════
# SELF-PAIR MODE
# ════════════════════════════════════════════════════════════════════════════

def make_synthetic_warp(img: np.ndarray, seed: Optional[int] = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a known random homography + photometric augmentation to img.
    Returns (warped_img, H_true).  TRUE RMSE is computed against H_true.

    Warp parameters:
    - Rotation: SP_ROT_RANGE degrees
    - Scale: SP_SCALE_RANGE
    - Aspect: SP_ASPECT_RANGE
    - Perspective shear: small values
    Photometric:
    - Gamma: SP_GAMMA_RANGE
    - Linear gradient
    - Gaussian noise σ ≤ SP_NOISE_SIGMA
    """
    rng = np.random.default_rng(seed)

    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    # --- Build homography ---
    angle = float(rng.uniform(*SP_ROT_RANGE)) * rng.choice([-1, 1])
    scale = float(rng.uniform(*SP_SCALE_RANGE))
    aspect = float(rng.uniform(*SP_ASPECT_RANGE))

    # Rotation + scale about image centre
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M_rot

    # Aspect ratio perturbation
    H_aspect = np.eye(3, dtype=np.float64)
    H_aspect[0, 0] = aspect
    H = H_aspect @ H

    # Small perspective shear
    H[2, 0] = rng.uniform(-2e-5, 2e-5)
    H[2, 1] = rng.uniform(-2e-5, 2e-5)

    # Warp
    warped = cv2.warpPerspective(img, H, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=int(np.median(img)))

    # --- Photometric augmentation ---
    # Gamma
    gamma = float(rng.uniform(*SP_GAMMA_RANGE))
    lut   = np.clip((np.arange(256) / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)
    warped = cv2.LUT(warped, lut)

    # Linear gradient (simulate illumination variation)
    grad = np.linspace(0.85, 1.15, w, dtype=np.float32)
    warped = np.clip(warped.astype(np.float32) * grad[np.newaxis, :], 0, 255).astype(np.uint8)

    # Gaussian noise
    noise  = rng.normal(0, SP_NOISE_SIGMA, warped.shape).astype(np.float32)
    warped = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return warped, H


def compute_true_rmse(H_est: Optional[np.ndarray],
                       H_true: np.ndarray,
                       pts_src: np.ndarray) -> float:
    """
    Compute RMSE between estimated and true homographies evaluated on pts_src.
    Lower is better; 0 = perfect.
    """
    if H_est is None or len(pts_src) == 0:
        return float("inf")
    pts_h = pts_src.reshape(-1, 1, 2).astype(np.float32)
    proj_est  = cv2.perspectiveTransform(pts_h, H_est.astype(np.float32)).reshape(-1, 2)
    proj_true = cv2.perspectiveTransform(pts_h, H_true.astype(np.float32)).reshape(-1, 2)
    errs = np.linalg.norm(proj_est - proj_true, axis=1)
    return float(np.sqrt(np.mean(errs ** 2)))


def run_selfpair(ohrc_u8: np.ndarray, out: Path) -> dict:
    """
    Self-pair benchmark mode.
    Warp OHRC window → use as "reference" → run both lanes → compute TRUE RMSE.
    """
    print("\n" + "=" * 60)
    print("SELF-PAIR BENCHMARK MODE")
    print("=" * 60)

    seed = 42
    warped, H_true = make_synthetic_warp(ohrc_u8, seed=seed)
    print(f"[selfpair] Synthetic warp generated (seed={seed})")
    print(f"           H_true[:2,:]=\n{H_true[:2,:]}")

    # Save warped reference image
    warp_path = out / "selfpair_warped_ref.png"
    from PIL import Image as PILImage
    PILImage.fromarray(warped).save(str(warp_path))
    print(f"[selfpair] Saved warped reference → {warp_path}")

    # Run both lanes (src=original, ref=warped)
    res_a = lane_a_classical(ohrc_u8, warped)
    res_b = lane_b_learned(ohrc_u8, warped)

    # Arbiter
    winner, reason, arb_log = arbiter(res_a, res_b)

    # Inlier points
    pts_s_in, pts_r_in = _extract_inlier_pts(winner)

    # Sub-pixel refinement
    pts_s_ref, pts_r_ref, residuals = subpixel_refine(winner, ohrc_u8, warped)

    # True RMSE (before refinement)
    true_rmse_coarse = compute_true_rmse(winner["H"], H_true,
                                          pts_s_in if len(pts_s_in) > 0 else np.zeros((1,2)))

    # True RMSE (after refinement)
    if winner["H"] is not None:
        H_refined, _ = cv2.findHomography(
            pts_s_ref.reshape(-1, 1, 2), pts_r_ref.reshape(-1, 1, 2),
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=MAGSAC_THRESH,
        )
    else:
        H_refined = None
    true_rmse_refined = compute_true_rmse(H_refined, H_true, pts_s_ref)

    print(f"\n[selfpair] TRUE RMSE (coarse):  {true_rmse_coarse:.4f} px  "
          f"(target < 1.5 px: {'PASS' if true_rmse_coarse < 1.5 else 'FAIL'})")
    print(f"[selfpair] TRUE RMSE (refined):  {true_rmse_refined:.4f} px  "
          f"(target < 1.0 px: {'PASS' if true_rmse_refined < 1.0 else 'FAIL'})")

    # Uniformity
    unif = uniformity_report(pts_r_ref, warped.shape[1], warped.shape[0])

    # Cell quota
    if len(pts_s_ref) > 0:
        pts_s_q, pts_r_q, res_q = apply_cell_quota(
            pts_s_ref, pts_r_ref, residuals, warped.shape[1], warped.shape[0])
    else:
        pts_s_q, pts_r_q, res_q = pts_s_ref, pts_r_ref, residuals

    # Save outputs
    save_matches_csv(pts_s_q, pts_r_q, res_q, winner.get("lane","?"),
                      out / "matches.csv")
    save_overlay_png(ohrc_u8, warped, pts_s_in, pts_r_in, None, None,
                      out / "matches_overlay.png")
    table = print_benchmark_table(res_a, res_b, winner, true_rmse_refined)
    (out / "benchmark_table.txt").write_text(table)

    metrics = {
        "mode": "selfpair",
        "winner_lane": winner.get("lane"),
        "decision": reason,
        "inlier_count": winner["inlier_count"],
        "inlier_ratio": winner["inlier_ratio"],
        "true_rmse_coarse_px": true_rmse_coarse,
        "true_rmse_refined_px": true_rmse_refined,
        "target_coarse_pass": true_rmse_coarse < 1.5,
        "target_refined_pass": true_rmse_refined < 1.0,
        "subpixel_median_residual_px": float(np.median(residuals)) if len(residuals) > 0 else None,
        "subpixel_p95_residual_px": float(np.percentile(residuals, 95)) if len(residuals) > 0 else None,
        "uniformity": unif,
        "lane_A": {k: v for k, v in res_a.items() if k not in ("kp_src","kp_ref","matches","mask","H","pts_src","pts_ref","scores_raw")},
        "lane_B": {k: v for k, v in res_b.items() if k not in ("kp_src","kp_ref","matches","mask","H","pts_src","pts_ref","scores_raw")},
        "arbiter_log": arb_log,
    }
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n[out] metrics.json saved → {metrics_path}")

    return metrics


# ════════════════════════════════════════════════════════════════════════════
# REAL MODE
# ════════════════════════════════════════════════════════════════════════════

def load_pair_meta(meta_path: str = "output/pair_meta.json") -> Optional[dict]:
    """Load pair metadata saved by pair_finder.py."""
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception as exc:
        print(f"[real] Cannot load pair_meta.json: {exc}")
        return None


def load_ohrc_window(xml_path: str, img_path: str,
                      window_size: int = WINDOW_SIZE) -> np.ndarray:
    """
    Load a centred window from the OHRC memmap and return uint8.
    Selects the densest-overlap region (centre strip) for initial matching.
    """
    print(f"[load] Opening OHRC memmap ...")
    mm = open_pds4_memmap(xml_path, img_path)
    h, w = mm.shape[-2:]
    # Centre strip
    r0 = max(0, h // 2 - window_size // 2)
    r1 = min(h, r0 + window_size)
    c0 = max(0, w // 2 - window_size // 2)
    c1 = min(w, c0 + window_size)
    print(f"[load] OHRC window [{r0}:{r1}, {c0}:{c1}] shape={r1-r0}×{c1-c0}")
    if mm.ndim == 3:
        tile = np.array(mm[:, r0:r1, c0:c1]).squeeze(axis=0)
    else:
        tile = np.array(mm[r0:r1, c0:c1])
    return normalize_to_uint8_percentile(tile)


def load_nac_window(meta: dict, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Load a centred window from the NAC memmap and return uint8."""
    nac_xml = meta["nac_xml"]
    nac_img = meta["nac_img"]

    print(f"[load] Opening NAC memmap ...")
    if Path(nac_xml).exists():
        mm = open_pds4_memmap(nac_xml, nac_img)
    else:
        # Hardcoded fallback shape for M1382366243RC
        mm = np.memmap(nac_img, dtype="<i2", mode="r", shape=(52224, 2532))

    h, w = mm.shape[-2:]
    r0 = max(0, h // 2 - window_size // 2)
    r1 = min(h, r0 + window_size)
    c0 = 0
    c1 = w
    print(f"[load] NAC window [{r0}:{r1}, {c0}:{c1}] shape={r1-r0}×{c1-c0}")
    if mm.ndim == 3:
        tile = np.array(mm[:, r0:r1, c0:c1]).squeeze(axis=0)
    else:
        tile = np.array(mm[r0:r1, c0:c1])
    return normalize_to_uint8_percentile(tile)


def resample_ohrc_to_nac_gsd(ohrc_u8: np.ndarray, nac_u8: np.ndarray) -> np.ndarray:
    """
    Resample OHRC to match NAC's spatial resolution.
    OHRC label GSD=0.23 m/px but actual empirical GSD ≈ 0.93 m/px
    (101,075 lines / 93.9 km); NAC CDR ≈ 0.5 m/px.
    Since empirical OHRC GSD > NAC GSD, no downsampling needed in this case —
    but we still resize to the same spatial window extent for matching.
    """
    # Scale OHRC window to same pixel dimensions as NAC window
    target_h, target_w = nac_u8.shape[:2]
    resampled = cv2.resize(ohrc_u8, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resampled


def run_real(meta: Optional[dict], out: Path) -> dict:
    """Real OHRC↔NAC matching mode."""
    print("\n" + "=" * 60)
    print("REAL OHRC↔NAC MATCHING MODE")
    print("=" * 60)
    print("[real] Note: TRUE RMSE is not computable on real pairs "
          "(no ground truth homography). Reporting inliers/ratio/uniformity only.")

    if meta is None:
        print("[real] No pair_meta.json — using default paths. "
              "Run pair_finder.py first for accurate overlap crop.")
        meta = {
            "ohrc_xml": OHRC_XML,
            "ohrc_img": OHRC_IMG,
            "nac_xml":  "output/M1382366243RC.xml",
            "nac_img":  "output/M1382366243RC.IMG",
        }

    ohrc_u8 = load_ohrc_window(meta["ohrc_xml"], meta["ohrc_img"])
    nac_u8  = load_nac_window(meta)

    # Resample OHRC to NAC spatial scale for matching
    ohrc_rs = resample_ohrc_to_nac_gsd(ohrc_u8, nac_u8)
    print(f"[real] OHRC resampled: {ohrc_rs.shape}  NAC: {nac_u8.shape}")

    # Save input windows for inspection
    from PIL import Image as PILImage
    PILImage.fromarray(ohrc_rs).save(str(out / "match_input_ohrc.png"))
    PILImage.fromarray(nac_u8 ).save(str(out / "match_input_nac.png"))

    res_a = lane_a_classical(ohrc_rs, nac_u8)
    res_b = lane_b_learned(ohrc_rs, nac_u8)
    winner, reason, arb_log = arbiter(res_a, res_b)

    pts_s_in, pts_r_in = _extract_inlier_pts(winner)
    pts_s_ref, pts_r_ref, residuals = subpixel_refine(winner, ohrc_rs, nac_u8)

    if len(pts_s_ref) > 0:
        pts_s_q, pts_r_q, res_q = apply_cell_quota(
            pts_s_ref, pts_r_ref, residuals, nac_u8.shape[1], nac_u8.shape[0])
    else:
        pts_s_q, pts_r_q, res_q = pts_s_ref, pts_r_ref, residuals

    unif = uniformity_report(pts_r_q, nac_u8.shape[1], nac_u8.shape[0])

    save_matches_csv(pts_s_q, pts_r_q, res_q, winner.get("lane","?"),
                      out / "matches.csv")
    save_overlay_png(ohrc_rs, nac_u8, pts_s_in, pts_r_in, None, None,
                      out / "matches_overlay.png")
    table = print_benchmark_table(res_a, res_b, winner, true_rmse=None)
    (out / "benchmark_table.txt").write_text(table)

    metrics = {
        "mode": "real",
        "note": "TRUE RMSE not computable on real pairs — no ground truth homography.",
        "winner_lane": winner.get("lane"),
        "decision": reason,
        "inlier_count": winner["inlier_count"],
        "inlier_ratio": winner["inlier_ratio"],
        "subpixel_median_residual_px": float(np.median(residuals)) if len(residuals) > 0 else None,
        "subpixel_p95_residual_px": float(np.percentile(residuals, 95)) if len(residuals) > 0 else None,
        "uniformity": unif,
        "lane_A": {k: v for k, v in res_a.items() if k not in ("kp_src","kp_ref","matches","mask","H","pts_src","pts_ref","scores_raw")},
        "lane_B": {k: v for k, v in res_b.items() if k not in ("kp_src","kp_ref","matches","mask","H","pts_src","pts_ref","scores_raw")},
        "arbiter_log": arb_log,
    }
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n[out] metrics.json saved → {metrics_path}")

    return metrics


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-lane OHRC↔LROC matcher (SIH26166)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selfpair", action="store_true",
                        help="Self-pair ground-truth benchmark (synthetic warp)")
    group.add_argument("--real", action="store_true",
                        help="Real OHRC↔NAC pair matching")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE,
                        help="Window size in pixels for OHRC/NAC tiles")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help="Output directory")
    parser.add_argument("--meta", default="output/pair_meta.json",
                        help="pair_meta.json written by pair_finder.py")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.selfpair:
        print("[main] Loading OHRC window for self-pair benchmark ...")
        ohrc_u8 = load_ohrc_window(OHRC_XML, OHRC_IMG, window_size=args.window)
        run_selfpair(ohrc_u8, out)
    else:
        meta = load_pair_meta(args.meta)
        run_real(meta, out)


if __name__ == "__main__":
    main()
