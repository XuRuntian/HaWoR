"""Cached metric Z-depth and validity adaptation for the original scale fitter."""

from pathlib import Path

import cv2
import numpy as np


def depth_path(directory, frame, frame_count=1800):
    if not isinstance(frame, (int, np.integer)) or not 0 <= frame < frame_count:
        raise ValueError("Depth must be addressed by an original integer video frame")
    return Path(directory)/f"frame_{frame:06d}_foundation_depth_m.npy"


def resize_depth_and_mask(depth, hand_mask, shape_hw):
    depth = np.asarray(depth)
    if depth.ndim != 2 or depth.dtype != np.float32 or np.asarray(hand_mask).ndim != 2:
        raise ValueError("Expected float32 Z-depth and a 2D native hand exclusion mask")
    h, w = shape_hw
    if h <= 0 or w <= 0:
        raise ValueError("Invalid target depth geometry")
    valid = np.isfinite(depth) & (depth > 0)
    if valid.all():
        resized = cv2.resize(depth, (w, h))
        support = np.ones((h, w), bool)
    else:
        # A resized pixel is usable only if all contributing source samples are valid.
        support = cv2.resize(valid.astype(np.float32), (w, h)) >= 1-1e-7
        resized = cv2.resize(np.where(valid, depth, 0), (w, h))
        resized[~support] = np.nan
    support &= np.isfinite(resized) & (resized > 0)
    mask = cv2.resize(np.asarray(hand_mask, dtype=np.uint8), (w, h))
    mask[~support] = 1
    if not (support & (mask < .5)).any():
        raise ValueError("No finite positive unmasked depth support")
    return resized, mask, support


def fit_one(slam_depth, metric_depth, mask, near=.4, far=.7, estimator=None, max_attempts=100):
    if estimator is None:
        from lib.pipeline.est_scale import est_scale_hybrid
        estimator = est_scale_hybrid
    if slam_depth.shape != metric_depth.shape or mask.shape != metric_depth.shape:
        raise ValueError("Depth and mask shapes must agree")
    valid = np.isfinite(slam_depth) & (slam_depth > 0) & np.isfinite(metric_depth) & (metric_depth > 0) & (mask < .5)
    if not valid.any():
        raise ValueError("No finite positive background support for native scale fit")
    if not np.isfinite(slam_depth).all() or not (slam_depth > 0).all():
        raise ValueError("Frozen DROID depths must remain finite and positive")
    empty_bands = 0
    for retries in range(max_attempts):
        band = valid & (metric_depth > near) & (metric_depth < far)
        if band.any():
            scale = estimator(slam_depth, metric_depth, sigma=.5, msk=mask, near_thresh=near, far_thresh=far)
        else:
            empty_bands += 1
            scale = np.nan
        if np.isfinite(scale) and scale > 0:
            return float(scale), near, far, {"retries": retries, "empty_bands_skipped": empty_bands,
                "positive_background_support": int(valid.sum()), "initial_band_support_at_success": int(band.sum())}
        near -= .1
        far += .1
    raise RuntimeError("Native scale fit failed after bounded threshold expansion; no fallback scale")


def scaled_camera_fields(raw, scale, k):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Camera scale must be finite and positive")
    # Store scale separately: load_slam_cam applies it once to camera translation.
    return {**raw, "scale": float(scale), "img_focal": k[0, 0],
            "img_center": k[:2, 2], "camera_intrinsics": k}
