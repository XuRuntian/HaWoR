"""Compare calibration hypotheses against the pixels of the existing rectified video."""

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RECT = Path("/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/rectified")
RAW = Path("/home/user/ego_data/session_20260804_224442_clip_0649_60s/session/videos/left.mp4")


def frame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    if not ok:
        raise ValueError(f"Cannot decode frame {index}")
    return image


def main():
    calib = yaml.safe_load((RECT / "calibrations/rectify_params.yaml").read_text())
    source = calib["source_cameras"]
    rectified = calib["rectified"]
    k = np.asarray(rectified["K1_rect"], dtype=np.float64)
    old = k.copy()
    old[0, 0] = old[1, 1] = 600
    old[0, 2], old[1, 2] = 960, 537
    shifted = k.copy()
    shifted[1, 2] -= 8
    hypotheses = {"stored_calibration": k, "previous_hawor_assumption": old,
                  "stored_calibration_minus_crop_y_again": shifted}
    maps = {name: cv2.fisheye.initUndistortRectifyMap(
        np.asarray(source["K_left"]), np.asarray(source["D_left"]),
        np.asarray(rectified["R1"]), matrix, (1920, 1074), cv2.CV_32FC1)
        for name, matrix in hypotheses.items()}
    raw, actual = cv2.VideoCapture(str(RAW)), cv2.VideoCapture(str(RECT / "videos/left_rectified.mp4"))
    rows = {}
    try:
        for fi in (0, 100, 599):
            src, reference = frame(raw, fi), frame(actual, fi)
            rows[str(fi)] = {}
            for name, (mx, my) in maps.items():
                predicted = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                error = abs(predicted.astype(np.float32) - reference.astype(np.float32))
                rows[str(fi)][name] = {"channel_MAE_0_255": float(error.mean()),
                                     "channel_median_absolute_error": float(np.median(error))}
            assert rows[str(fi)]["stored_calibration"]["channel_MAE_0_255"] < rows[str(fi)]["previous_hawor_assumption"]["channel_MAE_0_255"]
    finally:
        raw.release()
        actual.release()
    result = {"raw_video": str(RAW), "rectified_video": str(RECT / "videos/left_rectified.mp4"),
              "calibration": str(RECT / "calibrations/rectify_params.yaml"),
              "hypotheses": {name: value.tolist() for name, value in hypotheses.items()},
              "checks": rows, "no_network_inference_or_video_rewrite": True,
              "interpretation": "Pixel consistency with existing rectification, not a pose accuracy evaluation or an independent camera recalibration."}
    out = ROOT / "outputs/pre_frontend_comparison/external_vio_depth_check"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rectified_focal_evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
