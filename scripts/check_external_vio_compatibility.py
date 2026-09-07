"""Read-only source audit and isolated HaWoR camera-loader round-trip test."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.eval_utils.custom_utils import load_slam_cam

FOREIGN = Path("/home/user/roboego-hand-vis")
SESSION = Path("/home/user/ego_data/session_20260804_224442")
CLIP = Path("/home/user/ego_data/session_20260804_224442_clip_0649_60s/session")
RECT = FOREIGN / "tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/rectified"
VIO = FOREIGN / "tmp/openvins_session_224442_full/trajectory/trajectory.npz"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out):
    assert out.is_relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    source_paths = [VIO, CLIP / "meta.json", RECT / "calibrations/rectify_params.yaml",
                    RECT / "timestamps/left_timestamps.txt", CLIP / "timestamps/left_timestamps.txt",
                    SESSION / "timestamps/left_timestamps.txt"]
    hashes = {str(p): digest(p) for p in source_paths}
    meta = json.loads((CLIP / "meta.json").read_text())
    raw = np.loadtxt(SESSION / "timestamps/left_timestamps.txt")
    clip = np.loadtxt(CLIP / "timestamps/left_timestamps.txt")
    rect = np.loadtxt(RECT / "timestamps/left_timestamps.txt")
    assert np.array_equal(clip, rect)
    assert np.array_equal(clip[:, 0], np.arange(1800))
    start = meta["source_start_frame"]
    assert np.array_equal(raw[start:start + 1800, 1], clip[:, 1])
    with np.load(VIO) as archive:
        vio = {k: archive[k] for k in archive.files}
    ts = vio["timestamp_camera"]
    assert np.all(np.diff(ts) > 0)
    insertion = np.searchsorted(ts, clip[:, 1])
    before = np.clip(insertion - 1, 0, len(ts) - 1)
    after = np.clip(insertion, 0, len(ts) - 1)
    nearest = np.where(abs(ts[before] - clip[:, 1]) <= abs(ts[after] - clip[:, 1]), before, after)
    error = ts[nearest] - clip[:, 1]
    valid = abs(error) < 1e-6
    source_frame_agrees = vio["frame_index"][nearest] == start + np.arange(1800)
    missing = np.flatnonzero(~valid)
    calib = yaml.safe_load((RECT / "calibrations/rectify_params.yaml").read_text())
    r1 = np.asarray(calib["rectified"]["R1"])
    raw_from_rect = np.eye(4)
    raw_from_rect[:3, :3] = r1.T
    transforms = vio["T_world_camera"][nearest[valid]] @ raw_from_rect
    frames = np.flatnonzero(valid)
    # This sparse fixture only tests serialization. It is NOT a dense pipeline cache.
    traj = np.concatenate([transforms[:, :3, 3], Rotation.from_matrix(transforms[:, :3, :3]).as_quat()], axis=1)
    fixture = out / "vio_valid_pose_roundtrip_fixture.npz"
    np.savez(fixture, traj=traj, scale=1.0, source_frame_idx=frames,
             purpose="sparse loader roundtrip only; not a pipeline SLAM cache")
    rwc, twc, rcw, tcw = [x.numpy() for x in load_slam_cam(str(fixture))]
    matrix_error = float(abs(rcw - transforms[:, :3, :3]).max())
    translation_error = float(abs(tcw - transforms[:, :3, 3]).max())
    closure_error = float(abs(np.einsum("nij,nj->ni", rwc, tcw) + twc).max())
    assert max(matrix_error, translation_error, closure_error) < 1e-10
    rotation = transforms[:, :3, :3]
    time = clip[valid, 1]
    speed = np.linalg.norm(np.diff(transforms[:, :3, 3], axis=0), axis=1) / np.diff(time)
    angular = Rotation.from_matrix(rotation[:-1].transpose(0, 2, 1) @ rotation[1:]).magnitude() / np.diff(time)
    gaps = [{"clip_frame": int(i), "timestamp": float(clip[i, 1]),
             "nearest_pose_dt_s": float(error[i]), "nearest_source_frame": int(vio["frame_index"][nearest[i]])}
            for i in missing]
    k = np.asarray(calib["rectified"]["K1_rect"])
    result = {
        "source_sha256": hashes, "source_trajectory": str(VIO),
        "trajectory_schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in vio.items()},
        "clip_source_frames_inclusive": [start, start + 1799],
        "clip_and_rectified_timestamps_identical": True,
        "clip_timestamps_match_source_slice": True,
        "unique_clip_timestamps": len(np.unique(clip[:, 1])),
        "clip_timestamp_step_seconds_percentiles": np.percentile(np.diff(clip[:, 1]), [0,50,100]).tolist(),
        "exact_pose_count_tolerance_1us": int(valid.sum()), "missing_pose_count": len(missing),
        "missing_poses": gaps,
        "max_exact_alignment_error_s": float(abs(error[valid]).max()),
        "exact_matches_with_different_source_frame_index": np.flatnonzero(valid & ~source_frame_agrees).tolist(),
        "raw_camera_to_rectified_rotation_degrees": float(np.rad2deg(Rotation.from_matrix(r1).magnitude())),
        "pose_conversion": "T_world_rect = T_world_raw @ blockdiag(R1.T,1)",
        "loader_contract": "traj rows tx,ty,tz,qx,qy,qz,qw; camera-to-world; scale=1 for metric VIO",
        "fixture_is_production_cache": False,
        "loader_roundtrip": {"rotation_max_error": matrix_error, "translation_max_error_m": translation_error,
                             "inverse_translation_closure_error_m": closure_error, "passed": True},
        "clip_motion_consistency_not_accuracy": {
            "path_length_m": float(np.linalg.norm(np.diff(transforms[:, :3, 3], axis=0), axis=1).sum()),
            "net_displacement_m": float(np.linalg.norm(transforms[-1, :3, 3] - transforms[0, :3, 3])),
            "speed_m_s_p50_p95_max": np.percentile(speed, [50,95,100]).tolist(),
            "angular_speed_deg_s_p50_p95_max": np.rad2deg(np.percentile(angular, [50,95,100])).tolist(),
            "rotation_det_range": [float(np.linalg.det(rotation).min()), float(np.linalg.det(rotation).max())],
            "finite_transforms": bool(np.isfinite(transforms).all()),
        },
        "rectified_intrinsics_as_stored": k.tolist(), "crop_roi": calib["crop_roi"],
        "previous_hawor_intrinsics": {"focal": 600, "center": [960,537]},
        "focal_ratio_stored_calibration_over_previous": float(k[0,0]/600),
        "no_interpolation_or_network_inference": True,
    }
    assert all(digest(Path(path)) == value for path, value in hashes.items())
    result["source_hashes_unchanged"] = True
    (out / "vio_compatibility.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pre_frontend_comparison/external_vio_depth_check")
    main(parser.parse_args().output.resolve())
