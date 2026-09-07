"""Timestamp/coordinate adaptation only; no VIO estimation or hand correction."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def check_transforms(transforms):
    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("Expected camera-to-world transforms with shape (N,4,4)")
    if not np.isfinite(transforms).all():
        raise ValueError("Nonfinite camera transform")
    rotation = transforms[:, :3, :3]
    if not (np.allclose(transforms[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            and np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-6, rtol=0)
            and np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError("Camera transform is not proper SE(3)")
    return transforms


def adapt_camera_poses(timestamps, original_frames, source, rectification, tolerance=1e-6):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    original_frames = np.asarray(original_frames)
    source_time = np.asarray(source["timestamp_camera"], dtype=np.float64)
    source_frames = np.asarray(source["frame_index"])
    if (timestamps.ndim != 1 or len(timestamps) < 2 or original_frames.shape != timestamps.shape
            or source_time.ndim != 1 or len(source_time) < 2 or source_frames.shape != source_time.shape
            or not np.isfinite(timestamps).all() or not np.isfinite(source_time).all()
            or not (np.diff(timestamps) > 0).all() or not (np.diff(source_time) > 0).all()
            or len(np.unique(source_frames)) != len(source_frames)
            or not np.array_equal(np.diff(original_frames), np.ones(len(original_frames)-1))):
        raise ValueError("Invalid or ambiguous timestamp/frame sequence")
    transforms = check_transforms(source["T_world_camera"])
    if len(transforms) != len(source_time):
        raise ValueError("Pose and timestamp lengths differ")
    after = np.clip(np.searchsorted(source_time, timestamps), 0, len(source_time)-1)
    before = np.maximum(after-1, 0)
    nearest = np.where(abs(source_time[before]-timestamps) <= abs(source_time[after]-timestamps), before, after)
    error = source_time[nearest]-timestamps
    available = abs(error) < tolerance
    if not np.array_equal(source_frames[nearest[available]], original_frames[available]):
        raise ValueError("Matching timestamp has a different original frame ID")
    missing = np.flatnonzero(~available)
    if any(i == 0 or i == len(timestamps)-1 or not available[i-1] or not available[i+1] for i in missing):
        raise ValueError("Only isolated interior single-frame camera gaps may be interpolated")
    dense = np.full((len(timestamps), 4, 4), np.nan)
    dense[available] = transforms[nearest[available]]
    anchors = np.full((len(timestamps), 2), -1, dtype=np.int64)
    weights = np.zeros(len(timestamps), dtype=np.float64)
    for i in missing:
        left, right = i-1, i+1
        span = timestamps[right]-timestamps[left]
        offset = timestamps[i]-timestamps[left]
        weight = offset/span
        dense[i] = np.eye(4)
        dense[i, :3, 3] = (1-weight)*dense[left, :3, 3] + weight*dense[right, :3, 3]
        dense[i, :3, :3] = Slerp([0., span], Rotation.from_matrix(dense[[left, right], :3, :3]))([offset]).as_matrix()[0]
        anchors[i], weights[i] = [left, right], weight
    raw_from_rect = np.eye(4)
    raw_from_rect[:3, :3] = np.asarray(rectification, dtype=np.float64).T
    check_transforms(raw_from_rect[None])
    rectified = check_transforms(dense @ raw_from_rect)
    trajectory = np.concatenate([rectified[:, :3, 3], Rotation.from_matrix(rectified[:, :3, :3]).as_quat()], axis=1)
    return {"traj": trajectory, "scale": np.array(1., dtype=np.float64),
            "T_world_rect": rectified, "T_world_raw_dense": dense,
            "frame_idx": np.arange(len(timestamps)), "original_frame_idx": original_frames,
            "timestamp_camera": timestamps, "camera_pose_from_source": available,
            "camera_pose_interpolated": ~available, "camera_pose_missing": np.zeros(len(timestamps), bool),
            "source_pose_row": np.where(available, nearest, -1),
            "source_timestamp_error_s": np.where(available, error, np.nan),
            "interpolation_anchor_frames": anchors, "interpolation_weight": weights}
