# Existing VIO / Depth Compatibility Check

Checked 2026-09-07 against local source code and actual saved artifacts. No
pipeline replacement, network inference, source-cache modification or pose
interpolation was performed. A standalone sparse camera-format round-trip test
was executed, not a new world-space reconstruction experiment.

## Conclusion

The existing OpenVINS trajectory can replace HaWoR's scaled camera trajectory
after timestamp, rectified-camera and serialization adaptation. Existing
FoundationStereo metric depth can technically replace Metric3D's scale-estimation
input, but the saved stereo pairing has a serious timestamp mismatch that must
be investigated before treating that depth as a reliable replacement.

Replacing a metric camera trajectory makes HaWoR's Metric3D camera-scale step
unnecessary. It does not automatically depth-align the hand, fix MANO pose,
resolve hand identity or make the native infiller compatible with B fragments.

## Actual Assets

The user's `/home/user/ego_data/session_20260804_224442` directory contains the
original session videos, timestamps, IMU and calibration, not the main computed
VIO result. The relevant saved results are:

```text
VIO:
/home/user/roboego-hand-vis/tmp/openvins_session_224442_full/trajectory/trajectory.npz

Depth:
/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/depth_only/

Exact 60-second clip provenance:
/home/user/ego_data/session_20260804_224442_clip_0649_60s/session/meta.json

Rectified camera calibration:
/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/rectified/calibrations/rectify_params.yaml
```

The clip metadata identifies original frames 12270..14069, starting at session
409 seconds. Its 1800 left-camera timestamps exactly equal both the original
session slice and the rectified video's timestamp table. Do not take the first
1800 VIO poses or use a rounded 409-second pose-array offset.

## VIO Checks and Adapter Contract

The full NPZ contains 32448 poses with `timestamp_camera`, `timestamp_imu`,
`frame_index`, `right_frame_index`, `T_world_imu`, `T_world_camera`, and covariance.
The run manifest declares raw-equidistant input, metric camera/IMU extrinsics,
and camera-to-IMU time shift -0.047775533 s. The exporter already associates
camera timestamps after accounting for this shift; it must not be applied twice.
See [pose export source](/home/user/roboego-hand-vis/scripts/vio/replay_openvins_session.py:195)
and [VIO conventions](/home/user/roboego-hand-vis/docs/pipeline/openvins_vio.md:114).

Actual checks for the 60-second clip:

- 1798/1800 timestamps match saved VIO poses exactly, including original frame ID.
- Missing local frames are **550 and 559**. Their nearest saved poses are about
  33 ms away, not exact observations. No nearest-neighbor substitution was made.
- All matched transforms are finite; rotation determinants are within floating
  precision of 1. The clip path is 14.910 m, net displacement 0.758 m; these are
  consistency diagnostics, not trajectory accuracy or scale-ground-truth proof.
- The existing VIO matched 1796 clip pairs with right index = left index + 1,
  and two pairs with equal indices. It does not use blanket same-index pairing.

`T_world_camera` already includes the IMU-to-camera extrinsic; do not add that
extrinsic again. It refers to the raw left camera. Since rectification maps
`p_rect = R1 @ p_raw`, use:

```text
T_world_rect = T_world_raw @ blockdiag(R1.T, 1)
traj[i] = [tx, ty, tz, qx, qy, qz, qw]  # camera-to-world
scale = 1.0                           # VIO translation is already in meters
```

R1 is small but nonzero: approximately 0.060643 degrees. World origin/yaw remain
the VIO gauge, not an externally registered room/map frame. Recentring, if used,
must be one explicitly recorded rigid transform applied consistently.

HaWoR's [camera loader](/home/user/HaWoR/lib/eval_utils/custom_utils.py:129) reads
`traj` and `scale`, converts xyzw quaternions, and constructs world-to-camera by
inversion. It does not read `tstamp` to align rows and has no missing-pose mask.
Therefore a production replacement must provide all frame-indexed rows or add
an explicit validity-aware consumer. Dropping the two missing rows would shift
all later frames. A future dense export could interpolate these isolated gaps
with translation interpolation and rotation SLERP, recording them separately;
that was not done here and is a camera-pose completion choice, not hand recall.

The real `load_slam_cam` successfully loaded the converted 1798 valid poses:
maximum rotation error 1.23e-15; translation and inverse-closure errors zero.
The sparse fixture is deliberately named `vio_valid_pose_roundtrip_fixture.npz`
and marked **not a production SLAM cache**.

## Depth's Actual Role in HaWoR

[hawor_slam.py](/home/user/HaWoR/scripts/scripts_test_video/hawor_slam.py:86)
runs masked monocular DROID, predicts Metric3D depth at keyframes, fits metric
depth against inverse DROID disparity, then saves a median trajectory scale.
[load_slam_cam](/home/user/HaWoR/lib/eval_utils/custom_utils.py:133) multiplies
camera translation by this scalar. This is not a per-hand depth correction.

Two different experiments are possible:

1. Keep DROID and replace only Metric3D with FoundationStereo Z-depth, matched
   to the same keyframe pixels. Preserve valid-depth masks, camera geometry and
   explicit scale-fit failure handling. Do not replace `disps` with metric Z.
2. Replace DROID + its metric-scale stage with metric OpenVINS poses. Bypass
   Metric3D and use scale 1. Dense stereo depth can remain a separate diagnostic;
   using it to translate hands onto surfaces would be an additional algorithm.

See [depth findings](depth_replacement_findings.md) for actual file inventory,
depth units/resolution, stereo timing and rectification pixel checks. The saved
depth is full-size camera-axis Z, but 1798/1800 same-index stereo pairs differ
by more than 20 ms; the median right-minus-left timestamp is -33.3333 ms.
This timing evidence does not by itself prove exposure-time misalignment if
timestamps represent a different clock event. It does mean blindly reusing
the cache as synchronized metric depth is not justified.

## Two Additional Integration Boundaries

**Calibrated hand camera geometry:** prior A/B predictions used focal 600 px,
center (960,537). The actual rectified calibration is fx=fy=444.439636 px,
cx=959.954471, cy=543.940584. Focal ratio is 0.740733. These cannot be called
calibrated metric hand positions merely by attaching a metric VIO trajectory.
Intrinsics feed the network's bbox features as well as translation, so scaling
old translation is not equivalent to inference with calibrated intrinsics.
See [bbox conditioning](/home/user/HaWoR/lib/models/hawor.py:158) and
[translation](/home/user/HaWoR/lib/models/hawor.py:500).

A calibrated follow-up should use separate caches and keep previous A/B results
unchanged. Native entry points assume image-center principal points; calibrated
plumbing must cover inference, projection/rendering, and the left-hand mirror
round trip. The dataset flips about image width, while the model unflips bbox
center using `2*img_center_x`; do not change only one of these conventions.
The exported K is supported by direct raw-to-rectified pixel checks; do not
subtract crop_roi.y_min=8 again just because that crop is listed in metadata.

**Optimized fragments versus infiller:**
[hawor_infiller](/home/user/HaWoR/scripts/scripts_test_video/hawor_video.py:275)
allocates two anatomical-side arrays, indexes `frame_chunks_all[0/1]`, and reads
`cam_space/0` or `cam_space/1`. Optimized B instead preserves fragment groups,
including ambiguous handedness and same-side simultaneous observations. Camera
replacement alone does not make these interfaces compatible. First world-space
inspection should transform each observed fragment separately and keep the
infiller off; enabling it requires an explicit identity policy, not silent merging.

## Recommended Next Step and Missing Information

Start with a separate **optimized HaWoR + OpenVINS, observed-only world export**,
after calibrated camera handling and a declared two-frame VIO gap policy. This
tests the trajectory replacement without confounding it with depth alignment or
hand infilling. No new VIO run appears necessary for this clip.

Before trusting the current stereo depth, confirm whether the recording's left/
right timestamps are exposure/sensor timestamps or receive/FIFO timestamps, and
whether any external one-frame synchronization correction was applied before
the rectified videos were produced. If exposure timestamps are authoritative and
there is no prior correction, pair by timestamp and recompute only the affected
stereo-depth cache in an isolated output directory. The needed raw data and
calibrations already exist; no extra user file path is presently required.

## Reproduce This Check

```bash
conda activate hawor
python scripts/check_external_vio_compatibility.py
```

Machine-readable VIO checks and the isolated loader fixture are under
`outputs/pre_frontend_comparison/external_vio_depth_check/`. The checker verifies
source hashes before/after execution. These new checks/docs are local and have
not been committed or pushed automatically.
