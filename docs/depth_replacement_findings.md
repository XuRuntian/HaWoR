# Existing Stereo Depth Replacement Audit

Date: 2026-09-07. Scope: read-only inspection of local code and existing artifacts; no depth inference, no modification of `roboego-hand-vis`, session data, or shared caches.

## Conclusion

The existing full-60-second FoundationStereo output is structurally suitable as an alternative metric-depth source for HaWoR's Metric3D scale-estimation stage. It is **not a drop-in replacement file**, and the current cache should not yet be treated as a validated metric reference: its documented producer pairs equal video indices, while the supplied left/right timestamps differ by approximately one frame for almost the entire sequence.

If metric VIO poses replace DROID poses instead, Metric3D's purpose (recovering the arbitrary monocular SLAM scale) normally disappears. Do not estimate and apply another scene scale to already-metric VIO translations. Using stereo depth to correct hand translation would be a separate algorithmic change, not this replacement.

## Available Artifacts

The relevant full-sequence cache is:

`/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/depth_only/`

Actual filesystem and NumPy-header checks found:

- Exactly 1800 files named `frame_000000_foundation_depth_m.npy` through `frame_001799_foundation_depth_m.npy`; no missing indices.
- Every array is `float32`, shape `(1074, 1920)`; combined NPY file size 14,847,206,400 bytes.
- This is the same rectified left video as the existing HaWoR comparison, not an earlier hand window. The cache metadata names both full-60-second videos and the `session_20260804_224442_clip_0649_60s` sequence: [metadata](/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/depth_only/foundation_stereo_video_meta.json:2).
- A separate cache under the user's original session directory exists at `segments/seg_0000_t000006_000015_place_reach/hand_windows/handwin_0000_t000008_000015/foundation_depth/`, but contains only 240 arrays, frames 0..239. It is a different interval and must not be substituted for this full-60-second cache.

All 1800 headers were inspected. Pixel statistics below use ten complete arrays, not a claim of pixel-wise validation of all 1800 arrays:

| Frame | Finite Pixels | Median Depth (m) | Pixels in Native Initial 0.4..0.7 m Band |
| --- | ---: | ---: | ---: |
| 0 | 100.0000% | 0.486535 | 26.1195% |
| 88 | 99.9723% | 0.426660 | 34.7887% |
| 100 | 100.0000% | 0.256321 | 19.5635% |
| 114 | 100.0000% | 0.201615 | 6.9063% |
| 179 | 100.0000% | 0.594967 | 33.6604% |
| 196 | 100.0000% | 0.597487 | 29.7312% |
| 599 | 100.0000% | 0.401346 | 25.0855% |
| 900 | 99.4295% | 0.511467 | 23.0346% |
| 1200 | 100.0000% | 0.417211 | 22.9913% |
| 1799 | 100.0000% | 0.211423 | 12.6372% |

All sampled finite values are positive. Sampled maxima are below 4 m. These percentages are before any hand/dynamic-object mask and do not establish depth accuracy.

## Units and Camera Geometry

The producer resizes disparity to the original image dimensions, then computes `Z = inference_scale * fx * baseline_m / disparity`, preserving the scaled-disparity convention. Invalid disparity and depth outside the permitted positive range become NaN: [producer](/home/user/roboego-hand-vis/scripts/run_foundation_stereo_video.py:43). This is optical-axis Z in meters, not radial range, inverse depth, a visualization, or HaMeR camera translation.

The saved metadata records `scale=0.5`, `valid_iters=32`, `hiera=0`, left reference, and:

```text
fx = fy = 444.43963608648465
cx = 959.954470803018
cy = 543.9405836940584
baseline_m = 0.09978680432198256
width = 1920, height = 1074
```

Source: [cache metadata](/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/depth_only/foundation_stereo_video_meta.json:7), [rectification geometry](/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/rectified/calibrations/rectify_params.yaml:35). The positive baseline is intentional for depth conversion; the signed source projection baseline is preserved separately. The producer already applies the 0.5 factor: do not halve the saved metric arrays again.

The metadata does not record the executed producer's commit/hash, checkpoint path/hash, `min_disp`, or `max_depth_m`. The inspected source defaults are `min_disp=1e-3` and `max_depth_m=4.0`, but defaults and the observed numeric range alone are not proof of the original invocation: [arguments](/home/user/roboego-hand-vis/scripts/run_foundation_stereo_video.py:85).

### Crop Principal Point Check

The rectification YAML contains `crop_roi.y_min=8`, but this does **not** justify subtracting 8 from `cy` again. The dataset loader copies the exported K directly: [loader](/home/user/roboego-hand-vis/roboego_hand_vis/datasets/wumei01.py:19). The original rectification generator was not found in the searched local Python sources.

I independently decoded raw left frames from `/home/user/ego_data/session_20260804_224442_clip_0649_60s/session/videos/left.mp4` (1800 frames, 1920x1088), reconstructed rectified images with `cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R1, K1_rect, (1920,1074))`, and compared them to the actual full-60-second rectified MP4. RGB mean absolute differences, on 0..255 channels:

| Frame | Exported K | K with cy - 8 | K with cy + 8 |
| --- | ---: | ---: | ---: |
| 0 | 4.20555 | 14.72523 | 14.61545 |
| 100 | 2.66539 | 12.18863 | 12.07869 |
| 599 | 4.16602 | 13.30907 | 13.43316 |

Compression and interpolation prevent exact equality, but these tests strongly support the exported K as the appropriate image-space calibration. No evidence supports an additional crop shift.

## Stereo Timestamp Concern

The checked files are [left timestamps](/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/rectified/timestamps/left_timestamps.txt:1) and [right timestamps](/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/rectified/timestamps/right_timestamps.txt:1). Each contains indices 0..1799.

- Equal-index `right_timestamp - left_timestamp`: min -34.49583 ms, median -33.33330 ms, max 0 ms.
- 1798 / 1800 equal-index pairs differ by more than 20 ms.
- Only frames 551 and 560 match exactly; the right timestamp stream has two duplicate adjacent timestamps, the left has none. Neither stream runs backwards.
- Pairing `left[j]` with `right[j+1]` matches within 1 ms for 1797 / 1799 pairs. This observation is not authorization to silently shift arrays or assume all frames share a single offset.
- The documented producer reads `read_frame(left_cap, frame_idx)` and `read_frame(right_cap, frame_idx)` without consulting timestamps: [same-index pairing](/home/user/roboego-hand-vis/scripts/run_foundation_stereo_video.py:175). The Foundation input loader does not load timestamps: [input loader](/home/user/roboego-hand-vis/roboego_hand_vis/depth/foundation.py:21).

Therefore, **if the timestamp files represent actual exposure times and the saved cache was produced by this documented implementation**, the depth was generated from temporally mispaired stereo images for most frames. That can produce moving-camera/hand disparity bias despite visually dense, finite depth. Confirm timestamp semantics and the executed producer provenance before promoting this cache to a metric reference. Existing values cannot generally be repaired merely by renaming or shifting completed left-reference depth files; corrected stereo pairing would require depth recomputation into a new cache.

## HaWoR Replacement Contract

HaWoR's native pipeline first runs monocular masked DROID and obtains integer keyframe indices, disparity maps, and a dense camera trajectory. Metric3D is then called on those keyframe images only; the resulting depth arrays feed a per-keyframe scale estimator. The median scale is saved alongside the original trajectory: [native SLAM and scale pipeline](/home/user/HaWoR/scripts/scripts_test_video/hawor_slam.py:86).

There is no existing `depth_dir` argument or cached-depth provider in that stage. An adapter would need to:

1. Resolve each DROID keyframe index through the exact input-frame mapping and load the corresponding left-reference metric NPY, not index by keyframe-list position.
2. Match DROID's pixel geometry. For this video, DROID resizes 1920x1074 to 592x331 and crops the bottom three rows to 592x328: [image stream](/home/user/HaWoR/lib/pipeline/masked_droid_slam.py:115). The current Metric3D path directly resizes depth to 592x328: [native depth resize](/home/user/HaWoR/scripts/scripts_test_video/hawor_slam.py:104). Preserving that native operation and reproducing the exact DROID resize-plus-crop are distinct choices; the latter is geometrically consistent but must be reported as an interface correction, not attributed to the depth model.
3. Preserve invalid-depth information during resampling and explicitly require finite, positive stereo and SLAM depths. NaNs must not contaminate interpolation or become fake zero-depth support.
4. Retain equivalent dynamic-hand exclusion masks and adequate finite background support. The native estimator starts with 0.4..0.7 m, uses iterative median ratios and robust optimization, and expands thresholds on NaN: [scale estimator](/home/user/HaWoR/lib/pipeline/est_scale.py:74), [unbounded retry loop](/home/user/HaWoR/scripts/scripts_test_video/hawor_slam.py:123). Missing or entirely invalid cache input requires an explicit failure, not endless threshold expansion.
5. Use the calibrated rectified intrinsics consistently. The previous camera-only comparison's shared focal 600 and center (960,537) are not these calibrated intrinsics. Relabeling old camera outputs as calibrated would be incorrect.

Passing `depth=` to the existing `run_slam` API does not accomplish this: it resets `depth=None` internally and operates monocularly, [implementation](/home/user/HaWoR/lib/pipeline/masked_droid_slam.py:130). Feeding stereo depth into DROID tracking itself is a separate design from replacing Metric3D scale estimation.

## Information Still Needed

- Confirmation that the left/right timestamp files denote exposure times, including the known same-index one-frame offset and right-side duplicates, or the recording/synchronization explanation that makes current pairing valid.
- The executed FoundationStereo command, model/checkpoint version, and any pairing/reindexing steps not captured in the current metadata, if available.
- The intended next experiment: retain DROID and replace only Metric3D, or replace the metric camera trajectory with VIO and bypass scale estimation. These are different integration paths and different comparisons.

No additional raw depth is missing for frames 0..1799; the current blockers are synchronization/provenance confidence and explicit interface adaptation, not absent NPY files.
