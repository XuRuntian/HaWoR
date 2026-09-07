# Focal-Length Evidence for the Rectified 60-Second Experiment

Checked 2026-09-07. This is an evidence audit, not a corrected inference run.

## Finding

The previous HaWoR experiment's focal 600 px is an explicit native fallback,
not the calibration of the supplied rectified video. Using the same fallback
for A and B establishes equal settings, not calibrated geometry. The actual
rectified calibration is fx=fy=444.43963608648465, cx=959.954470803018,
cy=543.9405836940584 at 1920x1074.

Calling 600 the physical/calibrated focal for this input is incorrect. This does
not prove every predicted joint rotation is wrong, nor does it establish that
using calibrated intrinsics will improve learned pose estimates without testing.

## Direct Evidence

1. [Rectification YAML](/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/rectified/calibrations/rectify_params.yaml:39)
   stores the calibrated rectified K; the same focal appears in P1, P2 and stereo
   depth parameters. This is not the original raw fisheye K or a 640-pixel preview K.
2. [Dataset loader](/home/user/roboego-hand-vis/roboego_hand_vis/datasets/wumei01.py:19)
   extracts fx/fy/cx/cy from K1_rect, then passes these into the manifest at line 82.
3. [FoundationStereo input loader](/home/user/roboego-hand-vis/roboego_hand_vis/depth/foundation.py:30)
   reads the manifest's left-camera intrinsics. The depth producer uses fx in
   [Z = scale * fx * baseline / disparity](/home/user/roboego-hand-vis/scripts/run_foundation_stereo_video.py:59).
4. The actual [full-60-second depth metadata](/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/depth_only/foundation_stereo_video_meta.json:7)
   records all four calibrated values and names the exact rectified MP4.
5. Native HaWoR uses 600 only when no focal argument/file is available:
   [fallback](/home/user/HaWoR/scripts/scripts_test_video/hawor_video.py:56).
   Our [experiment runner](/home/user/HaWoR/scripts/run_pre_frontend_comparison.py:40)
   explicitly selected 600; its executed source snapshot agrees, and the
   [actual run manifest](/home/user/HaWoR/outputs/pre_frontend_comparison/full_60s/manifest.json:128)
   records 600 and center (960,537). This was not automatic calibration loading.

## Independent Pixel Check

I ran `scripts/check_rectified_focal_evidence.py` against the raw and rectified
videos. It uses the recorded raw K/distortion and R1 to recreate rectification,
then compares the resulting pixels with the existing rectified MP4 at the same
frame. No video is rewritten and no network is loaded.

Mean absolute channel differences on 0..255 image values:

| Frame | Recorded K | Previous HaWoR K (f=600) | Recorded K With cy Minus 8 Again |
|---|---:|---:|---:|
| 0 | 4.206 | 42.849 | 14.725 |
| 100 | 2.665 | 44.377 | 12.189 |
| 599 | 4.166 | 43.446 | 13.309 |

This strongly supports the recorded K as the one matching the existing image
geometry. Compression/interpolation preclude exact equality. It is not an
independent physical recalibration or a hand-pose accuracy measurement. It also
rules against subtracting the listed crop offset from cy a second time.
Results: `outputs/pre_frontend_comparison/external_vio_depth_check/rectified_focal_evidence.json`.

## Consequences for Previous Results

- Observation ID selection, fragment preservation, observed/missing counts,
  source provenance and successful execution remain valid measured facts.
- The camera-space predictions are predictions under the fallback camera model,
  not established calibrated metric hand geometry for this recording.
- The earlier zero-difference native pilot tested equivalence to native HaWoR.
  Its projection test compared two implementations with the same assumed K.
  Neither test validated that K against this video's calibration.
- HaWoR uses focal and principal point in bbox conditioning supplied to its
  space-time/motion modules: [forward path](/home/user/HaWoR/lib/models/hawor.py:158),
  [bbox features](/home/user/HaWoR/lib/models/hawor.py:517). They also affect the
  [camera translation conversion](/home/user/HaWoR/lib/models/hawor.py:500).
  Updating only the final renderer or multiplying old translations is not an
  equivalent calibrated inference experiment.
- A calibrated rerun should preserve the old artifacts, use fresh A/B caches,
  keep the same network/checkpoint/MANO, and audit principal-point and left-hand
  mirroring conventions. Correct calibration is not itself a measured pose gain.

## Do Not Treat the Other Repository as One Camera Convention

The FoundationStereo stage uses calibrated K, but HaMeR camera-relative rendering
uses the model's nominal focal convention rather than this physical K. Its
camera translation and render focal must be interpreted together, not mixed
with calibrated VIO/depth coordinates without an explicit conversion.

There is also separate evidence of a missing-calibration fallback in saved
original/optimized HaMeR downstream depth alignment. This is not the same as
HaMeR's intentional nominal camera convention. See the detailed
[HaMeR focal audit](hamer_focal_evidence.md) for exact source and saved-artifact
evidence, scope, and the distinction from the RGB-only comparison videos.

No inference outputs, external source files, shared caches or existing reports
were modified during this check. No new commit or push was made.
