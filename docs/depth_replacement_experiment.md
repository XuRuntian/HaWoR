# Independent Depth Replacement

Branch: `experiment/depth-replacement`, directly based on frozen tag
`baseline/calibrated-native-backend-full60s-v1`, commit
`1042818a927cbd366e3397aefbd8d44ab5ca3ea7`. No VIO branch changes or VIO poses are
included. This compares metric-depth SOURCES for native camera-scale recovery,
not depth injection into DROID or per-hand depth alignment.

## Fixed Inputs

Use the complete 1920x1074 rectified left video, frames 0..1799 at 30 FPS, and
the correct stored rectified K. Both arms reuse exactly the optimized B frontend,
physical fragments/side mapping, camera-space HaWoR predictions, raw observed
masks, checkpoint, MANO and physical-identity-bounded native infiller policy.

Both arms share the exact full-sequence baseline DROID raw trajectory, 126
keyframe IDs, disparities and native hand masks. Baseline depths are its saved
Metric3D outputs; replacement depths are the existing 1800 left-reference
FoundationStereo metric Z arrays. Resolve actual original keyframe IDs through
`keyframe_original`, not keyframe-list positions. Neither depth network, the
hand network, association, detection nor DROID is rerun. The full baseline's
world and completed hand exports are reused read-only.

## Adapter Contract

Depth metadata must match the video, K, left reference, frame interval and file
pattern. Inventory every source array and hash it before and after the run.
Arrays must be float32 at (1074,1920). They already contain Z in meters including
the producer's inference-scale conversion; do not multiply by 0.5 again.

Preserve native depth/mask direct resizing to (592,328). Do not silently change
it to DROID RGB's resize-to-(592,331)-then-crop operation. Fully valid arrays and
uint8 hand masks reproduce the original resize exactly. If invalid/nonpositive
source samples contribute to a target depth, that target is excluded and kept
invalid, not replaced with artificial fitting evidence. No valid background
support is an explicit error. The same adapter is also regression-tested on all
saved baseline Metric3D maps.

Call original `est_scale_hybrid`, sigma=0.5, thresholds initially 0.4/0.7,
threshold carry between keyframes and median global aggregation. Empty bands
skip an undefined empty BFGS call before the normal threshold expansion;
100-attempt finite-positive failure guard never substitutes a scale. The
baseline replay must reproduce its archived per-keyframe scales, thresholds,
retry counts and global median within recorded tolerances.

Save unchanged raw DROID trajectory plus a separate scale, applied once by the
native camera loader. Preserve native world conversion and original infiller
functions. Observed camera geometry, hand availability and ownership intervals
must remain identical. Observed world change must equal exactly the change in
camera translation scale within floating precision; camera rotations must not
change. Generated hand geometry can change downstream, without becoming a
detector/observed gain.

## Known Input Limitation

Existing equal-index left/right timestamp pairs differ by approximately one
frame for most frames. Exposure timestamp semantics and historical producer
pairing remain uncertain. The user authorized this experiment assuming cached
depth is reasonably accurate; this run does not validate that assumption or
rank depth accuracy. Do not shift or rename saved depth arrays to "repair" them.
Any corrected stereo generation must be a new, separate cache/experiment.

The supplied metadata lacks the historical FoundationStereo command and
checkpoint hash. Consumed cached outputs are reproducibly pinned by SHA256, but
recreating the historical depth-generation process is not promised. Source code
and metadata hashes document what was available for inspection, not proof of
which code originally produced the cache.

## Run and Inspect

Restore the archives/assets in [the baseline index](experiment_branches.md).
Commit executable changes before running; a clean checkout and a fixed execution
commit are required. All later stages validate executable source hashes.

```bash
cd /home/user/HaWoR
conda activate hawor
python -m unittest discover -s tests -v
OUT="$PWD/outputs/pre_frontend_comparison/depth_replacement_full_60s/run_001"
python scripts/run_depth_replacement_stages.py --output "$OUT" --phase pilot
# Inspect pilot/depth_samples.png, the shared world/RGB videos and scale regression.
python scripts/run_depth_replacement_stages.py --output "$OUT" --phase full
```

The pilot driver prepares and hashes inputs, replays the full baseline scale
fit, and fits both sources on the same keyframes within frames 88..201. It runs
both pilot world conversions/infillers and common rendering. This local scale
fit is interface validation, not the full-sequence result. The full driver
validates the pilot, fits all 126 stereo keyframes, transforms all 1800 frames,
runs the frozen native infiller policy and exports/checks full comparisons.
Each GPU stage is a fresh process. Commands, runtime, stdout/stderr and failures
are saved in the run's logs; pre-output failures also remain in sibling
`process_logs/`. Changed-code retries must use a new numbered run directory.

## Results

`comparison.mp4` is RGB / optimized with Metric3D / optimized with FoundationStereo.
`baseline_overlay.mp4` and `stereo_overlay.mp4` use identical native full-K render
parameters. `world_observed_comparison.mp4` excludes completion effects;
`world_completed_comparison.mp4` includes them. World views share one fixed
virtual camera and per-arm first-camera SE(3) origins, with no fitted scale
alignment. Hand detail is best inspected in the RGB overlays.

The five requested windows and 918..926 are exported for all five videos.
`depth_samples.png` uses a common displayed depth range, with excluded mask
contours; visualization clipping does not affect fitting. `depth_scale_comparison.png`
shows per-keyframe scales and the resulting camera paths. Exact transforms,
observed/generated/missing masks, source hashes, model/code/environment versions,
runtime, numerical checks and known issues are in NPZ/JSON archives and report.
These are availability and consistency diagnostics, not detection recall or
pose accuracy. Large datasets, predictions, videos and licensed models remain
outside Git; result publication records their paths and hashes.
