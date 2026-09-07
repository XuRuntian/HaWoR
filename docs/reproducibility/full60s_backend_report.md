# Calibrated Native Backend Frontend Comparison

Completed the same rectified frames 0..1799, 1920x1074, 30 FPS. Both branches use K=[[444.43963608648465, 0.0, 959.954470803018], [0.0, 444.43963608648465, 543.9405836940584], [0.0, 0.0, 1.0]].
The original frontend and optimized frontend camera predictions are reused exactly from `/home/user/HaWoR/outputs/pre_frontend_comparison/calibrated_full_60s`.
Native masked DROID, Metric3D scale recovery, world conversion and the native infiller were actually executed.
No external VIO, FoundationStereo, hand-depth correction, extra pose filter, gate, network or weight modification.

## What Is Compared

A: native frontend -> calibrated HaWoR camera predictions -> native per-hand union masks -> native
masked DROID -> Metric3D and native scale fit -> original whole-sequence two-side infiller.

B: optimized physical-fragment frontend -> calibrated HaWoR predictions -> the same mask procedure,
DROID and Metric3D algorithms -> the ORIGINAL infiller within explicitly known ownership intervals.
All observed B outputs are preserved. Same-side conflicts, fragment boundaries and side changes are
not silently merged. Generated predictions carry separate masks and their physical owner.

This is a full-backend experiment WITH AN EXPLICIT B INFILLER INTERFACE ADAPTATION, not an unmodified
two-anatomical-track B demo. Native A keeps original slot/JSON overwrite semantics. Lossless pre-infiller
observations remain in `world_observed.npz`; do not confuse their count with dense anatomical slots.
A's slots are anatomical-side hypotheses; B's are anonymous physical slots. Neither is a common
ground-truth identity set, and availability differences are not detection recall.
`world_observed_comparison.mp4` excludes completion effects; `world_completed_comparison.mp4` includes them.

## Native Behavior and Necessary Interfaces

- Calibrated fx/fy/cx/cy enter both DROID and Metric3D, not only hand inference.
- DROID uses each branch's own native hand masks. Therefore frontend effects also propagate through
  dynamic-region masking, keyframe selection, camera trajectory and scale estimation.
- Native DROID RGB resize/crop, mask resizing, Metric3D direct depth resizing and scale hyperparameters
  are preserved. Their existing small resize-versus-crop discrepancy is not silently fixed here.
- Scale estimation retains native threshold expansion and median aggregation; a 100-retry finite-positive
  guard fails explicitly instead of hanging forever. Per-keyframe retries and thresholds are recorded.
- No hand observations are promoted from missing to observed by the infiller. Native's exclusive end-index
  behavior is retained and generated availability is not detection recall.
- B completion has the same weights, 120-frame horizon, native preprocessing and inference. Its available
  context is restricted by ownership, so completion differences cannot be attributed to detection alone.
- B's four same-side observed conflicts are frames 1258, 1346, 1348 and 1545. The ownership-conflict
  exclusion also includes frame 1347 between observations; that is not a fifth observed conflict.

## Results

| Quantity | Native A | Optimized B |
|---|---:|---:|
| Lossless pre-infiller observed instances | 2364 | 2872 |
| Post-interface observed output instances | 2338 | 2872 |
| Generated output instances | 1261 | 42 |
| Missing slots | 1 | 686 |
| Infiller seconds | 6.038817319000373 | 5.026340006006649 |
| DROID keyframes | 129 | 126 |
| DROID seconds | 273.40885163300845 | 267.3330707260029 |
| Median metric scale | 0.267191082239151 | 0.26340703666210175 |
| Metric3D + scale seconds | 131.21439525901224 | 128.381706649001 |
| Scaled camera path length | 8.273460388183594 | 8.241960525512695 |
| Scaled camera net displacement | 0.44432589411735535 | 0.4433838427066803 |

All world transforms passed camera->world->camera and native MANO-parameter reconstruction checks.
Observed geometry is preserved through completion; errors are in `diagnostics.json`. These are numerical
consistency checks, not pose/depth/trajectory ground-truth accuracy. DROID/Metric3D scale remains estimated.
Raw median scale factors are attached to different DROID reconstructions and should not be compared as
standalone depth-model accuracy. Trajectory differences after first-pose rigid alignment are recorded
separately in diagnostics; no fitted scale is used to hide their differences.

## Known Generated-Pose Issues

Native completed outputs have nonpositive camera-space joint depth at frames
`[975]`; optimized outputs at
`[922]`. These generated outputs
are retained and flagged, not filtered, treated as detections, or claimed to be accurate. Finite world
coordinates and valid camera transformations do not establish plausible hand completion. Existing
frontend side ambiguity and degenerate/off-image observations also remain unchanged.

## Viewing

`comparison.mp4`: RGB / Native full backend / Optimized backend with the documented identity adapter.
`native_overlay.mp4` and `optimized_overlay.mp4`: original video dimensions, common shading and opacity.
Observed hands use pink/cyan; generated hands use pale variants. Headers report separate OBS/GEN counts.
`world_observed_comparison.mp4` and `world_completed_comparison.mp4`: Native left, Optimized right.
Both use a shared fixed virtual camera; each branch's first camera defines its origin using SE(3) only.
No scale alignment, dynamic view tracking or external world registration is applied. Green is camera path.
Virtual bounds derive from all observed wrists and camera poses, not from generated outliers.
The displayed reference grid is a coordinate aid, not an estimated physical floor.
Five inclusive windows 88-94, 100-114, 176-182, 193-201, 596-602 contain all five videos.

All full videos passed frame-count/FPS checks and complete FFmpeg decoding. Rendering took
781.672 seconds before validation/window exports. All 3837 protected
old-result, source, foreign-cache and model files retained their pre-run SHA256.
Source snapshots for each executed stage are under `provenance/`; process logs and commands under `logs/`.
The focused unit suite passed; its complete output is `provenance/environment/unittest.log`.
The first full B export encountered an empty generated-index dtype error in the new adapter.
The adapter was corrected and its empty-output case regression-tested; B's original infiller was
actually rerun, not replaced with cached timing. Failed and successful attempt logs are retained.

## Reproduce

Use the configured hawor environment and the unchanged calibrated camera archives; use a fresh output.

```bash
conda activate hawor
python scripts/run_native_backend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --stage prepare
python scripts/run_native_backend_stages.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --phase pilot
python scripts/run_native_backend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --stage render --phase pilot
# Inspect both pilot world validations, infiller audits and videos before full inference.
python scripts/run_native_backend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --stage validate
python scripts/run_native_backend_stages.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --phase full
python scripts/run_native_backend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --stage render
python scripts/run_native_backend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --stage report
mpv --speed=0.5 '/home/user/HaWoR/outputs/pre_frontend_comparison/calibrated_native_backend_full_60s/comparison.mp4'
```
