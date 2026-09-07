# Optimized HaWoR: Metric3D vs Cached FoundationStereo

Completed rectified frames 0..1799, 1920x1074, 30 FPS. Output: `/home/user/HaWoR/outputs/pre_frontend_comparison/depth_replacement_full_60s/run_001`.
Execution commit: `54decd4554fe22f2fdab405d08ba1f2f223584cb`. Baseline commit: `1042818a927cbd366e3397aefbd8d44ab5ca3ea7`.
Correct K: `[[444.43963608648465, 0.0, 959.954470803018], [0.0, 444.43963608648465, 543.9405836940584], [0.0, 0.0, 1.0]]`. See `manifest.json` for exact inputs, schema and hashes.

## Controlled Replacement

Both arms use identical optimized B observations, physical fragments, calibrated camera predictions,
the EXACT same frozen DROID trajectory/rotations/keyframes/disparities/masks, and the same identity-bounded
native infiller contexts/checkpoint. The baseline reuses its frozen full world/completed outputs.
Only the metric Z-depth SOURCE in the original scale estimator changes: Metric3D to cached FoundationStereo.
All 126 DROID keyframes are resolved by their original video frame IDs, never keyframe-list position.
The stereo arm actually executes native scale estimation, world conversion and native infiller.
No detector, association, HaWoR hand network, DROID, Metric3D or FoundationStereo network inference is rerun.
No VIO input, per-hand depth translation correction, new pose filter, gate, checkpoint or MANO change.

## Interface and Validity

Saved stereo values are left-reference camera-axis Z in meters. The producer already applied inference
scale; no extra 0.5 factor, disparity inversion or focal rescaling is applied to those metric arrays.
Native direct resize to 592x328 is retained equally; DROID RGB's 592x331-then-bottom-crop discrepancy
is documented, not silently corrected. Native uint8 hand-mask resizing is preserved.
Invalid/nonpositive depth contributions are excluded from fitting and retained as invalid in adapted
caches; they do not become fake zero-depth support. Valid full images reproduce native resampling exactly.
Scale fitting calls the unmodified est_scale_hybrid with sigma=.5, starting thresholds .4/.7, threshold
carry across keyframes, and median aggregation. Empty bands skip invalid BFGS calls before normal threshold
expansion; finite positive estimates are required within 100 attempts, with no fallback scale.
The full Metric3D cache replay reproduces the archived baseline scale and per-keyframe behavior;
measured differences are in `cache/baseline_scale_regression.json`.

Scale is saved separately from the UNCHANGED raw DROID trajectory and applied once by load_slam_cam.
Independent archive checks show that observed world-vertex changes equal only the changed camera
translation scale. Rotations and camera-space observations remain unchanged. Native generated geometry
may change downstream because its world-motion input changes, not because hand observations improved.

| Quantity | Metric3D baseline | FoundationStereo replacement |
|---|---:|---:|
| observed_instances | 2872 | 2872 |
| generated_instances | 42 | 42 |
| missing_slots | 686 | 686 |
| scale | 0.26340703666210175 | 0.4085933715105057 |
| observed_camera_geometry_max_error | 7.152557373046875e-07 | 5.960464477539062e-07 |
| nonpositive_depth_frames | [922] | [] |

Full estimated path lengths: baseline 8.241960 m;
stereo 12.784814 m.
Stereo/baseline scale ratio: 1.551186.
These are model/trajectory estimates, not ground-truth errors or depth-model accuracy rankings.

## Stereo Timing and Known Limits

The existing same-index left/right timestamps differ by more than 20 ms for
1798 of 1800 pairs; median right-minus-left is
-33.333302 ms.
Exposure timestamp semantics and historical producer pairing remain unresolved. This run proceeds under
the user's assumption that the depth cache is reasonably accurate, but does NOT validate that premise.
No existing arrays are shifted/renamed/repaired and no stereo cache is recomputed. Correct pairing, if
needed, requires a separate isolated depth-generation experiment. Historical FoundationStereo command
and checkpoint hash are absent from supplied metadata; cache hashes pin consumption, not producer reruns.

Generated behind-camera hands, if listed above, are retained rather than filtered into apparent gains.
Physical/handedness ambiguity and original B infiller ownership restrictions remain unchanged.
Observed/generated/missing masks and all completion intervals are exactly equal across these two arms.
More plausible scale or a visually preferable generated hand does not establish pose accuracy. Metric3D's
native role here is camera scale recovery, not per-joint depth alignment, so this does not test all possible
uses of stereo depth for hand/object geometry.

## Validation, Outputs and Reproduction

Pilot frames 88..201 fit each source on the same selected full-DROID keyframes, then run identical
pilot world/infiller contexts. Both-hand camera/world round trips, raw trajectory identity and native
baseline scale replay pass before full inference. Full processing uses every one of the 126 keyframes
and exports all 1800 frames. Main videos pass complete decoding, frame/FPS/size checks and sampled
RGB/column alignment. 3833 protected files retain their hashes and the foreign
repository status is unchanged. The focused unit suite passes; its log is under `provenance/`.

- `comparison.mp4`: RGB / optimized frontend with Metric3D / same frontend with FoundationStereo.
- `baseline_overlay.mp4`, `stereo_overlay.mp4`: full-resolution common native renderer.
- `world_observed_comparison.mp4`, `world_completed_comparison.mp4`: baseline left, stereo right.
- `depth_scale_comparison.png`, `aligned_camera_trajectories.npz`: scale/trajectory diagnostics.
- `arms/{baseline,stereo}/predictions/optimized/`: world observations, completions and ownership audits.
- `cache/{baseline_replay,pilot,full}/`: per-keyframe adapted depth/validity/mask caches and scale audits.
- `windows/`: five requested windows plus 918..926 around the known generated outlier.
- `logs/`, `provenance/`, `diagnostics.json`: exact commands, stage times, sources, versions and checks.

The world view uses one shared fixed camera and each first-camera SE(3) origin, with NO fitted scale
alignment. It is an overview with small hands; use RGB overlays for hand detail. Reference grid is not ground.
Full rendering took 814.891 seconds before decoding/window exports. Depth-cache adaptation
and scale timings exclude historical Metric3D/FoundationStereo inference; baseline reuse is not zero-cost
inference. All native inference-stage times remain separately recorded.

Reproduce from the execution commit using `docs/depth_replacement_experiment.md`, with a new output.
Playback: `mpv --speed=0.5 '/home/user/HaWoR/outputs/pre_frontend_comparison/depth_replacement_full_60s/run_001/world_observed_comparison.mp4'`.

## Post-run Inspection Addendum

This section was added after all seven execution stages passed, without changing
the executed scripts, predictions, depth cache, masks or videos. The full suite
contains 25 passing tests. All five full videos contain 1800 frames and pass
complete decoding; 30 clips cover six inclusive diagnostic windows.

The full 1800-frame depth inventory contains 353 frames with invalid pixels,
18,738,792 invalid source pixels in total. Among the 126 used keyframes, 31 have
invalid source pixels; the adapter excludes 95,081 resized pixels in aggregate.
These validity exclusions are not an extra pose gate or a hand-recall gain.

Four used stereo depth maps (890, 928, 931, 1004) have even their ORIGINAL
maximum depth below 0.1 m. Their original medians are approximately 0.05169,
0.04873, 0.05270 and 0.04979 m respectively. The original RGB at frame 890 is
visibly motion blurred; its saved depth is only 0.04548..0.06536 m across the
whole image. These are concerning source-cache values, not a resize-induced
unit error. Blur and timestamp mismatch are possible concerns, not proven
causes. No source frames were removed or repaired for this comparison.

Frame 890 triggers four empty-band threshold expansions, changing native
near/far from 0.4/0.7 to approximately 0.0/1.1 m. The ORIGINAL carry policy
keeps that expanded band for the remaining keyframes; baseline replay needs
no expansion. Consequently the final scale reflects both the changed depth
source and the native estimator's response to it. The median aggregation does
not eliminate that carried-threshold interaction. No fixed-band alternative,
outlier rejection or counterfactual scale was substituted.

Generated-joint camera-space change is median 0.000690 m, p95 0.008421 m,
maximum 0.162189 m. The maximum occurs at generated left frame 1344.
Inspected generated hands at frames 922 and 1344 remain visibly incorrect
in BOTH arms. Stereo frame 922 has positive minimum joint Z (0.013706 m,
versus baseline -0.024915 m), but still shows a badly placed generated hand.
This numerical sign change is not evidence of improved hand accuracy.

Pilot RGB/world screenshots, full RGB frames 599, 922 and 1344, world frames
599 and 1780, depth examples and the scale plot were manually inspected.
The common world overview has small hand meshes. The numerical trajectory
plot makes the 1.551186 scale ratio clearer than those tiny meshes.
See `visual_inspection.json` for inspected files and supplementary extraction
commands. One optional screenshot command first failed on shell/filter
escaping, then succeeded; no experiment stage or inference was rerun.

Cached stereo read/adaptation/native scale fit plus fit-cache export took
4.856793 s; world conversion took 2.385076 s; native filling plus export took
6.474061 s. The full rendering driver including checks/window exports took
847.326837 s. These timings exclude historical depth-network inference and
must not be interpreted as an end-to-end speedup over Metric3D.

Publication tag: `result/depth-replacement-full60s-run001`.
The result publication only adds documentation and a selected-artifact JSON
index; execution remains commit `54decd4554fe22f2fdab405d08ba1f2f223584cb`.
The VIO experiment remains on its independent branch and is not combined here.
