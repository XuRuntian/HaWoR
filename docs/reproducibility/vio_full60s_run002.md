# Optimized HaWoR: Native Camera Backend vs OpenVINS

Completed rectified frames 0..1799, 1920x1074, 30 FPS. Output: `/home/user/HaWoR/outputs/pre_frontend_comparison/vio_replacement_full_60s/run_002`.
Execution commit: `45aab4f1217eb286162ba5be5fe604929ebd9129`. Baseline commit: `1042818a927cbd366e3397aefbd8d44ab5ca3ea7`.
Correct K: `[[444.43963608648465, 0.0, 959.954470803018], [0.0, 444.43963608648465, 543.9405836940584], [0.0, 0.0, 1.0]]`. Full configuration, source hashes and schema: `manifest.json`.

## Controlled Replacement

Both arms use identical frozen optimized B HaWoR camera predictions, bbox/observation IDs,
physical fragments, handedness, masks and identity-bounded native infiller invocation intervals.
The baseline reuses its completed native masked DROID + Metric3D trajectory AND hand completion.
VIO replaces the complete scaled camera-trajectory input with existing metric OpenVINS poses.
DROID, Metric3D, detection, association and the HaWoR hand network were not rerun.
The VIO arm actually reruns native world conversion and native infiller with the unchanged checkpoint.
There is no hand-depth correction, stereo depth, new filtering, gate, MANO or network change.

VIO camera timestamps match original source frames 12270..14069. T_world_camera already includes
the IMU-camera extrinsic and timestamp convention. Neither is applied twice. Raw-left to rectified-left
conversion is T_world_rect = T_world_raw @ blockdiag(R1.T,1), with scale=1. Original source world
coordinates remain in predictions. Each first-camera SE(3) origin is used only for comparative rendering
and trajectory diagnostics; no fitted scale alignment is applied.
The float64 source/adapter archive is retained separately. Native execution cameras are float32,
matching the original HaWoR infiller's MANO tensors; no native model code is changed for this conversion.

## Missing Cameras Are Not Missing Hands

1798 source poses are available. Camera frames 550 and 559 use timestamp-weighted linear translation
and rotation SLERP between their immediate neighbors. No rows are dropped or nearest poses substituted.
`cache/vio_pose_adapter.npz` preserves source indices, exact/source flags, interpolation flags and anchors.
`arms/*/camera_poses.npz` has frame-level camera status; prediction `row_camera_status.npz` maps that status
to completed output rows, separately from hand observed/generated masks. Interpolated camera poses are
not additional hand detections. Trajectory differences are reported on the common 1798 source frames;
speed statistics exclude edges touching interpolated camera frames. Full path length explicitly includes them.

| Quantity | Frozen baseline | VIO replacement |
|---|---:|---:|
| observed_instances | 2872 | 2872 |
| generated_instances | 42 | 42 |
| missing_slots | 686 | 686 |
| camera_source_frames | 1800 | 1798 |
| camera_interpolated_frames | 0 | 2 |
| camera_missing_frames | 0 | 0 |
| observed_hands_on_interpolated_camera_frames | 0 | 2 |
| nonpositive_depth_frames | [922] | [] |

Hand availability masks and completion ownership are exactly identical. Generated hand geometry can
change because the same infiller receives a different world trajectory. Such changes are a downstream
effect of the camera replacement, not a new frontend gain. `world_observed_comparison.mp4` isolates the
world placement of unchanged observed hand geometry; `world_completed_comparison.mp4` also includes
that downstream completion effect. Observed RGB reprojection is unchanged apart from floating precision.

## Validation and Limitations

Before full inference, both arms ran frames 88..201 using slices of their FULL camera trajectories,
with identical pilot completion context, not the historical short-window DROID trajectory. The independent
full-timeline camera adapter projection check also covers both hands and the two camera gaps.
Pilot videos were checked for nonblank hands and shared projection. Full videos passed full decoding,
frame count/FPS checks and sampled RGB/column alignment. 1911 protected files
retained their hashes; foreign repository status is unchanged. Focused unit tests passed.

Generated behind-camera hands, if listed above, are retained and flagged rather than filtered.
Source handedness ambiguity and physical fragment boundaries remain unchanged. Camera interpolation
may also influence other generated frames inside an infiller invocation, not only the two gap rows.
The original B infiller restriction is deliberately frozen; this is not an unrestricted native demo.
VIO accuracy is assumed for integration, not established by these tests. Trajectory lengths, smoothness
or more pleasing world motion are not ground-truth pose accuracy. A metric camera path does not make
HaWoR's monocular hand translation metrically correct. The separate FoundationStereo experiment has NOT run.

## Outputs and Runtime

- `comparison.mp4`: RGB / optimized frontend + native camera backend / same frontend + VIO.
- `baseline_overlay.mp4`, `vio_overlay.mp4`: full-size common camera renderer.
- `world_observed_comparison.mp4`, `world_completed_comparison.mp4`: shared fixed view, baseline left/VIO right.
- `camera_trajectory_comparison.png`, `aligned_camera_trajectories.npz`: common rigid-origin trajectory diagnostics.
- `arms/{baseline,vio}/predictions/optimized/`: observed/completed world exports and camera-row status.
- `windows/`: requested five windows plus 544..564 covering camera gaps and 918..926 covering old generated outlier.
- `logs/`: exact commands, per-stage wall times, return codes; `provenance/`: code snapshots and environment.

VIO cache adaptation excludes the historical OpenVINS run cost, which is not measured here. Baseline
inference is cached, not zero-cost inference. Full VIO native infiller time:
3.956058 seconds;
common full rendering: 776.081 seconds before decode/window exports.
The stage JSON logs also include process startup, export and validation costs.

Reproduce from the execution commit using `docs/vio_replacement_experiment.md` with a fresh output.
Playback: `mpv --speed=0.5 '/home/user/HaWoR/outputs/pre_frontend_comparison/vio_replacement_full_60s/run_002/world_observed_comparison.mp4'`.

## Publication and Manual Inspection

This is an exact generated-report snapshot above, followed by a publication addendum.
The executed code is commit `45aab4f1217eb286162ba5be5fe604929ebd9129`; the result publication commit only adds
this report and its JSON index. Tag: `result/vio-replacement-full60s-run002`.

Manual screenshots inspected: pilot RGB/world frame 91 and world frame 196; full
camera frames 550, 559 and 922; world frames 599 and 1780; the trajectory plot.
Camera interpolation labels are present, and the inspected observed projections
coincide. The generated hand at frame 922 STILL visibly deviates from the real
hand in the VIO result. Passing positive-depth checks does not make it accurate.
The fixed full-sequence world view has small hand meshes; use RGB overlays for
hand detail and the trajectory plot for camera movement. This inspection is
recorded in the local `visual_inspection.json`, not an accuracy evaluation.

The first `run_001` pilot failed on float64 VIO versus float32 native infiller
tensors, before full execution. Its logs and outputs remain intact. Commit
`45aab4f` adds the serialization-only correction and regression test; all seven
stages in the fresh `run_002` passed. The final suite contains 24 passing tests.

| Trajectory diagnostic | Baseline | VIO |
|---|---:|---:|
| Full path length (m), including camera interpolation | 8.241960 | 14.910369 |
| Net displacement (m) | 0.443384 | 0.757852 |

These are estimates, not ground-truth error or a scale-accuracy ranking. Rotation
difference on common source frames has median 0.746511 degrees; position
difference has median 0.161760 m after first-camera SE(3) alignment only.
The median generated-joint camera-space change is 0.001279 m; the maximum is
0.613424 m. This outlier is a reason to inspect completion, not claim improvement.

Selected artifact hashes and exact stage commands/times are in
[vio_full60s_run002.json](vio_full60s_run002.json). The external depth-replacement
branch remains at the original baseline and has not executed a new experiment.
