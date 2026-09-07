# VIO Replacement Experiment

Branch: `experiment/vio-replacement`. Parent:
`baseline/calibrated-native-backend-full60s-v1` (commit `1042818a927cbd366e3397aefbd8d44ab5ca3ea7`).
This is the VIO-only experiment, not the independent FoundationStereo replacement.

## Frozen Variables

Both arms use the completed optimized B frontend, calibrated K and camera-space
HaWoR prediction archive from the complete rectified 1800-frame video. Network,
checkpoint, MANO, crop/projection, physical fragments, observed/missing masks and
the baseline B identity-bounded native infiller policy are unchanged.

Baseline: read-only reuse of `calibrated_native_backend_full_60s` optimized B
DROID/Metric3D camera trajectory and observed/completed world predictions.
VIO: timestamp/frame matched metric OpenVINS poses replace the scaled camera
trajectory. Native world conversion and native infiller actually run again.
The existing baseline helper functions are invoked without modification. Each
arm has its own isolated directory with the helper's `optimized` branch schema;
this label still means optimized frontend, not camera method.

No DROID, Metric3D, detector, association or HaWoR hand-network inference runs.
There is no new stereo depth input, hand-depth translation correction or filter.

## Camera Contract

Original clip frames 12270..14069 are matched using rectified left timestamps,
checked against the raw session and clip tables. Source poses already include
the IMU-camera extrinsic and camera timestamp convention. Do not apply either
twice. Convert raw-left camera to rectified-left with
`T_world_rect = T_world_raw @ blockdiag(R1.T,1)` and serialize camera-to-world
`[tx,ty,tz,qx,qy,qz,qw]`, `scale=1`. The exported K is used without subtracting
the historical rectification crop offset a second time.

Only isolated interior single-frame camera gaps may be filled, using linear
translation and SciPy SLERP weighted by actual timestamps. Expected gaps are
frames 550 and 559; any different gap pattern fails this fixed experiment.
No nearest-neighbor substitution, row deletion, endpoint extrapolation or
multi-frame gap bridging is allowed. Retain source indices, interpolation
anchors/weights and camera source/interpolated/missing masks. Camera flags are
distinct from hand observed/generated/missing masks and are exported per row.

Pose estimation accuracy is assumed for this integration experiment, not
validated as ground truth. A metric camera trajectory does not calibrate the
hand network's inferred scale or translation. Both arms' first camera origins
are rigidly normalized for shared world rendering and trajectory comparison;
there is no fitted scale alignment. Raw VIO world coordinates remain in exports.

## Run

Restore the frozen archives, source timestamps/calibration/VIO and model assets
described in [the baseline index](experiment_branches.md). The configured `hawor`
environment includes SciPy, native CUDA extensions and the unchanged infiller.
Commit all executable changes first; the runner requires a clean checkout and
pins that execution commit and source hashes for every stage.

```bash
cd /home/user/HaWoR
conda activate hawor
python -m unittest discover -s tests -v
OUT="$PWD/outputs/pre_frontend_comparison/vio_replacement_full_60s/run_001"
python scripts/run_vio_replacement.py --output "$OUT" --stage prepare
python scripts/run_vio_replacement.py --output "$OUT" --stage pilot
python scripts/run_vio_replacement.py --output "$OUT" --stage render-pilot
# Inspect pilot screenshots/videos before full execution.
python scripts/run_vio_replacement.py --output "$OUT" --stage validate
python scripts/run_vio_replacement.py --output "$OUT" --stage full
python scripts/run_vio_replacement.py --output "$OUT" --stage render
python scripts/run_vio_replacement.py --output "$OUT" --stage report
```

Use a fresh numbered run directory; never overwrite a completed run. Stage JSON
logs record commands, commits, elapsed times and failures. For fresh-process
execution with full stdout/stderr archival, use the equivalent driver:

```bash
python scripts/run_vio_replacement_stages.py --output "$OUT" --phase pilot
# Inspect the pilot before the full driver, which validates then runs the full sequence.
python scripts/run_vio_replacement_stages.py --output "$OUT" --phase full
```

Do not run both the manual commands and driver into the same output. Process
logs are copied into the run's `logs/` and also retained under the output parent's
`process_logs/` so failures before output creation remain traceable.
Pilot frames 88..201 use slices of each full camera trajectory and identical
pilot ownership context. Baseline pilot world conversion/filling is rerun for
that controlled window only; the full baseline uses its exact frozen results.
The preparation stage independently checks all 1800 camera transforms and both
hands' projection, including the camera gaps, before any full hand completion.

## Outputs

`comparison.mp4` is RGB / frozen optimized baseline / optimized with VIO.
Both overlays use native full-K rendering at 1920x1074, 30 FPS. Observed colors
are pink/cyan; generated hands use the baseline's pale variants. Interpolated
camera frames are explicitly labeled.

`world_observed_comparison.mp4` shows world placement of unchanged observed hands.
`world_completed_comparison.mp4` also shows the downstream effect on native
completion. Both use one fixed virtual camera and common grid, without fitted
scale. Camera trajectory arrays and a static comparison plot are exported.
Five requested windows plus 544..564 and 918..926 are exported for all videos.

`arms/{baseline,vio}/predictions/optimized/` contains observed/completed world
predictions, infiller ownership audit and per-output camera status. The baseline
NPZ/audit files are read-only symlinks to frozen results. `arms/*/camera_poses.npz`
contains camera-level availability; `cache/vio_pose_adapter.npz` retains the
source mapping and interpolation details. Model/world units are documented
without claiming hand accuracy. `diagnostics.json`, `report.md`, `logs/` and
`provenance/` contain the executed result, rather than anticipated outcomes.

Report observed/generated/missing hands separately from interpolated cameras.
Compare trajectories on common 1798 source frames; exclude edges touching
interpolated cameras for velocity statistics. Full path length includes the two
camera interpolations and is labeled accordingly. Infiller ownership and hand
availability must match exactly, while generated geometry may change. Keep and
report nonpositive-depth predictions; do not filter them into an apparent gain.
