# Calibrated Frontend Comparison

This opt-in experiment reruns the same camera-only A/B comparison with the
recorded rectified intrinsics, keeping each branch's observations fixed.
The original default-camera experiment remains unchanged.

The publication tag and required frozen archives are indexed in
[Experiment Baseline and Branches](experiment_branches.md).

| Experiment | Native Frontend | Optimized Frontend |
|---|---|---|
| Default camera: f=600, center=(960,537) | Existing A | Existing B |
| Calibrated camera: f=444.439636, center=(959.954471,543.940584) | New A | New B |

The image is the same 1920x1074 rectified left video, frames 0..1799 at 30 FPS.
The calibration source is `rectified/calibrations/rectify_params.yaml` in the
original full-60-second diagnostics. See [calibration evidence](focal_calibration_evidence.md).

## Fixed Inputs

New A reads the original saved native detector/tracker results; new B reads the
original selected observations and physical fragments. A and B do not have the
same observations as each other. Neither detector nor association is rerun.
The checkpoint, learned architecture and MANO assets are unchanged. Existing
JPEGs are shared read-only, with isolated new reconstruction/chunk caches.

Native side merging, optimized fragment separation, contiguous chunks, padding,
native bbox interpolation, and observed/generated/missing definitions are the
same as the reference experiment. World/SLAM/VIO/depth and infilling remain off.

## Camera Plumbing

The optional `camera_intrinsics` argument to `hawor_motion_estimation` validates
a finite, square-pixel, zero-skew K and passes it through inference and rendering.
Existing callers retain default behavior.

In calibrated mode, the evaluation dataset keeps both the original principal
point and the image width. A left-hand image crop is mirrored exactly as before,
but the conditioning principal point is also mirrored: `cx' = width - 1 - cx`.
After the network forward pass, the bbox center is restored about image width,
and original-camera translation/projection uses the original principal point.
This changes camera metadata handling in `HAWOR.forward_step`, not its learned
layers or parameters. It avoids assuming that `2*cx` equals image width.
The native renderer's existing full-K interface is used for both calibrated views.

The calibrated dataset copies each bbox center before mirroring, so repeated
index access is stable. Legacy behavior is left unchanged for compatibility.
Intrinsics enter bbox conditioning in the space-time/motion path, so the new
outputs come from actual inference, not rescaled old translations.

## Verification

The pilot uses reference PILOT tracks for 88..114, before full reconstruction.
It checks default-mode reproduction against both archived A/B predictions and
the pinned original native function. All camera/pose/shape parameter differences
in those default-mode regression checks were exactly zero.

Calibrated pilot checks cover both hands, exact crop tensor equality, stable
repeated crop access, mirrored principal points, image-width bbox unmirroring,
positive camera depth, and independent pinhole/PyTorch3D projection.
The full run then uses reference FULL tracks and checks identical source
metadata, frame/side/bbox/slot arrays, chunks and masks against each old branch.
Focused tests also execute the real forward geometry with a lightweight network
stub, using deliberately off-center intrinsics to expose mirror errors.

## Run and View

Prerequisites: the configured `hawor` environment, licensed MANO/checkpoint files,
source calibration, and the completed original experiment under
`outputs/pre_frontend_comparison/full_60s/`. The original input/video/model hashes
are validated. Use a fresh output directory for a new run.

```bash
conda activate hawor
python -m unittest discover -s tests -v
python scripts/run_calibrated_frontend_comparison.py --stage pilot
# Inspect the pilot before continuing.
python scripts/run_calibrated_frontend_comparison.py --stage full
python scripts/run_calibrated_frontend_comparison.py --stage render
mpv --speed=0.5 outputs/pre_frontend_comparison/calibrated_full_60s/four_way_default_vs_calibrated.mp4
```

Default new output directory: `outputs/pre_frontend_comparison/calibrated_full_60s/`.
`--output` allows another directory under HaWoR, outside the reference result.
Each run archives executable sources, configuration, versions, hashes and an
interface diff. Its generated `report.md` and `diagnostics.json` are the authority
for final counts, runtimes and measured prediction changes.

- `native_overlay.mp4`, `optimized_overlay.mp4`: calibrated full-size overlays.
- `comparison.mp4`: RGB / calibrated Native / calibrated Optimized.
- `four_way_default_vs_calibrated.mp4`: top=default, bottom=calibrated;
  left=native, right=optimized, at 3840x2148, 30 FPS.
- `windows/`: inclusive 88-94, 100-114, 176-182, 193-201, 596-602; four videos per window.
- `predictions/`: per-call and consolidated camera-space outputs with provenance/masks.
- `pilot/validation.json`, `full_observation_invariants.json`, `diagnostics.json`.

## Interpretation

OldA/newA and oldB/newB show camera-interface sensitivity within fixed observation
chains, including principal-point/mirroring adaptation, not an isolated focal-only
effect. NewA/newB compares frontend chains under the same calibrated camera.
It still mixes detector source, observation selection, trajectory adaptation and
changed temporal context; it is not a consolidation-only ablation.

Observed output counts remain 2364 for A and 2872 for B, with zero generated outputs.
Availability statistics are not recall. Calibrated inputs are not proof of metric
hand accuracy or improved pose without ground truth. Known ambiguous handedness,
four zero-width source bboxes, and native duplicate/export-collision behavior are
preserved and documented rather than silently corrected.
