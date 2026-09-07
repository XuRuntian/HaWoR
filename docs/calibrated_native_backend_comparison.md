# Calibrated Native Backend Comparison

This experiment extends the completed calibrated camera-space frontend comparison
through HaWoR's own masked DROID, Metric3D, world conversion and infiller.
It does not replace either backend with external VIO or stereo depth.

The frozen Git tag, independent replacement branches and archived-result hashes
are indexed in [Experiment Baseline and Branches](experiment_branches.md).

## Scope

Both branches reuse the frozen camera-space predictions in
`outputs/pre_frontend_comparison/calibrated_full_60s/`. These were actually inferred
from the same complete rectified video, frames 0..1799, at 1920x1074 and 30 FPS,
with the same checkpoint and calibrated K. The native branch's detector/tracker
results and the optimized branch's selected observation IDs remain unchanged.

The experiment renders the native per-hand silhouette union from those archived
MANO vertices. Each branch runs its own native masked DROID and native Metric3D
scale recovery, because changed hand masks are part of the frontend's downstream
effect. Sharing a single camera trajectory would be a different controlled
experiment. No hand/camera/depth/vertex predictions from HaMeR are substituted.

Calibrated fx, fy, cx and cy are passed to both DROID and Metric3D. Native resize,
masking, scale-fit thresholds and DROID parameters are retained, including the
existing small discrepancy between RGB resize-plus-crop and depth/mask direct
resizing. The scale retry loop has an explicit finite-positive failure guard;
it never silently substitutes a scene scale or a different trajectory.

## Infiller Identity Boundary

Native A invokes the unmodified native whole-sequence two-side infiller and its
original JSON/slot overwrite policy. Its pre-infiller lossless observed outputs
remain separately available; dense anatomical slots and observation instances
are not interchangeable counts.

Optimized B cannot be serialized into two permanent anatomical tracks without
discarding its physical identity/fragment contract. The adapter therefore:

- Retains every observed prediction, including same-side conflicts.
- Resolves side from observation metadata, never from physical slot 0/1.
- Splits ownership at fragment and handedness-change boundaries.
- Does not fill unowned intervals or intervals with conflicting side ownership.
- Invokes the ORIGINAL native infiller only within intervals with two known
  owners and at least two observed anchors per side, when any values are missing.
- Writes generated predictions separately, linked to the owning physical
  fragment, side and invocation interval. It never counts these as observations.

This changes the B infiller's available context, not its weights, architecture,
120-frame horizon, interpolation, attention or prediction logic. The comparison
is therefore a frontend replacement plus a necessary documented identity-aware
interface adaptation, not a strict detector-only ablation. Disabling both
infiller outputs is represented by the separately exported observed-only world
videos and predictions.

The native exclusive end-index behavior is retained. A completion mask need not
be entirely valid merely because the infiller ran. Observed geometry is checked
against pre-infiller world vertices, independently of generated availability.

## Run

Use the configured `hawor` conda environment and a fresh isolated output directory.
Source data, old results, frontend caches, settings, MANO and all weights are
read-only. The runner checks protected SHA256 values before and after execution.

```bash
conda activate hawor
python -m unittest discover -s tests -v
python scripts/run_native_backend_comparison.py --stage prepare
python scripts/run_native_backend_stages.py --phase pilot
python scripts/run_native_backend_comparison.py --stage render --phase pilot
# Inspect frames 88..201 and the numerical validation before proceeding.
python scripts/run_native_backend_comparison.py --stage validate
python scripts/run_native_backend_stages.py --phase full
python scripts/run_native_backend_comparison.py --stage render
python scripts/run_native_backend_comparison.py --stage report
```

All commands accept `--output`. The sequential stage driver starts a fresh
process for each GPU stage and archives its command, runtime, exit status and
complete log. The runner archives each executed interface source version
separately, including pilot and full stages. Checkpoint paths and hashes are in
the experiment manifest. No automatic Git commit or push is performed.

## Outputs and Interpretation

Default output: `outputs/pre_frontend_comparison/calibrated_native_backend_full_60s/`.

- `comparison.mp4`: RGB / native A after completion / optimized B after adapted completion.
- `native_overlay.mp4`, `optimized_overlay.mp4`: full-size, common calibrated renderer.
- `world_observed_comparison.mp4`: both branches before completion.
- `world_completed_comparison.mp4`: both branches after completion.
- `predictions/{native,optimized}/world_observed.npz` and `world_completed.npz`.
- `cache/{pilot,full}/{native,optimized}/`: isolated DROID, Metric3D, masks and infiller inputs.
- `windows/`: the five requested inclusive frame windows, for all five videos.
- `pilot/validation.json`, `diagnostics.json`, `report.md`, `logs/`, `provenance/`.

World videos use a common fixed virtual camera after mapping each branch's first
camera to identity by SE(3). There is no fitted scale alignment or external world
registration. A reference grid is a coordinate aid, not a floor estimate. The
image overlays remain the appropriate high-resolution view for inspecting hand
shape; the world views expose trajectory, scale and world-motion differences.

Camera-space observation availability, mask-induced camera changes and infiller
generation are separate mechanisms. Numerical consistency, smoother movement,
more rendered hands, or a different scale do not establish accuracy without
ground truth. Treat the generated report as the authority for actual run counts
and known issues, not this interface description.
