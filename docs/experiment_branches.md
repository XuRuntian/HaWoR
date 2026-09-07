# Experiment Baseline and Branches

## Frozen Baseline

Publication date: 2026-09-07. The immutable reference for subsequent backend
replacement experiments is the annotated tag:

`baseline/calibrated-native-backend-full60s-v1`

Resolve the exact publication commit with
`git rev-parse 'baseline/calibrated-native-backend-full60s-v1^{commit}'`.
The existing `experiment/pre-frontend-camera-comparison` branch contains the
completed camera-calibration and native-backend A/B code, tests and documentation.
The tag, rather than a moving branch tip, identifies the shared baseline.

The completed experiments ran with uncommitted interface code on HEAD
`823f4fda8f81667e070ec00c950bbd40635a087d`. Their old manifests correctly retain
that historical HEAD. Do not rewrite them to claim that the publication commit
was the execution HEAD. Archived executable source hashes and per-stage
snapshots identify what actually ran; the publication commit preserves the final
tested implementation. Failed and successful stage-attempt logs remain intact.

One final reporting-only addition postdates the last archived source snapshot:
`native_backend_infilling.report()` runs and archives the unit suite and includes
its status in diagnostics/report text. Its inference, ownership, world conversion
and rendering functions match that snapshot. Both hashes and this difference are
recorded in the lock; the old snapshot is not retroactively changed.

## Independent Branches

Both replacement branches are created directly from the same baseline tag, not
from each other. At creation they contain no replacement implementation or new
experimental results. The VIO/depth documents currently describe compatibility
audits only.

| Reference | Camera trajectory | Metric scale source | Status at publication |
|---|---|---|---|
| Baseline tag, optimized B | Native masked DROID | Native Metric3D | Completed, all 1800 frames |
| `experiment/vio-replacement` | External metric OpenVINS | Bypass Metric3D, scale=1 | Not run |
| `experiment/depth-replacement` | Exact frozen baseline B DROID cache | FoundationStereo replaces Metric3D depth only | Not run |

Freeze optimized B observations, correct K, checkpoint, MANO, camera-space
predictions, physical-fragment mapping and identity-bounded infiller policy in
both future experiments. Native A uses whole-sequence anatomical-side filling;
B restricts native filling to known ownership intervals. The completed A/B is
therefore not a frontend-only ablation under identical completion context.
See [backend interfaces](calibrated_native_backend_comparison.md).

Use new run directories, for example:

```text
outputs/pre_frontend_comparison/vio_replacement_full_60s/run_001/
outputs/pre_frontend_comparison/depth_replacement_full_60s/run_001/
```

Do not overwrite any completed result, foreign repository, shared cache or
settings file. Reuse frozen inputs read-only and record their hashes. Before each
future experiment, commit the executable changes on its own branch and record
the exact commit, command, environment, configuration and input hashes. If a
run fails and code is fixed, retain its failed logs and record the new commit
before rerunning. A completed result should receive its own result-linked tag.

VIO adaptation must retain all frame indices, explicitly handle missing camera
poses at frames 550 and 559, account for raw-to-rectified camera rotation, and
avoid applying metric scale or IMU extrinsics twice. Depth adaptation must reuse
baseline B DROID keyframes/disparities/masks and preserve the same scale-fit and
resizing policy. The existing stereo timestamp pairing concern remains open;
assuming reasonable depth accuracy does not resolve it. No new hand-depth
translation correction, pose filter or gate is part of either replacement.

## Result and Dependency Lock

[full60s_baseline.json](reproducibility/full60s_baseline.json) records configuration,
model/input/foreign-source hashes, final executable hashes, and selected archived
result/cache hashes. It also hashes the original manifests, which reference the
larger protected-file inventory. The selection is not a backup of every output.
[The archived backend report](reproducibility/full60s_backend_report.md) is included
in Git as a small result snapshot; full diagnostics remain in the output archive.

The three completed local archives are:

```text
/home/user/HaWoR/outputs/pre_frontend_comparison/full_60s/
/home/user/HaWoR/outputs/pre_frontend_comparison/calibrated_full_60s/
/home/user/HaWoR/outputs/pre_frontend_comparison/calibrated_native_backend_full_60s/
```

Videos, frame JPEGs, predictions, masks, model weights, licensed MANO data and
foreign caches are NOT in Git. Clone alone is insufficient. Preserve these
archives outside Git, including symlink targets. The calibrated runner requires
the original `full_60s` archive; the backend runner requires the calibrated archive.
They deliberately pin these source paths, and `--output` changes only the new
destination. Restoring these exact caches reproduces the controlled experiment;
rerunning the original detector creates a new baseline, not an identical frozen
input by definition. Another machine must restore or explicitly remap the
documented absolute paths and supply its own licensed model assets.

From the baseline checkout on the original machine, verify the selected files:

```bash
cd /home/user/HaWoR
jq -r '.file_sha256 | to_entries[] | "\(.value)  \(.key)"' docs/reproducibility/full60s_baseline.json | sha256sum --check
conda activate hawor
python -m unittest discover -s tests -v
```

Environment setup and native-extension builds are in [ENVIRONMENT.md](../ENVIRONMENT.md).
`requirements-cu117.lock` pins PyTorch3D to commit
`3145dd4d16edaceb394838364b8e87a440f83c10`, confirmed against the local build source.
The historical pip freeze instead names `/tmp/hawor-pytorch3d`; that temporary
path is provenance, not a portable installation requirement.

## Rerun the Backend Baseline

Keep the restored calibrated camera archive unchanged. Run from the baseline
checkout in the configured `hawor` environment, with a new destination:

```bash
OUT="$PWD/outputs/pre_frontend_comparison/backend_reproduction_run_001"
python scripts/run_native_backend_comparison.py --output "$OUT" --stage prepare
python scripts/run_native_backend_stages.py --output "$OUT" --phase pilot
python scripts/run_native_backend_comparison.py --output "$OUT" --stage render --phase pilot
# Inspect the pilot videos and validation before approving the full sequence.
python scripts/run_native_backend_comparison.py --output "$OUT" --stage validate
python scripts/run_native_backend_stages.py --output "$OUT" --phase full
python scripts/run_native_backend_comparison.py --output "$OUT" --stage render
python scripts/run_native_backend_comparison.py --output "$OUT" --stage report
```

For hand-network reinference with frozen observations, see
[calibrated camera reproduction](calibrated_frontend_comparison.md). For the
earlier detector/frontend experiment, see
[original frontend reproduction](pre_frontend_comparison.md).

Hashes establish artifact identity, not a promise of bitwise-identical GPU
reruns across machines. Compare frame/identity/mask invariants exactly and report
numerical tolerances, runtime and observed/generated/missing counts separately.
Observed availability is not detection recall; visual preference and transform
consistency do not establish pose accuracy without ground truth.
