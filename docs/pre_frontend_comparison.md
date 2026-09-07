# Camera-Space Observation Frontend Experiment

This branch adds an opt-in experimental entry point, not a replacement for the
default HaWoR demo. The network, checkpoint, MANO, native crop/mirroring and
projection are unchanged. Environment setup is in [ENVIRONMENT.md](../ENVIRONMENT.md).

## Branches

- Native A: original HaWoR YOLO detection/tracking, majority-handedness merge,
  camera-space reconstruction and native temporal processing.
- Optimized B: existing Consolidation + Offline Temporal Association selections,
  resolved by observation ID, followed by HaWoR reconstruction. The adapter
  preserves physical track slots, fragments and source handedness. It bypasses
  native side-based merging rather than pretending track 0/1 means left/right.

Only bbox and observation/trajectory metadata cross the adapter boundary.
Detectron2, ViTPose and the offline selector are not rerun; cached HaMeR poses,
cameras, vertices and crop tensors are not used to predict HaWoR outputs.
The external frontend implementation and its existing caches are read-only.

Both branches retain native 16-frame inference/padding and bbox interpolation.
Missing frame rows are not inserted. Infilling is disabled in both because the
native infiller consumes SLAM/world coordinates. The unused SLAM mask pass is
also disabled in both. No Stereo/VIO, Metric3D, new gate, filtering or correction
is added. A common native fallback focal length of 600 px is used, not calibrated
metric depth. Singleton physical groups are accepted in B without identity merging.

## Local Inputs

The runner deliberately describes one fixed experiment; it is not a generic
video CLI. Another machine must supply these files or explicitly adapt the
input constants and validation together:

```text
/home/user/roboego-hand-vis/docs/pipeline/pre_hamer_observation_frontend_migration.md
/home/user/roboego-hand-vis/roboego_hand_vis/egohand/pre_hamer_observation_frontend.py
/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/rectified/videos/left_rectified.mp4
/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/failure_mining/full_60s/cache/full_sequence_temporal_pipeline.json
```

The input is the rectified video: frames 0..1799, 1920x1074, 30 FPS, 60 seconds.
Video SHA-256: `e8fe5443027c1876dc3c46d7bcae50b72e94eb07f4f9c69309820f716c41de6d`.
The temporal JSON schema is 1.0. Resolve `temporal_tracks.selected_candidate_id`
using original/official observation IDs, never array position or baseline ID.

## Reproduce

Run from the repository root in the configured `hawor` environment. Use a fresh
output directory for each run; do not reuse completed branch/chunk caches.

```bash
conda activate hawor
python -m unittest discover -s tests -p test_frontend_comparison_adapter.py -v
python scripts/run_pre_frontend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/reproduction" --stage pilot
# Inspect pilot/validation.json and pilot/comparison.mp4 before continuing.
python scripts/run_pre_frontend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/reproduction" --stage full
python scripts/run_pre_frontend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/reproduction" --stage render
python scripts/audit_pre_frontend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/reproduction"
```

The pilot covers frames 88..114 and compares against the untouched native
function pinned at `66c7d4108d58a716deccd192cb7645170cdc7bd7`, including its mask
pass. Keep that commit available locally; a shallow clone may need its history
fetched. Pinning prevents this oracle from changing when this branch advances.
The initial completed run used that same commit as HEAD; pinning was added for
publication without rerunning or rewriting the archived experiment.

## Completed Run

Completed locally on 2026-09-06; packaged for publication on 2026-09-07.
Results remain at `/home/user/HaWoR/outputs/pre_frontend_comparison/full_60s/`.

| Quantity | Native A | Optimized B |
|---|---:|---:|
| Observed inference instances | 2364 | 2872 |
| Generated instances | 0 | 0 |
| Occupied slots | 2338 | 2872 |
| Missing slots | 1262 | 728 |
| Frames with no output | 122 | 75 |
| Contiguous chunks | 297 | 128 |
| Detection / cached-read seconds | 82.945 | 0.274 |
| Reconstruction and export seconds | 141.210 | 131.310 |

A slots are routed anatomical-side hypotheses; B slots are anonymous physical
tracks. These are not matched ground-truth identities or detection recall.
Cached-read time excludes the historical frontend inference/association cost.
Different chunk boundaries alter HaWoR temporal context; the comparison does
not isolate selection gains from detector differences or trajectory adaptation.

The pilot camera-parameter difference was zero; projection disagreement was
0.0001221 px. All 2872 B observations map one-to-one to predictions with unchanged
bboxes and handedness. All three full videos decoded successfully. Requested
inclusive windows 88-94, 100-114, 176-182, 193-201 and 596-602 were exported.
No pose-accuracy improvement is claimed.

Known limitations:

- B retains 59 fragments and unresolved handedness/identity ambiguities, including
  four frames with two same-side hypotheses. Dominant side labels are diagnostic.
- Four zero-width, positive-height source boxes are preserved because native
  square crops support them. They are not corrected or evidence of good localization.
- Native side merging overrides 165 source-side hypotheses. Frame 1569 also has
  a native JSON filename collision. Lossless per-call exports and overlays retain
  both predictions: 2364 inference instances versus 2363 recoverable from native
  camera JSON files alone. This export difference is documented, not a model gain.
- The fixed-experiment audit checks executed source snapshots; auditing an older
  run with changed executable sources intentionally fails the integrity check.
  Use that run's archived sources when re-auditing it.

## Outputs and HaMeR Comparison

Each run writes `native_overlay.mp4`, `optimized_overlay.mp4`, `comparison.mp4`,
`predictions/`, `windows/`, `diagnostics.json`, `report.md`, and provenance hashes,
source snapshots and package versions. Prediction files retain raw observed,
generated and missing masks separately from camera-space geometry.
All experiment outputs are ignored by Git. Videos, predictions, source data,
licensed MANO files and model checkpoints are not distributed in this branch.

The optional `scripts/stitch_optimized_hawor_hamer.py` directly stitches the
completed optimized HaWoR video with the optimized column of the existing
HaMeR camera-relative RGB video. It performs no model inference or mesh
re-rendering. Run it only after both source videos exist:

```bash
python scripts/stitch_optimized_hawor_hamer.py
```

This makes a 1280x360, 1800-frame comparison and five windows. Existing HaMeR
resolution, rendering, colors, opacity and focal conventions differ from HaWoR;
it is a visual comparison, not a common-renderer accuracy benchmark. It does
not use HaMeR's separate depth-aligned/smoothed final-output video.
