# HaMeR Focal-Length Evidence

Date: 2026-09-07. Read-only audit of local source and saved full-60-second results. No inference or external-file modifications.

## Three Different Quantities

| Quantity | Full-Resolution Value | Meaning |
| --- | ---: | --- |
| Calibrated rectified image focal | 444.43963608648465 px | Physical pinhole intrinsics of the 1920x1074 stereo-rectified video |
| HaMeR model nominal crop focal | 5000 px at model image size 256 | Model camera parameterization, not this camera's physical calibration |
| HaMeR full-image nominal focal | 37500 px | `5000 / 256 * max(1920,1074)`, paired with HaMeR's nominal full-image camera translation |

The previous HaWoR experiment's 600 px therefore was not this video's calibrated physical focal. However, HaMeR's much larger 37500 px does not mean it used a physically more accurate camera: it is a different nominal projection convention. A visually aligned overlay is not evidence that its camera translation is metric-correct.

The physical calibration, including `cx=959.954470803018`, `cy=543.9405836940584`, comes from [rectification YAML](/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/rectified/calibrations/rectify_params.yaml:35). FoundationStereo's saved metadata uses the same values: [depth metadata](/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/depth_only/foundation_stereo_video_meta.json:7).

## Actual HaMeR Inference Convention

The actual cached candidate report is:

`/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/hand_only/ego_hand_candidates.json`

Its saved `hamer_config` records `/home/user/models/hamer`, `checkpoint=null` (default checkpoint), `batch_size=1`, and `rescale_factor=2.0`. Its `video_info` records the same 1920x1074, 30 FPS, 1800-frame rectified video.

The installed checkpoint configuration explicitly contains `EXTRA.FOCAL_LENGTH: 5000` and `MODEL.IMAGE_SIZE: 256`: [focal](/home/user/models/hamer/_DATA/hamer_ckpts/model_config.yaml:59), [image size](/home/user/models/hamer/_DATA/hamer_ckpts/model_config.yaml:81). Do not substitute the generic config class's default image size 224 for this loaded checkpoint's 256.

The execution path computes:

```python
scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
```

Source: [candidate reconstruction](/home/user/roboego-hand-vis/scripts/run_ropedia_hamer_ego_filter.py:366). The same focal is used for full-image joint projection and mesh rendering: [joint projection](/home/user/roboego-hand-vis/scripts/run_ropedia_hamer_ego_filter.py:405), [overlay rendering](/home/user/roboego-hand-vis/scripts/run_ropedia_hamer_ego_filter.py:453).

`cam_crop_to_full` sets `tz = 2*f/(b*s + 1e-9)` and computes lateral translation using box center relative to image center: [camera conversion](/home/user/models/hamer/hamer/utils/renderer.py:12). The renderer uses image-center principal point and the supplied nominal focal: [renderer](/home/user/models/hamer/hamer/utils/renderer.py:365).

### Independent Saved-Camera Check

I checked all **3489 cached HaMeR candidate instances**, without running HaMeR:

1. Reconstructed the actual input box size from the saved original bbox and `rescale_factor=2.0`.
2. Applied the model loader's `[192,256]` aspect-ratio override, giving `b = max(2*bbox_width*256/192, 2*bbox_height)`. This is the original input-box convention, not the later diagnostic crop replay metadata: [loader override](/home/user/models/hamer/hamer/models/__init__.py:35), [dataset box construction](/home/user/models/hamer/hamer/datasets/vitdet_dataset.py:39).
3. Computed `f_inferred = pred_cam_t_full[2] * b * mano_params.pred_cam[0] / 2` from the saved predictions.

Results:

```text
instances checked:                  3489
inferred f min:                     37499.9927631963 px
inferred f median:                  37500.00008859515 px
inferred f max:                     37500.00581118852 px
maximum absolute difference:        0.00723680370 px
instances within 0.02 px of 37500:   3489 / 3489
```

Thus 37500 is supported by executed-result algebra, not only current source/config defaults. Small errors reflect saved float32 arithmetic. Across the same candidates, nominal `pred_cam_t_full.z` has min 10.60088, median 37.32008, max 430.99112. These numbers must not be interpreted as measured hand distances in this scene.

## Existing Optimized HaMeR RGB Video

The HaMeR column in the stitched HaWoR/HaMeR comparison comes from:

`/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/three_way_baseline_comparison/full_60s/comparison/raw_vs_original_vs_optimized_rgb.mp4`

Its RGB-view producer uses selected cached vertices and `pred_cam_t_full`, not the downstream depth-aligned meshes: [render inputs](/home/user/roboego-hand-vis/scripts/diagnostics/run_three_way_baseline_comparison.py:304). It computes full-image nominal focal 37500 and scales it by `640/1920`, giving **12500 px for its 640-pixel-wide panel**: [RGB-video focal](/home/user/roboego-hand-vis/scripts/diagnostics/run_three_way_baseline_comparison.py:414).

That is internally paired with nominal HaMeR camera translation. Replacing only its renderer focal with 444.44 while keeping those saved translations would break the established projection, not calibrate the output. Physical placement requires an explicit camera/translation conversion or calibrated reconstruction, followed by validation.

## Additional Finding: Wrong Fallback K in Existing Depth Alignment

This is separate from the FoundationStereo depth arrays and separate from the nominal RGB video above.

Both existing full-60-second downstream alignment reports record:

```json
"intrinsics_json": null,
"intrinsics": {"fx": 200.0, "fy": 200.0, "cx": 256.0, "cy": 256.0}
```

Affected artifacts:

- [Optimized alignment report](/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/three_way_baseline_comparison/full_60s/comparison/optimized_pipeline/depth_alignment/hand_depth_mesh_alignment_report.json:5)
- [Original alignment report](/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/three_way_baseline_comparison/full_60s/comparison/original_pipeline/depth_alignment/hand_depth_mesh_alignment_report.json:5)

These reports each contain 1800 frames and explicitly name the full 1920x1074 rectified video and matching full-size depth directory. This is not a valid 512x512 working-resolution calibration: the implementation reads RGB with `resize_scale=1.0` and loads the original-size depth: [alignment input](/home/user/roboego-hand-vis/scripts/hand/mesh_depth_alignment_impl.py:167), [RGB loader](/home/user/roboego-hand-vis/scripts/export_depth_pointcloud.py:19).

The cause is traceable:

1. The manifest adapter passes through `camera_model.calibration_source` only if its filename ends in `.json`, ignoring the manifest's already-populated intrinsics. This sequence points to a `.yaml`, so it returns `intrinsics_json=None`: [adapter](/home/user/roboego-hand-vis/roboego_hand_vis/egohand/alignment_interface.py:23).
2. Without an intrinsics JSON, the alignment implementation falls back to `ROPEDIA_K`: [fallback](/home/user/roboego-hand-vis/scripts/hand/mesh_depth_alignment_impl.py:100).
3. `ROPEDIA_K` is exactly the 200/200/256/256 constant: [definition](/home/user/roboego-hand-vis/scripts/export_depth_pointcloud.py:11).

### Actual Geometry, Not Just Metadata

The saved first optimized hand's `shape_preserved_fit_transform` records:

```text
target_pixel_xy: [903.65966796875, 521.7327880859375]
target_z_median: 0.47879308462142944
target_anchor:   [1.5504748821258545, 0.6361551284790039, 0.47879308462142944]
```

Source: [saved target anchor](/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/three_way_baseline_comparison/full_60s/comparison/optimized_pipeline/depth_alignment/hand_depth_mesh_alignment_report.json:167).

Backprojecting that pixel/depth with the recorded fallback K yields `[1.550474851, 0.636155106, 0.478793085]`, matching the saved anchor to float precision. The actual rectified K instead gives `[-0.060646171, -0.023924371, 0.478793085]`. The original branch stores the same first-hand anchor. Therefore the mismatch affected actual camera-space placement, not only a stale report label.

The optimized report also records `hamer_depth_lift_focal_length=37500` for all its 2872 hand instances. The code deliberately uses nominal HaMeR focal for projecting the original mesh back into source pixels, then a separate K for depth backprojection: [two-stage placement](/home/user/roboego-hand-vis/scripts/hand/mesh_depth_alignment_impl.py:285). That separation is sensible in principle, but the second K here fell back to the wrong camera.

Consequences are scoped: the checked original/optimized **depth-aligned** full-60-second outputs must not be treated as calibrated geometry without correction and rerun. This finding does not imply that FoundationStereo used K=200; its saved metadata and formula use fx=444.44. It also does not retroactively introduce depth alignment into the existing raw-camera RGB comparison.

## What This Means for the 600 px Question

- If 600 was described as this video's measured/calibrated focal, that description would be wrong. The calibration is available and is approximately 444.44 px.
- Keeping 600 equal in the two original HaWoR A/B runs preserves a native-default, camera-only frontend comparison, but does not establish calibrated hand placement or metric depth.
- HaMeR uses its own nominal 37500 px full-image convention. One must not compare these numbers as though both were physical focal estimates.
- A next calibrated world/VIO experiment should carry the correct rectified K consistently through camera transforms, hand placement and projection, and should not reuse the affected K=200 depth-aligned outputs as ground truth.
