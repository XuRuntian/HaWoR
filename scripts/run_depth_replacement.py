"""Independent cached FoundationStereo-for-Metric3D scale-source experiment."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

import run_native_backend_comparison as backend
import native_backend_infilling as completion
from cached_depth_adapter import depth_path, fit_one, resize_depth_and_mask, scaled_camera_fields

base = backend.base
ROOT, BASELINE = base.ROOT, backend.OUTPUT
TAG = "baseline/calibrated-native-backend-full60s-v1"
PARENT = ROOT/"outputs/pre_frontend_comparison/depth_replacement_full_60s"
RECT = base.VIDEO.parents[1]
DEPTH = RECT.parent/"depth_only"
RAW = BASELINE/"cache/full/optimized/droid_raw.npz"
MASKS = BASELINE/"cache/full/optimized/model_masks.npy"
ARMS = ("baseline", "stereo")


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def read_npz(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def arm_root(out, arm):
    return out/"arms"/arm


def prediction_dir(out, phase, arm):
    return arm_root(out, arm)/("predictions" if phase == "full" else "pilot/predictions")/"optimized"


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if base.sha(path) != expected:
            raise ValueError(f"Protected file changed: {path}")


def statistics(values):
    values = np.asarray(values)
    return {"count": int(values.size), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())} if values.size else {"count": 0}


def prepare(out):
    if out.exists() or git("status", "--porcelain"):
        raise ValueError("Commit code first and use a fresh isolated run directory")
    frozen = git("rev-parse", f"{TAG}^{{commit}}")
    subprocess.run(["git", "merge-base", "--is-ancestor", frozen, "HEAD"], cwd=ROOT, check=True)
    lock = base.read(ROOT/"docs/reproducibility/full60s_baseline.json")
    protected = {str(ROOT/p) if not Path(p).is_absolute() else p: digest for p, digest in lock["file_sha256"].items()}
    verify_hashes(protected)
    k = np.asarray(lock["configuration"]["camera_intrinsics"])
    metadata_path = DEPTH/"foundation_stereo_video_meta.json"
    meta = base.read(metadata_path)
    assert meta["left_video"] == str(base.VIDEO) and meta["right_video"] == str(RECT/"videos/right_rectified.mp4")
    assert meta["reference_camera"] == "left" and meta["start"] == 0 and meta["num_frames"] == base.N
    assert meta["depth_filename_pattern"] == "frame_%06d_foundation_depth_m.npy"
    np.testing.assert_array_equal([meta["intrinsics"][x] for x in ("fx", "fy", "cx", "cy")], [k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    calibration = yaml.safe_load((RECT/"calibrations/rectify_params.yaml").read_text())
    np.testing.assert_array_equal(k, calibration["rectified"]["K1_rect"])
    left, right = [np.loadtxt(RECT/f"timestamps/{side}_timestamps.txt") for side in ("left", "right")]
    for stamps in (left, right):
        np.testing.assert_array_equal(stamps[:, 0], np.arange(base.N))
    timestamp_delta = right[:, 1]-left[:, 1]
    additional = [metadata_path, Path(meta["manifest_path"]), RECT/"timestamps/left_timestamps.txt",
        RECT/"timestamps/right_timestamps.txt", RECT/"videos/right_rectified.mp4",
        base.FOREIGN/"scripts/run_foundation_stereo_video.py", base.FOREIGN/"roboego_hand_vis/depth/foundation.py"]
    protected.update({str(p): base.sha(p) for p in additional})
    protected.update({str(p): base.sha(p) for p in (backend.SOURCE/"cache/frames").glob("*.jpg")})
    metric_files = sorted((BASELINE/"cache/full/optimized/metric3d").glob("*.npy"))
    protected.update({str(p): base.sha(p) for p in metric_files})
    expected = {depth_path(DEPTH, i) for i in range(base.N)}
    assert set(DEPTH.glob("frame_*_foundation_depth_m.npy")) == expected
    inventory = []
    begin = time.perf_counter()
    for fi in range(base.N):
        path = depth_path(DEPTH, fi)
        depth = np.load(path, mmap_mode="r")
        assert depth.shape == (base.HEIGHT, base.WIDTH) and depth.dtype == np.float32, str(path)
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise ValueError(f"Entirely invalid source depth: {path}")
        digest = base.sha(path)
        protected[str(path)] = digest
        inventory.append({"frame_idx": fi, "path": str(path), "sha256": digest,
            "shape": list(depth.shape), "dtype": str(depth.dtype), "valid_positive_pixels": int(valid.sum()),
            "invalid_pixels": int((~valid).sum())})
        if fi % 200 == 0:
            print(f"DEPTH INVENTORY {fi}/1800", flush=True)
    raw = read_npz(RAW)
    np.testing.assert_array_equal(raw["frame_idx"], np.arange(base.N))
    np.testing.assert_array_equal(raw["keyframe_original"], raw["frame_idx"][raw["tstamp"]])
    assert len(raw["keyframe_original"]) == 126 and (np.diff(raw["keyframe_original"]) > 0).all()
    assert raw["disps"].shape == (126, 328, 592) and np.isfinite(raw["disps"]).all() and (raw["disps"] > 0).all()
    assert len(metric_files) == 126
    out.mkdir(parents=True)
    base.dump(out/"cache/depth_inventory.json", {"frames": 1800, "seconds": time.perf_counter()-begin, "rows": inventory})
    for arm in ARMS:
        root = arm_root(out, arm)
        (root/"cache").mkdir(parents=True)
        (root/"cache/frames").symlink_to(backend.SOURCE/"cache/frames", target_is_directory=True)
        for phase in ("pilot", "full"):
            backend.folder(root, phase, "optimized")
            dest = prediction_dir(out, phase, arm)
            dest.mkdir(parents=True)
            (dest/"camera_source").symlink_to(backend.SOURCE/"predictions/optimized", target_is_directory=True)
    (backend.folder(arm_root(out, "baseline"), "full", "optimized")/"slam_scaled.npz").symlink_to(BASELINE/"cache/full/optimized/slam_scaled.npz")
    for name in ("world_observed.npz", "world_completed.npz", "world_validation.json", "infiller_audit.json"):
        (prediction_dir(out, "full", "baseline")/name).symlink_to(BASELINE/"predictions/optimized"/name)
    code = list(dict.fromkeys(base.CORE_FILES+["lib/pipeline/est_scale.py", "lib/eval_utils/custom_utils.py",
        "lib/eval_utils/filling_utils.py", "scripts/run_native_backend_comparison.py", "scripts/native_backend_infilling.py",
        "scripts/cached_depth_adapter.py", "scripts/run_depth_replacement.py", "scripts/render_depth_replacement.py",
        "scripts/run_depth_replacement_stages.py", "tests/test_cached_depth_adapter.py", "docs/depth_replacement_experiment.md"]))
    code_hashes = {}
    for name in code:
        target = out/"provenance/code"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
        code_hashes[str(ROOT/name)] = base.sha(target)
    for filename, command in {"pip-freeze.txt": [sys.executable, "-m", "pip", "freeze"],
                              "submodules.txt": ["git", "submodule", "status", "--recursive"]}.items():
        (out/"provenance"/filename).write_text(subprocess.check_output(command, cwd=ROOT, text=True))
    manifest = {"schema_version": 1, "git_head": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
        "baseline_tag": TAG, "baseline_commit": frozen, "baseline_archive": str(BASELINE),
        "input": str(base.VIDEO), "frames_inclusive": [0, 1799], "fps": 30, "image_size": [1920, 1074],
        "camera_intrinsics": k.tolist(), "depth_source": str(DEPTH), "depth_metadata": meta,
        "depth_units": "Saved left-reference camera-axis Z in meters; inference scale already applied; no second 0.5 multiplier",
        "foundation_producer_checkpoint_and_command": "Not recoverable from provided metadata; cached file hashes pin consumed results, not producer reproduction",
        "protected_sha256": protected, "code_sha256": code_hashes, "droid_cache": str(RAW), "droid_keyframes": raw["keyframe_original"].tolist(),
        "geometry_policy": "Preserve native direct depth/mask resize to 592x328; do not silently replace with RGB resize-to-592x331 then crop",
        "invalid_depth_policy": "Exclude invalid contributors from resized support; NaN outside support, never fake zero-depth fit evidence",
        "scale_fit": {"function": "unmodified lib.pipeline.est_scale.est_scale_hybrid", "sigma": .5, "initial_near": .4,
                      "initial_far": .7, "threshold_carry_across_keyframes": True, "aggregate": "median", "max_attempts": 100},
        "same_index_stereo_timestamp_audit": {"median_right_minus_left_s": float(np.median(timestamp_delta)),
            "pairs_over_20ms": int((abs(timestamp_delta)>.02).sum()), "right_duplicate_adjacent_timestamps": int((np.diff(right[:, 1])==0).sum()),
            "interpretation": "Unresolved exposure/producer pairing semantics; cached-depth integration under user accuracy assumption, not depth ground truth"},
        "hand_policy": "Frozen optimized B observations, camera predictions and physical/side ownership-bounded native infiller",
        "pilot_policy": "Same full DROID trajectory sliced to 88..201; both pilot scales fitted on the same selected original keyframes",
        "forbidden_stages_executed": [], "vio_used": False, "python": sys.version, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "seed": 0,
        "foreign_git_head": git("rev-parse", "HEAD", cwd=base.FOREIGN), "foreign_git_status_before": git("status", "--short", cwd=base.FOREIGN)}
    base.dump(out/"manifest.json", manifest)


def fit_depths(out, phase, arm, raw, indices):
    destination = out/"cache"/phase/arm
    if destination.exists():
        raise ValueError("Scale attempt already exists; use a fresh run for changed code")
    destination.mkdir(parents=True)
    masks = np.load(MASKS, mmap_mode="r")
    near, far = .4, .7
    rows, scales = [], []
    begin = time.perf_counter()
    for index in indices:
        fi = int(raw["keyframe_original"][index])
        path = BASELINE/f"cache/full/optimized/metric3d/{fi:06d}.npy" if arm == "baseline" else depth_path(DEPTH, fi)
        original = np.load(path)
        depth, mask, valid = resize_depth_and_mask(original, masks[fi], raw["disps"].shape[-2:])
        scale, near, far, audit = fit_one(1/raw["disps"][index], depth, mask, near, far)
        np.savez_compressed(destination/f"{fi:06d}.npz", depth_m=depth, exclusion_mask=mask, positive_valid_mask=valid)
        rows.append({"frame_idx": fi, "droid_keyframe_position": int(index), "scale": scale, "near": near, "far": far,
            **audit, "source_depth_sha256": base.sha(path), "resized_invalid_pixels": int((~valid).sum()),
            "source_invalid_pixels": int((~(np.isfinite(original)&(original>0))).sum()),
            "resized_depth_m": statistics(depth[valid]), "fit_cache_sha256": base.sha(destination/f"{fi:06d}.npz")})
        scales.append(scale)
        print(f"SCALE {phase} {arm} {len(rows)}/{len(indices)} frame={fi} scale={scale:.9f}", flush=True)
    result = {"median_scale": float(np.median(scales)), "rows": rows, "seconds": time.perf_counter()-begin,
        "source_inference_rerun": False, "scale_estimator_original": True,
        "empty_band_guard": "Skip empty BFGS inputs and expand native thresholds; no replacement estimate",
        "interpretation": "Cached depth read, adaptation, native scale fit and diagnostic export time; excludes original depth inference"}
    base.dump(destination/"scale_audit.json", result)
    return result


def save_camera(out, phase, arm, raw, scale, k):
    fields = scaled_camera_fields(raw, scale, k)
    if phase == "pilot":
        frames = backend.frame_ids(phase)
        fields = {"traj": raw["traj"][frames], "frame_idx": frames, "scale": scale,
                  "camera_intrinsics": k, "img_focal": k[0, 0], "img_center": k[:2, 2]}
    np.savez_compressed(backend.folder(arm_root(out, arm), phase, "optimized")/"slam_scaled.npz", **fields)


def predict(out, phase):
    manifest = base.read(out/"manifest.json")
    k = np.asarray(manifest["camera_intrinsics"])
    raw = read_npz(RAW)
    if phase == "pilot":
        replay = fit_depths(out, "baseline_replay", "baseline", raw, np.arange(len(raw["tstamp"])))
        expected = base.read(BASELINE/"cache/full/optimized/scale_audit.json")
        errors = [abs(a["scale"]-b["scale"]) for a, b in zip(replay["rows"], expected["keyframes"])]
        assert max(errors) < 1e-6 and abs(replay["median_scale"]-expected["median_scale"]) < 1e-7
        for actual, original in zip(replay["rows"], expected["keyframes"]):
            for key in ("frame_idx", "near", "far", "retries"):
                assert actual[key] == original[key], key
        base.dump(out/"cache/baseline_scale_regression.json", {"passed": True, "keyframes": len(errors),
            "per_keyframe_max_difference": max(errors), "median_difference": abs(replay["median_scale"]-expected["median_scale"]),
            "same_threshold_expansion": True, "no_metric3d_network_rerun": True})
        selected = np.flatnonzero(np.isin(raw["keyframe_original"], backend.frame_ids("pilot")))
    else:
        assert base.read(out/"pilot/validation.json")["passed"]
        selected = np.arange(len(raw["tstamp"]))
    for arm in ARMS:
        if phase == "full" and arm == "baseline":
            continue
        scale = fit_depths(out, phase, arm, raw, selected)
        save_camera(out, phase, arm, raw, scale["median_scale"], k)
        begin = time.perf_counter()
        backend.world(arm_root(out, arm), phase, "optimized")
        world_seconds = time.perf_counter()-begin
        begin = time.perf_counter()
        completion.fill(arm_root(out, arm), phase, "optimized", k)
        base.dump(prediction_dir(out, phase, arm)/"runtime.json", {"world_seconds": world_seconds,
            "fill_export_seconds": time.perf_counter()-begin, "baseline_full_reused": False})
    audit_predictions(out, phase)


def audit_predictions(out, phase):
    source = backend.load_predictions("optimized")
    frames = backend.frame_ids(phase)
    expected_ids = np.flatnonzero(np.isin(source["frame_idx"], frames))
    result, arrays, observed_arrays, cameras, policies = {}, {}, {}, {}, {}
    k = np.asarray(base.read(out/"manifest.json")["camera_intrinsics"])
    renderer, camera, _, _ = backend.render_setup(k)
    for arm in ARMS:
        dest = prediction_dir(out, phase, arm)
        observed, data = [read_npz(dest/f"world_{mode}.npz") for mode in ("observed", "completed")]
        observed_arrays[arm], arrays[arm] = observed, data
        cameras[arm] = read_npz(backend.folder(arm_root(out, arm), phase, "optimized")/"slam_scaled.npz")
        obs = data["observed"]
        np.testing.assert_array_equal(data["generated"], ~obs)
        np.testing.assert_array_equal(observed["source_prediction_index"], expected_ids)
        np.testing.assert_array_equal(np.sort(data["source_prediction_index"][obs]), expected_ids)
        ids = data["source_prediction_index"][obs]
        for key in ("frame_idx", "side", "slot"):
            np.testing.assert_array_equal(data[key][obs], source[key][ids])
        np.testing.assert_array_equal(data["source_prediction_index"][~obs], -1)
        geometry_error = float(abs(data["vertices_camera"][obs]-source["vertices"][ids]).max())
        assert geometry_error < .0002
        joints = data["joints_camera"][obs]
        assert set(data["side"][obs]) == {0, 1} and (joints[..., 2] > 0).all()
        direct = joints[..., :2]/joints[..., 2:]*k[0, 0]+k[:2, 2]
        screen = camera.transform_points_screen(torch.as_tensor(joints, device="cuda"))[..., :2].cpu().numpy()
        projection_error = float(abs(screen-direct).max())
        assert projection_error < .005, projection_error
        masks = [np.zeros((base.N, 2), bool) for _ in range(2)]
        for mask, choose in zip(masks, (obs, ~obs)):
            mask[data["frame_idx"][choose], data["slot"][choose]] = True
        assert not (masks[0]&masks[1]).any()
        for expected, name in zip((*masks, ~(masks[0]|masks[1])), ("observed_mask", "generated_mask", "missing_mask")):
            np.testing.assert_array_equal(expected, data[name])
        np.testing.assert_array_equal(masks[0][frames], source["raw_observed_mask"][frames])
        policy = base.read(dest/"infiller_audit.json")
        policies[arm] = {key: policy[key] for key in ("policy", "intervals", "ambiguities", "completion_parts")}
        parts = {p["ownership_interval"]: p for p in policy["completion_parts"]}
        for i in np.flatnonzero(~obs):
            part = parts[int(data["ownership_interval"][i])]
            fi, side, slot = [int(data[key][i]) for key in ("frame_idx", "side", "slot")]
            assert part["first"] <= fi <= part["last"] and part["owners"][side][0] == slot
            assert part["owners"][side][2] == side and fi not in policy["ambiguities"]["side_conflict_frames"]
        for key in ("vertices", "vertices_camera", "joints_camera", "joints_world"):
            assert np.isfinite(data[key]).all()
        nonpositive = (data["joints_camera"][..., 2] <= 0).any(axis=-1)
        result[arm] = {"observed_instances": int(obs.sum()), "generated_instances": int((~obs).sum()),
            "missing_slots": int(data["missing_mask"][frames].sum()), "observed_camera_geometry_max_error": geometry_error,
            "both_hands_pinhole_vs_native_projection_max_px": projection_error,
            "generated_owners_checked": int((~obs).sum()), "nonpositive_depth_frames": np.unique(data["frame_idx"][nonpositive]).tolist(),
            "nonpositive_observed_instances": int((nonpositive&obs).sum()), "scale": float(cameras[arm]["scale"])}
    assert policies["baseline"] == policies["stereo"], "Physical infiller context changed"
    a, b = [arrays[x] for x in ARMS]
    for key in ("frame_idx", "side", "slot", "observed", "generated", "source_prediction_index", "ownership_interval", "observed_mask", "generated_mask", "missing_mask"):
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(cameras["baseline"]["traj"], cameras["stereo"]["traj"])
    raw = read_npz(RAW)
    for arm in ARMS:
        np.testing.assert_array_equal(cameras[arm]["traj"], raw["traj"][frames])
        if phase == "full":
            for key in raw:
                np.testing.assert_array_equal(cameras[arm][key], raw[key])
    offsets = raw["traj"][source["frame_idx"][expected_ids], :3].astype(float)*(result["stereo"]["scale"]-result["baseline"]["scale"])
    delta = observed_arrays["stereo"]["vertices"]-observed_arrays["baseline"]["vertices"]
    scale_error = float(abs(delta-offsets[:, None]).max())
    assert scale_error < .0001, scale_error
    result.update(passed=True, same_hand_masks_ids_and_infiller_context=True, same_raw_droid_fields=True,
        observed_world_change_matches_only_camera_translation_scale_max_error=scale_error,
        generated_joint_camera_change_m=statistics(np.linalg.norm(a["joints_camera"][a["generated"]]-b["joints_camera"][b["generated"]], axis=-1)))
    base.dump(out/("prediction_audit.json" if phase == "full" else "pilot/prediction_audit.json"), result)
    return result


def plots_and_trajectory(out):
    fits = {"baseline": base.read(out/"cache/baseline_replay/baseline/scale_audit.json"),
            "stereo": base.read(out/"cache/full/stereo/scale_audit.json")}
    raw = read_npz(RAW)
    rotation = Rotation.from_quat(raw["traj"][:, 3:]).as_matrix()
    positions, result = {}, {"alignment": "First-camera SE(3) origin only; no fitted scale alignment", "arms": {}}
    for arm in ARMS:
        scale = fits[arm]["median_scale"]
        xyz = raw["traj"][:, :3].astype(float)*scale
        positions[arm] = (xyz-xyz[0])@rotation[0]
        result["arms"][arm] = {"median_scale": scale, "path_length_m": float(np.linalg.norm(np.diff(xyz, axis=0), axis=-1).sum()),
            "net_displacement_m": float(np.linalg.norm(xyz[-1]-xyz[0])), "per_keyframe_scale": statistics([x["scale"] for x in fits[arm]["rows"]]),
            "total_retries": sum(x["retries"] for x in fits[arm]["rows"]),
            "resized_invalid_pixels_total": sum(x["resized_invalid_pixels"] for x in fits[arm]["rows"])}
    result["stereo_over_baseline_scale"] = fits["stereo"]["median_scale"]/fits["baseline"]["median_scale"]
    result["position_difference_m"] = statistics(np.linalg.norm(positions["stereo"]-positions["baseline"], axis=-1))
    result["camera_rotations_identical"] = True
    result["interpretation"] = "Scale-source effects on the same monocular trajectory, not depth or pose accuracy"
    np.savez_compressed(out/"aligned_camera_trajectories.npz", frame_idx=np.arange(base.N),
        baseline_position=positions["baseline"], stereo_position=positions["stereo"], rotations=rotation,
        baseline_scale=fits["baseline"]["median_scale"], stereo_scale=fits["stereo"]["median_scale"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for arm in ARMS:
        points = positions[arm]
        axes[0].plot(points[:, 0], points[:, 2], label=arm)
        rows = fits[arm]["rows"]
        axes[1].plot([x["frame_idx"] for x in rows], [x["scale"] for x in rows], label=arm)
        axes[1].axhline(fits[arm]["median_scale"], linestyle="--", alpha=.5)
    axes[0].set(xlabel="First-camera X (m)", ylabel="First-camera Z (m)", title="Same DROID, different depth-derived scale")
    axes[0].axis("equal")
    axes[1].set(xlabel="Original video frame", ylabel="Scale (m / DROID unit)", title="126 keyframes; dashed lines: global medians")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out/"depth_scale_comparison.png", dpi=150)
    plt.close(fig)
    return result


def report(out):
    from render_depth_replacement import audit_video_alignment
    prediction = audit_predictions(out, "full")
    render = base.read(out/"render_audit.json")
    assert render["frame_count"] == base.N
    trajectory = plots_and_trajectory(out)
    video = audit_video_alignment(out)
    manifest = base.read(out/"manifest.json")
    verify_hashes(manifest["protected_sha256"])
    assert git("status", "--short", cwd=base.FOREIGN) == manifest["foreign_git_status_before"]
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, capture_output=True, text=True)
    (out/"provenance/unittest.log").write_text(tests.stdout+tests.stderr)
    assert tests.returncode == 0
    result = {"passed": True, "git_head": manifest["git_head"], "baseline_commit": manifest["baseline_commit"],
        "frames": 1800, "fps": 30, "predictions": prediction, "trajectory": trajectory, "rendering": render,
        "video_alignment": video, "baseline_scale_regression": base.read(out/"cache/baseline_scale_regression.json"),
        "protected_files_unchanged": len(manifest["protected_sha256"]), "unit_tests_exit_code": tests.returncode,
        "stereo_timestamp_risk": manifest["same_index_stereo_timestamp_audit"], "vio_used": False,
        "detector_hawor_droid_metric3d_foundation_networks_rerun": False,
        "accuracy_claim": "No accuracy claim; cached stereo depth is assumed usable for this integration test"}
    base.dump(out/"diagnostics.json", result)
    rows = "\n".join(f"| {key} | {prediction['baseline'][key]} | {prediction['stereo'][key]} |" for key in
        ("observed_instances", "generated_instances", "missing_slots", "scale", "observed_camera_geometry_max_error", "nonpositive_depth_frames"))
    text = f"""# Optimized HaWoR: Metric3D vs Cached FoundationStereo

Completed rectified frames 0..1799, 1920x1074, 30 FPS. Output: `{out}`.
Execution commit: `{manifest['git_head']}`. Baseline commit: `{manifest['baseline_commit']}`.
Correct K: `{manifest['camera_intrinsics']}`. See `manifest.json` for exact inputs, schema and hashes.

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
{rows}

Full estimated path lengths: baseline {trajectory['arms']['baseline']['path_length_m']:.6f} m;
stereo {trajectory['arms']['stereo']['path_length_m']:.6f} m.
Stereo/baseline scale ratio: {trajectory['stereo_over_baseline_scale']:.6f}.
These are model/trajectory estimates, not ground-truth errors or depth-model accuracy rankings.

## Stereo Timing and Known Limits

The existing same-index left/right timestamps differ by more than 20 ms for
{manifest['same_index_stereo_timestamp_audit']['pairs_over_20ms']} of 1800 pairs; median right-minus-left is
{manifest['same_index_stereo_timestamp_audit']['median_right_minus_left_s']*1000:.6f} ms.
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
RGB/column alignment. {len(manifest['protected_sha256'])} protected files retain their hashes and the foreign
repository status is unchanged. The focused unit suite passes; its log is under `provenance/`.

- `comparison.mp4`: RGB / optimized frontend with Metric3D / same frontend with FoundationStereo.
- `baseline_overlay.mp4`, `stereo_overlay.mp4`: full-resolution common native renderer.
- `world_observed_comparison.mp4`, `world_completed_comparison.mp4`: baseline left, stereo right.
- `depth_scale_comparison.png`, `aligned_camera_trajectories.npz`: scale/trajectory diagnostics.
- `arms/{{baseline,stereo}}/predictions/optimized/`: world observations, completions and ownership audits.
- `cache/{{baseline_replay,pilot,full}}/`: per-keyframe adapted depth/validity/mask caches and scale audits.
- `windows/`: five requested windows plus 918..926 around the known generated outlier.
- `logs/`, `provenance/`, `diagnostics.json`: exact commands, stage times, sources, versions and checks.

The world view uses one shared fixed camera and each first-camera SE(3) origin, with NO fitted scale
alignment. It is an overview with small hands; use RGB overlays for hand detail. Reference grid is not ground.
Full rendering took {render['seconds']:.3f} seconds before decoding/window exports. Depth-cache adaptation
and scale timings exclude historical Metric3D/FoundationStereo inference; baseline reuse is not zero-cost
inference. All native inference-stage times remain separately recorded.

Reproduce from the execution commit using `docs/depth_replacement_experiment.md`, with a new output.
Playback: `mpv --speed=0.5 '{out}/world_observed_comparison.mp4'`.
"""
    (out/"report.md").write_text(text)


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PARENT/"run_001")
    parser.add_argument("--stage", required=True, choices=("prepare", "pilot", "render-pilot", "validate", "full", "render", "report"))
    args = parser.parse_args()
    out = args.output.resolve()
    if out == PARENT or not out.is_relative_to(PARENT) or out.is_relative_to(PARENT/"process_logs"):
        raise ValueError("Use an isolated numbered depth-experiment run directory")
    if args.stage != "prepare":
        manifest = base.read(out/"manifest.json")
        assert git("rev-parse", "HEAD") == manifest["git_head"], "Execution commit changed"
        verify_hashes(manifest["code_sha256"])
    begin = time.perf_counter()
    try:
        if args.stage == "prepare":
            prepare(out)
        elif args.stage in ("pilot", "full"):
            predict(out, args.stage)
        elif args.stage in ("render-pilot", "render"):
            from render_depth_replacement import render
            render(out, "pilot" if args.stage == "render-pilot" else "full")
        elif args.stage == "validate":
            assert base.read(out/"pilot/prediction_audit.json")["passed"]
            assert base.read(out/"cache/baseline_scale_regression.json")["passed"]
            rendering = base.read(out/"pilot/render_audit.json")
            assert rendering["frame_count"] == 114 and all(max(v)>0 for v in rendering["mesh_pixels"].values())
            base.dump(out/"pilot/validation.json", {"passed": True, "frames": [88, 201], "before_full": True,
                "same_raw_droid": True, "same_hand_masks_and_infiller_intervals": True, "native_scale_replay_passed": True,
                "nonblank_shared_renderer": True})
        else:
            report(out)
    except Exception:
        if out.exists():
            base.dump(out/f"logs/{time.time_ns()}_{args.stage}_failed.json", {"stage": args.stage,
                "command": [sys.executable, *sys.argv], "git_head": git("rev-parse", "HEAD"), "seconds": time.perf_counter()-begin, "exit_code": 1})
        raise
    base.dump(out/f"logs/{time.time_ns()}_{args.stage}.json", {"stage": args.stage, "command": [sys.executable, *sys.argv],
        "git_head": git("rev-parse", "HEAD"), "seconds": time.perf_counter()-begin, "exit_code": 0})
    print(f"DONE {args.stage}: {out}", flush=True)


if __name__ == "__main__":
    main()
