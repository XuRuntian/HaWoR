"""Frozen optimized HaWoR baseline versus metric OpenVINS, full 60 seconds."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

import run_native_backend_comparison as backend
import native_backend_infilling as completion
from vio_camera_adapter import adapt_camera_poses

base = backend.base
ROOT = base.ROOT
BASELINE = backend.OUTPUT
TAG = "baseline/calibrated-native-backend-full60s-v1"
OUTPUT_PARENT = ROOT / "outputs/pre_frontend_comparison/vio_replacement_full_60s"
RECT = base.VIDEO.parents[1]
SESSION = Path("/home/user/ego_data/session_20260804_224442")
CLIP = Path("/home/user/ego_data/session_20260804_224442_clip_0649_60s/session")
VIO_ROOT = base.FOREIGN / "tmp/openvins_session_224442_full"
VIO = VIO_ROOT / "trajectory/trajectory.npz"
ARMS = ("baseline", "vio")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def arm_root(out, arm):
    return out / "arms" / arm


def prediction_dir(out, phase, arm):
    return arm_root(out, arm) / ("predictions" if phase == "full" else "pilot/predictions") / "optimized"


def read_npz(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if base.sha(path) != expected:
            raise ValueError(f"Protected file changed: {path}")


def prepare(out):
    if out.exists():
        raise ValueError("Use a new isolated run directory, not an existing result")
    if git("status", "--porcelain"):
        raise ValueError("Commit experiment code before execution")
    frozen = git("rev-parse", f"{TAG}^{{commit}}")
    subprocess.run(["git", "merge-base", "--is-ancestor", frozen, "HEAD"], cwd=ROOT, check=True)
    lock = base.read(ROOT / "docs/reproducibility/full60s_baseline.json")
    protected = {str(ROOT/path) if not Path(path).is_absolute() else path: digest
                 for path, digest in lock["file_sha256"].items()}
    source_paths = [VIO, VIO_ROOT/"run_manifest.json", VIO_ROOT/"trajectory/run_report.json",
        VIO_ROOT/"config/kalibr_imucam_chain.yaml", VIO_ROOT/"config/estimator_config.yaml",
        VIO_ROOT/"config/kalibr_imu_chain.yaml", CLIP/"meta.json",
        RECT/"timestamps/left_timestamps.txt", CLIP/"timestamps/left_timestamps.txt",
        SESSION/"timestamps/left_timestamps.txt", base.FOREIGN/"scripts/vio/replay_openvins_session.py"]
    protected.update({str(p): base.sha(p) for p in source_paths})
    protected.update({str(p): base.sha(p) for p in (backend.SOURCE/"cache/frames").glob("*.jpg")})
    start = time.perf_counter()
    verify_hashes(protected)
    stamps = np.loadtxt(RECT/"timestamps/left_timestamps.txt")
    clip_stamps = np.loadtxt(CLIP/"timestamps/left_timestamps.txt")
    original = np.loadtxt(SESSION/"timestamps/left_timestamps.txt")
    meta = base.read(CLIP/"meta.json")
    first = meta["source_start_frame"]
    np.testing.assert_array_equal(stamps, clip_stamps)
    np.testing.assert_array_equal(stamps[:, 0], np.arange(base.N))
    np.testing.assert_array_equal(original[first:first+base.N, 1], stamps[:, 1])
    calibration = yaml.safe_load((RECT/"calibrations/rectify_params.yaml").read_text())
    k = np.asarray(calibration["rectified"]["K1_rect"])
    np.testing.assert_array_equal(k, lock["configuration"]["camera_intrinsics"])
    source = read_npz(VIO)
    adapted = adapt_camera_poses(stamps[:, 1], first+np.arange(base.N), source, calibration["rectified"]["R1"])
    gaps = np.flatnonzero(adapted["camera_pose_interpolated"])
    np.testing.assert_array_equal(gaps, [550, 559])
    out.mkdir(parents=True)
    (out/"cache").mkdir()
    np.savez_compressed(out/"cache/vio_pose_adapter.npz", **adapted)
    from lib.eval_utils.custom_utils import load_slam_cam
    rwc, twc, rcw, tcw = [x.numpy() for x in load_slam_cam(out/"cache/vio_pose_adapter.npz")]
    expected = adapted["T_world_rect"]
    loader_error = float(abs(rcw-expected[:, :3, :3]).max())
    np.testing.assert_allclose(tcw, expected[:, :3, 3], atol=1e-12, rtol=0)
    assert loader_error < 1e-12
    assert float(abs(np.einsum("nij,nj->ni", rwc, tcw)+twc).max()) < 1e-12
    # This independent full-timeline round trip also covers both camera gaps.
    hands = backend.load_predictions("optimized")
    ids = hands["frame_idx"]
    world = np.einsum("nij,nvj->nvi", rcw[ids], hands["joints_camera"]) + tcw[ids, None]
    restored = np.einsum("nij,nvj->nvi", rwc[ids], world) + twc[ids, None]
    uv = restored[..., :2]/restored[..., 2:]*k[0, 0]+k[:2, 2]
    direct_uv = hands["joints_camera"].astype(float)
    direct_uv = direct_uv[..., :2]/direct_uv[..., 2:]*k[0, 0]+k[:2, 2]
    projection_error = float(abs(uv-direct_uv).max())
    assert projection_error < 1e-5 and set(hands["side"]) == {0, 1}
    gap_rows = [{"frame_idx": int(i), "original_frame_idx": int(first+i),
                 "timestamp_camera": float(stamps[i, 1]),
                 "anchor_frames": adapted["interpolation_anchor_frames"][i].tolist(),
                 "weight": float(adapted["interpolation_weight"][i])} for i in gaps]
    base.dump(out/"cache/vio_adapter_audit.json", {"passed": True,
        "source_pose_count": int(adapted["camera_pose_from_source"].sum()),
        "interpolated_pose_count": len(gaps), "missing_after_adaptation": 0,
        "gaps": gap_rows, "loader_rotation_max_error": loader_error,
        "full_timeline_both_hands_projection_roundtrip_max_px": projection_error,
        "exact_frame_ids_and_timestamps": True, "seconds_including_input_verification": time.perf_counter()-start,
        "coordinate_conversion": "T_world_rect = T_world_raw @ blockdiag(R1.T,1)",
        "scale": 1., "extra_imu_extrinsic_or_time_offset_applied": False,
        "source_world_origin_retained": True, "camera_gaps_are_not_hand_observations": True})
    with np.load(BASELINE/"cache/full/optimized/slam_scaled.npz") as original_slam:
        baseline_camera = {"traj": original_slam["traj"], "scale": original_slam["scale"]}
    for arm in ARMS:
        root = arm_root(out, arm)
        (root/"cache").mkdir(parents=True)
        (root/"cache/frames").symlink_to(backend.SOURCE/"cache/frames", target_is_directory=True)
        camera = baseline_camera if arm == "baseline" else adapted
        for phase in ("pilot", "full"):
            frames = backend.frame_ids(phase)
            cache = backend.folder(root, phase, "optimized")
            path = cache/"slam_scaled.npz"
            if phase == "full" and arm == "baseline":
                path.symlink_to(BASELINE/"cache/full/optimized/slam_scaled.npz")
            else:
                np.savez_compressed(path, traj=camera["traj"][frames], scale=camera["scale"],
                    frame_idx=frames, camera_intrinsics=k, img_focal=k[0, 0], img_center=k[:2, 2])
            dest = prediction_dir(out, phase, arm)
            dest.mkdir(parents=True)
            (dest/"camera_source").symlink_to(backend.SOURCE/"predictions/optimized", target_is_directory=True)
        if arm == "baseline":
            for name in ("world_observed.npz", "world_completed.npz", "infiller_audit.json", "world_validation.json"):
                (prediction_dir(out, "full", arm)/name).symlink_to(BASELINE/"predictions/optimized"/name)
        flags = {key: (adapted[key] if arm == "vio" else np.full(base.N, key == "camera_pose_from_source", bool))
                 for key in ("camera_pose_from_source", "camera_pose_interpolated", "camera_pose_missing")}
        np.savez_compressed(root/"camera_poses.npz", traj=camera["traj"], scale=camera["scale"],
            frame_idx=np.arange(base.N), timestamp_camera=stamps[:, 1], **flags)
    code_paths = list(dict.fromkeys(base.CORE_FILES + ["lib/eval_utils/custom_utils.py", "lib/eval_utils/filling_utils.py",
        "scripts/native_backend_infilling.py", "scripts/run_native_backend_comparison.py",
        "scripts/run_vio_replacement.py", "scripts/render_vio_replacement.py", "scripts/vio_camera_adapter.py",
        "scripts/run_vio_replacement_stages.py",
        "tests/test_vio_camera_adapter.py", "docs/vio_replacement_experiment.md"]))
    code_hashes = {}
    for name in code_paths:
        path = out/"provenance/code"/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
        code_hashes[str(ROOT/name)] = base.sha(path)
    for name, command in {"pip-freeze.txt": [sys.executable, "-m", "pip", "freeze"],
                          "submodules.txt": ["git", "submodule", "status", "--recursive"]}.items():
        (out/"provenance"/name).write_text(subprocess.check_output(command, cwd=ROOT, text=True))
    manifest = {"schema_version": 1, "git_head": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
        "baseline_tag": TAG, "baseline_commit": frozen, "baseline_archive": str(BASELINE),
        "input": str(base.VIDEO), "frame_range_inclusive": [0, 1799], "fps": 30, "image_size": [1920, 1074],
        "camera_intrinsics": k.tolist(), "camera_prediction_source": str(backend.SOURCE/"predictions/optimized"),
        "protected_sha256": protected, "code_sha256": code_hashes,
        "source_trajectory": str(VIO), "vio_source_schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in source.items()},
        "source_original_frames": [first, first+1799], "camera_gap_policy": "Isolated single-frame linear translation + timestamp SLERP; flags retained",
        "hand_infiller_policy": "Unchanged frozen optimized B physical/side ownership bounded native infiller",
        "baseline_reuse": "Full predictions, infiller outputs and scaled DROID copied by read-only symlinks; no new DROID/Metric3D inference",
        "pilot_policy": "Both arms slice their FULL camera trajectories at 88..201 and run identical pilot ownership intervals; not the historical short DROID run",
        "forbidden_stages_executed": [], "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "seed": 0,
        "foreign_git_head": git_foreign("rev-parse", "HEAD"), "foreign_git_status_before": git_foreign("status", "--short")}
    base.dump(out/"manifest.json", manifest)


def git_foreign(*args):
    return subprocess.check_output(["git", *args], cwd=base.FOREIGN, text=True).strip()


def predict(out, phase):
    if phase == "full":
        assert base.read(out/"pilot/validation.json")["passed"], "Inspect/validate pilot first"
    manifest = base.read(out/"manifest.json")
    k = np.asarray(manifest["camera_intrinsics"])
    for arm in ARMS:
        if phase == "full" and arm == "baseline":
            continue
        destination = prediction_dir(out, phase, arm)
        if (destination/"world_observed.npz").exists():
            raise ValueError("Predictions already exist; retain this attempt and use a fresh run")
        begin = time.perf_counter()
        backend.world(arm_root(out, arm), phase, "optimized")
        world_seconds = time.perf_counter()-begin
        begin = time.perf_counter()
        completion.fill(arm_root(out, arm), phase, "optimized", k)
        base.dump(destination/"stage_runtime.json", {"world_seconds": world_seconds,
            "fill_and_export_seconds": time.perf_counter()-begin, "baseline_full_reused": False})
    audit_predictions(out, phase)


def statistics(values):
    values = np.asarray(values)
    return {"count": int(values.size), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())} if values.size else {"count": 0}


def audit_predictions(out, phase):
    source = backend.load_predictions("optimized")
    frames = backend.frame_ids(phase)
    source_ids = np.flatnonzero(np.isin(source["frame_idx"], frames))
    result, completions, policies = {}, {}, {}
    for arm in ARMS:
        dest = prediction_dir(out, phase, arm)
        observed, completed = [read_npz(dest/f"world_{mode}.npz") for mode in ("observed", "completed")]
        completions[arm] = completed
        camera = read_npz(arm_root(out, arm)/"camera_poses.npz")
        obs = completed["observed"]
        np.testing.assert_array_equal(completed["generated"], ~obs)
        np.testing.assert_array_equal(observed["source_prediction_index"], source_ids)
        np.testing.assert_array_equal(np.sort(completed["source_prediction_index"][obs]), source_ids)
        ids = completed["source_prediction_index"][obs]
        for key in ("frame_idx", "side", "slot"):
            np.testing.assert_array_equal(completed[key][obs], source[key][ids])
        np.testing.assert_array_equal(completed["source_prediction_index"][~obs], -1)
        error = float(abs(completed["vertices_camera"][obs]-source["vertices"][ids]).max())
        assert error < .0002, (arm, error)
        masks = [np.zeros((base.N, 2), bool) for _ in range(2)]
        for mask, selected in zip(masks, (obs, ~obs)):
            mask[completed["frame_idx"][selected], completed["slot"][selected]] = True
        assert not (masks[0] & masks[1]).any()
        for expected, name in zip((*masks, ~(masks[0]|masks[1])), ("observed_mask", "generated_mask", "missing_mask")):
            np.testing.assert_array_equal(expected, completed[name])
        np.testing.assert_array_equal(masks[0][frames], source["raw_observed_mask"][frames])
        policy = base.read(dest/"infiller_audit.json")
        policies[arm] = {key: policy[key] for key in ("policy", "intervals", "ambiguities", "completion_parts")}
        parts = {p["ownership_interval"]: p for p in policy["completion_parts"]}
        for i in np.flatnonzero(~obs):
            part = parts[int(completed["ownership_interval"][i])]
            fi, side, slot = [int(completed[key][i]) for key in ("frame_idx", "side", "slot")]
            assert part["first"] <= fi <= part["last"] and part["owners"][side][0] == slot
            assert part["owners"][side][2] == side and fi not in policy["ambiguities"]["side_conflict_frames"]
        for key in ("vertices", "vertices_camera", "joints_camera", "joints_world"):
            assert np.isfinite(completed[key]).all()
        flags = {key: camera[key][completed["frame_idx"]] for key in
                 ("camera_pose_from_source", "camera_pose_interpolated", "camera_pose_missing")}
        flag_path = dest/"row_camera_status.npz"
        if not flag_path.exists():
            np.savez_compressed(flag_path, frame_idx=completed["frame_idx"], hand_observed=obs,
                hand_generated=~obs, source_prediction_index=completed["source_prediction_index"], **flags)
        nonpositive = (completed["joints_camera"][..., 2] <= 0).any(axis=-1)
        result[arm] = {"observed_instances": int(obs.sum()), "generated_instances": int((~obs).sum()),
            "missing_slots": int(completed["missing_mask"][frames].sum()), "observed_geometry_max_error": error,
            "generated_owners_checked": int((~obs).sum()), "nonpositive_depth_frames": np.unique(completed["frame_idx"][nonpositive]).tolist(),
            "nonpositive_observed_instances": int((nonpositive & obs).sum()),
            "camera_source_frames": int(camera["camera_pose_from_source"][frames].sum()),
            "camera_interpolated_frames": int(camera["camera_pose_interpolated"][frames].sum()),
            "camera_missing_frames": int(camera["camera_pose_missing"][frames].sum()),
            "observed_hands_on_interpolated_camera_frames": int((obs & flags["camera_pose_interpolated"]).sum()),
            "generated_hands_on_interpolated_camera_frames": int((~obs & flags["camera_pose_interpolated"]).sum())}
    assert policies["baseline"] == policies["vio"], "Infiller ownership/context changed between arms"
    a, b = [completions[arm] for arm in ARMS]
    for key in ("frame_idx", "side", "slot", "observed", "generated", "source_prediction_index", "ownership_interval", "observed_mask", "generated_mask", "missing_mask"):
        np.testing.assert_array_equal(a[key], b[key])
    generated = a["generated"]
    result["generated_joint_camera_change_m"] = statistics(np.linalg.norm(a["joints_camera"][generated]-b["joints_camera"][generated], axis=-1))
    result["passed"] = True
    result["same_hand_masks_ids_and_infiller_context"] = True
    base.dump(out/("prediction_audit.json" if phase == "full" else "pilot/prediction_audit.json"), result)
    return result


def trajectory_diagnostics(out):
    cameras, positions, rotations = {}, {}, {}
    for arm in ARMS:
        cameras[arm] = read_npz(arm_root(out, arm)/"camera_poses.npz")
        c = cameras[arm]
        r = Rotation.from_quat(c["traj"][:, 3:]).as_matrix()
        t = c["traj"][:, :3]*c["scale"]
        positions[arm], rotations[arm] = (t-t[0])@r[0], r[0].T@r
    common = cameras["vio"]["camera_pose_from_source"]
    timestamps = cameras["vio"]["timestamp_camera"]
    edges = common[:-1] & common[1:]
    result = {"alignment": "Each first camera to identity using SE(3) only, no fitted scale", "common_source_frames": int(common.sum()),
        "interpretation": "Estimated trajectory and between-method differences, not ground-truth errors", "arms": {}}
    for arm in ARMS:
        step = np.linalg.norm(np.diff(positions[arm], axis=0), axis=-1)
        angular = Rotation.from_matrix(rotations[arm][:-1].transpose(0, 2, 1)@rotations[arm][1:]).magnitude()
        result["arms"][arm] = {"path_length_m_full_including_camera_interpolation": float(step.sum()),
            "net_displacement_m": float(np.linalg.norm(positions[arm][-1])),
            "speed_m_s_common_adjacent_source_frames": statistics((step/np.diff(timestamps))[edges]),
            "angular_speed_deg_s_common_adjacent_source_frames": statistics(np.rad2deg(angular/np.diff(timestamps))[edges])}
    result["position_difference_m_common_source_frames"] = statistics(np.linalg.norm(positions["baseline"]-positions["vio"], axis=-1)[common])
    angle = Rotation.from_matrix(rotations["baseline"].transpose(0, 2, 1)@rotations["vio"]).magnitude()
    result["rotation_difference_deg_common_source_frames"] = statistics(np.rad2deg(angle)[common])
    np.savez_compressed(out/"aligned_camera_trajectories.npz", frame_idx=np.arange(base.N), timestamp_camera=timestamps,
        baseline_position=positions["baseline"], vio_position=positions["vio"], common_source_pose_mask=common,
        baseline_rotation=rotations["baseline"], vio_rotation=rotations["vio"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for arm in ARMS:
        p = positions[arm]
        axes[0].plot(p[:, 0], p[:, 2], label=arm)
        axes[1].plot(timestamps-timestamps[0], np.linalg.norm(p, axis=-1), label=arm)
    axes[0].set(xlabel="First-camera X (m)", ylabel="First-camera Z (m)", title="SE(3) origin only; no scale fitting")
    axes[0].axis("equal")
    axes[1].set(xlabel="Time (s)", ylabel="Displacement from start (m)", title="Trajectory estimates, not ground truth")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out/"camera_trajectory_comparison.png", dpi=150)
    plt.close(fig)
    return result


def report(out):
    from render_vio_replacement import audit_video_alignment
    predictions = audit_predictions(out, "full")
    rendering = base.read(out/"render_audit.json")
    assert rendering["frame_count"] == base.N
    trajectory = trajectory_diagnostics(out)
    video_alignment = audit_video_alignment(out)
    manifest = base.read(out/"manifest.json")
    verify_hashes(manifest["protected_sha256"])
    assert git_foreign("status", "--short") == manifest["foreign_git_status_before"]
    test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, capture_output=True, text=True)
    (out/"provenance/unittest.log").write_text(test.stdout+test.stderr)
    assert test.returncode == 0
    stages = [base.read(p) for p in sorted((out/"logs").glob("*.json"))]
    result = {"passed": True, "experiment": "VIO replacement only", "frames": 1800, "fps": 30,
        "predictions": predictions, "camera_adapter": base.read(out/"cache/vio_adapter_audit.json"),
        "trajectory": trajectory, "video_alignment": video_alignment, "rendering": rendering,
        "protected_files_unchanged": len(manifest["protected_sha256"]), "git_head": manifest["git_head"],
        "baseline_commit": manifest["baseline_commit"], "stage_records_before_report": stages,
        "unit_tests_exit_code": test.returncode, "droid_metric3d_or_hand_network_rerun": False,
        "accuracy_claim": "None; user assumes VIO reasonably accurate for this integration test"}
    base.dump(out/"diagnostics.json", result)
    a, b = predictions["baseline"], predictions["vio"]
    rows = "\n".join(f"| {key} | {a[key]} | {b[key]} |" for key in
        ("observed_instances", "generated_instances", "missing_slots", "camera_source_frames", "camera_interpolated_frames",
         "camera_missing_frames", "observed_hands_on_interpolated_camera_frames", "nonpositive_depth_frames"))
    text = f"""# Optimized HaWoR: Native Camera Backend vs OpenVINS

Completed rectified frames 0..1799, 1920x1074, 30 FPS. Output: `{out}`.
Execution commit: `{manifest['git_head']}`. Baseline commit: `{manifest['baseline_commit']}`.
Correct K: `{manifest['camera_intrinsics']}`. Full configuration, source hashes and schema: `manifest.json`.

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
{rows}

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
frame count/FPS checks and sampled RGB/column alignment. {len(manifest['protected_sha256'])} protected files
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
- `arms/{{baseline,vio}}/predictions/optimized/`: observed/completed world exports and camera-row status.
- `windows/`: requested five windows plus 544..564 covering camera gaps and 918..926 covering old generated outlier.
- `logs/`: exact commands, per-stage wall times, return codes; `provenance/`: code snapshots and environment.

VIO cache adaptation excludes the historical OpenVINS run cost, which is not measured here. Baseline
inference is cached, not zero-cost inference. Full VIO native infiller time:
{base.read(prediction_dir(out, 'full', 'vio')/'infiller_audit.json')['seconds']:.6f} seconds;
common full rendering: {rendering['seconds']:.3f} seconds before decode/window exports.
The stage JSON logs also include process startup, export and validation costs.

Reproduce from the execution commit using `docs/vio_replacement_experiment.md` with a fresh output.
Playback: `mpv --speed=0.5 '{out}/world_observed_comparison.mp4'`.
"""
    (out/"report.md").write_text(text)


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PARENT/"run_001")
    parser.add_argument("--stage", required=True, choices=("prepare", "pilot", "render-pilot", "validate", "full", "render", "report"))
    args = parser.parse_args()
    out = args.output.resolve()
    if out == OUTPUT_PARENT or not out.is_relative_to(OUTPUT_PARENT):
        raise ValueError(f"Use a fresh child of {OUTPUT_PARENT}")
    if args.stage != "prepare":
        manifest = base.read(out/"manifest.json")
        assert git("rev-parse", "HEAD") == manifest["git_head"], "Execution commit changed"
        verify_hashes(manifest["code_sha256"])
    start = time.perf_counter()
    try:
        if args.stage == "prepare":
            prepare(out)
        elif args.stage in ("pilot", "full"):
            predict(out, args.stage)
        elif args.stage in ("render-pilot", "render"):
            from render_vio_replacement import render
            render(out, "pilot" if args.stage == "render-pilot" else "full")
        elif args.stage == "validate":
            prediction = base.read(out/"pilot/prediction_audit.json")
            rendering = base.read(out/"pilot/render_audit.json")
            assert prediction["passed"] and rendering["frame_count"] == 114
            assert all(max(values) > 0 for values in rendering["mesh_pixels"].values())
            base.dump(out/"pilot/validation.json", {"passed": True, "frames": [88, 201],
                "camera_gap_frames_numerically_checked": [550, 559], "before_full": True,
                "same_infiller_intervals": True, "nonblank_shared_renderer": True})
        else:
            report(out)
    except Exception:
        if out.exists():
            base.dump(out/f"logs/{time.time_ns()}_{args.stage}_failed.json", {"stage": args.stage, "command": sys.argv,
                "git_head": git("rev-parse", "HEAD"), "seconds": time.perf_counter()-start, "exit_code": 1})
        raise
    base.dump(out/f"logs/{time.time_ns()}_{args.stage}.json", {"stage": args.stage, "command": [sys.executable, *sys.argv],
        "git_head": git("rev-parse", "HEAD"), "seconds": time.perf_counter()-start, "exit_code": 0})
    print(f"DONE {args.stage}: {out}", flush=True)


if __name__ == "__main__":
    main()
