"""Isolated native DROID/Metric3D backend for calibrated frontend A/B predictions."""

import argparse
from collections import defaultdict
import gc
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

import run_pre_frontend_comparison as base

ROOT = base.ROOT
SOURCE = ROOT / "outputs/pre_frontend_comparison/calibrated_full_60s"
OUTPUT = ROOT / "outputs/pre_frontend_comparison/calibrated_native_backend_full_60s"
PILOT = (88, 201)
EXTRA_ASSETS = ["weights/external/droid.pth", "weights/hawor/checkpoints/infiller.pt",
                "thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth"]
PARAMETERS = ("init_root_orient", "init_hand_pose", "init_trans", "init_betas")


def frame_ids(phase):
    return np.arange(base.N) if phase == "full" else np.arange(PILOT[0], PILOT[1] + 1)


def folder(out, phase, branch):
    path = out / "cache" / phase / branch
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_predictions(branch):
    with np.load(SOURCE / "predictions" / branch / "camera_predictions.npz") as data:
        return {key: data[key] for key in data.files}


def prepare(out):
    if (out / "manifest.json").exists():
        return base.read(out / "manifest.json")
    if out.exists():
        raise ValueError("Use a fresh output directory")
    source = base.read(SOURCE / "manifest.json")
    assert base.read(SOURCE / "diagnostics.json")["passed"]
    protected = dict(source["reference_file_sha256"])
    protected.update(source["asset_sha256"])
    protected.update(source["foreign_file_sha256"])
    protected.update({source["input"]: source["input_sha256"],
                      source["calibration"]: source["calibration_sha256"]})
    protected.update({str(p): base.sha(p) for p in SOURCE.rglob("*") if p.is_file() and not p.is_symlink()})
    protected.update({str(ROOT / name): base.sha(ROOT / name) for name in EXTRA_ASSETS})
    for name, digest in protected.items():
        assert base.sha(name) == digest, name
    out.mkdir(parents=True)
    (out / "cache").mkdir()
    (out / "cache/frames").symlink_to(SOURCE / "cache/frames", target_is_directory=True)
    (out / "predictions").mkdir()
    for branch in ("native", "optimized"):
        p = out / "predictions" / branch
        p.mkdir()
        (p / "camera_source").symlink_to(SOURCE / "predictions" / branch, target_is_directory=True)
    manifest = {"source": str(SOURCE), "camera_intrinsics": source["camera_intrinsics"],
        "video": str(base.VIDEO), "frames": [0, 1799], "fps": 30, "size": [1920, 1074],
        "protected_sha256": protected, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "world_backend": "native masked DROID and native Metric3D scale estimation",
        "camera_prediction_reuse": "Exact calibrated A/B archives; no detection, association or hand-network rerun",
        "external_depth_or_vio": False,
        "geometry_policy": "Full calibrated K; native mask/depth resizing and all DROID/scale hyperparameters retained",
        "pilot_frames": list(PILOT)}
    base.dump(out / "manifest.json", manifest)
    return manifest


def provenance(out, stage, phase, branch):
    path = out / "provenance" / f"{stage}_{phase}_{branch}_{time.time_ns()}"
    path.mkdir(parents=True, exist_ok=True)
    names = base.CORE_FILES + ["scripts/run_native_backend_comparison.py", "scripts/scripts_test_video/hawor_slam.py",
        "lib/pipeline/masked_droid_slam.py", "lib/pipeline/est_scale.py", "lib/eval_utils/custom_utils.py",
        "lib/eval_utils/filling_utils.py", "thirdparty/Metric3D/metric.py"]
    for extra_name in ("scripts/native_backend_infilling.py", "scripts/run_native_backend_stages.py",
                       "scripts/audit_native_backend_comparison.py", "tests/test_native_backend_comparison.py"):
        if (ROOT/extra_name).exists():
            names.append(extra_name)
    for name in names:
        dest = path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    base.dump(path / "hashes.json", {name: base.sha(ROOT / name) for name in names})


def render_setup(k, width=base.WIDTH, height=base.HEIGHT):
    renderer = base.Renderer(width, height, float(k[0, 0]), "cuda", bin_size=128, max_faces_per_bin=20000)
    camera, lights = renderer.create_camera_from_cv(torch.eye(3, device="cuda")[None],
        torch.zeros(1, 3, device="cuda"), K=torch.as_tensor(k, device="cuda", dtype=torch.float32)[None])
    closure = np.array([[92,38,234],[234,38,239],[38,122,239],[239,122,279],
        [122,118,279],[279,118,215],[118,117,215],[215,117,214],[117,119,214],
        [214,119,121],[119,120,121],[121,120,78],[120,108,78],[78,108,79]])
    right = np.concatenate([base.get_mano_faces(), closure])
    faces = [torch.as_tensor(right[:, [0,2,1]].copy(), device="cuda"), torch.as_tensor(right, device="cuda")]
    return renderer, camera, lights, faces


@torch.no_grad()
def masks(out, branch, k):
    p = folder(out, "full", branch)
    if (p / "mask_audit.json").exists():
        assert base.sha(p / "model_masks.npy") == base.read(p / "mask_audit.json")["sha256"]
        return
    data = load_predictions(branch)
    table = defaultdict(list)
    for i, fi in enumerate(data["frame_idx"]):
        table[int(fi)].append(i)
    renderer, camera, lights, faces = render_setup(k)
    mask = np.lib.format.open_memmap(p / "model_masks.npy", mode="w+", dtype=bool,
                                   shape=(base.N, base.HEIGHT, base.WIDTH))
    counts, start = [], time.perf_counter()
    for fi in range(base.N):
        combined = np.zeros((base.HEIGHT, base.WIDTH), dtype=bool)
        # Native masking renders each hand separately, then unions the silhouettes.
        for i in table[fi]:
            vertices = torch.as_tensor(data["vertices"][i], device="cuda")[None, None]
            _, silhouette = renderer.render_multiple(vertices, faces[int(data["side"][i])],
                torch.tensor([[0., 0., 1., 1.]], device="cuda"), camera, lights)
            combined |= silhouette
        mask[fi] = combined
        counts.append(int(combined.sum()))
        if fi % 100 == 0:
            print(f"MASK {branch} {fi}/1800", flush=True)
    mask.flush()
    del mask
    base.dump(p / "mask_audit.json", {"frames": base.N, "shape": [base.N, base.HEIGHT, base.WIDTH],
        "dtype": "bool", "method": "Unmodified native renderer per-instance union from archived MANO vertices",
        "sha256": base.sha(p / "model_masks.npy"), "pixel_counts": counts,
        "seconds": time.perf_counter() - start})


def slam(out, phase, branch, k, buffer):
    import lib.pipeline.masked_droid_slam as droid_module
    p = folder(out, phase, branch)
    if (p / "droid_audit.json").exists():
        assert base.sha(p / "droid_raw.npz") == base.read(p / "droid_audit.json")["sha256"]
        return
    frames = frame_ids(phase)
    images = [str(out / f"cache/frames/{fi:04d}.jpg") for fi in frames]
    masks_path = folder(out, "full", branch) / "model_masks.npy"
    raw_masks = np.load(masks_path, mmap_mode="r")
    selected_masks = torch.from_numpy(np.asarray(raw_masks[frames]).copy())
    calib = np.asarray([k[0,0], k[1,1], k[0,2], k[1,2]])
    droid_module.args.buffer = buffer
    base.seed()
    start = time.perf_counter()
    droid, trajectory = droid_module.run_slam(images, selected_masks, calib=calib)
    count = droid.video.counter.value
    keyframes = droid.video.tstamp.cpu().int().numpy()[:count]
    disps = droid.video.disps_up.cpu().numpy()[:count]
    assert trajectory.shape == (len(frames), 7) and np.isfinite(trajectory).all()
    assert np.isfinite(disps).all() and (disps > 0).all()
    errors = [float(x) for x in droid.backend.errors]
    np.savez_compressed(p / "droid_raw.npz", traj=trajectory, tstamp=keyframes,
        keyframe_original=frames[keyframes], frame_idx=frames, disps=disps, calib=calib)
    base.dump(p / "droid_audit.json", {"seconds": time.perf_counter()-start, "frames": len(frames),
        "keyframes": count, "droid_args": vars(droid_module.args), "backend_errors": errors,
        "sha256": base.sha(p / "droid_raw.npz"), "mask_sha256": base.sha(masks_path),
        "intrinsics_input": calib.tolist(), "resized_shape_hw": list(disps.shape[-2:])})
    del droid, selected_masks
    gc.collect()
    torch.cuda.empty_cache()


def metric(out, phase, branch, k):
    sys.path.insert(0, str(ROOT / "thirdparty/Metric3D"))
    from metric import Metric3D
    from lib.pipeline.est_scale import est_scale_hybrid
    p = folder(out, phase, branch)
    if (p / "scale_audit.json").exists():
        assert base.sha(p / "slam_scaled.npz") == base.read(p / "scale_audit.json")["sha256"]
        return
    d = np.load(p / "droid_raw.npz")
    model = Metric3D(str(ROOT / EXTRA_ASSETS[2]))
    raw_masks = np.load(folder(out, "full", branch) / "model_masks.npy", mmap_mode="r")
    h, w = d["disps"].shape[-2:]
    calib = np.array([k[0,0], k[1,1], k[0,2], k[1,2]])
    depth_dir = p / "metric3d"
    depth_dir.mkdir(exist_ok=True)
    scales, rows = [], []
    near, far = .4, .7
    start = time.perf_counter()
    for index, fi in enumerate(d["keyframe_original"]):
        path = depth_dir / f"{fi:06d}.npy"
        if path.exists():
            depth = np.load(path)
        else:
            depth = model(str(out / f"cache/frames/{fi:04d}.jpg"), calib.copy())
            depth = cv2.resize(depth, (w, h))
            np.save(path, depth)
        assert depth.shape == (h, w) and np.isfinite(depth).all()
        slam_depth = 1 / d["disps"][index]
        mask = raw_masks[fi].astype(np.uint8)
        attempts = 0
        while True:
            scale = est_scale_hybrid(slam_depth, depth, sigma=.5, msk=mask, near_thresh=near, far_thresh=far)
            if np.isfinite(scale) and scale > 0:
                break
            attempts += 1
            if attempts >= 100:
                raise RuntimeError(f"Native scale estimator failed at frame {fi}; stopped instead of infinite retry")
            near -= .1
            far += .1
        scales.append(scale)
        rows.append({"frame_idx": int(fi), "scale": scale, "near": near, "far": far, "retries": attempts,
                     "depth_sha256": base.sha(path)})
        print(f"METRIC {phase} {branch} {index+1}/{len(d['tstamp'])} frame={fi} scale={scale:.6f}", flush=True)
    scale = float(np.median(scales))
    np.savez_compressed(p / "slam_scaled.npz", **{key: d[key] for key in d.files},
        scale=scale, img_focal=k[0,0], img_center=k[:2,2], camera_intrinsics=k)
    base.dump(p / "scale_audit.json", {"seconds": time.perf_counter()-start, "median_scale": scale,
        "keyframes": rows, "sha256": base.sha(p / "slam_scaled.npz"),
        "algorithm": "Native est_scale_hybrid, thresholds carried across keyframes, median aggregation",
        "geometry": "Native direct resize of Metric3D depth to DROID output shape retained equally",
        "failure_guard": "At most 100 retries, finite positive scale required; no replacement scale"})


@torch.no_grad()
def world(out, phase, branch):
    from lib.eval_utils.custom_utils import load_slam_cam, cam2world_convert
    p = folder(out, phase, branch)
    frames = frame_ids(phase)
    rwc, twc, rcw, tcw = load_slam_cam(p / "slam_scaled.npz")
    rwc, twc, rcw, tcw = [v.float() for v in (rwc, twc, rcw, tcw)]
    data = load_predictions(branch)
    selected = np.flatnonzero(np.isin(data["frame_idx"], frames))
    data = {key: value[selected] for key, value in data.items() if len(value) == len(data["frame_idx"])}
    local = data["frame_idx"] - frames[0]
    vertices = np.einsum("nij,nvj->nvi", rcw[local].numpy(), data["vertices"]) + tcw[local].numpy()[:,None]
    joints = np.einsum("nij,nvj->nvi", rcw[local].numpy(), data["joints_camera"]) + tcw[local].numpy()[:,None]
    parameters = {"init_trans": np.zeros_like(data["init_trans"]), "init_root_orient": np.zeros((len(local),3),np.float32),
                  "init_hand_pose": np.zeros((len(local),15,3),np.float32), "init_betas": data["init_betas"].copy()}
    mano_error = 0.
    for side in (0, 1):
        ids = np.flatnonzero(data["side"] == side)
        for start in range(0, len(ids), 128):
            take = ids[start:start+128]
            inputs = {key: torch.from_numpy(data[key][take])[None] for key in PARAMETERS}
            converted = cam2world_convert(rcw[local[take]], tcw[local[take]], inputs, "right" if side else "left")
            for key, value in converted.items():
                parameters[key][take] = value[0].numpy()
            result = (base.run_mano if side else base.run_mano_left)(converted["init_trans"],
                converted["init_root_orient"], converted["init_hand_pose"], betas=converted["init_betas"])
            mano_error = max(mano_error, float(abs(result["vertices"][0].cpu().numpy()-vertices[take]).max()))
    restored = np.einsum("nij,nvj->nvi", rwc[local].numpy(), vertices) + twc[local].numpy()[:,None]
    roundtrip = float(abs(restored-data["vertices"]).max())
    assert roundtrip < .0001 and mano_error < .0002, (roundtrip, mano_error)
    destination = out / ("predictions" if phase == "full" else "pilot/predictions") / branch
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "world_observed.npz", **parameters, vertices=vertices, joints_world=joints,
        frame_idx=data["frame_idx"], side=data["side"], slot=data["slot"], source_prediction_index=selected,
        observed=np.ones(len(local),bool), generated=np.zeros(len(local),bool))
    base.dump(destination / "world_validation.json", {"passed": True, "camera_world_camera_max_error": roundtrip,
        "native_parameter_conversion_vs_rigid_vertices_error": mano_error, "observed_instances": len(local),
        "generated_instances": 0, "finite": bool(np.isfinite(vertices).all())})


def validate_pilot(out):
    rendering = base.read(out/"pilot/render_audit.json")
    assert rendering["frame_count"] == PILOT[1]-PILOT[0]+1
    checks = {}
    for branch in ("native","optimized"):
        p = out/"pilot/predictions"/branch
        world_check = base.read(p/"world_validation.json")
        fill_check = base.read(p/"infiller_audit.json")
        assert world_check["passed"] and world_check["finite"]
        assert fill_check["observed_geometry_preservation_error"] < .0002
        assert max(rendering["mesh_pixels"][f"{branch}_camera"]) > 0
        assert max(rendering["mesh_pixels"][f"{branch}_completed"]) > 0
        checks[branch] = {"world":world_check,"infiller_observed_preserved":True,"nonblank_camera_and_world":True}
    base.dump(out/"pilot/validation.json",{"passed":True,"frames":list(PILOT),"checks":checks,
              "rendering_and_complete_decode":True,"performed_before_full_slam":True})


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stage", required=True, choices=["prepare", "masks", "slam", "metric", "world", "fill", "render", "report", "validate"])
    parser.add_argument("--phase", choices=["pilot", "full"], default="full")
    parser.add_argument("--branch", choices=["native", "optimized"], default="native")
    parser.add_argument("--buffer", type=int, default=512)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / "outputs") or out == SOURCE or out.is_relative_to(SOURCE):
        raise ValueError("Output must be a separate HaWoR outputs directory")
    manifest = prepare(out)
    k = np.asarray(manifest["camera_intrinsics"])
    provenance(out, args.stage, args.phase, args.branch)
    if args.stage == "masks":
        masks(out, args.branch, k)
    elif args.stage == "slam":
        if args.phase == "full":
            assert base.read(out/"pilot/validation.json")["passed"], "Validate both pilot branches first"
        slam(out, args.phase, args.branch, k, args.buffer)
    elif args.stage == "metric":
        metric(out, args.phase, args.branch, k)
    elif args.stage == "world":
        world(out, args.phase, args.branch)
    elif args.stage == "validate":
        validate_pilot(out)
    elif args.stage in ("fill", "render", "report"):
        import native_backend_infilling as completion
        getattr(completion, args.stage)(out, args.phase, args.branch, k)


if __name__ == "__main__":
    main()
