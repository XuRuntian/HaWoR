"""Calibrated rerun with frozen A/B observations and isolated reconstruction caches."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch
import yaml

import run_pre_frontend_comparison as base
from lib.datasets.track_dataset import TrackDatasetEval

REFERENCE = base.ROOT / "outputs/pre_frontend_comparison/full_60s"
DEFAULT_OUTPUT = base.ROOT / "outputs/pre_frontend_comparison/calibrated_full_60s"
CALIBRATION = base.VIDEO.parents[1] / "calibrations/rectify_params.yaml"
PARAMETERS = ("init_root_orient", "init_hand_pose", "init_trans", "init_betas")
IDENTITY_ARRAYS = ("frame_idx", "bbox", "side", "slot", "chunk_id", "observed", "generated",
                   "raw_observed_mask", "source_hypothesis_side_mask", "present_mask", "generated_mask", "missing_mask")


def prepare(out):
    if (out / "manifest.json").exists():
        manifest = base.read(out / "manifest.json")
        for name, digest in manifest["code_sha256"].items():
            assert base.sha(base.ROOT / name) == digest, f"Code changed during experiment: {name}"
        assert base.sha(CALIBRATION) == manifest["calibration_sha256"]
        return np.asarray(manifest["camera_intrinsics"])
    if out.exists():
        raise ValueError("Use a fresh output directory")
    old = base.read(REFERENCE / "manifest.json")
    assert base.read(REFERENCE / "audit.json")["passed"]
    assert base.sha(base.VIDEO) == old["input_sha256"]
    assert base.sha(base.CACHE) == old["cache_sha256"]
    calib = yaml.safe_load(CALIBRATION.read_text())
    k = np.asarray(calib["rectified"]["K1_rect"], dtype=np.float64)
    assert calib["rectified"]["image_size"] == [base.WIDTH, base.HEIGHT]
    assert k[0, 0] == k[1, 1] and k[0, 0] > 0
    for path, digest in {**old["asset_sha256"], **old["foreign_file_sha256"]}.items():
        assert base.sha(path) == digest, path
    old_hashes = {str(p): base.sha(p) for p in sorted(REFERENCE.rglob("*")) if p.is_file() and not p.is_symlink()}
    out.mkdir(parents=True)
    (out / "cache").mkdir()
    (out / "cache/frames").symlink_to(REFERENCE / "cache/frames", target_is_directory=True)
    for phase in ("pilot", "full"):
        folder = out / "cache" / phase
        folder.mkdir()
        (folder / "native_raw_tracks.npy").write_bytes((REFERENCE / "cache" / phase / "native_raw_tracks.npy").read_bytes())
    (out / "cache/frontend_selection.json").write_bytes((REFERENCE / "cache/frontend_selection.json").read_bytes())
    code = list(dict.fromkeys(base.CORE_FILES + [
        "scripts/run_calibrated_frontend_comparison.py", "tests/test_calibrated_camera.py",
        "tests/test_frontend_comparison_adapter.py"]))
    manifest = {
        "reference_experiment": str(REFERENCE), "reference_manifest": old,
        "reference_file_sha256": old_hashes, "input": str(base.VIDEO),
        "input_sha256": old["input_sha256"], "cache_sha256": old["cache_sha256"],
        "calibration": str(CALIBRATION), "calibration_sha256": base.sha(CALIBRATION),
        "camera_intrinsics": k.tolist(), "previous_intrinsics": [[600,0,960],[0,600,537],[0,0,1]],
        "code_sha256": {name: base.sha(base.ROOT / name) for name in code},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "asset_sha256": old["asset_sha256"], "foreign_file_sha256": old["foreign_file_sha256"],
        "observation_source": "Frozen reference native tracks and optimized selected observations; no detector rerun",
        "frames": [0,1799], "fps": 30, "image_size": [1920,1074],
        "infiller": False, "world_slam_depth_vio": False,
        "network_architecture_checkpoint_and_mano_unchanged": True,
        "camera_geometry_interface": "Opt-in mirrored principal point for conditioning; original principal point for output; exact image-width unmirror",
        "python": base.sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    for name in code:
        path = out / "provenance/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((base.ROOT / name).read_bytes())
    (out / "provenance/pip-freeze.txt").write_text(subprocess.check_output([base.sys.executable, "-m", "pip", "freeze"], text=True))
    (out / "provenance/interface.patch").write_text(subprocess.check_output(["git", "diff"], text=True))
    base.dump(out / "manifest.json", manifest)
    return k


def tracks(out, phase):
    native = np.load(out / "cache" / phase / "native_raw_tracks.npy", allow_pickle=True).item()
    selection = base.read(out / "cache/frontend_selection.json")
    allowed = set(range(base.PILOT[0], base.PILOT[1] + 1)) if phase == "pilot" else None
    optimized = base.optimized_groups(selection["selected"], allowed)
    return {"native": native, "optimized": optimized}


def matching_predictions(old, new, phase):
    results = {}
    for branch in ("native", "optimized"):
        old_dir, new_dir = base.prediction_dir(old, phase, branch), base.prediction_dir(new, phase, branch)
        assert base.read(old_dir / "index.json") == base.read(new_dir / "index.json"), branch
        assert base.read(old_dir / "chunks.json") == base.read(new_dir / "chunks.json"), branch
        with np.load(old_dir / "camera_predictions.npz") as a, np.load(new_dir / "camera_predictions.npz") as b:
            for key in IDENTITY_ARRAYS:
                assert np.array_equal(a[key], b[key]), (branch, key)
            results[branch] = {"observations_metadata_chunks_masks_and_crops_unchanged": True,
                               "instances": len(b["frame_idx"]),
                               "parameter_max_abs_difference": {key: float(abs(a[key] - b[key]).max()) for key in PARAMETERS}}
    return results


def crop_and_mirror_checks(out, k):
    result = {}
    for branch in ("native", "optimized"):
        with np.load(base.prediction_dir(out, "pilot", branch) / "camera_predictions.npz") as data:
            count, sides, max_error = 0, set(), 0.0
            for fi, box, side in zip(data["frame_idx"], data["bbox"], data["side"]):
                kwargs = dict(imgfiles=[str(out / f"cache/frames/{fi:04d}.jpg")], boxes=box[None].copy(),
                              img_focal=float(k[0,0]), img_center=k[:2,2].tolist(), dilate=1.2, do_flip=not bool(side))
                legacy = TrackDatasetEval(**kwargs)[0]
                dataset = TrackDatasetEval(**kwargs, calibrated_camera=True)
                calibrated, repeated = dataset[0], dataset[0]
                assert torch.equal(legacy["img"], calibrated["img"])
                assert torch.equal(calibrated["img"], repeated["img"])
                original_center = np.asarray([(box[0]+box[2])/2, (box[1]+box[3])/2])
                mirrored = calibrated["center"].numpy().copy()
                if side == 0:
                    mirrored[0] = base.WIDTH - 1 - mirrored[0]
                    assert abs(float(calibrated["img_center"][0]) - (base.WIDTH - 1 - k[0,2])) < .0002
                else:
                    assert abs(float(calibrated["img_center"][0]) - k[0,2]) < .0002
                max_error = max(max_error, float(abs(mirrored - original_center).max()))
                assert max_error < .0002
                count += 1
                sides.add(int(side))
            assert sides == {0,1}
            result[branch] = {"crop_tensor_comparisons": count, "crop_pixels_exactly_unchanged": True,
                              "sides_checked": sorted(sides), "bbox_mirror_roundtrip_max_error_px": max_error}
    return result


def pilot(out, k):
    frozen = tracks(out, "pilot")
    regression = out / "default_regression"
    (regression / "cache").mkdir(parents=True)
    (regression / "cache/frames").symlink_to(out / "cache/frames", target_is_directory=True)
    for branch in ("native", "optimized"):
        base.run_branch(regression, "pilot", branch, deepcopy(frozen[branch]), 0.0)
    default_match = matching_predictions(REFERENCE, regression, "pilot")
    for result in default_match.values():
        assert max(result["parameter_max_abs_difference"].values()) < 1e-5
    golden = base.golden_pilot(regression, deepcopy(frozen["native"]))
    for branch in ("native", "optimized"):
        base.run_branch(out, "pilot", branch, deepcopy(frozen[branch]), 0.0, camera_intrinsics=k)
    invariant = matching_predictions(REFERENCE, out, "pilot")
    crop_checks = crop_and_mirror_checks(out, k)
    projection = base.projection_checks(out, camera_intrinsics=k)
    for value in projection.values():
        assert value["positive_joint_depth_fraction"] == 1
    base.render(out, "pilot", camera_intrinsics=k)
    base.dump(out / "pilot/validation.json", {"passed": True, "frames": list(base.PILOT),
        "frozen_original_pilot_tracks": True, "default_mode_regression": default_match,
        "pinned_native_default_oracle_errors": golden,
        "observation_invariants": invariant, "crop_and_mirror": crop_checks,
        "projection": projection, "performed_before_full": True})


def full(out, k):
    assert base.read(out / "pilot/validation.json")["passed"]
    start = time.perf_counter()
    frozen = tracks(out, "full")
    read_seconds = time.perf_counter() - start
    for branch in ("native", "optimized"):
        base.run_branch(out, "full", branch, deepcopy(frozen[branch]), read_seconds, camera_intrinsics=k)
    base.dump(out / "full_observation_invariants.json", matching_predictions(REFERENCE, out, "full"))


def statistics(values):
    return {"median": float(np.median(values)), "p95": float(np.percentile(values,95)), "max": float(np.max(values))}


def changes(out, k):
    result = {}
    for branch in ("native", "optimized"):
        old_path = base.prediction_dir(REFERENCE, "full", branch) / "camera_predictions.npz"
        new_path = base.prediction_dir(out, "full", branch) / "camera_predictions.npz"
        with np.load(old_path) as a, np.load(new_path) as b:
            angles = {}
            for key in ("init_root_orient", "init_hand_pose"):
                relative = a[key].astype(float).swapaxes(-1,-2) @ b[key].astype(float)
                angle = np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1)))
                angles[key] = statistics(angle)
            joints = b["joints_camera"]
            direct = joints[...,:2] / joints[...,2:] * k[0,0] + k[:2,2]
            renderer = base.Renderer(base.WIDTH, base.HEIGHT, float(k[0,0]), "cuda", bin_size=128, max_faces_per_bin=20000)
            camera, _ = renderer.create_camera_from_cv(torch.eye(3,device="cuda")[None], torch.zeros(1,3,device="cuda"),
                K=torch.tensor(k,dtype=torch.float32,device="cuda")[None])
            screen = camera.transform_points_screen(torch.as_tensor(joints,device="cuda"))[...,:2].cpu().numpy()
            projection_error = float(abs(screen-direct).max())
            assert projection_error < .002
            assert np.isfinite(joints).all() and (joints[...,2] > 0).all()
            result[branch] = {
                "interpretation": "Differences between model predictions, not accuracy gains",
                "translation_difference_model_units": statistics(np.linalg.norm(b["init_trans"]-a["init_trans"],axis=-1)),
                "joint_difference_model_units": statistics(np.linalg.norm(joints-a["joints_camera"],axis=-1)),
                "projected_joint_difference_px": statistics(np.linalg.norm(b["joints_2d"]-a["joints_2d"],axis=-1)),
                "orientation_difference_degrees": angles,
                "old_median_root_translation_z": float(np.median(a["init_trans"][...,2])),
                "calibrated_median_root_translation_z": float(np.median(b["init_trans"][...,2])),
                "max_native_renderer_projection_error_px": projection_error,
                "positive_joint_depth_fraction": float((joints[...,2]>0).mean()),
            }
    return result


def four_way(out):
    inputs = [REFERENCE / "native_overlay.mp4", REFERENCE / "optimized_overlay.mp4",
              out / "native_overlay.mp4", out / "optimized_overlay.mp4"]
    names = ["Native | Default K, f=600", "Optimized | Default K, f=600",
             "Native | Calibrated K, f=444.44", "Optimized | Calibrated K, f=444.44"]
    graph = []
    for i, name in enumerate(names):
        graph.append(f"[{i}:v]setpts=PTS-STARTPTS,setsar=1,drawbox=x=0:y=0:w=iw:h=44:color=0x191919:t=fill,"
            f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{name} | frame %{{n}}':x=16:y=8:fontsize=26:fontcolor=white[v{i}]")
    graph.extend(["[v0][v1]hstack=inputs=2[top]", "[v2][v3]hstack=inputs=2[bottom]", "[top][bottom]vstack=inputs=2[result]"])
    command = ["ffmpeg", "-v", "error", "-n"]
    for path in inputs:
        command.extend(["-i", str(path)])
    target = out / "four_way_default_vs_calibrated.mp4"
    command.extend(["-filter_complex_threads", "2", "-filter_complex", ";".join(graph), "-map", "[result]", "-an",
                    "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)])
    subprocess.run(command, check=True)
    for first,last in base.WINDOWS:
        folder = out / f"windows/{first:04d}-{last:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(target), "-vf",
            f"trim=start_frame={first}:end_frame={last+1},setpts=PTS-STARTPTS", "-an", "-c:v", "libx264",
            "-threads", "4", "-preset", "fast", "-crf", "20", str(folder / target.name)], check=True)
    base.dump(out / "four_way_command.json", command)


def finish(out, k):
    from audit_pre_frontend_comparison import video_frame
    invariant = matching_predictions(REFERENCE, out, "full")
    delta = changes(out, k)
    video_info = {}
    for name, width, height in (("native_overlay",1920,1074), ("optimized_overlay",1920,1074),
                               ("comparison",5760,1074), ("four_way_default_vs_calibrated",3840,2148)):
        path = out / f"{name}.mp4"
        info = base.probe(path)
        stream = info["streams"][0]
        assert (stream["width"],stream["height"],int(stream["nb_frames"]),stream["r_frame_rate"]) == (width,height,1800,"30/1")
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"], check=True)
        video_info[name] = info
    alignment = {}
    for fi in (0,88,114,196,599,1799):
        rgb = video_frame(base.VIDEO,fi)
        comparison = video_frame(out / "comparison.mp4",fi)
        error = float(abs(comparison[48:,:1920].astype(float)-rgb[48:].astype(float)).mean())
        assert error < 8
        four = video_frame(out / "four_way_default_vs_calibrated.mp4",fi)
        errors = []
        for i, (source, name) in enumerate(((REFERENCE,"native"),(REFERENCE,"optimized"),(out,"native"),(out,"optimized"))):
            panel = four[(i//2)*1074:(i//2+1)*1074,(i%2)*1920:(i%2+1)*1920]
            expected = video_frame(source / f"{name}_overlay.mp4",fi)
            error_panel = float(abs(panel[48:].astype(float)-expected[48:].astype(float)).mean())
            assert error_panel < 8
            errors.append(error_panel)
        alignment[str(fi)] = {"RGB_MAE": error, "four_way_panel_codec_MAE": errors}
        if fi in (88,196,599):
            cv2.imwrite(str(out / f"screenshots/four_way_{fi:06d}.jpg"), four)
    windows = {}
    for first,last in base.WINDOWS:
        paths = sorted((out / f"windows/{first:04d}-{last:04d}").glob("*.mp4"))
        assert len(paths) == 4
        windows[f"{first:04d}-{last:04d}"] = {}
        for path in paths:
            count = int(base.probe(path)["streams"][0]["nb_frames"])
            assert count == last-first+1
            windows[f"{first:04d}-{last:04d}"][path.name] = count
    manifest = base.read(out / "manifest.json")
    integrity = {path: base.sha(path) == digest for path,digest in
                 {**manifest["reference_file_sha256"], **manifest["asset_sha256"],
                  **manifest["foreign_file_sha256"], str(CALIBRATION): manifest["calibration_sha256"],
                  str(base.VIDEO): manifest["input_sha256"], str(base.CACHE): manifest["cache_sha256"]}.items()}
    assert all(integrity.values())
    summaries = {branch: base.read(base.prediction_dir(out,"full",branch) / "summary.json") for branch in ("native","optimized")}
    diagnostics = {"passed": True, "manifest": manifest, "observation_invariants": invariant,
        "pilot": base.read(out / "pilot/validation.json"), "native": summaries["native"], "optimized": summaries["optimized"],
        "old_vs_calibrated_prediction_differences": delta, "video_info": video_info, "frame_alignment": alignment,
        "windows": windows, "all_reference_model_and_foreign_hashes_unchanged": True,
        "integrity_checked_file_count": len(integrity), "render": base.read(out / "render_diagnostics.json"),
        "forbidden_stages_executed": [], "accuracy_claim": "not evaluated; no ground truth"}
    base.dump(out / "diagnostics.json", diagnostics)
    table = "\n".join(f"| {key} | {summaries['native'][key]} | {summaries['optimized'][key]} |" for key in
        ("output_observed_instances","output_generated_instances","observed_slots","missing_slots","empty_output_frames","chunk_count","reconstruction_with_capture_seconds"))
    change_table = "\n".join(f"| {branch} | {value['old_median_root_translation_z']:.6f} | {value['calibrated_median_root_translation_z']:.6f} | {value['projected_joint_difference_px']['median']:.3f} |" for branch,value in delta.items())
    (out / "report.md").write_text(f"""# Calibrated Camera-Space A/B Rerun

Completed frames 0..1799, 1920x1074, 30 FPS, 60 seconds from the same rectified video.
Reference: `{REFERENCE}`. New outputs: `{out}`.
K: `{k.tolist()}`. Old K: focal600, principal point(960,537).

## Controlled Inputs

Native A reuses exactly the reference native detector/tracker rows, including majority-side routing.
Optimized B reuses exactly the selected observation IDs, bboxes, side hypotheses, physical slots and
fragments from the reference frontend. No detection, ViTPose, consolidation or association was rerun.
Full prediction source metadata, bbox/side/frame arrays, chunk boundaries, padding and all masks are
identical to each respective old branch. A and B do NOT use identical observations to each other.
Both use the unchanged checkpoint, architecture and MANO. Crop tensors are unchanged in the pilot.
Only calibrated camera interpretation and the necessary mirror/principal-point plumbing changed.
No VIO, world/SLAM, Metric3D, stereo depth, infiller, new filtering, correction or gate is introduced.
Native bbox interpolation and 16-frame temporal inference remain; missing frames stay missing.

## Camera Interface Changes

`hawor_motion_estimation(camera_intrinsics=K)` is opt-in and accepts square-pixel, zero-skew K.
`TrackDatasetEval(calibrated_camera=True)` preserves the original principal point and image width.
For left crops, both the bbox center and the conditioning principal point are mirrored with
`x_mirror = width - 1 - x`. After inference, bbox center is unmirrored using image width, and
projection/translation use the ORIGINAL principal point. The previous default path is unchanged.
`HAWOR.forward_step` changes only this camera metadata handling, not learned layers or parameters.
The renderer uses its existing full-K interface. Intrinsics affect bbox conditioning as well as
translation; this is fresh HaWoR inference, not a resize of old poses or just a renderer change.
Therefore old/new differences include calibrated conditioning AND the explicit mirror adaptation,
not a claimed isolated focal-only effect.

## Validation Before Full Inference

Pilot frames88..114 reused the OLD PILOT tracks, separate from full-sequence tracking history.
Default-camera regression against archived A/B predictions passed (<1e-5 parameter error).
The pinned original native function including its mask pass also matched the default regression.
The calibrated pilot verified both sides, exact crop tensor equality, repeatable crop access,
bbox mirror round-trip and independent full-K pinhole/PyTorch3D projection. It completed before full.
The full run reuses the OLD FULL tracks, not the pilot subset. All outputs have finite geometry,
positive joint depth and projection error below0.002px. See `pilot/validation.json` and diagnostics.

## Counts

| Quantity | Native A | Optimized B |
|---|---:|---:|
{table}

These counts are unchanged from the respective default-camera runs. A slots are majority-routed
side hypotheses, B slots are anonymous physical tracks. They are not common ground-truth identities.
Generated count is zero, not a recall gain. Fragment ambiguity, same-side conflicts and the four
zero-width source boxes remain. Native's existing frame1569 JSON collision is still preserved in
lossless per-call exports/overlays, with the same policy as the old experiment.

## Changes Are Not Accuracy Improvements

| Branch | Old Median Root Z | Calibrated Median Root Z | Median Joint UV Change(px) |
|---|---:|---:|---:|
{change_table}

Z values are model coordinate units with unchanged MANO conventions, not validated metric accuracy.
Diagnostics also record translation, local/root rotation and joint differences. No pose or depth
ground truth is available. Correct calibration does not prove that every learned pose improves.
Compare oldA/newA and oldB/newB for camera-interface sensitivity; compare newA/newB for frontend
differences under the same calibrated camera. Frontend differences still include detector source,
selection, trajectory adaptation and resulting temporal context, not consolidation alone.

## Videos and Reproducibility

`comparison.mp4`: RGB / calibrated Native / calibrated Optimized,5760x1074.
`native_overlay.mp4` and `optimized_overlay.mp4`:1920x1074, common original shader, colors and opacity.
`four_way_default_vs_calibrated.mp4`:3840x2148; TOP=old default, BOTTOM=calibrated;
LEFT=native, RIGHT=optimized. Five inclusive windows each contain all four videos.
All full videos decode completely, and sampled RGB/four-way frames align with their sources.
All {len(integrity)} checked old-result/model/input/foreign files retained their pre-run hashes.
No old video/prediction was overwritten. The recorded default regression source depends on the
current calibrated-interface code, whose exact version is archived under `provenance/code/`.
Rendering two calibrated overlays and RGB comparison took {diagnostics['render']['seconds']:.3f}s;
this excludes the subsequent four-way composition and window exports.

```bash
conda activate hawor
python scripts/run_calibrated_frontend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/calibrated_reproduction" --stage pilot
# Inspect pilot/validation.json and pilot/comparison.mp4 before the full run.
python scripts/run_calibrated_frontend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/calibrated_reproduction" --stage full
python scripts/run_calibrated_frontend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/calibrated_reproduction" --stage render
mpv --speed=0.5 '{out / 'four_way_default_vs_calibrated.mp4'}'
```
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare","pilot","full","render","report"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    os.chdir(base.ROOT)
    out = args.output.resolve()
    assert out.is_relative_to(base.ROOT) and out != REFERENCE and not out.is_relative_to(REFERENCE)
    torch.set_num_threads(4)
    base.seed()
    k = prepare(out)
    if args.stage == "pilot":
        pilot(out,k)
    elif args.stage == "full":
        full(out,k)
    elif args.stage == "render":
        assert (out / "full_observation_invariants.json").exists()
        base.render(out,"full",camera_intrinsics=k)
        four_way(out)
        finish(out,k)
    elif args.stage == "report":
        finish(out,k)


if __name__ == "__main__":
    main()
