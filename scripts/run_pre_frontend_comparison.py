"""Camera-only, cache-isolated native/front-end HaWoR migration experiment."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2
import numpy as np
import torch

from frontend_comparison_adapter import (
    native_groups_for_audit, optimized_groups, read_frontend,
)
from scripts.scripts_test_video import hawor_video as hv
from scripts.scripts_test_video.detect_track_video import extract_frames
from lib.pipeline.tools import detect_track
from hawor.utils.process import run_mano, run_mano_left, get_mano_faces
from hawor.utils.rotation import rotation_matrix_to_angle_axis
from lib.vis.renderer import Renderer, create_meshes
from pytorch3d.renderer import Materials

VIDEO = Path("/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/rectified/videos/left_rectified.mp4")
CACHE = Path("/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/failure_mining/full_60s/cache/full_sequence_temporal_pipeline.json")
FOREIGN = Path("/home/user/roboego-hand-vis")
DEFAULT_OUTPUT = ROOT / "outputs/pre_frontend_comparison/full_60s"
N, FPS, WIDTH, HEIGHT, FOCAL = 1800, 30, 1920, 1074, 600.0
WINDOWS = [(88, 94), (100, 114), (176, 182), (193, 201), (596, 602)]
PILOT = (88, 114)
NATIVE_REFERENCE_COMMIT = "66c7d4108d58a716deccd192cb7645170cdc7bd7"
COLORS = [[0.85, 0.25, 0.55], [0.15, 0.70, 0.90]]
CORE_FILES = [
    "scripts/scripts_test_video/hawor_video.py", "scripts/scripts_test_video/detect_track_video.py",
    "lib/pipeline/tools.py", "lib/models/hawor.py", "lib/datasets/track_dataset.py",
    "lib/utils/imutils.py", "lib/models/mano_wrapper.py", "hawor/utils/process.py",
    "lib/vis/renderer.py", "scripts/frontend_comparison_adapter.py",
    "scripts/run_pre_frontend_comparison.py",
]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,avg_frame_rate:format=duration",
        "-of", "json", str(path),
    ]))


def seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False


def prepare(out):
    if (out / "manifest.json").exists():
        manifest = read(out / "manifest.json")
        if manifest["input_sha256"] != sha(VIDEO) or manifest["cache_sha256"] != sha(CACHE):
            raise ValueError("Input changed; use a new experiment directory")
        return manifest
    out.mkdir(parents=True, exist_ok=True)
    info = probe(VIDEO)
    stream = info["streams"][0]
    assert (stream["width"], stream["height"], int(stream["nb_frames"])) == (WIDTH, HEIGHT, N)
    assert stream["r_frame_rate"] == "30/1" and float(info["format"]["duration"]) == 60.0
    payload = read(CACHE)
    frontend = read_frontend(payload, VIDEO, (WIDTH, HEIGHT), N)
    video_sha, cache_sha = sha(VIDEO), sha(CACHE)
    expected = next(r["sha256"] for r in payload["source_signature"] if r["path"] == str(VIDEO))
    assert video_sha == expected, "Cache was produced using different video bytes"
    dump(out / "cache/frontend_selection.json", frontend)
    assets = [ROOT / name for name in (
        "weights/external/detector.pt", "weights/hawor/checkpoints/hawor.ckpt",
        "weights/hawor/model_config.yaml", "_DATA/data/mano/MANO_RIGHT.pkl",
        "_DATA/data_left/mano_left/MANO_LEFT.pkl", "_DATA/data/mano_mean_params.npz",
    )]
    foreign_files = [FOREIGN / name for name in (
        "docs/pipeline/pre_hamer_observation_frontend_migration.md",
        "roboego_hand_vis/egohand/pre_hamer_observation_frontend.py",
        "roboego_hand_vis/egohand/hand_observation_consolidation.py",
        "roboego_hand_vis/egohand/physical_hand_temporal_association.py",
        "roboego_hand_vis/settings.py",
    )]
    manifest = {
        "input": str(VIDEO), "input_sha256": video_sha, "video_info": info,
        "cache": str(CACHE), "cache_sha256": cache_sha,
        "cache_schema": payload["schema_version"], "cache_mode": payload["mode"],
        "cache_sources": payload["sources"], "cache_source_signature": payload["source_signature"],
        "consolidation_config": payload["consolidation_config"],
        "temporal_association_config": payload["temporal_association_config"],
        "hawor_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "frontend_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=FOREIGN, text=True).strip(),
        "frontend_git_status_before": subprocess.check_output(["git", "status", "--short"], cwd=FOREIGN, text=True),
        "asset_sha256": {str(p): sha(p) for p in assets},
        "foreign_file_sha256": {str(p): sha(p) for p in foreign_files},
        "code_sha256": {name: sha(ROOT / name) for name in CORE_FILES},
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "platform": platform.platform(),
        "focal_length": FOCAL, "principal_point": [WIDTH / 2, HEIGHT / 2],
        "focal_source": "HaWoR native fallback, explicitly shared; not calibrated depth",
        "seed": 0, "frame_range_inclusive": [0, N - 1],
        "infiller": "disabled in both: native infiller consumes SLAM/world coordinates",
        "rendering": {"renderer": "native PyTorch3D Renderer", "bin_size": 128,
                      "max_faces_per_bin": 20000, "alpha": 0.65, "colors_rgb": COLORS},
    }
    dump(out / "manifest.json", manifest)
    (out / "provenance").mkdir(exist_ok=True)
    (out / "provenance/pip-freeze.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))
    (out / "provenance/interface.patch").write_text(subprocess.check_output(
        ["git", "diff", "--", "scripts/scripts_test_video/hawor_video.py"], cwd=ROOT, text=True))
    for name in CORE_FILES:
        target = out / "provenance/code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    frames = out / "cache/frames"
    if frames.exists():
        raise ValueError("Unmanifested image cache already exists")
    t = time.perf_counter()
    extract_frames(str(VIDEO), str(frames))
    assert len(list(frames.glob("*.jpg"))) == N
    dump(out / "cache/extraction.json", {"seconds": time.perf_counter() - t,
         "method": "unmodified native extract_frames: ffmpeg fps=30, start_number=0, JPEG defaults",
         "source": str(VIDEO), "frame_count": N})
    return manifest


def case_paths(out, phase, branch, tracks):
    base = out / "cache" / phase / branch
    base.mkdir(parents=True, exist_ok=True)
    video = base / "sequence.mp4"
    if not video.exists():
        video.symlink_to(VIDEO)
    seq = base / "sequence"
    seq.mkdir(exist_ok=True)
    link = seq / "extracted_images"
    if not link.exists():
        link.symlink_to(out / "cache/frames", target_is_directory=True)
    folder = seq / f"tracks_0_{N}"
    folder.mkdir(exist_ok=True)
    if (folder / "frame_chunks_all.npy").exists():
        raise ValueError(f"Refusing stale reconstruction cache: {folder}")
    np.save(folder / "model_tracks.npy", tracks)
    np.save(folder / "model_boxes.npy", np.array([], dtype=object))
    return video, seq


def prediction_dir(out, phase, branch):
    return out / ("predictions" if phase == "full" else "pilot/predictions") / branch


def run_branch(out, phase, branch, tracks, tracking_seconds, *, camera_intrinsics=None):
    focal = FOCAL if camera_intrinsics is None else float(camera_intrinsics[0, 0])
    principal = [WIDTH / 2, HEIGHT / 2] if camera_intrinsics is None else camera_intrinsics[:2, 2]
    target = prediction_dir(out, phase, branch)
    if (target / "summary.json").exists():
        raise ValueError(f"Branch already completed: {target}; do not overwrite silently")
    target.mkdir(parents=True, exist_ok=True)
    (target / "chunks").mkdir(exist_ok=True)
    video, seq = case_paths(out, phase, branch, tracks)
    groups, mapping = native_groups_for_audit(tracks) if branch == "native" else (tracks, [])
    lookup = defaultdict(list)
    for group, rows in groups.items():
        for row in rows:
            lookup[(str(group), int(row["frame"]))].append(row)
    pieces, metadata, chunks = [], [], []

    def capture(group, frame_ids, boxes, flipped, data, raw):
        chunk_id = len(chunks)
        side = int(not flipped)
        cpu = {key: value.detach().cpu().numpy()[0].copy() for key, value in data.items()}
        trans = torch.from_numpy(cpu["init_trans"])[None]
        root = rotation_matrix_to_angle_axis(torch.from_numpy(cpu["init_root_orient"]))[None]
        pose = rotation_matrix_to_angle_axis(torch.from_numpy(cpu["init_hand_pose"]))[None]
        betas = torch.from_numpy(cpu["init_betas"])[None]
        with torch.no_grad():
            result = (run_mano_left if flipped else run_mano)(trans, root, pose, betas=betas)
        vertices = result["vertices"][0].cpu().numpy()
        joints = result["joints"][0].cpu().numpy()
        projected = joints[..., :2] / joints[..., 2:] * focal + principal
        assert np.isfinite(vertices).all() and np.isfinite(projected).all()
        observed, slots = [], []
        rows_meta = []
        for fi, box in zip(frame_ids, boxes):
            matches = lookup[(str(group), int(fi))]
            exact = [r for r in matches if np.array_equal(r["det_box"][0], box)]
            matches = exact or matches
            if not matches:
                raise ValueError("Output has no auditable input row")
            sources = [r["provenance"] for r in matches]
            obs = any(r["det"] for r in matches)
            slot = side if branch == "native" else sources[0]["physical_track_id"]
            observed.append(obs)
            slots.append(slot)
            rows_meta.append({"prediction_index": len(metadata) + len(rows_meta),
                "frame_idx": int(fi), "group": str(group), "chunk_id": chunk_id,
                "hawor_side": side, "state": "observed" if obs else "generated",
                "physical_slot_or_native_side": slot,
                "source_observations": sources,
                "source_mapping_ambiguous": len(sources) != 1})
        piece = {**cpu, "frame_idx": np.asarray(frame_ids), "bbox": np.asarray(boxes),
                 "side": np.full(len(frame_ids), side, dtype=np.int8),
                 "slot": np.asarray(slots, dtype=np.int8),
                 "observed": np.asarray(observed), "generated": ~np.asarray(observed),
                 "vertices": vertices, "joints_camera": joints, "joints_2d": projected.astype(np.float32),
                 "pred_cam": raw["pred_cam"].cpu().numpy(),
                 "chunk_id": np.full(len(frame_ids), chunk_id, dtype=np.int32)}
        for key, value in piece.items():
            assert np.isfinite(value).all(), key
        np.savez_compressed(target / f"chunks/{chunk_id:05d}.npz", **piece)
        chunks.append({"chunk_id": chunk_id, "group": str(group), "hawor_side": side,
                       "frame_idx": [int(x) for x in frame_ids], "length": len(frame_ids),
                       "padded_network_frames": ((len(frame_ids) + 15) // 16) * 16})
        pieces.append(piece)
        metadata.extend(rows_meta)

    args = SimpleNamespace(video_path=str(video), checkpoint=str(ROOT / "weights/hawor/checkpoints/hawor.ckpt"), img_focal=focal)
    seed()
    t = time.perf_counter()
    hv.hawor_motion_estimation(args, 0, N, str(seq),
        track_groups=None if branch == "native" else tracks,
        prediction_callback=capture, render_masks=False, camera_intrinsics=camera_intrinsics)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t
    if not pieces:
        raise ValueError("Branch produced no camera predictions")
    arrays = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
    allowed = set(range(N)) if phase == "full" else set(range(PILOT[0], PILOT[1] + 1))
    assert set(arrays["frame_idx"]).issubset(allowed)
    input_mask = np.zeros((N, 2), dtype=bool)
    source_side_mask = np.zeros((N, 2), dtype=bool)
    for group, rows in groups.items():
        for row in rows:
            if row["det"]:
                slot = int(group) if branch == "native" else row["provenance"]["physical_track_id"]
                input_mask[row["frame"], slot] = True
                source_side_mask[row["frame"], int(row["det_handedness"][0])] = True
    present = np.zeros_like(input_mask)
    present[arrays["frame_idx"], arrays["slot"]] = True
    arrays.update({"raw_observed_mask": input_mask, "source_hypothesis_side_mask": source_side_mask,
                   "present_mask": present, "generated_mask": present & ~input_mask,
                   "missing_mask": ~present})
    np.savez_compressed(target / "camera_predictions.npz", **arrays)
    dump(target / "index.json", metadata)
    dump(target / "chunks.json", chunks)
    duplicate_slots = Counter(zip(arrays["frame_idx"].tolist(), arrays["slot"].tolist()))
    same_side = Counter(zip(arrays["frame_idx"].tolist(), arrays["side"].tolist()))
    summary = {
        "input_observation_instances": sum(bool(row["det"]) for rows in groups.values() for row in rows),
        "output_observed_instances": int(arrays["observed"].sum()),
        "output_generated_instances": int(arrays["generated"].sum()),
        "observed_slots": int(input_mask[list(allowed)].sum()),
        "present_slots": int(present[list(allowed)].sum()),
        "missing_slots": int((~present[list(allowed)]).sum()),
        "generated_slots": int((present & ~input_mask)[list(allowed)].sum()),
        "observed_but_no_output_slots": int((input_mask & ~present)[list(allowed)].sum()),
        "empty_output_frames": int((~present[list(allowed)].any(axis=1)).sum()),
        "duplicate_output_slot_frames": sorted({f for (f, _), count in duplicate_slots.items() if count > 1}),
        "same_hypothesis_side_multi_output_frames": sorted({f for (f, _), count in same_side.items() if count > 1}),
        "source_mapping_ambiguous_instances": sum(r["source_mapping_ambiguous"] for r in metadata),
        "chunk_count": len(chunks), "singleton_chunks": sum(c["length"] == 1 for c in chunks),
        "padded_network_frames": sum(c["padded_network_frames"] for c in chunks),
        "tracking_or_cache_read_seconds": tracking_seconds,
        "reconstruction_with_capture_seconds": seconds, "native_track_mapping": mapping,
        "slot_semantics": "native majority-routed handedness" if branch == "native" else "anonymous physical track slot, not anatomical side",
        "render_masks": False, "mask_contract": "native SLAM mask would be N x H x W bool; skipped symmetrically; observed masks are N x 2",
    }
    dump(target / "summary.json", summary)
    torch.cuda.empty_cache()
    return summary


def get_tracks(out, phase):
    frames = sorted((out / "cache/frames").glob("*.jpg"))
    assert len(frames) == N
    selected = frames if phase == "full" else frames[PILOT[0]:PILOT[1] + 1]
    t = time.perf_counter()
    seed()
    _, array = detect_track([str(f) for f in selected], thresh=0.2)
    tracks = array.item()
    if phase == "pilot":
        for rows in tracks.values():
            for row in rows:
                row["frame"] += PILOT[0]
    seconds = time.perf_counter() - t
    folder = out / "cache" / phase
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "native_raw_tracks.npy", tracks)
    dump(folder / "native_detection.json", {"seconds": seconds, "threshold": 0.2,
        "detector": "unmodified lib.pipeline.tools.detect_track / YOLO.track(persist=True)",
        "frame_range": [0, N - 1] if phase == "full" else list(PILOT)})
    return tracks, seconds


def golden_pilot(out, tracks):
    """Execute the pinned native function, including its mask pass, as an oracle."""
    from copy import deepcopy
    base = out / "pilot/upstream_reference"
    seq = base / "sequence"
    frames = seq / "extracted_images"
    frames.mkdir(parents=True, exist_ok=True)
    for local, fi in enumerate(range(PILOT[0], PILOT[1] + 1)):
        (frames / f"{local:04d}.jpg").symlink_to(out / f"cache/frames/{fi:04d}.jpg")
    count = PILOT[1] - PILOT[0] + 1
    adjusted = deepcopy(tracks)
    for rows in adjusted.values():
        for row in rows:
            row["frame"] -= PILOT[0]
    folder = seq / f"tracks_0_{count}"
    folder.mkdir()
    np.save(folder / "model_tracks.npy", adjusted)
    source = subprocess.check_output(["git", "show", f"{NATIVE_REFERENCE_COMMIT}:scripts/scripts_test_video/hawor_video.py"], cwd=ROOT, text=True)
    namespace = {"__name__": "upstream_camera_reference", "__file__": str(ROOT / "scripts/scripts_test_video/hawor_video.py")}
    exec(compile(source, f"{NATIVE_REFERENCE_COMMIT}:hawor_video.py", "exec"), namespace)
    seed()
    args = SimpleNamespace(video_path=str(base / "sequence.mp4"), checkpoint=str(ROOT / "weights/hawor/checkpoints/hawor.ckpt"), img_focal=FOCAL)
    namespace["hawor_motion_estimation"](args, 0, count, str(seq))
    native = prediction_dir(out, "pilot", "native")
    errors = {}
    for chunk in read(native / "chunks.json"):
        ids = chunk["frame_idx"]
        reference = read(seq / f"cam_space/{chunk['group']}/{ids[0]-PILOT[0]}_{ids[-1]-PILOT[0]}.json")
        with np.load(native / f"chunks/{chunk['chunk_id']:05d}.npz") as result:
            for key, values in reference.items():
                error = float(np.max(np.abs(np.asarray(values)[0] - result[key])))
                errors[key] = max(errors.get(key, 0), error)
    assert max(errors.values()) < 1e-5, errors
    torch.cuda.empty_cache()
    return errors


class VideoWriter:
    def __init__(self, path, width, height):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.path.with_suffix(".ffmpeg.log"), "w")
        self.process = subprocess.Popen([
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(FPS), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
        ], stdin=subprocess.PIPE, stderr=self.log)

    def write(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.process.stdin.close()
        code = self.process.wait()
        self.log.close()
        if code:
            raise RuntimeError(f"ffmpeg failed for {self.path}")


def annotate(frame, title, fi):
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (WIDTH, 44), (25, 25, 25), -1)
    cv2.putText(result, f"{title} | frame {fi:04d} | {fi/FPS:.3f}s", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def render(out, phase, *, camera_intrinsics=None):
    result_dir = out if phase == "full" else out / "pilot"
    arrays = {}
    for branch in ("native", "optimized"):
        with np.load(prediction_dir(out, phase, branch) / "camera_predictions.npz") as archive:
            arrays[branch] = {key: archive[key] for key in ("vertices", "side", "frame_idx")}
    indices = {}
    for branch, data in arrays.items():
        table = defaultdict(list)
        for index, fi in enumerate(data["frame_idx"]):
            table[int(fi)].append(index)
        indices[branch] = table
    focal = FOCAL if camera_intrinsics is None else float(camera_intrinsics[0, 0])
    renderer = Renderer(WIDTH, HEIGHT, focal, "cuda", bin_size=128, max_faces_per_bin=20000)
    render_K = None if camera_intrinsics is None else torch.as_tensor(camera_intrinsics, device="cuda", dtype=torch.float32)[None]
    camera, lights = renderer.create_camera_from_cv(torch.eye(3, device="cuda")[None], torch.zeros(1, 3, device="cuda"), K=render_K)
    faces = get_mano_faces()
    # Same wrist closure and reversed left winding used by native HaWoR.
    closure = np.array([[92,38,234],[234,38,239],[38,122,239],[239,122,279],
        [122,118,279],[279,118,215],[118,117,215],[215,117,214],[117,119,214],
        [214,119,121],[119,120,121],[121,120,78],[120,108,78],[78,108,79]])
    right = np.concatenate([faces, closure])
    face_tensors = [torch.as_tensor(right[:, [0,2,1]].copy(), device="cuda"), torch.as_tensor(right, device="cuda")]
    material = Materials(device="cuda", shininess=0)
    writers = {"native": VideoWriter(result_dir / "native_overlay.mp4", WIDTH, HEIGHT),
               "optimized": VideoWriter(result_dir / "optimized_overlay.mp4", WIDTH, HEIGHT),
               "comparison": VideoWriter(result_dir / "comparison.mp4", WIDTH * 3, HEIGHT)}
    cap = cv2.VideoCapture(str(VIDEO))
    start, end = (0, N - 1) if phase == "full" else PILOT
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    audit = {b: {"rendered_frames": 0, "nonzero_mesh_frames": 0, "mesh_pixel_counts": []} for b in arrays}
    t = time.perf_counter()
    try:
        for fi in range(start, end + 1):
            ok, rgb_bgr = cap.read()
            assert ok and rgb_bgr.shape[:2] == (HEIGHT, WIDTH), fi
            views = {}
            for branch, data in arrays.items():
                frame = rgb_bgr.copy()
                ids = indices[branch][fi]
                pixel_count = 0
                if ids:
                    verts = [torch.as_tensor(data["vertices"][i], device="cuda") for i in ids]
                    sides = [int(data["side"][i]) for i in ids]
                    colors = [torch.tensor(COLORS[s], device="cuda").expand(778, 3) for s in sides]
                    mesh = create_meshes(verts, [face_tensors[s] for s in sides], colors)
                    with torch.no_grad():
                        result = renderer.renderer(mesh, cameras=camera, lights=lights, materials=material)[0].cpu().numpy()
                    mask = result[..., 3] > 0
                    pixel_count = int(mask.sum())
                    rendered_bgr = np.clip(result[..., :3] * 255, 0, 255).astype(np.uint8)[..., ::-1]
                    frame[mask] = np.clip(frame[mask] * 0.35 + rendered_bgr[mask] * 0.65, 0, 255).astype(np.uint8)
                audit[branch]["rendered_frames"] += 1
                audit[branch]["nonzero_mesh_frames"] += int(pixel_count > 0)
                audit[branch]["mesh_pixel_counts"].append(pixel_count)
                title = "Native HaWoR" if branch == "native" else "Optimized HaWoR"
                views[branch] = annotate(frame, title, fi)
                writers[branch].write(views[branch])
            comparison = np.concatenate([annotate(rgb_bgr, "RGB", fi), views["native"], views["optimized"]], axis=1)
            writers["comparison"].write(comparison)
            if fi in (88, 91, 100, 107, 114, 179, 196, 599):
                screenshot = result_dir / "screenshots" / f"frame_{fi:06d}.jpg"
                screenshot.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(screenshot), comparison)
            if fi % 100 == 0:
                print(f"Rendered {fi}/{end}", flush=True)
    finally:
        cap.release()
        for writer in writers.values():
            writer.close()
    audit["seconds"] = time.perf_counter() - t
    for name in ("native_overlay", "optimized_overlay", "comparison"):
        info = probe(result_dir / f"{name}.mp4")
        stream = info["streams"][0]
        assert int(stream["nb_frames"]) == end - start + 1
        assert stream["r_frame_rate"] == "30/1"
        assert (stream["width"], stream["height"]) == (WIDTH * (3 if name == "comparison" else 1), HEIGHT)
    dump(result_dir / "render_diagnostics.json", audit)
    if phase == "full":
        for first, last in WINDOWS:
            folder = out / f"windows/{first:04d}-{last:04d}"
            folder.mkdir(parents=True, exist_ok=True)
            for name in ("native_overlay", "optimized_overlay", "comparison"):
                path = folder / f"{name}.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / f"{name}.mp4"),
                    "-vf", f"trim=start_frame={first}:end_frame={last+1},setpts=PTS-STARTPTS",
                    "-an", "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(path)], check=True)
                assert int(probe(path)["streams"][0]["nb_frames"]) == last - first + 1
    return audit


def projection_checks(out, *, camera_intrinsics=None):
    result = {}
    for branch in ("native", "optimized"):
        with np.load(prediction_dir(out, "pilot", branch) / "camera_predictions.npz") as data:
            assert data["frame_idx"].min() >= PILOT[0] and data["frame_idx"].max() <= PILOT[1]
            focal = FOCAL if camera_intrinsics is None else float(camera_intrinsics[0, 0])
            renderer = Renderer(WIDTH, HEIGHT, focal, "cuda", bin_size=128, max_faces_per_bin=20000)
            render_K = None if camera_intrinsics is None else torch.as_tensor(camera_intrinsics, device="cuda", dtype=torch.float32)[None]
            camera, _ = renderer.create_camera_from_cv(torch.eye(3, device="cuda")[None], torch.zeros(1, 3, device="cuda"), K=render_K)
            points = torch.tensor(data["joints_camera"], device="cuda")
            screen = camera.transform_points_screen(points)[..., :2].cpu().numpy()
            error = float(np.max(np.abs(screen - data["joints_2d"])))
            assert error < 0.002, error
            result[branch] = {"max_opencv_vs_native_pytorch3d_projection_error_px": error,
                "positive_joint_depth_fraction": float((data["joints_camera"][..., 2] > 0).mean()),
                "frame_range": [int(data["frame_idx"].min()), int(data["frame_idx"].max())],
                "predicted_side_counts": {str(k): int(v) for k,v in Counter(data["side"].tolist()).items()}}
    return result


def finish(out):
    manifest = read(out / "manifest.json")
    frontend = read(out / "cache/frontend_selection.json")
    summaries = {b: read(prediction_dir(out, "full", b) / "summary.json") for b in ("native", "optimized")}
    unchanged = {p: sha(p) == digest for p, digest in {**manifest["asset_sha256"], **manifest["foreign_file_sha256"]}.items()}
    unchanged[str(VIDEO)] = sha(VIDEO) == manifest["input_sha256"]
    unchanged[str(CACHE)] = sha(CACHE) == manifest["cache_sha256"]
    assert all(unchanged.values()), unchanged
    assert not any(name in sys.modules for name in ("metric", "droid", "lib.pipeline.masked_droid_slam"))
    diagnostics = {"manifest": manifest, "native": summaries["native"], "optimized": summaries["optimized"],
        "frontend_fragment_mapping": frontend["fragment_mapping"],
        "frontend_same_side_conflict_frames": frontend["same_side_conflict_frames"],
        "frontend_cache_summary": frontend["cache_association_summary"],
        "input_and_model_integrity": unchanged, "pilot": read(out / "pilot/validation.json"),
        "render": read(out / "render_diagnostics.json"),
        "forbidden_stages_executed": [], "pose_accuracy_claim": "not evaluated: no ground truth",
        "frontend_git_status_after": subprocess.check_output(["git", "status", "--short"], cwd=FOREIGN, text=True)}
    dump(out / "diagnostics.json", diagnostics)
    write_report(out, diagnostics)


def write_report(out, d):
    a, b = d["native"], d["optimized"]
    rows = []
    for key in ("input_observation_instances", "output_observed_instances", "output_generated_instances",
                "observed_slots", "generated_slots", "missing_slots", "observed_but_no_output_slots",
                "empty_output_frames", "chunk_count", "singleton_chunks", "padded_network_frames",
                "tracking_or_cache_read_seconds", "reconstruction_with_capture_seconds"):
        rows.append(f"| {key} | {a[key]} | {b[key]} |")
    text = "\n".join(rows)
    report = f"""# Camera-Space Frontend Migration Experiment

## Scope and Result

Completed both branches for original frames **0..1799**, **1920x1074**, **30 FPS**, 60 seconds.
Input: `{VIDEO}`. This is the rectified video, not the raw left video.
No world/SLAM/Metric3D, Stereo/VIO, filters, gate, ROI/pose corrections or network/checkpoint/MANO changes.
All predictions come from HaWoR, not cached HaMeR pose, camera or vertices.
Successful execution is not evidence of improved pose accuracy. There is no pose/identity ground truth here.

## Three Distinct Contributions

1. **Frontend replacement**: A uses native YOLO detector/tracker, threshold 0.2. B reads the existing
   Consolidation + Offline Temporal Association selection, using `(frame_idx, selected_candidate_id)`
   against the original/official observation-ID namespace. It consumes only allowlisted bbox, side,
   confidence, keypoint/provenance and track/fragment/state metadata. Detectron2/ViTPose and the offline
   DP were not rerun. Consequently this A/B also changes the upstream detector/keypoint-based bbox
   source, not just consolidation. It cannot isolate the benefit of the two selectors by themselves.
2. **Trajectory adaptation**: B preserves anonymous physical slot and fragment. HaWoR's anatomical-side
   merge is bypassed. Groups split at fragment boundaries, missing frames and side changes; no identity
   is inferred across a long-gap slot reuse. Per-observation handedness determines native crop mirroring
   and MANO side, NOT track 0/1 or the dominant diagnostic label. Both same-side observations are retained.
3. **HaWoR temporal processing**: both branches use the same checkpoint, 16-frame native inference batches,
   repeated-last-frame padding, space-time and motion modules, crop dilation 1.2 and native projection.
   Different chunks change temporal context and padding workload. Such effects cannot be attributed
   solely to observation selection. The native interpolation implementation is retained in both.

## Interface Changes and Fairness

`hawor_motion_estimation` adds keyword-only `track_groups`, `prediction_callback`, `render_masks`.
Default A detection, majority-side merging, interpolation/valid rewrite, `parse_chunks(min_len=1)`,
prediction, mirrored-left rotation conversion and camera-space JSON output remain the original path.
B explicitly bypasses majority-side merging and permits singleton physical groups (native outer guard
requires two rows per merged side, while its chunk inference already supports length one). It does NOT
bridge missing frames or merge independent identities to satisfy that guard.
The original bbox interpolation only replaces all-zero rows ALREADY in the sparse input list;
it does not create absent frame rows. Original observed masks are saved before the valid rewrite.
With observed-only rows, missing states stay missing. Network temporal predictions at observed frames
are not relabeled as generated; generated means an output lacking a source observation.

Both branches disable the unused SLAM segmentation-mask pass to avoid allocating an N x H x W float64
array. This does not feed back into camera pose. The original mask contract is an N x H x W boolean union
of rendered meshes, distinct from the N x 2 observed/generated/missing masks saved here.
Both retain original camera-space processing but **disable the pretrained infiller**, whose native entry
loads SLAM cameras and converts poses to world coordinates. Supplying identity SLAM would create a
different camera-space infiller experiment and is deliberately not done.
Focal length is the native fallback **600 px**, explicitly shared, principal point `(960,537)`.
No calibration-derived metric depth claim is made. Source frames are extracted using the native JPEG
extraction settings; overlays decode the original rectified MP4 at full resolution.

## Counts (Not Detection Recall or Pose Accuracy)

| Quantity | Native A | Frontend B |
|---|---:|---:|
{text}

Instance counts count model outputs; slot counts use 1800 x 2 masks. A slots mean native majority-routed
left/right, whereas B slots mean anonymous physical tracks. They are NOT matched ground-truth identities.
Multiple outputs in one slot can make instance counts exceed occupied slots. No generated frame is
counted as an observed detection. Neither output availability nor absence is a labeled recall measure.

## Mapping Ambiguity

B has {len(d['frontend_fragment_mapping'])} physical fragments. All fragment mappings are in
`diagnostics.json` and all selected observations/states in `cache/frontend_selection.json`.
Same-side hypothesis conflict frames: `{d['frontend_same_side_conflict_frames']}`.
Track-level dominant labels were never used to flip crops or replace original handedness. Mixed-side
fragments and suspected identity switches remain unresolved, explicitly recorded rather than corrected.
Native multi-output slot frames: `{a['duplicate_output_slot_frames']}`.
Every prediction index maps to source observation(s), physical fragment (B), routed side and chunk in
`predictions/<branch>/index.json`. No synthetic anatomical identity is assigned to an ambiguous fragment.

## Validation

Pilot frames 88..114 ran BEFORE full-sequence reconstruction, with absolute frame indices retained.
The untouched native function at `{NATIVE_REFERENCE_COMMIT}` (including its original mask pass) was executed as an oracle on
the same pilot image/track sequence. Camera-parameter differences: `{d['pilot']['native_reference_max_abs_error']}`.
Projection checks compare pinhole UV with native PyTorch3D camera projection:
`{d['pilot']['projection']}`. Pilot overlays/screenshots are under `pilot/`.
Full overlays have the same 1920x1074 dimensions/FPS, shader/material/light, opacity and side colors.
Comparison is RGB / Native HaWoR / Optimized HaWoR, 5760x1074, 1800 frames at 30 FPS.
All requested inclusive windows were encoded and frame-count checked: {WINDOWS}.
All source-cache, source-video, foreign configuration/module and model hashes match pre-run values.

## Runtime and Reproducibility

Per-stage measured times are in the table and `diagnostics.json`. Reconstruction time includes model
loading and extra prediction/MANO export, not just neural forward time. Rendering both branches and
encoding comparison took {d['render']['seconds']:.3f} seconds. A detector/tracker time and B cached-read
time are different workloads: these are NOT end-to-end frontend speedup numbers. The historical cost
of Detectron2/ViTPose/consolidation/association is excluded and not estimated.
`manifest.json` records input/cache/model/code SHA-256, source git revisions, the dirty upstream state,
package/runtime/GPU versions and all configs. `provenance/` contains the exact adapter/runner/core source
snapshot, interface diff and pip freeze. Existing source caches and old results were never written.
A/B caches are isolated under `cache/full/native` and `cache/full/optimized`; only newly extracted,
identical input JPEGs are shared read-only. Reconstruction refuses existing branch/chunk caches.

```bash
conda activate hawor
cd /home/user/HaWoR
python scripts/run_pre_frontend_comparison.py --output /home/user/HaWoR/outputs/pre_frontend_comparison/reproduction --stage pilot
# Inspect pilot/validation.json and pilot/comparison.mp4 first.
python scripts/run_pre_frontend_comparison.py --output /home/user/HaWoR/outputs/pre_frontend_comparison/reproduction --stage full
python scripts/run_pre_frontend_comparison.py --output /home/user/HaWoR/outputs/pre_frontend_comparison/reproduction --stage render
mpv --speed=0.5 '{out / 'comparison.mp4'}'
```

## Artifacts and Known Limits

- `native_overlay.mp4`, `optimized_overlay.mp4`, `comparison.mp4`
- `predictions/{{native,optimized}}/camera_predictions.npz`: native camera parameters, bbox, camera-space
  vertices/joints, UV, frame indices, routed side, physical-slot/native-side, chunk index, original masks.
- `predictions/<branch>/chunks/`: lossless per-call predictions; `index.json`: source mapping per output.
- `windows/<inclusive-range>/`: the three videos for each requested window; `screenshots/`: selected frames.
- `diagnostics.json`, `manifest.json`, `report.md`, `render_diagnostics.json`, `pilot/validation.json`.

No automatic posture-improvement verdict is produced. Ambiguous handedness, fragmented temporal
context, original native merges, unassigned cache observations, lack of GT and uncalibrated focal length
limit interpretation. Pink/cyan encode the model-input side hypothesis, not verified physical identity.
Neither branch imputes missing poses in this camera-only test. Numerical reproducibility is checked at
pilot tolerance 1e-5, not promised bitwise across GPUs, PyTorch releases or detector/tracker versions.
"""
    (out / "report.md").write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "pilot", "full", "render", "report"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    os.chdir(ROOT)
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Experiment output must stay inside HaWoR")
    seed()
    torch.set_num_threads(4)
    prepare(out)
    if args.stage in ("pilot", "full"):
        phase = args.stage
        if phase == "full":
            assert read(out / "pilot/validation.json")["passed"]
        tracks, seconds = get_tracks(out, phase)
        run_branch(out, phase, "native", tracks, seconds)
        t = time.perf_counter()
        frontend = read(out / "cache/frontend_selection.json")
        allowed = None if phase == "full" else set(range(PILOT[0], PILOT[1] + 1))
        groups = optimized_groups(frontend["selected"], allowed)
        seconds = time.perf_counter() - t
        run_branch(out, phase, "optimized", groups, seconds)
        if phase == "pilot":
            errors = golden_pilot(out, tracks)
            projection = projection_checks(out)
            render(out, phase)
            dump(out / "pilot/validation.json", {"passed": True,
                 "native_reference_max_abs_error": errors, "projection": projection,
                 "frame_range": list(PILOT), "performed_before_full_run": True})
    elif args.stage == "render":
        render(out, "full")
        finish(out)
    elif args.stage == "report":
        finish(out)


if __name__ == "__main__":
    main()
