"""Common native renderer for frozen-baseline and VIO replacement archives."""

from collections import defaultdict
import subprocess
import time

import cv2
import numpy as np
import torch

import run_vio_replacement as experiment

base, backend, completion = experiment.base, experiment.backend, experiment.completion
FILES = ("baseline_overlay.mp4", "vio_overlay.mp4", "comparison.mp4",
         "world_observed_comparison.mp4", "world_completed_comparison.mp4")
EXTRA_WINDOWS = [(544, 564), (918, 926)]


@torch.no_grad()
def render(out, phase):
    from lib.eval_utils.custom_utils import load_slam_cam
    frames = backend.frame_ids(phase)
    dest = out if phase == "full" else out/"pilot"
    if any((dest/name).exists() for name in FILES):
        raise ValueError("Refusing to overwrite rendered videos")
    k = np.asarray(base.read(out/"manifest.json")["camera_intrinsics"])
    archives, tables, paths, bounds, flags = {}, {}, {}, [], {}
    for arm in experiment.ARMS:
        root = experiment.arm_root(out, arm)
        _, _, rcw, tcw = load_slam_cam(root/f"cache/{phase}/optimized/slam_scaled.npz")
        rcw, tcw = rcw.numpy(), tcw.numpy()
        paths[arm] = (tcw-tcw[0])@rcw[0]
        bounds.append(paths[arm])
        flags[arm] = experiment.read_npz(root/"camera_poses.npz")["camera_pose_interpolated"]
        for mode in ("observed", "completed"):
            data = experiment.read_npz(experiment.prediction_dir(out, phase, arm)/f"world_{mode}.npz")
            data["aligned_vertices"] = completion.camera_relative_world(data["vertices"], rcw, tcw)
            archives[arm, mode] = data
            table = defaultdict(list)
            for i, fi in enumerate(data["frame_idx"]):
                table[int(fi)].append(i)
            tables[arm, mode] = table
            if mode == "observed":
                bounds.append((data["joints_world"][:, 0]-tcw[0])@rcw[0])
    vr, vt, vk, view = completion.virtual_camera(np.concatenate(bounds))
    background = completion.reference_grid(view, vr, vt, vk)
    world_renderer, _, world_lights, world_faces = backend.render_setup(vk, 960, 720)
    world_camera, _ = world_renderer.create_camera_from_cv(torch.as_tensor(vr, dtype=torch.float32, device="cuda")[None],
        torch.as_tensor(vt, dtype=torch.float32, device="cuda")[None], K=torch.as_tensor(vk, dtype=torch.float32, device="cuda")[None])
    renderer, camera, lights, faces = backend.render_setup(k)
    writers = {arm: base.VideoWriter(dest/f"{arm}_overlay.mp4", base.WIDTH, base.HEIGHT) for arm in experiment.ARMS}
    writers.update(comparison=base.VideoWriter(dest/"comparison.mp4", base.WIDTH*3, base.HEIGHT))
    writers.update({mode: base.VideoWriter(dest/f"world_{mode}_comparison.mp4", 1920, 720) for mode in ("observed", "completed")})
    audit = {"frame_count": len(frames), "frames_inclusive": [int(frames[0]), int(frames[-1])],
        "view": view, "mesh_pixels": {f"{arm}_{mode}": [] for arm in experiment.ARMS for mode in ("camera", "observed", "completed")},
        "common_rendering": {"camera_intrinsics": k.tolist(), "opacity": .65, "observed_colors_rgb": base.COLORS,
                             "generated_colors": "0.35*observed + 0.65", "renderer": "Native PyTorch3D Renderer; native closed wrist faces"}}
    cap = cv2.VideoCapture(str(base.VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    start = time.perf_counter()
    try:
        for local, fi in enumerate(frames):
            fi = int(fi)
            ok, rgb = cap.read()
            assert ok and rgb.shape[:2] == (base.HEIGHT, base.WIDTH), fi
            overlays, worlds = {}, {"observed": [], "completed": []}
            for arm in experiment.ARMS:
                data = archives[arm, "completed"]
                ids = tables[arm, "completed"][fi]
                image, count = completion.mesh_image(rgb.copy(), data["vertices_camera"][ids], data["side"][ids], data["generated"][ids],
                    renderer, camera, lights, faces)
                audit["mesh_pixels"][f"{arm}_camera"].append(count)
                label = "Optimized frontend | DROID + Metric3D" if arm == "baseline" else "Optimized frontend | OpenVINS (scale=1)"
                overlays[arm] = completion.title(image, label, fi, int(data["observed"][ids].sum()), int(data["generated"][ids].sum()))
                if flags[arm][fi]:
                    cv2.putText(overlays[arm], "CAMERA POSE INTERPOLATED", (18, 108), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 190, 255), 2, cv2.LINE_AA)
                writers[arm].write(overlays[arm])
                for mode in worlds:
                    data = archives[arm, mode]
                    ids = tables[arm, mode][fi]
                    image, count = completion.mesh_image(background.copy(), data["aligned_vertices"][ids], data["side"][ids], data["generated"][ids],
                        world_renderer, world_camera, world_lights, world_faces)
                    completion.draw_path(image, paths[arm], vr, vt, vk, local)
                    audit["mesh_pixels"][f"{arm}_{mode}"].append(count)
                    label = f"{'Baseline DROID + Metric3D' if arm == 'baseline' else 'OpenVINS'} | {mode}"
                    image = completion.title(image, label, fi, int(data["observed"][ids].sum()), int(data["generated"][ids].sum()))
                    if flags[arm][fi]:
                        cv2.putText(image, "CAMERA POSE INTERPOLATED", (18, 108), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 125, 185), 2, cv2.LINE_AA)
                    cv2.putText(image, "Shared fixed view | first-camera SE(3) origin | NO fitted scale", (16, 701),
                                cv2.FONT_HERSHEY_SIMPLEX, .53, (40, 40, 40), 1, cv2.LINE_AA)
                    cv2.putText(image, f"Reference grid: {view['reference_grid']['spacing_model_units']:g} m (not ground)", (16, 677),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (80, 80, 80), 1, cv2.LINE_AA)
                    worlds[mode].append(image)
            comparison = np.concatenate([completion.title(rgb, "RGB", fi), overlays["baseline"], overlays["vio"]], axis=1)
            writers["comparison"].write(comparison)
            for mode, panels in worlds.items():
                writers[mode].write(np.concatenate(panels, axis=1))
            if fi in (88, 91, 100, 114, 179, 196, 549, 550, 551, 558, 559, 560, 599, 922, 974, 1258, 1545, 1780):
                screenshots = dest/"screenshots"
                screenshots.mkdir(exist_ok=True)
                cv2.imwrite(str(screenshots/f"comparison_{fi:06d}.jpg"), comparison)
                cv2.imwrite(str(screenshots/f"world_observed_{fi:06d}.jpg"), np.concatenate(worlds["observed"], axis=1))
                cv2.imwrite(str(screenshots/f"world_completed_{fi:06d}.jpg"), np.concatenate(worlds["completed"], axis=1))
            if local % 100 == 0:
                print(f"RENDER {phase} {fi}/{int(frames[-1])}", flush=True)
    finally:
        cap.release()
        for writer in writers.values():
            writer.close()
    audit["seconds"] = time.perf_counter()-start
    audit["videos"] = {}
    for name in FILES:
        probe = base.probe(dest/name)
        stream = probe["streams"][0]
        assert int(stream["nb_frames"]) == len(frames) and stream["avg_frame_rate"] == "30/1"
        expected = (5760, 1074) if name == "comparison.mp4" else ((1920, 720) if name.startswith("world_") else (1920, 1074))
        assert (stream["width"], stream["height"]) == expected
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(dest/name), "-f", "null", "-"], check=True)
        audit["videos"][name] = {"probe": probe, "full_decode_passed": True, "sha256": base.sha(dest/name)}
    if phase == "full":
        for first, last in base.WINDOWS+EXTRA_WINDOWS:
            window = dest/f"windows/{first:04d}-{last:04d}"
            window.mkdir(parents=True)
            for name in FILES:
                subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(dest/name), "-vf",
                    f"trim=start_frame={first}:end_frame={last+1},setpts=PTS-STARTPTS", "-an", "-c:v", "libx264",
                    "-threads", "4", "-preset", "fast", "-crf", "20", str(window/name)], check=True)
                assert int(base.probe(window/name)["streams"][0]["nb_frames"]) == last-first+1
    base.dump(dest/"render_audit.json", audit)


def read_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    cap.release()
    assert ok, (str(path), index)
    return image


def audit_video_alignment(out):
    records = []
    for fi in (0, 88, 114, 196, 550, 559, 599, 922, 974, 1258, 1545, 1780, 1799):
        rgb = read_frame(base.VIDEO, fi)
        joined = read_frame(out/"comparison.mp4", fi)
        error = {"rgb": float(abs(rgb[120:].astype(float)-joined[120:, :base.WIDTH].astype(float)).mean())}
        for column, arm in enumerate(experiment.ARMS, 1):
            overlay = read_frame(out/f"{arm}_overlay.mp4", fi)
            panel = joined[120:, column*base.WIDTH:(column+1)*base.WIDTH]
            error[arm] = float(abs(overlay[120:].astype(float)-panel.astype(float)).mean())
        assert max(error.values()) < 8, (fi, error)
        records.append({"frame": fi, "mean_absolute_rgb_errors": error})
    return {"passed": True, "sampled_frames": records}
