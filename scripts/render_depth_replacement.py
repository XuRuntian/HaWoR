"""Shared native rendering for one frozen frontend and two depth scale sources."""

from collections import defaultdict
import subprocess
import time

import cv2
import numpy as np
import torch

import run_depth_replacement as experiment

base, backend, completion = experiment.base, experiment.backend, experiment.completion
FILES = ("baseline_overlay.mp4", "stereo_overlay.mp4", "comparison.mp4",
         "world_observed_comparison.mp4", "world_completed_comparison.mp4")


def depth_samples(out, phase):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = base.read(out/f"cache/{phase}/stereo/scale_audit.json")["rows"]
    frames = np.array([row["frame_idx"] for row in rows])
    wanted = (88, 100, 196) if phase == "pilot" else (100, 599, 922)
    selected = sorted({int(frames[np.argmin(abs(frames-fi))]) for fi in wanted})
    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 3.4*len(selected)), squeeze=False)
    for row, fi in enumerate(selected):
        rgb = cv2.imread(str(backend.SOURCE/f"cache/frames/{fi:04d}.jpg"))
        axes[row, 0].imshow(cv2.resize(rgb, (592, 328))[..., ::-1])
        axes[row, 0].set_title(f"Frame {fi}: RGB on depth resize grid")
        for column, arm in enumerate(experiment.ARMS, 1):
            directory = "baseline_replay" if phase == "full" and arm == "baseline" else phase
            data = experiment.read_npz(out/f"cache/{directory}/{arm}/{fi:06d}.npz")
            depth = np.ma.masked_invalid(data["depth_m"])
            plot = axes[row, column].imshow(depth, vmin=.2, vmax=3., cmap="viridis")
            if data["exclusion_mask"].any():
                axes[row, column].contour(data["exclusion_mask"], levels=[.5], colors="red", linewidths=.5)
            axes[row, column].set_title("Metric3D Z (m)" if arm == "baseline" else "FoundationStereo Z (m)")
            fig.colorbar(plot, ax=axes[row, column], fraction=.03)
        for ax in axes[row]:
            ax.set_axis_off()
    fig.suptitle("Common 0.2..3 m display range; red: excluded from scale fit; no accuracy claim")
    fig.tight_layout()
    destination = out if phase == "full" else out/"pilot"
    fig.savefig(destination/"depth_samples.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def render(out, phase):
    from lib.eval_utils.custom_utils import load_slam_cam
    frames = backend.frame_ids(phase)
    dest = out if phase == "full" else out/"pilot"
    if any((dest/name).exists() for name in FILES):
        raise ValueError("Refusing to overwrite rendered videos")
    k = np.asarray(base.read(out/"manifest.json")["camera_intrinsics"])
    archives, tables, paths, bounds = {}, {}, {}, []
    for arm in experiment.ARMS:
        _, _, rcw, tcw = load_slam_cam(experiment.arm_root(out, arm)/f"cache/{phase}/optimized/slam_scaled.npz")
        rcw, tcw = rcw.numpy(), tcw.numpy()
        paths[arm] = (tcw-tcw[0])@rcw[0]
        bounds.append(paths[arm])
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
    writers["comparison"] = base.VideoWriter(dest/"comparison.mp4", base.WIDTH*3, base.HEIGHT)
    writers.update({mode: base.VideoWriter(dest/f"world_{mode}_comparison.mp4", 1920, 720) for mode in ("observed", "completed")})
    audit = {"frame_count": len(frames), "frames_inclusive": [int(frames[0]), int(frames[-1])], "view": view,
        "mesh_pixels": {f"{arm}_{mode}": [] for arm in experiment.ARMS for mode in ("camera", "observed", "completed")},
        "common_rendering": {"camera_intrinsics": k.tolist(), "opacity": .65, "observed_colors_rgb": base.COLORS,
            "generated_colors": "0.35*observed + 0.65", "renderer": "Native PyTorch3D renderer and closed-wrist faces"}}
    cap = cv2.VideoCapture(str(base.VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    begin = time.perf_counter()
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
                label = "Optimized frontend | DROID + Metric3D" if arm == "baseline" else "Optimized frontend | DROID + FoundationStereo"
                overlays[arm] = completion.title(image, label, fi, int(data["observed"][ids].sum()), int(data["generated"][ids].sum()))
                writers[arm].write(overlays[arm])
                for mode in worlds:
                    data = archives[arm, mode]
                    ids = tables[arm, mode][fi]
                    image, count = completion.mesh_image(background.copy(), data["aligned_vertices"][ids], data["side"][ids], data["generated"][ids],
                        world_renderer, world_camera, world_lights, world_faces)
                    completion.draw_path(image, paths[arm], vr, vt, vk, local)
                    audit["mesh_pixels"][f"{arm}_{mode}"].append(count)
                    label = f"{'Metric3D' if arm == 'baseline' else 'FoundationStereo'} | world {mode}"
                    image = completion.title(image, label, fi, int(data["observed"][ids].sum()), int(data["generated"][ids].sum()))
                    cv2.putText(image, "Same DROID | shared fixed view | SE(3) origin | NO fitted scale", (16, 701),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 40, 40), 1, cv2.LINE_AA)
                    cv2.putText(image, f"Reference grid: {view['reference_grid']['spacing_model_units']:g} m (not ground)", (16, 677),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (80, 80, 80), 1, cv2.LINE_AA)
                    worlds[mode].append(image)
            joined = np.concatenate([completion.title(rgb, "RGB", fi), overlays["baseline"], overlays["stereo"]], axis=1)
            writers["comparison"].write(joined)
            for mode, panels in worlds.items():
                writers[mode].write(np.concatenate(panels, axis=1))
            if fi in (88, 91, 100, 114, 179, 196, 550, 559, 599, 922, 974, 1258, 1545, 1780):
                screenshots = dest/"screenshots"
                screenshots.mkdir(exist_ok=True)
                cv2.imwrite(str(screenshots/f"comparison_{fi:06d}.jpg"), joined)
                for mode, panels in worlds.items():
                    cv2.imwrite(str(screenshots/f"world_{mode}_{fi:06d}.jpg"), np.concatenate(panels, axis=1))
            if local % 100 == 0:
                print(f"RENDER {phase} {fi}/{int(frames[-1])}", flush=True)
    finally:
        cap.release()
        for writer in writers.values():
            writer.close()
    audit["seconds"] = time.perf_counter()-begin
    audit["videos"] = {}
    for name in FILES:
        probe = base.probe(dest/name)
        stream = probe["streams"][0]
        assert int(stream["nb_frames"]) == len(frames) and stream["avg_frame_rate"] == "30/1"
        size = (5760, 1074) if name == "comparison.mp4" else ((1920, 720) if name.startswith("world_") else (1920, 1074))
        assert (stream["width"], stream["height"]) == size
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(dest/name), "-f", "null", "-"], check=True)
        audit["videos"][name] = {"probe": probe, "full_decode_passed": True, "sha256": base.sha(dest/name)}
    if phase == "full":
        for first, last in base.WINDOWS+[(918, 926)]:
            directory = dest/f"windows/{first:04d}-{last:04d}"
            directory.mkdir(parents=True)
            for name in FILES:
                subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(dest/name), "-vf",
                    f"trim=start_frame={first}:end_frame={last+1},setpts=PTS-STARTPTS", "-an", "-c:v", "libx264",
                    "-threads", "4", "-preset", "fast", "-crf", "20", str(directory/name)], check=True)
                assert int(base.probe(directory/name)["streams"][0]["nb_frames"]) == last-first+1
    depth_samples(out, phase)
    base.dump(dest/"render_audit.json", audit)


def read_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    cap.release()
    assert ok, (str(path), index)
    return image


def audit_video_alignment(out):
    rows = []
    for fi in (0, 88, 114, 196, 550, 559, 599, 922, 974, 1258, 1545, 1780, 1799):
        rgb, joined = read_frame(base.VIDEO, fi), read_frame(out/"comparison.mp4", fi)
        errors = {"rgb": float(abs(rgb[100:].astype(float)-joined[100:, :base.WIDTH].astype(float)).mean())}
        for column, arm in enumerate(experiment.ARMS, 1):
            image = read_frame(out/f"{arm}_overlay.mp4", fi)
            panel = joined[100:, column*base.WIDTH:(column+1)*base.WIDTH]
            errors[arm] = float(abs(image[100:].astype(float)-panel.astype(float)).mean())
        assert max(errors.values()) < 8, (fi, errors)
        rows.append({"frame": fi, "mean_absolute_rgb_errors": errors})
    return {"passed": True, "sampled_frames": rows}
