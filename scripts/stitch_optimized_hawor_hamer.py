"""Stitch existing camera-relative overlays without model inference or re-rendering."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/pre_frontend_comparison/full_60s"
HAMER = Path("/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/three_way_baseline_comparison/full_60s/comparison/raw_vs_original_vs_optimized_rgb.mp4")
HAWOR = BASE / "optimized_overlay.mp4"
RGB = Path("/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/decoupled_diagnostics/full_60s/rectified/videos/left_rectified.mp4")
WINDOWS = [(88, 94), (100, 114), (176, 182), (193, 201), (596, 602)]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate:format=duration", "-of", "json", str(path)], text=True))


def frame(path, index):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
        assert ok, (path, index)
        return image
    finally:
        cap.release()


def main(out):
    assert out.is_relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "optimized_hawor_vs_optimized_hamer.mp4"
    if target.exists():
        raise ValueError("Use a fresh output directory; existing results are not overwritten")
    started = time.perf_counter()
    sources = {str(p): {"sha256": sha(p), "probe": probe(p)} for p in (HAWOR, HAMER, RGB)}
    for info in sources.values():
        stream = info["probe"]["streams"][0]
        assert stream["r_frame_rate"] == "30/1" and int(stream["nb_frames"]) == 1800
        assert float(info["probe"]["format"]["duration"]) == 60
    assert sources[str(HAMER)]["probe"]["streams"][0]["width"] == 1920
    assert sources[str(HAMER)]["probe"]["streams"][0]["height"] == 360
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    def label(title):
        return ("drawbox=x=0:y=0:w=iw:h=64:color=0x191919:t=fill,"
                f"drawtext=fontfile={font}:text='{title}':x=12:y=8:fontsize=20:fontcolor=white,"
                f"drawtext=fontfile={font}:text='frame %{{n}} | 30 FPS':x=12:y=37:fontsize=15:fontcolor=white")

    graph = (f"[0:v]setpts=PTS-STARTPTS,scale=640:360:flags=area,setsar=1,{label('Optimized HaWoR')}[a];"
             f"[1:v]setpts=PTS-STARTPTS,crop=640:360:1280:0,setsar=1,{label('Optimized HaMeR')}[b];"
             "[a][b]hstack=inputs=2[v]")
    command = ["ffmpeg", "-v", "error", "-n", "-i", str(HAWOR), "-i", str(HAMER),
               "-filter_complex_threads", "2", "-filter_complex", graph, "-map", "[v]", "-an",
               "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    subprocess.run(command, check=True)
    result_info = probe(target)
    stream = result_info["streams"][0]
    assert (stream["width"], stream["height"], int(stream["nb_frames"]), stream["r_frame_rate"]) == (1280, 360, 1800, "30/1")
    checks = {}
    for fi in (0, 88, 100, 196, 599, 1799):
        actual = frame(target, fi)
        hawor = cv2.resize(frame(HAWOR, fi), (640, 360), interpolation=cv2.INTER_AREA)
        hamer = frame(HAMER, fi)[:, 1280:1920]
        rgb = cv2.resize(frame(RGB, fi), (640, 360), interpolation=cv2.INTER_AREA)
        checks[str(fi)] = {}
        for name, panel, expected in (("hawor", actual[:, :640], hawor), ("hamer", actual[:, 640:], hamer)):
            error = np.abs(panel[64:].astype(float) - expected[64:].astype(float))
            assert error.mean() < 6, (fi, name, error.mean())
            background_error = np.abs(expected[64:].astype(float) - rgb[64:].astype(float))
            assert np.median(background_error) < 10, (fi, name, "source background misalignment")
            checks[str(fi)][name] = {"stitch_codec_MAE": float(error.mean()),
                                    "same_frame_RGB_background_median_error": float(np.median(background_error))}
        (out / "screenshots").mkdir(exist_ok=True)
        cv2.imwrite(str(out / f"screenshots/frame_{fi:06d}.jpg"), actual)
    windows = []
    for first, last in WINDOWS:
        folder = out / "windows"
        folder.mkdir(exist_ok=True)
        path = folder / f"{first:04d}-{last:04d}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(target), "-vf",
                        f"trim=start_frame={first}:end_frame={last+1},setpts=PTS-STARTPTS",
                        "-an", "-c:v", "libx264", "-threads", "4", "-crf", "18", str(path)], check=True)
        assert int(probe(path)["streams"][0]["nb_frames"]) == last - first + 1
        windows.append(str(path))
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(target), "-f", "null", "-"], check=True)
    assert all(sha(Path(p)) == info["sha256"] for p, info in sources.items())
    manifest = {"sources": sources, "output": str(target), "output_probe": result_info,
                "output_sha256": sha(target), "ffmpeg_command": command,
                "frame_alignment_checks": checks, "windows": windows,
                "source_files_unchanged": True, "full_decode_passed": True,
                "inference_or_mesh_rerender_executed": False,
                "script_sha256": sha(Path(__file__)), "seconds": time.perf_counter() - started}
    (out / "diagnostics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "stitch_optimized_hawor_hamer.py").write_bytes(Path(__file__).read_bytes())
    (out / "report.md").write_text(f"""# Optimized HaWoR vs Optimized HaMeR

Direct stitching only; no inference, mesh re-render, filtering, interpolation or pose modification.
Left: optimized HaWoR from `{HAWOR}`.
Right: third column (x=1280..1919) of `{HAMER}`.
The HaMeR RGB-view generator uses selected original camera-relative meshes and cameras, not its
separate metric/depth/smoothing final-output view. Source: migration guide section 7 and
`scripts/diagnostics/run_three_way_baseline_comparison.py::_make_rgb_videos` in roboego-hand-vis.
Both sources cover the same rectified input, frames 0..1799, 30 FPS, 60 seconds.

Output: `{target}`, 1280x360; each panel is 640x360, matching the existing HaMeR panel.
HaWoR is resized from 1920x1074; the old HaMeR view already resized this aspect ratio to 640x360.
This small aspect-ratio distortion is matched, not corrected or mistaken for geometry quality.
The top 64 pixels of both panels replace existing labels with model names and absolute frame index.
No image region is cropped from HaWoR; HaMeR crop selects only its existing optimized column.

Important limits: existing mesh colors, opacity, lighting, focal conventions and renderer differ.
HaWoR uses its native temporal network, while HaMeR uses its cached reconstruction pipeline.
This is a convenient visual comparison of existing outputs, not a controlled common-renderer
benchmark, metric-depth comparison, or proof that either model has better pose accuracy.
No Stereo/VIO/Metric3D or final depth-aligned HaMeR video is introduced here.

All 1800 frames and the five inclusive windows were exported; sampled frames were checked against
both source videos and the same-frame RGB background. Full video decoding passed. See diagnostics.json
for source/output hashes, exact FFmpeg invocation and numeric checks. All source files are unchanged.
Runtime: {manifest['seconds']:.3f} seconds.

```bash
mpv --speed=0.5 '{target}'
/home/user/miniconda3/bin/conda run -n hawor python scripts/stitch_optimized_hawor_hamer.py --output /home/user/HaWoR/outputs/pre_frontend_comparison/hawor_hamer_reproduction
```
""")
    print(json.dumps({"output": str(target), "passed": True, "seconds": manifest["seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BASE / "optimized_hawor_vs_hamer")
    main(parser.parse_args().output.resolve())
