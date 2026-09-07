"""Check the HaWoR inference environment, including native CUDA kernels."""

import importlib
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import torch

    print(f"Python: {sys.version.split()[0]} ({sys.executable})", flush=True)
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
    assert torch.cuda.is_available(), "NVIDIA GPU is not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    assert shutil.which("ffmpeg"), "ffmpeg is not on PATH"

    for name in (
        "torchvision", "pytorch_lightning", "cv2", "smplx", "chumpy",
        "timm", "mmcv", "mmengine", "ultralytics", "supervision",
        "pytorch3d._C", "torch_scatter", "droid_backends", "lietorch",
        "aitviewer", "demo",
    ):
        importlib.import_module(name)
        print(f"OK import {name}", flush=True)

    from pytorch3d.ops import knn_points
    from torch_scatter import scatter_add
    from lietorch import SE3
    import droid_backends

    points = torch.rand(1, 16, 3, device="cuda")
    distances = knn_points(points, points).dists
    assert torch.allclose(distances, torch.zeros_like(distances))
    values = scatter_add(
        torch.ones(4, device="cuda"), torch.tensor([0, 0, 1, 1], device="cuda")
    )
    assert torch.equal(values, torch.tensor([2., 2.], device="cuda"))
    pose = SE3.exp(torch.zeros(1, 6, device="cuda")).matrix()
    assert torch.allclose(pose, torch.eye(4, device="cuda")[None])
    volume = torch.ones(1, 2, 2, 2, 2, device="cuda")
    coords = torch.zeros(1, 2, 2, 2, device="cuda")
    correlation = droid_backends.corr_index_forward(volume, coords, 1)[0]
    assert torch.isfinite(correlation).all() and correlation.abs().sum() > 0
    torch.cuda.synchronize()
    print("OK PyTorch3D, torch-scatter, LieTorch and DROID CUDA kernels", flush=True)

    from hawor.utils.process import run_mano, run_mano_left

    for run in (run_mano, run_mano_left):
        result = run(
            torch.zeros(1, 1, 3), torch.zeros(1, 1, 3),
            torch.zeros(1, 1, 45), betas=torch.zeros(1, 1, 10),
        )
        assert result["vertices"].shape == (1, 1, 778, 3)
        assert torch.isfinite(result["vertices"]).all()
    print("OK left and right MANO GPU forward passes", flush=True)

    from aitviewer.headless import HeadlessRenderer

    renderer = HeadlessRenderer(size=(128, 128))
    print(f"OK headless OpenGL: {renderer.ctx.info['GL_RENDERER']}", flush=True)
    renderer.on_close()

    for filename in (
        "weights/external/detector.pt", "weights/external/droid.pth",
        "weights/hawor/checkpoints/hawor.ckpt",
        "weights/hawor/checkpoints/infiller.pt", "weights/hawor/model_config.yaml",
        "thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth",
    ):
        path = ROOT / filename
        assert path.is_file() and path.stat().st_size > 1024, f"Missing: {path}"
    print("OK model assets exist (checkpoint loading is tested by the demo)")
    print("Environment checks passed.")


if __name__ == "__main__":
    main()
