# Local HaWoR Environment

## Use

```bash
conda activate hawor
cd /home/user/HaWoR
python scripts/check_environment.py
python demo.py --video_path example/video_0.mp4 --vis_mode world
```

Reactivate `hawor` if it was already active during installation. Its Conda
environment variables select the environment-local CUDA compiler, GCC 11,
and EGL rendering. `PYTHONPATH` and `LD_LIBRARY_PATH` are cleared inside this
environment to prevent ROS, system Qt and another environment's TensorRT/CUDA
libraries from overriding its packages. The inherited system Qt path caused an
`xcb` plugin crash before this isolation was applied.
No system CUDA installation or other Conda environment was changed.

## Versions

- Environment: `/home/user/miniconda3/envs/hawor`
- Python 3.10.13, PyTorch 1.13.0+cu117, torchvision 0.14.0+cu117
- CUDA compiler/runtime 11.7, GCC/G++ 11
- NumPy 1.26.4, PyTorch3D 0.7.2, torch-scatter 2.1.2
- PyTorch Lightning 2.2.4, torchmetrics 1.4.0
- DROID-SLAM and LieTorch compiled from the checked-out repository
- GPU: NVIDIA GeForce RTX 4070 (12 GB)

Compatibility pins are in `constraints-cu117.txt`; `requirements-cu117.lock`
records the installed Python packages. The DROID/LieTorch entries in that
snapshot must be built locally, not downloaded from PyPI. In particular, the original
unpinned `pytorch3d@stable` should not be used with this older PyTorch release.
[PyTorch3D 0.7.2 documents support for PyTorch 1.13.0](https://github.com/facebookresearch/pytorch3d/blob/v0.7.2/INSTALL.md).
Matplotlib is pinned because the viewer uses `matplotlib.cm.get_cmap`.

## Reinstall Dependencies

With a Python 3.10 Conda environment named `hawor` activated, and GCC/G++ 11 and
ffmpeg installed on the host:

```bash
conda install -c nvidia/label/cuda-11.7.1 cuda-nvcc cuda-cudart \
  cuda-cudart-dev cuda-cccl cuda-cupti libcublas-dev libcusparse-dev \
  libcusolver-dev libcurand-dev
export CUDA_HOME="$CONDA_PREFIX"
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST='8.6+PTX' MAX_JOBS=4
export PYTHONPATH= LD_LIBRARY_PATH= PYOPENGL_PLATFORM=egl
python -m pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install numpy==1.26.4 setuptools==69.5.1 wheel==0.45.1 ninja Cython
python -m pip install --no-build-isolation -c constraints-cu117.txt \
  -r <(sed '/pytorch3d/d; /torch-scatter/d; /chumpy/d' requirements.txt) \
  'fsspec[http]' fvcore iopath gdown
python -m pip install --no-build-isolation -c constraints-cu117.txt \
  'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2' \
  torch-scatter==2.1.2 \
  'chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da'
python -m pip install pytorch-lightning==2.2.4 --no-deps
python -m pip install lightning-utilities torchmetrics==1.4.0
(cd thirdparty/DROID-SLAM && python setup.py install)
```

These commands require Bash. They intentionally keep the upstream requirements
file unchanged and install the native extensions without build isolation so
they link against the installed PyTorch.

## Assets

Public checkpoints are downloaded to the paths documented in `README.md`:

- `weights/external/{detector.pt,droid.pth}`
- `weights/hawor/checkpoints/{hawor.ckpt,infiller.pt}`
- `weights/hawor/model_config.yaml`
- `thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth`

MANO files are symlinked from the existing local licensed models:

- `_DATA/data/mano/MANO_RIGHT.pkl` -> `/home/user/models/mano/MANO_RIGHT.pkl`
- `_DATA/data_left/mano_left/MANO_LEFT.pkl` -> `/home/user/models/mano/MANO_LEFT.pkl`

These model files are not committed. The symlinks require the local source
files to remain available; a different machine needs its own licensed MANO copy.
The environment check verifies CUDA kernels, MANO forward passes, EGL creation,
imports and asset presence; full checkpoint loading and video processing require
running `demo.py`.

## Verification (2026-09-06)

- `python -m pip check`: no broken requirements.
- `python scripts/check_environment.py`: all checks passed.
- Qt desktop viewer creation: passed on the RTX 4070 after library isolation.
- All five model checkpoints loaded successfully; detector and Metric3D GPU
  forward passes passed independently.
- Processed all 121 frames of `example/video_0.mp4` resized to 960x540, using
  focal length 300: tracking, HaWoR, masked DROID-SLAM, metric depth, infilling,
  and world-view rendering completed. The smoke test selected the renderer's
  existing `interactive=False` option at runtime; `demo.py` was not modified.
- Test output: `/tmp/hawor-smoke/input/vis_0_120/aitviewer/video_0.mp4`
  (1600x900, 60 fps, 240 frames). Temporary output may be cleared on reboot.

The standard demo opens an interactive viewer. Close its window to end the
process. Full-resolution or longer inputs were not part of the smoke test.
