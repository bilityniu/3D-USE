# Installation

## Tested environment

The release is validated with:

- Linux and an NVIDIA CUDA-capable GPU;
- Python 3.11.11;
- PyTorch 2.5.1 and torchvision 0.20.1;
- CUDA 12.4;
- GCC/G++ 10 for CUDA extension compilation;
- Nerfstudio and gsplat at the revisions in `UPSTREAM_LOCKS.md`.

Clone the repository:

```bash
git clone https://github.com/bilityniu/3D-USE.git
cd 3D-USE
```

Create the Conda environment from the repository root:

```bash
conda env create -f environment/environment.yml
conda activate 3duse
bash environment/install.sh
```

The repository bundles the modified Nerfstudio source used for the experiments and installs the unmodified `gsplat==1.5.3` package. The installer then installs 3D-USE and runs `scripts/validate_install.py`, which imports and validates the upstream `gsplat_cuda` module and the project-owned `threeduse_cuda` module for spherical-harmonics decoding and underwater compositing. Their first import triggers CUDA compilation, which is CPU-heavy and may show little GPU utilization for several minutes.

## Existing environment

If PyTorch 2.5.1 with CUDA 12.4 is already installed, activate that Python 3.11
environment and run:

```bash
bash environment/install.sh
```

Set `NERFSTUDIO_ROOT` before running the installer only when testing an
equivalent external checkout.

## Additional tools

- Depth Anything V2 is needed only to generate Stage-1 pseudo-depth.

## Validate an updated checkout

Reinstall the editable package whenever entry points or dependencies change:

```bash
python3 -m pip install -e . --no-build-isolation
python3 scripts/validate_install.py
ns-train --help | grep 3duse
```

The installer uses GCC/G++ 10 automatically when both executables are
available. Otherwise, set compatible host compilers explicitly before
building, for example `CC=gcc-10 CXX=g++-10 CUDAHOSTCXX=g++-10`.
