# Upstream compatibility locks

The released implementation was validated with the following upstream
revisions:

| Component | Revision | Role |
|---|---|---|
| Nerfstudio | `50e0e3c70c775e89333256213363badbf074f29d` | CLI, trainer, and data pipeline |
| gsplat | `937e29912570c372bed6747a5c9bf85fed877bae` (`v1.5.3`) | Gaussian projection, sorting, and rasterization support |
| PyTorch | `2.5.1+cu124` | Validated CUDA runtime |
| GCC/G++ | `10.5.0` | Validated CUDA host compiler |

The corresponding modified Nerfstudio source is included directly under
`third_party/`, following Plenodium's release layout. gsplat is unmodified and
installed at its fixed package version. No recursive clone or post-install
source patch is required:

```bash
python -m pip install -e third_party/nerfstudio --no-deps
BUILD_NO_CUDA=1 python -m pip install gsplat==1.5.3 --no-build-isolation
python -m pip install -e . --no-build-isolation
python scripts/validate_install.py
```

The bundled Nerfstudio contains only the compatibility changes required by the
released pipeline: Stage 2 loads the frozen Stage-1 pipeline without restoring
its optimizer state, full-image progress reporting handles scalar cameras, and
the training recipes can disable graph compilation. Checkpoints are loaded in
restricted tensor-only mode, and model-state loading remains strict by default.

Other CUDA/PyTorch combinations may work but are not part of the tested
release matrix.
