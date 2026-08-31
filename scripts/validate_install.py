"""Fast dependency and method-registration gate for the clean release."""

from __future__ import annotations

import importlib.metadata
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("PYTORCH_JIT", "0")
import torch


def main() -> None:
    assert sys.version_info[:2] == (3, 11), sys.version
    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert torch.version.cuda == "12.4", torch.version.cuda
    import gsplat
    import nerfstudio
    from gsplat.cuda._backend import _C as gsplat_cuda
    from threeduse.config import stage1, stage2
    from threeduse.cuda._backend import _C as threeduse_cuda

    methods = {
        entry.name
        for entry in importlib.metadata.entry_points(group="nerfstudio.method_configs")
        if entry.dist is not None and entry.dist.name == "3d-use"
    }
    assert methods == {"3duse-stage1", "3duse-stage2"}, methods
    assert importlib.metadata.version("gsplat") == "1.5.3"
    assert stage1.config.method_name == "3duse-stage1"
    assert stage2.config.method_name == "3duse-stage2"
    assert gsplat_cuda is not None, "gsplat CUDA extension is unavailable"
    assert threeduse_cuda is not None, "3D-USE CUDA extension is unavailable"
    for symbol in (
        "compute_sh_forward",
        "compute_sh_backward",
        "sparse_compute_sh_forward",
        "sparse_compute_sh_backward",
        "rasterize_to_pixels_3dgs_underwater_fwd",
        "rasterize_to_pixels_3dgs_underwater_bwd",
    ):
        assert hasattr(
            threeduse_cuda, symbol
        ), f"3D-USE CUDA symbol is unavailable: {symbol}"
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("gsplat", importlib.metadata.version("gsplat"), gsplat.__file__)
    print("gsplat CUDA", gsplat_cuda.__file__)
    print("nerfstudio", nerfstudio.__file__)
    print("3D-USE CUDA", threeduse_cuda.__file__)
    print("3D-USE methods", sorted(methods))


if __name__ == "__main__":
    main()
