# Adapted from gsplat v1.5.3 under Apache-2.0 and modified for 3D-USE.
"""JIT loader for the project CUDA extension."""

import json
import os
import time
from subprocess import DEVNULL, call

import torch
from packaging import version
from rich.console import Console
from torch.utils.cpp_extension import _find_cuda_home
from torch.utils.cpp_extension import (
    _TORCH_PATH,
    _get_build_directory,
    _import_module_from_library,
    _jit_compile,
)

PATH = os.path.dirname(os.path.abspath(__file__))
NO_FAST_MATH = os.getenv("NO_FAST_MATH", "0") == "1"
MAX_JOBS = os.getenv("MAX_JOBS")
USE_PRECOMPILED_HEADERS = os.getenv("USE_PRECOMPILED_HEADERS", "0") == "1"
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"

# torch has bugs on precompiled headers before 2.2, see:
# https://github.com/nerfstudio-project/gsplat/pull/583#issuecomment-2732597080
if version.parse(torch.__version__) < version.parse("2.2") and USE_PRECOMPILED_HEADERS:
    Console().print(
        "[yellow]3D-USE: Precompiled headers require torch 2.2 or newer; disabling them.[/yellow]"
    )
    USE_PRECOMPILED_HEADERS = False


def load_extension(
    name,
    sources,
    extra_cflags=None,
    extra_cuda_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
    verbose=False,
):
    """Load a JIT compiled extension."""
    # Make sure the build directory exists.
    if build_directory:
        os.makedirs(build_directory, exist_ok=True)

    # If the JIT build happens concurrently in multiple processes,
    # race conditions can occur when removing the lock file at:
    # https://github.com/pytorch/pytorch/blob/e3513fb2af7951ddf725d8c5b6f6d962a053c9da/torch/utils/cpp_extension.py#L1736
    # But it's ok so we catch this exception and ignore it.
    try:
        if USE_PRECOMPILED_HEADERS:
            from torch.utils.cpp_extension import (
                _check_and_build_extension_h_precompiler_headers,
            )

            # Using PreCompiled Header('torch/extension.h') to reduce compile time.
            _check_and_build_extension_h_precompiler_headers(
                extra_cflags, extra_include_paths
            )
            head_file = os.path.join(_TORCH_PATH, "include", "torch", "extension.h")
            extra_cflags += ["-include", head_file, "-Winvalid-pch"]

        try:
            compiled = _jit_compile(
                name,
                sources,
                extra_cflags,
                extra_cuda_cflags,
                extra_ldflags,
                extra_include_paths,
                build_directory,
                verbose,
                with_cuda=None,
                is_python_module=True,
                is_standalone=False,
                keep_intermediates=True,
            )
        except (
            TypeError
        ) as e:  # torch>=2.7.0 has added arguments to _jit_compile to support SYCL.
            # Narrow the scope of catch: only retry if it's due to unexpected argument(s)
            if "_jit_compile() missing" in str(e):
                compiled = _jit_compile(
                    name,
                    sources,
                    extra_cflags,
                    extra_cuda_cflags,
                    None,  # SYCL fallback
                    extra_ldflags,
                    extra_include_paths,
                    build_directory,
                    verbose,
                    with_cuda=None,
                    with_sycl=None,
                    is_python_module=True,
                    is_standalone=False,
                    keep_intermediates=True,
                )
            else:
                raise e

        return compiled
    except OSError:
        # The module should already be compiled if we get OSError
        return _import_module_from_library(name, build_directory, True)


def cuda_toolkit_available():
    """
    Check more robustly if the CUDA toolkit is available.
    1. Attempt to locate `CUDA_HOME` using PyTorch’s internal method.
    2. Check if nvcc is present in that location.
    """
    cuda_home = _find_cuda_home()  # This tries various heuristics
    if not cuda_home:
        return False

    # If we have a cuda_home, check if nvcc exists there:
    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc_path):
        # Maybe still on PATH, try calling "nvcc" directly:
        try:
            call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
            return True
        except FileNotFoundError:
            return False
    return True


def cuda_toolkit_version():
    """Get the CUDA toolkit version if we found CUDA home."""
    cuda_home = _find_cuda_home()
    if not cuda_home:
        return None

    if os.path.exists(os.path.join(cuda_home, "version.txt")):
        with open(os.path.join(cuda_home, "version.txt")) as f:
            cuda_version = f.read().strip().split()[-1]
    elif os.path.exists(os.path.join(cuda_home, "version.json")):
        with open(os.path.join(cuda_home, "version.json")) as f:
            cuda_version = json.load(f)["cuda"]["version"]
    else:
        raise RuntimeError("Cannot find the CUDA version file in CUDA_HOME.")
    return cuda_version


_C = None

try:
    # Try to import the compiled module (via setup.py or pre-built .so)
    from threeduse import cuda_csrc as _C
except ImportError:
    # If that fails, build the local CUDA extension once for this environment.
    if cuda_toolkit_available():
        name = "threeduse_cuda"
        build_dir = _get_build_directory(name, verbose=False)

        # GLM is header-only and is vendored with this project-owned CUDA
        # extension. The standard gsplat package remains an ordinary runtime
        # dependency and does not need to be bundled in the repository.
        extra_include_paths = [os.path.join(PATH, "include/")]
        extra_cflags = ["-O3", "-Wno-attributes"]
        extra_cuda_cflags = ["-O3"]
        if not NO_FAST_MATH:
            extra_cuda_cflags += ["-use_fast_math"]
        sources = [
            os.path.join(PATH, "csrc", "RasterizeToPixels3DGSUnderwaterFwd.cu"),
            os.path.join(PATH, "csrc", "RasterizeToPixels3DGSUnderwaterBwd.cu"),
            os.path.join(PATH, "csrc", "UnderwaterRasterization.cpp"),
            os.path.join(PATH, "csrc", "SphericalHarmonics.cu"),
            os.path.join(PATH, "ext.cpp"),
        ]

        if os.path.exists(os.path.join(build_dir, f"{name}.so")) or os.path.exists(
            os.path.join(build_dir, f"{name}.lib")
        ):
            # If the build exists, we assume the extension has been built
            # and we can load it.
            _C = load_extension(
                name=name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_dir,
                verbose=False,
            )
        else:
            # Keep PyTorch's inter-process build lock intact. Deleting another
            # process's live lock can let concurrent imports corrupt the build.
            os.makedirs(build_dir, exist_ok=True)
            tic = time.time()
            with Console().status(
                f"[bold yellow]3D-USE: compiling the unified CUDA extension with MAX_JOBS={os.environ['MAX_JOBS']}",
                spinner="bouncingBall",
            ):
                _C = load_extension(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_dir,
                    verbose=False,
                )
            toc = time.time()
            Console().print(
                f"[green]3D-USE CUDA extension compiled in {toc - tic:.2f} seconds.[/green]"
            )

    else:
        Console().print(
            "[yellow]3D-USE: no CUDA toolkit found; CUDA rendering is unavailable.[/yellow]"
        )

if need_to_unset_max_jobs:
    os.environ.pop("MAX_JOBS")


__all__ = ["_C"]
