from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    scripts = str(path.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    specification = importlib.util.spec_from_file_location(path.stem, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_only_final_nerfstudio_methods_are_published() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    methods = project["project"]["entry-points"]["nerfstudio.method_configs"]
    assert methods == {
        "3duse-stage1": "threeduse.config:stage1",
        "3duse-stage2": "threeduse.config:stage2",
    }


def test_pseudodepth_normalization_is_uint16_and_robust() -> None:
    import numpy as np

    module = _load_script("generate_da2_pseudodepth.py")
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    values[0, 0] = 10_000
    encoded, low, high = module.robust_u16(values)
    assert encoded.dtype == np.uint16
    assert encoded.shape == values.shape
    assert low < high < 10_000
    assert int(encoded.min()) == 0
    assert int(encoded.max()) == 65535


def test_documented_files_exist() -> None:
    for relative in (
        "docs/INSTALLATION.md",
        "docs/DATA.md",
        "docs/TRAINING.md",
        "UPSTREAM_LOCKS.md",
    ):
        assert (ROOT / relative).is_file(), relative


def test_release_runtime_has_no_submodule_or_patcher() -> None:
    assert not (ROOT / ".gitmodules").exists()
    assert not (ROOT / "scripts" / "patch_nerfstudio.py").exists()
    for relative in (
        "third_party/nerfstudio/nerfstudio/__init__.py",
        "third_party/nerfstudio/nerfstudio/data/datamanagers/base_datamanager.py",
        "third_party/nerfstudio/nerfstudio/data/dataparsers/base_dataparser.py",
        "third_party/nerfstudio/nerfstudio/data/datasets/base_dataset.py",
        "third_party/nerfstudio/nerfstudio/data/scene_box.py",
        "third_party/nerfstudio/pyproject.toml",
        "threeduse/cuda/include/glm/glm.hpp",
    ):
        assert (ROOT / relative).is_file(), relative
    assert not list((ROOT / "third_party").rglob(".git"))
    assert not list((ROOT / "third_party").rglob(".gitmodules"))
    assert not (ROOT / "third_party" / "gsplat-v1.5.3").exists()


def test_dataset_ignore_rules_do_not_hide_vendored_sources() -> None:
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/" not in rules
    assert "datasets/" not in rules
    assert "/data/" in rules
    assert "/datasets/" in rules


def test_install_validation_covers_every_runtime_extension() -> None:
    validation = (ROOT / "scripts" / "validate_install.py").read_text(
        encoding="utf-8"
    )
    for extension in ("gsplat_cuda", "threeduse_cuda"):
        assert extension in validation
    for removed_extension in ("water_cuda", "threeduse_underwater_cuda"):
        assert removed_extension not in validation


def test_retired_release_wrappers_are_absent() -> None:
    for name in (
        "_release_runtime.py",
        "benchmark_render.py",
        "prepare_cuda.py",
        "reconstruction_metrics.py",
        "render.py",
        "render_video.py",
    ):
        assert not (ROOT / "scripts" / name).exists(), name
