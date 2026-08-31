from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np
import torch

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.checkpoint_utils import safe_torch_load

from threeduse.enhance.transition_calibrator import (
    DeterministicTransitionCalibrator,
    SCHEMA_NAME,
    SCHEMA_VERSION,
)
from threeduse.model import ThreeDUSEModel, ThreeDUSEModelConfig
from threeduse.rendering.stage_renderers import (
    ReconstructionRender,
    StageRenderBundle,
    UnderwaterRasterContext,
)


ROOT = Path(__file__).resolve().parents[1]


def _small_model(*, stage2: bool):
    config = ThreeDUSEModelConfig(
        random_init=True,
        num_random=8,
        enable_stage2_enhancement=stage2,
    )
    return config.setup(
        scene_box=SceneBox(
            aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        ),
        num_train_data=1,
        metadata={},
        device="cpu",
        grad_scaler=None,
    )


def test_checkpoint_loading_is_strict_but_accepts_stage1_to_stage2() -> None:
    stage1_state = _small_model(stage2=False).state_dict()
    _small_model(stage2=True).load_state_dict(stage1_state, strict=True)

    missing = dict(stage1_state)
    missing.pop("gauss_params.scales")
    with pytest.raises(RuntimeError, match="gauss_params.scales"):
        _small_model(stage2=False).load_state_dict(missing, strict=True)

    unexpected = dict(stage1_state)
    unexpected["not_a_real_parameter"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="not_a_real_parameter"):
        _small_model(stage2=False).load_state_dict(unexpected, strict=True)


def test_rgba_ground_truth_is_composited() -> None:
    image = torch.tensor([[[0.8, 0.2, 0.4, 0.25]]])
    background = torch.tensor([0.0, 0.4, 1.0])
    result = ThreeDUSEModel.composite_with_background(None, image, background)
    expected = image[..., :3] * 0.25 + background * 0.75
    assert torch.allclose(result, expected)


def test_clear_diagnostics_remain_finite_for_zero_radiance() -> None:
    rgb = torch.zeros(2, 3, 3)
    scalar = torch.zeros(2, 3, 1)
    context = UnderwaterRasterContext(
        projection={},
        medium_source=rgb,
        beta_bs=rgb,
        beta_attn=rgb,
        far_depths=scalar,
        ray_depth_scales=torch.ones_like(scalar),
        width=3,
        height=2,
        tile_size=16,
    )
    reconstruction = ReconstructionRender(
        composite=rgb,
        object=rgb,
        medium=rgb,
        clear_linear=rgb,
        alpha=scalar,
        depth=scalar,
        context=context,
    )
    outputs = StageRenderBundle(reconstruction).as_output_dict(
        enhanced_prediction=False
    )
    assert torch.isfinite(outputs["rgb_clear_unclamp"]).all()


def test_released_calibrator_uses_the_current_safe_schema() -> None:
    path = ROOT / "weights" / "transition_calibrator.pth"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == SCHEMA_NAME
    assert payload["schema_version"] == SCHEMA_VERSION
    model, metadata = DeterministicTransitionCalibrator.from_checkpoint(path)
    assert isinstance(model, DeterministicTransitionCalibrator)
    assert isinstance(metadata, dict)


def test_safe_checkpoint_loader_accepts_nerfstudio_numpy_scalars(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.ckpt"
    torch.save({"step": np.float64(15_000), "value": torch.ones(1)}, path)
    payload = safe_torch_load(path)
    assert float(payload["step"]) == 15_000
    assert torch.equal(payload["value"], torch.ones(1))
