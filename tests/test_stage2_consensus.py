from __future__ import annotations

import torch

from threeduse.enhance.consensus_appearance import ConsensusStage2Appearance
from threeduse.enhance.effective_transition import (
    GLOBAL_CONDITION_DIM,
    GLOBAL_OPERATOR_DIM,
    LOCAL_CONDITION_DIM,
    LOCAL_OPERATOR_DIM,
    robust_scene_operator_consensus,
)
from threeduse.enhance.multiview_consensus import MultiViewOperatorConsensus
from threeduse.enhance.transition_calibrator import (
    DeterministicTransitionCalibrator,
)
from threeduse.rendering.stage_renderers import (
    EnhancementRender,
    ReconstructionRender,
    StageRenderBundle,
    UnderwaterRasterContext,
)


def test_residual_calibrator_adds_uie_proposal() -> None:
    model = DeterministicTransitionCalibrator(
        prediction_mode="proposal_residual"
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    global_condition = torch.randn(2, GLOBAL_CONDITION_DIM + GLOBAL_OPERATOR_DIM)
    global_proposal = global_condition[..., -GLOBAL_OPERATOR_DIM:].clone()
    assert torch.allclose(model.global_target(global_condition), global_proposal)

    local_width = (
        LOCAL_CONDITION_DIM
        + GLOBAL_CONDITION_DIM
        + GLOBAL_OPERATOR_DIM
        + LOCAL_OPERATOR_DIM
        + GLOBAL_OPERATOR_DIM
    )
    local_condition = torch.randn(2, 3, 4, local_width)
    proposal_start = LOCAL_CONDITION_DIM + GLOBAL_CONDITION_DIM + GLOBAL_OPERATOR_DIM
    local_proposal = local_condition[
        ..., proposal_start : proposal_start + LOCAL_OPERATOR_DIM
    ].clone()
    assert torch.allclose(model.local_target(local_condition), local_proposal)


def test_effect_space_consensus_rejects_one_outlier() -> None:
    torch.manual_seed(4)
    inliers = 0.02 * torch.randn(9, GLOBAL_OPERATOR_DIM)
    outlier = torch.full((1, GLOBAL_OPERATOR_DIM), 3.0)
    target, dispersion = robust_scene_operator_consensus(
        torch.cat((inliers, outlier), dim=0)
    )
    assert target.shape == (1, GLOBAL_OPERATOR_DIM)
    assert target.norm() < 0.25
    assert dispersion.isfinite()


def test_consensus_appearance_is_identity_at_initialization() -> None:
    torch.manual_seed(2)
    model = ConsensusStage2Appearance(12, cache_compiled=False)
    means = torch.randn(12, 3)
    colors = torch.rand(12, 3)
    compiled = model.object_appearance(
        colors, colors, means, strength=1.0
    )
    assert torch.allclose(compiled.enhanced_linear, colors, atol=1.0e-6)

    medium = torch.rand(8, 9, 3)
    enhanced_medium, _ = model.medium_appearance(
        medium, torch.zeros(3), strength=1.0
    )
    assert torch.allclose(enhanced_medium, medium, atol=1.0e-6)
    beta = torch.rand(8, 9, 3).clamp_min(1.0e-3)
    beta_bs, beta_attn, _ = model.transport(beta, beta, strength=1.0)
    assert torch.allclose(beta_bs, beta)
    assert torch.allclose(beta_attn, beta)


def test_public_enhanced_rgb_is_underwater_composite() -> None:
    image = torch.rand(5, 7, 3)
    zeros = torch.zeros_like(image)
    scalar = torch.zeros(5, 7, 1)
    context = UnderwaterRasterContext(
        projection={},
        medium_source=zeros,
        beta_bs=zeros,
        beta_attn=zeros,
        far_depths=scalar,
        ray_depth_scales=torch.ones_like(scalar),
        width=7,
        height=5,
        tile_size=16,
    )
    reconstruction = ReconstructionRender(
        composite=zeros,
        object=zeros,
        medium=zeros,
        clear_linear=zeros,
        alpha=scalar,
        depth=scalar,
        context=context,
    )
    enhancement = EnhancementRender(
        composite=image,
        object=zeros,
        medium=image,
        medium_source=image,
        clear_linear=torch.full_like(image, 9.0),
        beta_bs=zeros,
        beta_attn=zeros,
        alpha=scalar,
        depth=scalar,
    )
    outputs = StageRenderBundle(reconstruction, enhancement).as_output_dict(
        enhanced_prediction=True
    )
    assert outputs["enhanced_rgb"] is image
    assert outputs["pred_image"] is image
    assert not torch.allclose(outputs["enhanced_rgb"], outputs["enhanced_clear_display"])


def test_consensus_storage_resizes_for_densified_stage1() -> None:
    consensus = MultiViewOperatorConsensus(3)
    consensus.resize_num_gaussians(11)
    assert consensus.local_mean.shape == (11, LOCAL_OPERATOR_DIM)
    assert consensus.local_weight.shape == (11, 1)
    assert consensus.local_views.shape == (11, 1)
