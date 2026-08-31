"""Differentiable effective photometric transitions for Stage-2.

The paired prior and the scene optimizer must observe exactly the same object:
an image-space transform realized *after* underwater compositing.  This module
therefore extracts an identity-regularized global affine grade followed by
local diagonal gain/bias residuals.  The source image is treated as fixed, so
the closed-form estimators remain differentiable with respect to the target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

GLOBAL_OPERATOR_DIM = 12
LOCAL_OPERATOR_DIM = 6
GLOBAL_CONDITION_DIM = 18
LOCAL_CONDITION_DIM = 6
GLOBAL_TRANSITION_CONDITION_DIM = GLOBAL_CONDITION_DIM + GLOBAL_OPERATOR_DIM
LOCAL_TRANSITION_CONDITION_DIM = (
    LOCAL_CONDITION_DIM
    + GLOBAL_CONDITION_DIM
    + GLOBAL_OPERATOR_DIM
    + LOCAL_OPERATOR_DIM
    + GLOBAL_OPERATOR_DIM
)


@dataclass(frozen=True)
class EffectiveTransition:
    """Effective source-to-target transition and fit diagnostics."""

    global_operator: Tensor
    local_operator: Tensor
    global_condition: Tensor
    local_condition: Tensor
    local_confidence: Tensor
    global_residual: Tensor
    local_residual: Tensor


def _as_bchw(image: Tensor, name: str) -> tuple[Tensor, bool]:
    if image.ndim == 3 and image.shape[-1] == 3:
        return image.permute(2, 0, 1)[None].contiguous(), True
    if image.ndim == 4 and image.shape[-1] == 3:
        return image.permute(0, 3, 1, 2).contiguous(), False
    if image.ndim == 4 and image.shape[1] == 3:
        return image.contiguous(), False
    raise ValueError(f"{name} must be [H,W,3], [B,H,W,3], or [B,3,H,W], got {tuple(image.shape)}")


def _resize_long_side(image: Tensor, max_side: int) -> Tensor:
    if max_side <= 0 or max(image.shape[-2:]) <= max_side:
        return image
    scale = float(max_side) / float(max(image.shape[-2:]))
    size = tuple(max(8, int(round(value * scale))) for value in image.shape[-2:])
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


def _resolve_grid(height: int, width: int, long_side_cells: int) -> tuple[int, int]:
    if long_side_cells < 2:
        raise ValueError("long_side_cells must be at least two")
    scale = float(long_side_cells) / float(max(height, width))
    return (
        max(2, int(round(height * scale))),
        max(2, int(round(width * scale))),
    )


def global_source_descriptor(image: Tensor) -> Tensor:
    """Return an 18D camera-image condition without a learned UIE encoder."""

    value, _ = _as_bchw(image, "image")
    flat = value.flatten(2)
    mean = flat.mean(dim=-1)
    std = flat.std(dim=-1, unbiased=False)
    quantiles = torch.quantile(
        flat.float(),
        flat.new_tensor([0.1, 0.5, 0.9], dtype=torch.float32),
        dim=-1,
    ).to(flat)
    quantiles = quantiles.permute(1, 0, 2).reshape(flat.shape[0], -1)
    positive = F.softplus(value * 50.0) / 50.0
    luminance = 0.299 * positive[:, 0] + 0.587 * positive[:, 1] + 0.114 * positive[:, 2]
    lum_flat = luminance.flatten(1)
    lum_mean = lum_flat.mean(dim=-1, keepdim=True)
    lum_std = lum_flat.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-5)
    lum_skew = (((lum_flat - lum_mean) / lum_std).pow(3)).mean(dim=-1, keepdim=True)
    descriptor = torch.cat((mean, std, quantiles, lum_mean, lum_std, lum_skew), dim=-1)
    if descriptor.shape[-1] != GLOBAL_CONDITION_DIM:
        raise RuntimeError(f"Expected {GLOBAL_CONDITION_DIM} global conditions, got {descriptor.shape[-1]}")
    return descriptor


def local_source_descriptor(image: Tensor, grid: tuple[int, int]) -> Tensor:
    """Return per-patch log-luma/opponent means and standard deviations."""

    value, _ = _as_bchw(image, "image")
    positive = value.clamp_min(0.0)
    log_rgb = torch.log(positive + 1e-4)
    red, green, blue = log_rgb.unbind(dim=1)
    luminance = (
        0.299 * positive[:, 0]
        + 0.587 * positive[:, 1]
        + 0.114 * positive[:, 2]
    )
    feature = torch.stack(
        (
            torch.log(luminance + 1e-4),
            (red - green) / math.sqrt(2.0),
            (red + green - 2.0 * blue) / math.sqrt(6.0),
        ),
        dim=1,
    )
    mean = F.adaptive_avg_pool2d(feature, grid)
    second = F.adaptive_avg_pool2d(feature.square(), grid)
    std = (second - mean.square()).clamp_min(1e-5).sqrt()
    return torch.cat((mean, std), dim=1).permute(0, 2, 3, 1).contiguous()


def _global_affine_fit(source: Tensor, target: Tensor, ridge: float) -> tuple[Tensor, Tensor]:
    """Fit target ~= M [source,1] with an identity-centered ridge prior."""

    batch = source.shape[0]
    x = source.permute(0, 2, 3, 1).reshape(batch, -1, 3)
    y = target.permute(0, 2, 3, 1).reshape(batch, -1, 3)
    ones = torch.ones_like(x[..., :1])
    xh = torch.cat((x, ones), dim=-1)

    luminance = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    lower = torch.sigmoid((luminance - 0.005) / 0.02)
    upper = torch.sigmoid((1.2 - luminance) / 0.10)
    weight = (0.05 + 0.95 * lower * upper).detach()
    weighted_x = xh * weight[..., None]
    xtwx = torch.einsum("bpi,bpj->bij", xh, weighted_x)
    ytwx = torch.einsum("bpc,bpi->bci", y * weight[..., None], xh)

    identity = source.new_zeros((batch, 3, 4))
    identity[:, :, :3] = torch.eye(3, device=source.device, dtype=source.dtype)[None]
    scale = xtwx.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-4)
    regularizer = float(ridge) * scale
    system = xtwx + regularizer[:, None, None] * torch.eye(
        4, device=source.device, dtype=source.dtype
    )[None]
    rhs = ytwx + regularizer[:, None, None] * identity
    matrix = torch.linalg.solve(system.transpose(-1, -2), rhs.transpose(-1, -2)).transpose(-1, -2)
    prediction = torch.einsum("bci,bpi->bpc", matrix, xh)
    residual = ((prediction - y).square().mean(dim=-1) * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1e-6)
    return matrix, residual


def _apply_global_affine(source: Tensor, matrix: Tensor) -> Tensor:
    batch, _, height, width = source.shape
    flat = source.permute(0, 2, 3, 1).reshape(batch, -1, 3)
    homogeneous = torch.cat((flat, torch.ones_like(flat[..., :1])), dim=-1)
    result = torch.einsum("bci,bpi->bpc", matrix, homogeneous)
    return result.reshape(batch, height, width, 3).permute(0, 3, 1, 2).contiguous()


def _patchify(image: Tensor, grid: tuple[int, int], samples_per_patch: int = 8) -> Tensor:
    grid_h, grid_w = grid
    pooled = F.adaptive_avg_pool2d(
        image,
        (grid_h * samples_per_patch, grid_w * samples_per_patch),
    )
    batch, channels = pooled.shape[:2]
    return (
        pooled.reshape(batch, channels, grid_h, samples_per_patch, grid_w, samples_per_patch)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch, grid_h, grid_w, samples_per_patch**2, channels)
        .contiguous()
    )


def _local_diagonal_fit(
    source: Tensor,
    target: Tensor,
    grid: tuple[int, int],
    ridge: float,
) -> tuple[Tensor, Tensor, Tensor]:
    x = _patchify(source, grid)
    y = _patchify(target, grid)
    count = x.shape[-2]
    sum_x2 = x.square().sum(dim=-2)
    sum_x = x.sum(dim=-2)
    sum_yx = (y * x).sum(dim=-2)
    sum_y = y.sum(dim=-2)

    scale = x.square().mean(dim=-2).clamp_min(1e-4)
    regularizer = float(ridge) * float(count) * scale
    a00 = sum_x2 + regularizer
    a01 = sum_x
    a11 = x.new_full(sum_x.shape, float(count)) + float(ridge) * float(count)
    rhs0 = sum_yx + regularizer
    rhs1 = sum_y
    determinant = (a00 * a11 - a01.square()).clamp_min(1e-8)
    gain = (rhs0 * a11 - rhs1 * a01) / determinant
    bias = (a00 * rhs1 - a01 * rhs0) / determinant
    prediction = gain[..., None, :] * x + bias[..., None, :]
    residual = (prediction - y).square().mean(dim=(-2, -1))
    variance = x.var(dim=-2, unbiased=False).mean(dim=-1)
    confidence = (variance / (variance + 1e-3)) * torch.exp(-residual.detach() / 0.02)
    operator = torch.cat((gain - 1.0, bias), dim=-1)
    return operator, residual, confidence.detach().clamp(0.0, 1.0)


def extract_effective_transition(
    source: Tensor,
    target: Tensor,
    *,
    local_grid_long_side: int = 16,
    max_side: int = 256,
    global_ridge: float = 1e-3,
    local_ridge: float = 1e-2,
) -> EffectiveTransition:
    """Extract the effective full-render transition used by prior and student."""

    source_bchw, _ = _as_bchw(source, "source")
    target_bchw, _ = _as_bchw(target, "target")
    if source_bchw.shape != target_bchw.shape:
        raise ValueError(f"source and target must match, got {tuple(source_bchw.shape)} and {tuple(target_bchw.shape)}")
    source_bchw = _resize_long_side(source_bchw.float(), int(max_side))
    target_bchw = F.interpolate(
        target_bchw.float(),
        size=source_bchw.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    grid = _resolve_grid(*source_bchw.shape[-2:], int(local_grid_long_side))

    matrix, global_residual = _global_affine_fit(
        source_bchw.detach(), target_bchw, float(global_ridge)
    )
    identity = torch.eye(3, 4, device=matrix.device, dtype=matrix.dtype)[None]
    global_operator = (matrix - identity).reshape(matrix.shape[0], GLOBAL_OPERATOR_DIM)
    globally_graded = _apply_global_affine(source_bchw.detach(), matrix)
    local_operator, local_residual, local_confidence = _local_diagonal_fit(
        globally_graded,
        target_bchw,
        grid,
        float(local_ridge),
    )

    local_condition = local_source_descriptor(source_bchw, grid)
    return EffectiveTransition(
        global_operator=global_operator,
        local_operator=local_operator,
        global_condition=global_source_descriptor(source_bchw),
        local_condition=local_condition,
        local_confidence=local_confidence,
        global_residual=global_residual,
        local_residual=local_residual,
    )


def global_transition_condition(
    source: EffectiveTransition,
    uie_proposal: EffectiveTransition,
) -> Tensor:
    """Condition a paired transition on source appearance and a UIE proposal."""

    if (
        source.global_condition.shape[:-1]
        != uie_proposal.global_operator.shape[:-1]
    ):
        raise ValueError("Source and UIE proposal global batches must match")
    result = torch.cat(
        (source.global_condition, uie_proposal.global_operator), dim=-1
    )
    if result.shape[-1] != GLOBAL_TRANSITION_CONDITION_DIM:
        raise RuntimeError(
            f"Expected {GLOBAL_TRANSITION_CONDITION_DIM} global transition conditions, "
            f"got {result.shape[-1]}"
        )
    return result


def local_transition_condition(
    source: EffectiveTransition,
    uie_proposal: EffectiveTransition,
    target_global_operator: Tensor,
) -> Tensor:
    """Build the joint local condition without inventing intermediate grades.

    The calibrator observes the raw patch, scene-wide raw statistics, the frozen
    UIE global/local proposal, and the selected target global grade. Its value
    remains the local operator fitted from the paired enhanced target.
    """

    if source.local_condition.shape != uie_proposal.local_condition.shape:
        raise ValueError("Source and UIE proposal local grids must match")
    if (
        source.local_condition.shape[:3]
        != uie_proposal.local_operator.shape[:3]
    ):
        raise ValueError("UIE local proposal does not match the source grid")
    batch, grid_h, grid_w, _ = source.local_condition.shape
    if target_global_operator.shape != (batch, GLOBAL_OPERATOR_DIM):
        raise ValueError(
            f"Expected target global operator {(batch, GLOBAL_OPERATOR_DIM)}, "
            f"got {tuple(target_global_operator.shape)}"
        )
    global_source = source.global_condition[:, None, None, :].expand(
        batch, grid_h, grid_w, GLOBAL_CONDITION_DIM
    )
    proposal_global = uie_proposal.global_operator[:, None, None, :].expand(
        batch, grid_h, grid_w, GLOBAL_OPERATOR_DIM
    )
    target_global = target_global_operator[:, None, None, :].expand(
        batch, grid_h, grid_w, GLOBAL_OPERATOR_DIM
    )
    result = torch.cat(
        (
            source.local_condition,
            global_source,
            proposal_global,
            uie_proposal.local_operator,
            target_global,
        ),
        dim=-1,
    )
    if result.shape[-1] != LOCAL_TRANSITION_CONDITION_DIM:
        raise RuntimeError(
            f"Expected {LOCAL_TRANSITION_CONDITION_DIM} local transition conditions, "
            f"got {result.shape[-1]}"
        )
    return result


def center_local_operator(
    local_operator: Tensor,
    weight: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Remove the spatially constant mode from a local operator map.

    ``SceneAppearanceField`` removes the constant local mode structurally so
    that the scene-global affine is identifiable. Target operators must use
    the same gauge; otherwise their nominally local target still contains a
    second global colour grade that the 3D field cannot represent without
    spatial artefacts.

    Returns the centred map and the removed per-channel centre.
    """

    if local_operator.ndim != 4 or local_operator.shape[-1] != LOCAL_OPERATOR_DIM:
        raise ValueError(
            "Expected local operator [B,H,W,6], got "
            f"{tuple(local_operator.shape)}"
        )
    batch, height, width, _ = local_operator.shape
    if weight is None:
        spatial_weight = local_operator.new_ones((batch, height, width, 1))
    else:
        spatial_weight = weight.detach().to(local_operator)
        if spatial_weight.ndim == 3:
            spatial_weight = spatial_weight[..., None]
        if spatial_weight.shape != (batch, height, width, 1):
            raise ValueError(
                "Expected local weight [B,H,W] or [B,H,W,1], got "
                f"{tuple(weight.shape)}"
            )
        spatial_weight = spatial_weight.clamp_min(0.0)
    denominator = spatial_weight.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    center = (local_operator * spatial_weight).sum(
        dim=(1, 2), keepdim=True
    ) / denominator
    return local_operator - center, center[:, 0, 0]


def _global_operator_probes(operator: Tensor) -> tuple[Tensor, Tensor]:
    """Return a fixed RGB probe chart and the operator responses on it."""

    if operator.ndim != 2 or operator.shape[-1] != GLOBAL_OPERATOR_DIM:
        raise ValueError(
            f"Expected global operators [V,{GLOBAL_OPERATOR_DIM}], got "
            f"{tuple(operator.shape)}"
        )
    if operator.shape[0] == 0:
        raise ValueError("At least one view operator is required")

    probes = operator.new_tensor(
        [
            [0.0, 0.0, 0.0],
            [0.18, 0.18, 0.18],
            [0.50, 0.50, 0.50],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    identity = torch.eye(
        3, 4, device=operator.device, dtype=operator.dtype
    )
    matrices = identity[None] + operator.reshape(-1, 3, 4)
    homogeneous = torch.cat((probes, torch.ones_like(probes[:, :1])), dim=-1)
    responses = torch.einsum("vci,pi->vpc", matrices, homogeneous)
    return probes, responses


def robust_scene_operator_consensus(
    global_operators: Tensor,
    *,
    iterations: int = 32,
) -> tuple[Tensor, Tensor]:
    """Fuse view operators through their observable action, not coefficients.

    A geometric median is computed on a fixed RGB probe chart.  The consensus
    responses are then projected back to the renderer-expressible 3x4 affine
    family.  This avoids both failure modes already observed in the project:
    averaging matrices into a dataset-wide colour filter and selecting the
    arbitrary colour mode of one medoid/support view.
    """

    probes, responses = _global_operator_probes(global_operators)
    observable = responses.reshape(responses.shape[0], -1).float()
    center = observable.median(dim=0).values
    eps = torch.finfo(observable.dtype).eps * 64.0
    for _ in range(max(1, int(iterations))):
        distance = (observable - center).norm(dim=-1).clamp_min(eps)
        weight = distance.reciprocal()
        updated = (weight[:, None] * observable).sum(dim=0) / weight.sum()
        if (updated - center).norm() <= 1.0e-6:
            center = updated
            break
        center = updated

    target = center.to(global_operators).reshape(probes.shape[0], 3)
    homogeneous = torch.cat((probes, torch.ones_like(probes[:, :1])), dim=-1)
    # The chart is full rank; lstsq is the unique least-squares projection of
    # the robust observable consensus back to one scene affine.
    matrix = torch.linalg.lstsq(homogeneous, target).solution.T
    identity = torch.eye(
        3, 4, device=matrix.device, dtype=matrix.dtype
    )
    operator = (matrix - identity).reshape(1, GLOBAL_OPERATOR_DIM)
    dispersion = (observable - center).norm(dim=-1).median().to(global_operators)
    return operator, dispersion


__all__ = [
    "EffectiveTransition",
    "GLOBAL_CONDITION_DIM",
    "GLOBAL_TRANSITION_CONDITION_DIM",
    "GLOBAL_OPERATOR_DIM",
    "LOCAL_CONDITION_DIM",
    "LOCAL_TRANSITION_CONDITION_DIM",
    "LOCAL_OPERATOR_DIM",
    "extract_effective_transition",
    "center_local_operator",
    "global_transition_condition",
    "global_source_descriptor",
    "local_source_descriptor",
    "local_transition_condition",
    "robust_scene_operator_consensus",
]
