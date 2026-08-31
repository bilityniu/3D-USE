"""Typed rendering branches for reconstruction and underwater enhancement.

The public contract deliberately separates semantic branches from CUDA
implementation details.  Stage-1 owns projection, visibility, opacity and the
medium.  Stage-2 consumes the resulting immutable raster context and replaces
only Gaussian appearance in the same attenuated object compositor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
from torch import Tensor

from threeduse.rendering.underwater import (
    prepare_underwater_gsplat_projection,
    rasterize_clear_from_underwater_meta,
    rasterize_underwater_gsplat,
)


@dataclass(frozen=True)
class GradientPolicy:
    """Explicit ownership of gradients through a rendering branch."""

    geometry: bool
    opacity: bool
    medium: bool

    @classmethod
    def stage2(cls) -> "GradientPolicy":
        """Stage-2 changes appearance and nothing in reconstruction state."""

        return cls(geometry=False, opacity=False, medium=False)


@dataclass(frozen=True)
class UnderwaterRasterContext:
    """Projection/sort state shared by two semantic renderer heads."""

    projection: Mapping[str, Tensor]
    medium_source: Tensor
    beta_bs: Tensor
    beta_attn: Tensor
    far_depths: Tensor
    ray_depth_scales: Tensor
    width: int
    height: int
    tile_size: int


@dataclass(frozen=True)
class ReconstructionRender:
    """Linear Stage-1 channels.  Display transforms never enter this object."""

    composite: Tensor
    object: Tensor
    medium: Tensor
    clear_linear: Tensor
    alpha: Tensor
    depth: Tensor
    context: UnderwaterRasterContext


@dataclass(frozen=True)
class EnhancementRender:
    """Complete Stage-2 underwater render with trainable source appearance."""

    composite: Tensor
    object: Tensor
    medium: Tensor
    medium_source: Tensor
    clear_linear: Tensor
    beta_bs: Tensor
    beta_attn: Tensor
    alpha: Tensor
    depth: Tensor


@dataclass(frozen=True)
class StageRenderBundle:
    """Immutable Stage-1/Stage-2 branch bundle used by losses and viewers."""

    reconstruction: ReconstructionRender
    enhancement: EnhancementRender | None = None

    def as_output_dict(
        self,
        *,
        enhanced_prediction: bool,
    ) -> dict[str, Tensor]:
        """Expose stable names while retaining Nerfstudio boundary aliases.

        ``rgb`` always means the Stage-1 underwater reconstruction.  Only
        ``pred_image`` is a task-dependent boundary selection.
        """

        reconstruction = self.reconstruction
        clear_linear = reconstruction.clear_linear
        clear_display = clear_linear / (clear_linear + 1.0)
        clear_scale = torch.quantile(clear_linear, 0.995).clamp_min(1e-6)
        clear_unclamp = clear_linear / clear_scale
        outputs = {
            "rgb": reconstruction.composite,
            "pred_image": reconstruction.composite,
            "stage1_rgb": reconstruction.composite,
            "stage1_object": reconstruction.object,
            "stage1_medium": reconstruction.medium,
            "stage1_clear_linear": clear_linear,
            # Backwards-compatible diagnostic names are explicit about where
            # clipping/tone mapping occurs.  Losses must use the stage names.
            "rgb_object": reconstruction.object,
            "rgb_medium": reconstruction.medium,
            "rgb_clear": clear_display,
            "rgb_clear_unclamp": clear_unclamp,
            "rgb_clear_clamp": clear_linear.clamp(0.0, 1.0),
            "depth": reconstruction.depth,
            "accumulation": reconstruction.alpha,
        }
        if self.enhancement is not None:
            enhancement = self.enhancement
            enhanced_clear_display = enhancement.clear_linear / (
                enhancement.clear_linear + 1.0
            )
            # Preserve Plenodium's three clear branches exactly.  They remain
            # diagnostics and never participate in the enhancement loss.
            enhanced_clear_scale = torch.quantile(
                enhancement.clear_linear,
                0.995,
            ).clamp_min(1e-6)
            enhanced_clear_unclamp = (
                enhancement.clear_linear / enhanced_clear_scale
            )
            outputs.update(
                {
                    "stage2_source_rgb": reconstruction.composite,
                    "enhanced_object": enhancement.object,
                    "enhanced_object_rgb": enhancement.object,
                    # ``enhanced_rgb`` is the public Stage-2 prediction alias.
                    # The explicit keys below preserve both semantic branches.
                    "enhanced_rgb": enhancement.composite,
                    "enhanced_underwater_rgb": enhancement.composite,
                    "enhanced_medium_rgb": enhancement.medium,
                    "enhanced_medium_source": enhancement.medium_source,
                    "enhanced_clear_linear": enhancement.clear_linear,
                    "enhanced_clear_display": enhanced_clear_display,
                    # Exact analogue of Plenodium's ``rgb_clear_unclamp``:
                    # divide by the per-view 99.5-percentile.
                    "enhanced_clear_unclamp": enhanced_clear_unclamp,
                    # Explicit legacy-style display diagnostic.  This is never
                    # selected by the training loss or the public enhanced_rgb
                    # alias; it exists only for controlled visualization and
                    # metric comparisons against clear-clamp baselines.
                    "enhanced_clear_clamp": enhancement.clear_linear.clamp(
                        0.0, 1.0
                    ),
                    "enhanced_beta_bs": enhancement.beta_bs,
                    "enhanced_beta_attn": enhancement.beta_attn,
                    "enhanced_accumulation": enhancement.alpha,
                    "stage2_expected_depth": enhancement.depth,
                }
            )
            if enhanced_prediction:
                outputs["pred_image"] = enhancement.composite
        return outputs


class UnderwaterReconstructionRenderer(nn.Module):
    """Stage-1 renderer that owns projection and underwater compositing."""

    def prepare_context(
        self, *, means: Tensor, scales: Tensor, quats: Tensor, opacities: Tensor,
        viewmat: Tensor, intrinsics: Tensor, medium_rgb: Tensor, beta_bs: Tensor,
        beta_attn: Tensor, width: int, height: int, far_depths: Tensor,
        ray_depth_scales: Tensor, tile_size: int, near_plane: float,
        projection_far_plane: float, antialiased: bool,
    ) -> UnderwaterRasterContext:
        """Build the shared raster context without launching a pixel compositor."""
        projection = prepare_underwater_gsplat_projection(
            means, scales, quats, opacities, viewmat, intrinsics, width, height,
            tile_size=tile_size, near_plane=near_plane,
            projection_far_plane=projection_far_plane, antialiased=antialiased,
        )
        return UnderwaterRasterContext(
            projection=projection, medium_source=medium_rgb, beta_bs=beta_bs,
            beta_attn=beta_attn, far_depths=far_depths,
            ray_depth_scales=ray_depth_scales, width=int(width), height=int(height),
            tile_size=int(tile_size),
        )

    def forward(
        self,
        *,
        means: Tensor,
        scales: Tensor,
        quats: Tensor,
        opacities: Tensor,
        colors: Tensor,
        viewmat: Tensor,
        intrinsics: Tensor,
        medium_rgb: Tensor,
        beta_bs: Tensor,
        beta_attn: Tensor,
        width: int,
        height: int,
        far_depths: Tensor,
        ray_depth_scales: Tensor,
        tile_size: int,
        near_plane: float,
        projection_far_plane: float,
        antialiased: bool,
        absgrad: bool,
        optical_depth_grad_scale: float,
    ) -> ReconstructionRender:
        channels, projection = rasterize_underwater_gsplat(
            means,
            scales,
            quats,
            opacities,
            colors,
            viewmat,
            intrinsics,
            medium_rgb,
            beta_bs,
            beta_attn,
            width,
            height,
            far_depths=far_depths,
            ray_depth_scales=ray_depth_scales,
            tile_size=tile_size,
            near_plane=near_plane,
            projection_far_plane=projection_far_plane,
            antialiased=antialiased,
            absgrad=absgrad,
            optical_depth_grad_scale=optical_depth_grad_scale,
        )
        composite, object_rgb, clear, medium, alpha, depth_accum = (
            tensor[0] for tensor in channels
        )
        expected_depth = torch.where(
            alpha > 1e-6,
            depth_accum / alpha.clamp_min(1e-6),
            torch.zeros_like(depth_accum),
        )
        context = UnderwaterRasterContext(
            projection=projection,
            medium_source=medium_rgb,
            beta_bs=beta_bs,
            beta_attn=beta_attn,
            far_depths=far_depths,
            ray_depth_scales=ray_depth_scales,
            width=int(width),
            height=int(height),
            tile_size=int(tile_size),
        )
        return ReconstructionRender(
            composite=composite,
            object=object_rgb,
            medium=medium,
            clear_linear=clear,
            alpha=alpha,
            depth=expected_depth,
            context=context,
        )


class UnderwaterEnhancementRenderer(nn.Module):
    """Render C+, B+ and beta+ with the frozen Stage-1 raster context."""

    _REQUIRED_CONTEXT = (
        "means2d",
        "conics",
        "depths",
        "opacities",
        "isect_offsets",
        "flatten_ids",
    )

    @staticmethod
    def _owned(tensor: Tensor, allow_gradient: bool) -> Tensor:
        return tensor if allow_gradient else tensor.detach()

    def forward(
        self,
        colors: Tensor,
        context: UnderwaterRasterContext,
        *,
        medium_source: Tensor | None = None,
        beta_bs_source: Tensor | None = None,
        beta_attn_source: Tensor | None = None,
        policy: GradientPolicy,
    ) -> EnhancementRender:
        from threeduse.cuda._wrapper import rasterize_to_pixels_underwater

        projection = context.projection
        missing = [name for name in self._REQUIRED_CONTEXT if name not in projection]
        if missing:
            raise ValueError(f"Underwater raster context is missing fields: {missing}")
        if colors.ndim != 2 or colors.shape[-1] != 3:
            raise ValueError(
                f"Expected enhanced Gaussian RGB [N,3], got {tuple(colors.shape)}"
            )

        means2d = projection["means2d"]
        if means2d.ndim != 3 or means2d.shape[0] != 1:
            raise ValueError(
                f"Expected one-camera means2d [1,N,2], got {tuple(means2d.shape)}"
            )
        if colors.shape[0] != means2d.shape[1]:
            raise ValueError(
                f"Enhanced color count {colors.shape[0]} does not match projected "
                f"Gaussians {means2d.shape[1]}"
            )

        means2d = self._owned(means2d, policy.geometry)
        conics = self._owned(projection["conics"], policy.geometry)
        depths = self._owned(projection["depths"], policy.geometry)
        opacities = self._owned(projection["opacities"], policy.opacity)
        beta_bs_source = (
            self._owned(context.beta_bs, policy.medium)
            if beta_bs_source is None
            else beta_bs_source
        )
        beta_attn_source = (
            self._owned(context.beta_attn, policy.medium)
            if beta_attn_source is None
            else beta_attn_source
        )
        expected_transport_shape = (context.height, context.width, 3)
        if beta_bs_source.shape != expected_transport_shape:
            raise ValueError(
                f"Expected enhanced beta_bs {expected_transport_shape}, "
                f"got {tuple(beta_bs_source.shape)}"
            )
        if beta_attn_source.shape != expected_transport_shape:
            raise ValueError(
                f"Expected enhanced beta_attn {expected_transport_shape}, "
                f"got {tuple(beta_attn_source.shape)}"
            )
        if medium_source is None:
            medium_source = context.medium_source.detach()
        if medium_source.shape != context.medium_source.shape:
            raise ValueError(
                f"Expected enhanced medium source {tuple(context.medium_source.shape)}, "
                f"got {tuple(medium_source.shape)}"
            )

        composite, object_rgb, clear_linear, medium_rgb, alpha, depth_accum = (
            rasterize_to_pixels_underwater(
                means2d,
                conics,
                colors.unsqueeze(0),
                opacities,
                depths,
                medium_source.unsqueeze(0),
                beta_bs_source.unsqueeze(0),
                beta_attn_source.unsqueeze(0),
                context.far_depths.detach().unsqueeze(0),
                context.ray_depth_scales.detach().unsqueeze(0),
                context.width,
                context.height,
                context.tile_size,
                projection["isect_offsets"],
                projection["flatten_ids"],
                packed=False,
                absgrad=False,
                optical_depth_grad_scale=0.0,
            )
        )
        expected_depth = torch.where(
            alpha > 1e-6,
            depth_accum / alpha.clamp_min(1e-6),
            torch.zeros_like(depth_accum),
        )
        return EnhancementRender(
            composite=composite[0],
            object=object_rgb[0],
            medium=medium_rgb[0],
            medium_source=medium_source,
            clear_linear=clear_linear[0],
            beta_bs=beta_bs_source,
            beta_attn=beta_attn_source,
            alpha=alpha[0],
            depth=expected_depth[0],
        )
