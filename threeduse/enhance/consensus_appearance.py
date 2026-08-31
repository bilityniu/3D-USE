"""Persistent full underwater appearance used by the single Stage-2 mainline."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from threeduse.enhance.appearance_compiler import (
    CompiledAppearance,
    Stage2AppearanceCompiler,
)
from threeduse.enhance.scene_appearance_field import SceneAppearanceField


class ConsensusStage2Appearance(nn.Module):
    """Persistent U-BAF state for object, water, and transport.

    Global colour is owned by a single scene affine.  The zero-constant-mode
    4D field owns only local Gaussian/medium residuals.  Transport retains two
    scalar strengths, so it cannot become a second wavelength-dependent colour
    filter.  Every quantity is view independent and survives into novel views.
    """

    def __init__(
        self,
        num_gaussians: int,
        *,
        grid_resolution: int = 16,
        rank: int = 8,
        hidden_dim: int = 64,
        coordinate_quantile: float = 0.01,
        cache_compiled: bool = True,
        max_transport_log_ratio: float = 1.3862943611198906,
    ) -> None:
        super().__init__()
        field = SceneAppearanceField(
            grid_resolution=int(grid_resolution),
            rank=int(rank),
            hidden_dim=int(hidden_dim),
        )
        self.compiler = Stage2AppearanceCompiler(
            field,
            coordinate_quantile=float(coordinate_quantile),
            cache_compiled=bool(cache_compiled),
            max_transport_log_ratio=float(max_transport_log_ratio),
        )

    @property
    def field(self) -> SceneAppearanceField:
        field = self.compiler.field
        if not isinstance(field, SceneAppearanceField):
            raise RuntimeError("Consensus appearance lost its 4D field")
        return field

    def clear_cache(self) -> None:
        self.compiler.clear_cache()

    def object_appearance(
        self,
        raw_rgb: Tensor,
        base_rgb: Tensor,
        means: Tensor,
        *,
        strength: float,
    ) -> CompiledAppearance:
        return self.compiler(
            raw_rgb,
            base_rgb,
            means,
            strength=float(strength),
            cache_token=0,
        )

    def medium_appearance(
        self,
        medium_source: Tensor,
        camera_origin: Tensor,
        *,
        strength: float,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        endpoint, auxiliary = self.field.enhance_medium(
            medium_source.detach(),
            camera_origin=camera_origin.reshape(3).detach(),
        )
        amount = max(0.0, min(1.0, float(strength)))
        enhanced = medium_source.detach() + amount * (
            endpoint - medium_source.detach()
        )
        auxiliary["medium_endpoint_delta_abs_mean"] = (
            endpoint - medium_source.detach()
        ).abs().mean()
        return enhanced, auxiliary

    def transport(
        self,
        beta_bs: Tensor,
        beta_attn: Tensor,
        *,
        strength: float,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        return self.compiler.enhance_transport(
            beta_bs,
            beta_attn,
            strength=float(strength),
        )

    def tv_loss(self) -> Tensor:
        return self.field.tv_loss()


__all__ = ["ConsensusStage2Appearance"]
