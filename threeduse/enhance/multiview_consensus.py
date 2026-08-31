"""Scene-global and Gaussian-local consensus for Stage-2 enhancement.

The paired calibrator proposes an operator independently for every captured
view.  This module is the only place where those view proposals become scene
state: global operators are fused in observable effect space, while local
operators are back-projected through the transpose of the frozen Stage-1
rasterizer and robustly accumulated on Gaussians.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from threeduse.enhance.effective_transition import (
    GLOBAL_OPERATOR_DIM,
    LOCAL_OPERATOR_DIM,
    robust_scene_operator_consensus,
)
from threeduse.rendering.underwater import (
    rasterize_clear_from_underwater_meta,
)


class MultiViewOperatorConsensus(nn.Module):
    """Appearance Transition Consensus (ATC) stored as persistent targets.

    This module has buffers but no trainable weights. It robustly fits one
    scene-global operator and fuses centered view residuals into one local
    target per Gaussian.
    """

    def __init__(self, num_gaussians: int) -> None:
        super().__init__()
        if num_gaussians < 1:
            raise ValueError("Stage-2 consensus requires at least one Gaussian")
        self.register_buffer(
            "global_target", torch.zeros(1, GLOBAL_OPERATOR_DIM), persistent=True
        )
        self.register_buffer("global_ready", torch.tensor(False), persistent=True)
        self.register_buffer("global_dispersion", torch.zeros(()), persistent=True)
        self.register_buffer(
            "local_mean",
            torch.zeros(num_gaussians, LOCAL_OPERATOR_DIM),
            persistent=True,
        )
        self.register_buffer(
            "local_weight", torch.zeros(num_gaussians, 1), persistent=True
        )
        self.register_buffer(
            "local_views", torch.zeros(num_gaussians, 1, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "local_scale", torch.ones(LOCAL_OPERATOR_DIM), persistent=True
        )

    @torch.no_grad()
    def resize_num_gaussians(self, num_gaussians: int) -> None:
        """Resize consensus storage before checkpoint restoration.

        Stage-1 checkpoints may contain a densified Gaussian set that differs
        from the COLMAP seed count used to construct the model.
        """

        num_gaussians = int(num_gaussians)
        if num_gaussians < 1:
            raise ValueError("Stage-2 consensus requires at least one Gaussian")
        if self.local_mean.shape[0] == num_gaussians:
            return
        device, dtype = self.local_mean.device, self.local_mean.dtype
        self.local_mean = torch.zeros(
            num_gaussians, LOCAL_OPERATOR_DIM, device=device, dtype=dtype
        )
        self.local_weight = torch.zeros(
            num_gaussians, 1, device=device, dtype=dtype
        )
        self.local_views = torch.zeros(
            num_gaussians, 1, device=device, dtype=torch.long
        )

    @torch.no_grad()
    def set_global(self, view_operators: Tensor) -> Tensor:
        target, dispersion = robust_scene_operator_consensus(view_operators)
        self.global_target.copy_(target.to(self.global_target))
        self.global_dispersion.copy_(dispersion.to(self.global_dispersion))
        self.global_ready.fill_(True)
        return target

    @torch.no_grad()
    def set_local_scale(self, scale: Tensor) -> None:
        scale = torch.as_tensor(scale, device=self.local_scale.device, dtype=self.local_scale.dtype)
        if scale.shape != (LOCAL_OPERATOR_DIM,):
            raise ValueError(
                f"Expected local robust scale [{LOCAL_OPERATOR_DIM}], got {tuple(scale.shape)}"
            )
        self.local_scale.copy_(scale.clamp_min(1.0e-4))

    @staticmethod
    def _upsample_guidance(
        local_operator: Tensor,
        confidence: Tensor,
        *,
        height: int,
        width: int,
    ) -> tuple[Tensor, Tensor]:
        if local_operator.ndim != 4 or local_operator.shape[0] != 1:
            raise ValueError(
                "Expected one local operator grid [1,Hg,Wg,C], got "
                f"{tuple(local_operator.shape)}"
            )
        if local_operator.shape[-1] != LOCAL_OPERATOR_DIM:
            raise ValueError(
                f"Expected {LOCAL_OPERATOR_DIM} local channels, got {local_operator.shape[-1]}"
            )
        if confidence.ndim == 3:
            confidence = confidence[..., None]
        if confidence.shape != (*local_operator.shape[:3], 1):
            raise ValueError(
                "Local confidence must match the operator grid, got "
                f"{tuple(confidence.shape)}"
            )
        operator = F.interpolate(
            local_operator.permute(0, 3, 1, 2),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)
        weight = F.interpolate(
            confidence.permute(0, 3, 1, 2),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0).clamp_min(0.0)
        return operator, weight

    def backproject_and_update(
        self,
        local_operator: Tensor,
        confidence: Tensor,
        projection: Mapping[str, Tensor],
        *,
        width: int,
        height: int,
        tile_size: int,
        gaussian_indices: Tensor,
    ) -> dict[str, Tensor]:
        """Back-project one view with exact frozen raster visibility.

        The gradient of a linear splat with respect to its input feature is the
        transpose visibility operator.  Applying it to the view target gives a
        per-Gaussian visibility-weighted numerator without an approximate depth
        test, a view embedding, or geometry gradients.
        """

        operator, weight_map = self._upsample_guidance(
            local_operator,
            confidence,
            height=int(height),
            width=int(width),
        )
        count = int(projection["means2d"].shape[1])
        if gaussian_indices.shape != (count,):
            raise ValueError(
                f"Expected {count} projected Gaussian indices, got {tuple(gaussian_indices.shape)}"
            )
        with torch.enable_grad():
            probe = torch.zeros(
                count,
                LOCAL_OPERATOR_DIM + 1,
                device=operator.device,
                dtype=operator.dtype,
                requires_grad=True,
            )
            rendered, _, _ = rasterize_clear_from_underwater_meta(
                probe,
                dict(projection),
                int(width),
                int(height),
                tile_size=int(tile_size),
                return_expected_depth=False,
                detach_geometry=True,
            )
            adjoint_target = torch.cat(
                (operator * weight_map, weight_map), dim=-1
            )
            objective = (rendered[0] * adjoint_target).sum()
            backprojected = torch.autograd.grad(
                objective,
                probe,
                create_graph=False,
                retain_graph=False,
            )[0].detach()
        view_weight = backprojected[:, -1:].clamp_min(0.0)
        observation = (
            backprojected[:, :LOCAL_OPERATOR_DIM]
            / view_weight.clamp_min(1.0e-8)
        )
        valid = view_weight[:, 0] > 1.0e-8
        full_index = gaussian_indices[valid].long()
        if full_index.numel() == 0:
            return {
                "visible_gaussians": view_weight.new_zeros(()),
                "mean_weight": view_weight.new_zeros(()),
            }

        value = observation[valid].to(self.local_mean)
        base_weight = view_weight[valid].to(self.local_weight)
        old_mean = self.local_mean[full_index]
        old_weight = self.local_weight[full_index]
        old_views = self.local_views[full_index]

        # Paired-data IQR defines the units.  Cauchy influence gives robust
        # multi-view fusion without a scene-specific threshold.
        standardized = (value - old_mean) / self.local_scale.to(value)
        robust_weight = 1.0 / (1.0 + standardized.square().mean(dim=-1, keepdim=True))
        robust_weight = torch.where(old_views >= 2, robust_weight, torch.ones_like(robust_weight))
        effective_weight = base_weight * robust_weight
        total_weight = old_weight + effective_weight
        updated = old_mean + effective_weight / total_weight.clamp_min(1.0e-8) * (
            value - old_mean
        )
        self.local_mean[full_index] = updated
        self.local_weight[full_index] = total_weight
        self.local_views[full_index] = old_views + 1
        return {
            "visible_gaussians": view_weight.new_tensor(float(full_index.numel())),
            "mean_weight": effective_weight.mean().to(view_weight),
        }

    def rasterized_local_target(
        self,
        projection: Mapping[str, Tensor],
        *,
        width: int,
        height: int,
        tile_size: int,
        gaussian_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Render the fused Gaussian target into the current training view."""

        ready = (self.local_views[gaussian_indices] >= 2).to(self.local_mean)
        local = self.local_mean[gaussian_indices] * ready
        features = torch.cat((local, ready), dim=-1)
        rendered, _, _ = rasterize_clear_from_underwater_meta(
            features,
            dict(projection),
            int(width),
            int(height),
            tile_size=int(tile_size),
            return_expected_depth=False,
            detach_geometry=True,
        )
        support = rendered[0, ..., -1:].clamp(0.0, 1.0)
        target = rendered[0, ..., :LOCAL_OPERATOR_DIM] / support.clamp_min(1.0e-8)
        return target, support


__all__ = ["MultiViewOperatorConsensus"]
