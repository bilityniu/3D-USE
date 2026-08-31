"""Underwater Bilateral Appearance Field used by 3D-USE Stage 2."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SceneAppearanceField(nn.Module):
    """Underwater Bilateral Appearance Field (U-BAF).

    This low-rank 4D operator field represents one scene appearance transform,
    not independent object and medium edits. Gaussian radiance samples it at
    Gaussian world positions; MediumRBF samples it at the camera world
    position. Both use the same learned RGB guide and operator decoder, so the
    underwater compositor, rather than a hand-written visibility mask, routes
    the full-render supervision to the source appearance that produced a ray.

    The local field emits guide-dependent diagonal RGB gains, matching the
    global-affine/local-diagonal factorization of the distilled transition.
    Guide dependence supplies nonlinear tone/contrast behaviour while full
    cross-channel mixing stays in one shared global operator. The resulting
    transforms are applied to object and medium appearance before underwater
    compositing.
    """

    def __init__(
        self,
        *,
        grid_resolution: int = 16,
        rank: int = 12,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if grid_resolution < 4 or rank < 1 or hidden_dim < 8:
            raise ValueError("grid_resolution>=4, rank>=1 and hidden_dim>=8 are required")
        self.grid_resolution = int(grid_resolution)
        self.rank = int(rank)

        # A learned underwater RGB guide replaces fixed display-luminance
        # coefficients.  signed-log input remains defined for unclamped linear
        # radiance and tanh provides the bilateral range coordinate.
        self.guide_weight = nn.Parameter(torch.full((3,), 1.0 / 3.0))
        self.guide_bias = nn.Parameter(torch.zeros(()))

        # The global mode and every local mode are shared by object and medium.
        self.global_matrix_raw = nn.Parameter(torch.zeros(3, 3))
        self.global_bias_raw = nn.Parameter(torch.zeros(3))

        factor_shape = (1, self.rank, self.grid_resolution, 1)
        self.factor_x = nn.Parameter(torch.ones(factor_shape))
        self.factor_y = nn.Parameter(torch.ones(factor_shape))
        self.factor_z = nn.Parameter(torch.ones(factor_shape))
        self.factor_guide = nn.Parameter(torch.ones(factor_shape))
        self.operator_head = nn.Sequential(
            nn.Linear(self.rank, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )

        self.register_buffer("scene_position_min", torch.full((3,), -1.0))
        self.register_buffer("scene_position_max", torch.full((3,), 1.0))
        self.register_buffer("scene_bounds_ready", torch.tensor(False))

        with torch.no_grad():
            for factor in self.factors:
                factor.add_(0.01 * torch.randn_like(factor))
            nn.init.zeros_(self.operator_head[-1].weight)
            nn.init.zeros_(self.operator_head[-1].bias)

    @property
    def factors(self) -> tuple[Tensor, ...]:
        return self.factor_x, self.factor_y, self.factor_z, self.factor_guide

    @torch.no_grad()
    def set_scene_bounds(
        self,
        positions: Tensor,
        *,
        quantile: float = 0.01,
        max_samples: int = 200_000,
    ) -> None:
        """Fit one world-space domain for Gaussian and MediumRBF queries."""

        if positions.ndim != 2 or positions.shape[-1] != 3 or positions.shape[0] == 0:
            raise ValueError(
                f"Expected non-empty world positions [N,3], got {tuple(positions.shape)}"
            )
        stride = max(1, int(positions.shape[0]) // max(1, int(max_samples)))
        sample = positions.detach()[::stride].float()
        q = max(0.0, min(0.2, float(quantile)))
        lower = torch.quantile(sample, q, dim=0) if q > 0.0 else sample.amin(dim=0)
        upper = torch.quantile(sample, 1.0 - q, dim=0) if q > 0.0 else sample.amax(dim=0)
        padding = 0.02 * (upper - lower).clamp_min(1e-4)
        self.scene_position_min.copy_((lower - padding).to(self.scene_position_min))
        self.scene_position_max.copy_((upper + padding).to(self.scene_position_max))
        self.scene_bounds_ready.fill_(True)

    @staticmethod
    def _sample_factor(factor: Tensor, coordinate: Tensor) -> Tensor:
        grid = torch.stack((torch.zeros_like(coordinate), coordinate), dim=-1)
        sampled = F.grid_sample(
            factor,
            grid.reshape(1, -1, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(factor.shape[1], -1).transpose(0, 1)

    def _normalize_position(self, position: Tensor, dtype: torch.dtype) -> Tensor:
        lower = self.scene_position_min.to(device=position.device, dtype=dtype)
        upper = self.scene_position_max.to(device=position.device, dtype=dtype)
        return (
            2.0 * (position.to(dtype=dtype) - lower)
            / (upper - lower).clamp_min(1e-6)
            - 1.0
        ).clamp(-1.0, 1.0)

    def _learned_guide(self, rgb: Tensor) -> Tensor:
        signed_log = torch.sign(rgb) * torch.log1p(rgb.abs())
        projection = self.guide_weight.to(device=rgb.device, dtype=rgb.dtype)
        bias = self.guide_bias.to(device=rgb.device, dtype=rgb.dtype)
        return torch.tanh(torch.einsum("...c,c->...", signed_log, projection) + bias)

    def _global_affine(
        self,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        identity = torch.eye(
            3, device=self.global_matrix_raw.device, dtype=dtype
        )
        matrix = identity + self.global_matrix_raw.to(dtype=dtype)
        bias = self.global_bias_raw.to(dtype=dtype)
        return matrix, bias

    def _local_code_from_coordinates(
        self,
        position: Tensor,
        guide: Tensor,
    ) -> Tensor:
        if position.ndim != 2 or position.shape[-1] != 3:
            raise ValueError(f"Expected positions [N,3], got {tuple(position.shape)}")
        if guide.shape != position.shape[:-1]:
            raise ValueError(
                f"Guide shape {tuple(guide.shape)} does not match positions {tuple(position.shape)}"
            )
        dtype = guide.dtype
        normalized = self._normalize_position(position, dtype)
        return (
            self._sample_factor(self.factor_x.to(dtype=dtype), normalized[:, 0])
            * self._sample_factor(self.factor_y.to(dtype=dtype), normalized[:, 1])
            * self._sample_factor(self.factor_z.to(dtype=dtype), normalized[:, 2])
            * self._sample_factor(self.factor_guide.to(dtype=dtype), guide)
        )

    def _local_gain_from_coordinates(
        self,
        position: Tensor,
        guide: Tensor,
        *,
        log_gain_center: Tensor | None = None,
        bias_center: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        code = self._local_code_from_coordinates(position, guide)
        decoded = self.operator_head(code)
        raw_log_gain = decoded[..., :3]
        raw_bias = decoded[..., 3:]
        if log_gain_center is None:
            log_gain_center = raw_log_gain.mean(dim=0)
        if bias_center is None:
            bias_center = raw_bias.mean(dim=0)
        if log_gain_center.shape != (3,):
            raise ValueError(
                f"Expected one RGB local-gain center [3], got {tuple(log_gain_center.shape)}"
            )
        if bias_center.shape != (3,):
            raise ValueError(
                f"Expected one RGB local-bias center [3], got {tuple(bias_center.shape)}"
            )
        # Remove the constant mode from the local field.  This is a structural
        # gauge choice, not a penalty: the global affine exclusively owns the
        # scene-wide grade while the 4D field represents only residual spatial
        # and guide-dependent variation.
        log_gain = raw_log_gain - log_gain_center.to(raw_log_gain)
        local_bias = raw_bias - bias_center.to(raw_bias)
        return torch.exp(log_gain), log_gain, local_bias, log_gain_center, bias_center

    def _operator_from_coordinates(
        self,
        position: Tensor,
        guide: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        local_gain, log_gain, local_bias, log_gain_center, bias_center = self._local_gain_from_coordinates(
            position, guide
        )
        _, global_bias = self._global_affine(guide.dtype)
        # The local operator is diagonal. ``forward`` composes it with the
        # shared global affine before the transformed Gaussian radiance enters
        # underwater compositing.
        matrix = torch.diag_embed(local_gain)
        return matrix, log_gain, local_bias, global_bias, log_gain_center, bias_center

    def forward(
        self,
        means: Tensor,
        base_rgb: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not bool(self.scene_bounds_ready.detach().cpu().item()):
            self.set_scene_bounds(means)
        if base_rgb.shape != (means.shape[0], 3):
            raise ValueError(
                f"Expected base RGB {(means.shape[0], 3)}, got {tuple(base_rgb.shape)}"
            )
        guide = self._learned_guide(base_rgb)
        matrix, log_gain, local_bias, global_bias, log_gain_center, bias_center = self._operator_from_coordinates(
            means, guide
        )
        global_matrix = self._global_affine(base_rgb.dtype)[0]
        # Compile the complete global-then-local operator into Gaussian
        # radiance. No residual image-space affine is applied after rendering.
        affine_matrix = torch.einsum("nij,jk->nik", matrix, global_matrix)
        affine_bias = torch.einsum("nij,j->ni", matrix, global_bias) + local_bias
        renderer_global_matrix = torch.eye(
            3, device=base_rgb.device, dtype=base_rgb.dtype
        )
        renderer_global_bias = torch.zeros_like(global_bias)
        return log_gain, {
            "affine_matrix": affine_matrix,
            "affine_bias": affine_bias,
            "appearance_guide": guide[:, None],
            "operator_log_gain": log_gain,
            "local_log_gain_center": log_gain_center,
            "local_bias_center": bias_center,
            "global_matrix": renderer_global_matrix,
            "global_bias": renderer_global_bias,
            "gaussian_global_matrix": global_matrix,
            "gaussian_global_bias": global_bias,
        }

    def enhance_medium(
        self,
        medium_source: Tensor,
        *,
        camera_origin: Tensor,
        local_log_gain_center: Tensor | None = None,
        local_bias_center: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Slice the shared operator at the MediumRBF spatial argument.

        Only ``grid_resolution`` operator samples are decoded for a camera;
        full-resolution rays then use differentiable 1D bilateral slicing.
        This is the standard efficient bilateral-grid evaluation pattern and
        avoids a per-pixel MLP activation tensor during training.
        """

        if medium_source.ndim != 3 or medium_source.shape[-1] != 3:
            raise ValueError(
                f"Expected medium source [H,W,3], got {tuple(medium_source.shape)}"
            )
        if not bool(self.scene_bounds_ready.detach().cpu().item()):
            self.set_scene_bounds(camera_origin.reshape(1, 3))
        guide = self._learned_guide(medium_source)
        table_guide = torch.linspace(
            -1.0,
            1.0,
            self.grid_resolution,
            device=medium_source.device,
            dtype=medium_source.dtype,
        )
        table_position = camera_origin.reshape(1, 3).to(medium_source).expand(
            self.grid_resolution, 3
        )
        table_gain, table_log_gain, table_bias, _, _ = self._local_gain_from_coordinates(
            table_position,
            table_guide,
            log_gain_center=local_log_gain_center,
            bias_center=local_bias_center,
        )
        # ``grid_sample`` is the optimized bilateral slicing primitive.  The
        # coefficient table is a 1D image with RGB gain channels; the learned
        # guide supplies its vertical coordinate for every full-resolution ray.
        slice_grid = torch.stack((torch.zeros_like(guide), guide), dim=-1)[None]
        gain = F.grid_sample(
            table_gain.transpose(0, 1)[None, :, :, None],
            slice_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0].permute(1, 2, 0)
        bias = F.grid_sample(
            table_bias.transpose(0, 1)[None, :, :, None],
            slice_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0].permute(1, 2, 0)
        global_matrix, global_bias = self._global_affine(medium_source.dtype)
        identity = torch.eye(
            3, device=medium_source.device, dtype=medium_source.dtype
        )
        # Apply the same global-then-local transform used for Gaussian
        # radiance before underwater compositing.
        enhanced = gain * (
            torch.einsum("ij,...j->...i", global_matrix, medium_source)
            + global_bias
        ) + bias
        renderer_global_matrix = identity
        renderer_global_bias = torch.zeros_like(global_bias)
        return enhanced, {
            "medium_appearance_guide": guide,
            "medium_operator_log_gain": torch.log(gain.clamp_min(1e-8)),
            "medium_operator_bias": bias,
            "medium_operator_log_gain_abs_mean": table_log_gain.abs().mean(),
            "global_matrix": renderer_global_matrix,
            "global_bias": renderer_global_bias,
            "medium_global_matrix": global_matrix,
            "medium_global_bias": global_bias,
        }

    def tv_loss(self) -> Tensor:
        """Regularize adjacent samples of the low-rank 4D factors."""

        factor_tv = [
            (factor[:, :, 1:] - factor[:, :, :-1]).abs().mean()
            for factor in self.factors
        ]
        return torch.stack(factor_tv).mean()

__all__ = ["SceneAppearanceField"]
