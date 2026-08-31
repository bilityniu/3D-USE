"""Persistent Stage-2 Gaussian appearance state and compiler contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class CompiledAppearance:
    """One view's Gaussian colors generated from persistent 3D state."""

    source_linear: Tensor
    enhanced_linear: Tensor
    endpoint_linear: Tensor
    auxiliary: dict[str, Tensor]


class Stage2AppearanceCompiler(nn.Module):
    """Own the trainable U-BAF state independently of reconstruction."""

    def __init__(
        self,
        field: nn.Module,
        *,
        coordinate_quantile: float,
        cache_compiled: bool,
        max_transport_log_ratio: float = 1.3862943611198906,
    ) -> None:
        super().__init__()
        self.field = field
        self.coordinate_quantile = float(coordinate_quantile)
        self.cache_compiled = bool(cache_compiled)
        if max_transport_log_ratio <= 0.0:
            raise ValueError("max_transport_log_ratio must be positive")
        self.max_transport_log_ratio = float(max_transport_log_ratio)
        # Enhancement transport is an identity-centred residual over the
        # frozen Stage-1 optical fields.  A scene-level spectral ratio keeps
        # positivity, preserves Stage-1's learned spatial variation, and does
        # not introduce a camera/view embedding.
        # Transport owns visibility strength, not a second RGB grade. One
        # scalar per process preserves Stage-1's learned spectral ratios;
        # hue belongs to the calibrated scene/global appearance operator.
        self.transport_attn_log_ratio_raw = nn.Parameter(torch.zeros(1))
        self.transport_bs_log_ratio_raw = nn.Parameter(torch.zeros(1))
        self._compiled_cache: dict[str, Any] | None = None
        self._scene_bounds_ready = False

    def clear_cache(self) -> None:
        self._compiled_cache = None

    def train(self, mode: bool = True) -> "Stage2AppearanceCompiler":  # type: ignore[override]
        if mode:
            self.clear_cache()
        return super().train(mode)

    def enhance_transport(
        self,
        beta_bs: Tensor,
        beta_attn: Tensor,
        *,
        strength: float,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Create the independent Stage-2 underwater transport fields.

        Stage-1 optical fields are immutable sources for the Stage-2
        enhancement objective. The two learned scalar log ratios are
        persistent scene state shared by every training and novel camera.
        """

        if beta_bs.shape != beta_attn.shape or beta_bs.ndim != 3 or beta_bs.shape[-1] != 3:
            raise ValueError(
                "beta_bs and beta_attn must be matching [H,W,3] fields, got "
                f"{tuple(beta_bs.shape)} and {tuple(beta_attn.shape)}"
            )
        source_bs = beta_bs.detach()
        source_attn = beta_attn.detach()
        amount = max(0.0, min(1.0, float(strength)))
        bs_scalar = self.max_transport_log_ratio * torch.tanh(
            self.transport_bs_log_ratio_raw
        )
        attn_scalar = self.max_transport_log_ratio * torch.tanh(
            self.transport_attn_log_ratio_raw
        )
        bs_log_ratio = amount * bs_scalar.to(source_bs).expand(3)
        attn_log_ratio = amount * attn_scalar.to(source_attn).expand(3)
        enhanced_bs = source_bs * torch.exp(bs_log_ratio)[None, None]
        enhanced_attn = source_attn * torch.exp(attn_log_ratio)[None, None]
        return enhanced_bs, enhanced_attn, {
            "transport_bs_log_ratio": bs_log_ratio,
            "transport_attn_log_ratio": attn_log_ratio,
            "transport_bs_ratio_mean": torch.exp(bs_log_ratio).mean(),
            "transport_attn_ratio_mean": torch.exp(attn_log_ratio).mean(),
        }

    def _ensure_scene_bounds(self, means: Tensor) -> None:
        if self.field is None or not hasattr(self.field, "set_scene_bounds"):
            return
        if self._scene_bounds_ready:
            return
        ready = getattr(self.field, "scene_bounds_ready", None)
        if ready is None or not bool(ready.detach().cpu().item()):
            self.field.set_scene_bounds(
                means.detach(),
                quantile=self.coordinate_quantile,
            )
        self._scene_bounds_ready = True

    @staticmethod
    def _apply_affine(rgb: Tensor, matrix: Tensor, bias: Tensor) -> Tensor:
        return torch.einsum("...ij,...j->...i", matrix, rgb) + bias

    def _cache_key(self, means: Tensor, base_rgb: Tensor, cache_token: int) -> tuple[Any, ...]:
        return (
            int(means.data_ptr()),
            int(means.shape[0]),
            str(means.device),
            str(base_rgb.dtype),
            int(cache_token),
        )

    def forward(
        self,
        raw_rgb: Tensor,
        base_rgb: Tensor,
        means: Tensor,
        *,
        strength: float,
        cache_token: int = 0,
    ) -> CompiledAppearance:
        if raw_rgb.shape != base_rgb.shape or raw_rgb.ndim != 2 or raw_rgb.shape[-1] != 3:
            raise ValueError(
                "raw_rgb and base_rgb must be matching [N,3] tensors, got "
                f"{tuple(raw_rgb.shape)} and {tuple(base_rgb.shape)}"
            )
        if means.shape != (raw_rgb.shape[0], 3):
            raise ValueError(
                f"means must match Gaussian appearance [N,3], got {tuple(means.shape)}"
            )
        if not torch.isfinite(raw_rgb).all():
            raise FloatingPointError("Non-finite reconstruction appearance entering Stage-2")
        source_linear = raw_rgb
        # Both tensors remain in linear radiance space.  In particular, the
        # view-independent DC appearance used as the fourth field coordinate
        # is not silently clipped before compilation.
        source_guide = raw_rgb.detach()
        base_guide = base_rgb.detach()

        self._ensure_scene_bounds(means)
        key = self._cache_key(means, base_guide, cache_token)
        use_cache = self.cache_compiled and not torch.is_grad_enabled()
        cached = self._compiled_cache
        if use_cache and cached is not None and cached["key"] == key:
            auxiliary = dict(cached["auxiliary"])
            matrix = cached["matrix"]
            bias = cached["bias"]
            endpoint = self._apply_affine(source_linear, matrix, bias)
            base_endpoint = self._apply_affine(base_guide, matrix, bias)
            operator_log_gain = torch.log(
                torch.diagonal(matrix, dim1=-2, dim2=-1).clamp_min(1e-8)
            )
            identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
            auxiliary.update(
                {
                    "affine_matrix": matrix,
                    "affine_bias": bias,
                    "operator_log_gain": operator_log_gain,
                    "operator_matrix_delta": matrix - identity,
                }
            )
        else:
            _, auxiliary = self.field(means.detach(), base_guide)
            matrix = auxiliary["affine_matrix"]
            bias = auxiliary["affine_bias"]
            endpoint = self._apply_affine(source_linear, matrix, bias)
            base_endpoint = self._apply_affine(base_guide, matrix, bias)
            if use_cache:
                cached_auxiliary = {
                    name: value.detach()
                    for name, value in auxiliary.items()
                    if torch.is_tensor(value) and value.numel() == 1
                }
                cached_auxiliary["appearance_guide"] = auxiliary[
                    "appearance_guide"
                ].detach()
                if "local_log_gain_center" in auxiliary:
                    cached_auxiliary["local_log_gain_center"] = auxiliary[
                        "local_log_gain_center"
                    ].detach()
                if "local_bias_center" in auxiliary:
                    cached_auxiliary["local_bias_center"] = auxiliary[
                        "local_bias_center"
                    ].detach()
                cache: dict[str, Any] = {
                    "key": key,
                    "auxiliary": cached_auxiliary,
                    "matrix": matrix.detach(),
                    "bias": bias.detach(),
                }
                self._compiled_cache = cache

        strength = max(0.0, min(1.0, float(strength)))
        enhanced = source_linear + strength * (endpoint - source_linear)
        auxiliary.update(
            {
                "delta_rgb": endpoint - source_linear,
                "endpoint_linear": endpoint,
                "source_linear": source_guide,
                "physical_source_rgb": source_linear,
                "endpoint_rgb": endpoint,
                "base_source_linear": base_guide,
                "base_endpoint_rgb": base_endpoint,
                "enhancement_strength": source_linear.new_tensor(strength),
            }
        )
        return CompiledAppearance(
            source_linear=source_linear,
            enhanced_linear=enhanced,
            endpoint_linear=endpoint,
            auxiliary=auxiliary,
        )
