# Adapted from gsplat v1.5.3 under Apache-2.0 and modified for 3D-USE.
"""Autograd wrapper for the project-owned underwater pixel compositor.

Projection, culling, tile intersection, and sorting stay in official gsplat.
The same project CUDA module also provides the SH decoder while avoiding
duplicate registration of gsplat's standard pybind types.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        from ._backend import _C

        if _C is None:
            raise RuntimeError("3D-USE CUDA extension is unavailable")
        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


def rasterize_to_pixels_underwater(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    medium_rgb: Tensor,
    beta_bs: Tensor,
    beta_attn: Tensor,
    far_depths: Tensor,
    ray_depth_scales: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    masks: Optional[Tensor] = None,
    packed: bool = False,
    absgrad: bool = False,
    optical_depth_grad_scale: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Composite sorted RGB Gaussians through the per-pixel water medium.

    Returns ``(rgb, object, clear, medium, alpha, depth_accum)``. The operator
    supports float32 only; the public Stage-1 renderer owns projection/sorting.
    """
    image_dims = tuple(isect_offsets.shape[:-2])
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape == (nnz, 3), colors.shape
        assert opacities.shape == (nnz,), opacities.shape
        assert depths.shape == (nnz,), depths.shape
    else:
        n_gaussians = means2d.size(-2)
        assert means2d.shape == image_dims + (n_gaussians, 2), means2d.shape
        assert conics.shape == image_dims + (n_gaussians, 3), conics.shape
        assert colors.shape == image_dims + (n_gaussians, 3), colors.shape
        assert opacities.shape == image_dims + (n_gaussians,), opacities.shape
        assert depths.shape == image_dims + (n_gaussians,), depths.shape

    rgb_shape = image_dims + (image_height, image_width, 3)
    scalar_shape = image_dims + (image_height, image_width, 1)
    assert medium_rgb.shape == rgb_shape, medium_rgb.shape
    assert beta_bs.shape == rgb_shape, beta_bs.shape
    assert beta_attn.shape == rgb_shape, beta_attn.shape
    if far_depths.shape == image_dims + (image_height, image_width):
        far_depths = far_depths.unsqueeze(-1)
    assert far_depths.shape == scalar_shape, far_depths.shape
    assert ray_depth_scales.shape == scalar_shape, ray_depth_scales.shape
    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()
    if optical_depth_grad_scale < 0.0:
        raise ValueError("optical_depth_grad_scale must be non-negative")

    tensors = (
        means2d,
        conics,
        colors,
        opacities,
        depths,
        medium_rgb,
        beta_bs,
        beta_attn,
        far_depths,
        ray_depth_scales,
    )
    if any(t.dtype != torch.float32 for t in tensors):
        raise TypeError("underwater rasterization currently supports float32 only")

    tile_height, tile_width = isect_offsets.shape[-2:]
    assert tile_height * tile_size >= image_height
    assert tile_width * tile_size >= image_width

    return _RasterizeToPixelsUnderwater.apply(
        *(tensor.contiguous() for tensor in tensors),
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
        float(optical_depth_grad_scale),
    )


class _RasterizeToPixelsUnderwater(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        conics: Tensor,
        colors: Tensor,
        opacities: Tensor,
        depths: Tensor,
        medium_rgb: Tensor,
        beta_bs: Tensor,
        beta_attn: Tensor,
        far_depths: Tensor,
        ray_depth_scales: Tensor,
        masks: Optional[Tensor],
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        absgrad: bool,
        optical_depth_grad_scale: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        outputs = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_underwater_fwd")(
            means2d,
            conics,
            colors,
            opacities,
            depths,
            medium_rgb,
            beta_bs,
            beta_attn,
            far_depths,
            ray_depth_scales,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )
        (
            render_rgb,
            render_object,
            render_clear,
            render_medium,
            render_medium_weight,
            render_alphas,
            render_depth,
            last_ids,
        ) = outputs
        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            depths,
            medium_rgb,
            beta_bs,
            beta_attn,
            far_depths,
            ray_depth_scales,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            render_medium_weight,
        )
        ctx.masks = masks
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.optical_depth_grad_scale = optical_depth_grad_scale
        return (
            render_rgb,
            render_object,
            render_clear,
            render_medium,
            render_alphas.float(),
            render_depth,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_rgb: Optional[Tensor],
        v_render_object: Optional[Tensor],
        v_render_clear: Optional[Tensor],
        v_render_medium: Optional[Tensor],
        v_render_alphas: Optional[Tensor],
        v_render_depth: Optional[Tensor],
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            depths,
            medium_rgb,
            beta_bs,
            beta_attn,
            far_depths,
            ray_depth_scales,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            render_medium_weight,
        ) = ctx.saved_tensors

        def grad_or_zeros(grad: Optional[Tensor], like: Tensor) -> Tensor:
            return torch.zeros_like(like) if grad is None else grad.contiguous()

        image_dims = tuple(isect_offsets.shape[:-2])
        rgb_like = medium_rgb.new_empty(image_dims + (ctx.height, ctx.width, 3))
        scalar_like = far_depths.new_empty(image_dims + (ctx.height, ctx.width, 1))
        gradients = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_underwater_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            depths,
            medium_rgb,
            beta_bs,
            beta_attn,
            far_depths,
            ray_depth_scales,
            ctx.masks,
            ctx.width,
            ctx.height,
            ctx.tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            render_medium_weight,
            grad_or_zeros(v_render_rgb, rgb_like),
            grad_or_zeros(v_render_object, rgb_like),
            grad_or_zeros(v_render_clear, rgb_like),
            grad_or_zeros(v_render_medium, rgb_like),
            grad_or_zeros(v_render_alphas, scalar_like),
            grad_or_zeros(v_render_depth, scalar_like),
            ctx.absgrad,
            ctx.optical_depth_grad_scale,
        )
        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_depths,
            v_medium_rgb,
            v_beta_bs,
            v_beta_attn,
        ) = gradients
        if ctx.absgrad:
            means2d.absgrad = v_means2d_abs

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_depths,
            v_medium_rgb,
            v_beta_bs,
            v_beta_attn,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


__all__ = ["rasterize_to_pixels_underwater"]
