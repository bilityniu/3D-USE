"""Modern gsplat projection/intersection with 3D-USE underwater compositing."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor


def prepare_underwater_gsplat_projection(
    means: Tensor,
    scales: Tensor,
    quats: Tensor,
    opacities: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    *,
    tile_size: int = 16,
    near_plane: float = 0.01,
    projection_far_plane: float = 1.0e4,
    eps2d: float = 0.3,
    radius_clip: float = 0.0,
    antialiased: bool = False,
 ) -> Dict[str, Tensor]:
    """Prepare one-camera projection, intersections, and depth ordering only.

    MediumRBF/SH decoding remains outside CUDA. Official gsplat v1.5.3
    performs projection, culling, tile intersection, and depth sorting. The
    project-owned CUDA extension consumes those sorted intersections.
    """
    try:
        from gsplat.cuda._wrapper import (
            fully_fused_projection,
            isect_offset_encode,
            isect_tiles,
        )
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "stage1_renderer='gsplat_underwater' requires the locked gsplat v1.5.3 runtime"
        ) from exc
    if viewmat.shape == (4, 4):
        viewmats = viewmat.unsqueeze(0)
    else:
        assert viewmat.shape == (1, 4, 4), viewmat.shape
        viewmats = viewmat
    if K.shape == (3, 3):
        Ks = K.unsqueeze(0)
    else:
        assert K.shape == (1, 3, 3), K.shape
        Ks = K

    assert means.ndim == 2 and means.shape[-1] == 3, means.shape
    assert scales.shape == means.shape, scales.shape
    assert quats.shape == (means.shape[0], 4), quats.shape
    assert opacities.shape == (means.shape[0],), opacities.shape

    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=projection_far_plane,
        radius_clip=radius_clip,
        packed=False,
        sparse_grad=False,
        calc_compensations=antialiased,
        camera_model="pinhole",
        opacities=opacities,
    )

    projected_opacities = opacities.unsqueeze(0)
    if compensations is not None:
        projected_opacities = projected_opacities * compensations
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
    )
    isect_offsets = isect_offset_encode(
        isect_ids, 1, tile_width, tile_height
    ).reshape(1, tile_height, tile_width)

    if radii.ndim == means2d.ndim and radii.shape[-1] == 2:
        visible = (radii > 0).all(dim=-1)
    else:
        visible = radii > 0
    meta = {
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": projected_opacities,
        "compensations": compensations,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
    }
    return meta


def rasterize_underwater_gsplat(
    means: Tensor, scales: Tensor, quats: Tensor, opacities: Tensor, colors: Tensor,
    viewmat: Tensor, K: Tensor, medium_rgb: Tensor, beta_bs: Tensor, beta_attn: Tensor,
    width: int, height: int, *, far_depths: Tensor | None = None,
    ray_depth_scales: Tensor | None = None, tile_size: int = 16,
    near_plane: float = 0.01, projection_far_plane: float = 1.0e4,
    eps2d: float = 0.3, radius_clip: float = 0.0, antialiased: bool = False,
    absgrad: bool = True, optical_depth_grad_scale: float = 0.0,
) -> Tuple[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Dict[str, Tensor]]:
    """Project/sort once and run the Stage-1 underwater pixel compositor."""
    from threeduse.cuda._wrapper import rasterize_to_pixels_underwater

    assert colors.shape == (means.shape[0], 3), colors.shape
    assert medium_rgb.shape == (height, width, 3), medium_rgb.shape
    assert beta_bs.shape == medium_rgb.shape, beta_bs.shape
    assert beta_attn.shape == medium_rgb.shape, beta_attn.shape
    meta = prepare_underwater_gsplat_projection(
        means, scales, quats, opacities, viewmat, K, width, height,
        tile_size=tile_size, near_plane=near_plane,
        projection_far_plane=projection_far_plane, eps2d=eps2d,
        radius_clip=radius_clip, antialiased=antialiased,
    )
    depths = meta["depths"]
    if ray_depth_scales is None:
        ray_depth_scales = torch.ones((height, width, 1), device=depths.device, dtype=depths.dtype)
    if far_depths is None:
        far_depths = -torch.ones((height, width, 1), device=depths.device, dtype=depths.dtype)
    if far_depths.shape != (height, width, 1) or not bool((far_depths < 0).all()):
        raise ValueError("3D-USE Stage 1 supports only infinite-tail compositing")
    assert ray_depth_scales.shape == (height, width, 1), ray_depth_scales.shape
    outputs = rasterize_to_pixels_underwater(
        meta["means2d"], meta["conics"], colors.unsqueeze(0), meta["opacities"],
        depths, medium_rgb.unsqueeze(0), beta_bs.unsqueeze(0), beta_attn.unsqueeze(0),
        far_depths.unsqueeze(0), ray_depth_scales.unsqueeze(0), width, height, tile_size,
        meta["isect_offsets"], meta["flatten_ids"], packed=False, absgrad=absgrad,
        optical_depth_grad_scale=optical_depth_grad_scale,
    )
    return outputs, meta


def rasterize_clear_from_underwater_meta(
    colors: Tensor,
    meta: Dict[str, Tensor],
    width: int,
    height: int,
    *,
    tile_size: int = 16,
    return_expected_depth: bool = True,
    detach_geometry: bool = True,
    geometry_grad_scale: float | None = None,
    opacity_grad_scale: float | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Render a standard-gsplat clear head from an existing Stage-1 context.

    ``rasterize_underwater_gsplat`` already exposes the official gsplat v1.5.3
    projection, tile-intersection and depth-sorting buffers in ``meta``.  This
    helper reuses those buffers and launches only the official clear pixel
    compositor for a second set of Gaussian colors.  No attenuation,
    backscatter, water-column tail, or image-space post-processing is applied.

    The reconstruction and enhancement heads reuse one projection/sort.
    ``detach_geometry=True`` makes this head update appearance only while the
    underwater head retains exclusive responsibility for geometry and opacity.

    Returns ``(clear, alpha, expected_depth)`` with a leading camera dimension.
    ``expected_depth`` is zero on empty pixels.
    """

    try:
        from gsplat.cuda._wrapper import rasterize_to_pixels
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "rasterize_clear_from_underwater_meta requires the locked gsplat v1.5.3 runtime"
        ) from exc

    required = (
        "means2d",
        "conics",
        "depths",
        "opacities",
        "isect_offsets",
        "flatten_ids",
    )
    missing = [name for name in required if name not in meta]
    if missing:
        raise ValueError(f"Underwater raster context is missing fields: {missing}")
    if colors.ndim != 2 or colors.shape[-1] <= 0:
        raise ValueError(f"Expected Gaussian features [N,C], got {tuple(colors.shape)}")

    means2d = meta["means2d"]
    conics = meta["conics"]
    depths = meta["depths"]
    opacities = meta["opacities"]
    if means2d.ndim != 3 or means2d.shape[0] != 1:
        raise ValueError(f"Expected unpacked one-camera means2d [1,N,2], got {tuple(means2d.shape)}")
    if colors.shape[0] != means2d.shape[1]:
        raise ValueError(
            f"Color count {colors.shape[0]} does not match projected Gaussians {means2d.shape[1]}"
        )

    projected_colors = colors.unsqueeze(0)
    if geometry_grad_scale is None:
        geometry_grad_scale = 0.0 if detach_geometry else 1.0
    if opacity_grad_scale is None:
        opacity_grad_scale = 0.0 if detach_geometry else 1.0
    geometry_grad_scale = max(0.0, min(1.0, float(geometry_grad_scale)))
    opacity_grad_scale = max(0.0, min(1.0, float(opacity_grad_scale)))

    def scaled_gradient(tensor: Tensor, scale: float) -> Tensor:
        detached = tensor.detach()
        return detached + scale * (tensor - detached)

    means2d = scaled_gradient(means2d, geometry_grad_scale)
    conics = scaled_gradient(conics, geometry_grad_scale)
    depths = scaled_gradient(depths, geometry_grad_scale)
    opacities = scaled_gradient(opacities, opacity_grad_scale)

    feature_channels = int(projected_colors.shape[-1])
    if return_expected_depth:
        projected_colors = torch.cat([projected_colors, depths.unsqueeze(-1)], dim=-1)
    render, alpha = rasterize_to_pixels(
        means2d,
        conics,
        projected_colors,
        opacities,
        width,
        height,
        tile_size,
        meta["isect_offsets"],
        meta["flatten_ids"],
        backgrounds=None,
        packed=False,
        absgrad=False,
    )
    clear = render[..., :feature_channels]
    if return_expected_depth:
        depth_accum = render[..., feature_channels : feature_channels + 1]
        expected_depth = torch.where(
            alpha > 1e-6,
            depth_accum / alpha.clamp_min(1e-6),
            torch.zeros_like(depth_accum),
        )
    else:
        expected_depth = clear.new_zeros(*clear.shape[:-1], 1)
    return clear, alpha, expected_depth


def rasterize_object_from_underwater_meta(
    colors: Tensor,
    meta: Dict[str, Tensor],
    beta_attn: Tensor,
    far_depths: Tensor,
    ray_depth_scales: Tensor,
    width: int,
    height: int,
    *,
    tile_size: int = 16,
    detach_geometry: bool = True,
    geometry_grad_scale: float | None = None,
    opacity_grad_scale: float | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Re-render only the attenuated object term from a Stage-1 context.

    Projection, intersection ordering, opacity, depth and ``beta_attn`` are
    exactly those of the frozen underwater reconstruction.  Only Gaussian RGB
    is replaced.  The returned image is therefore in the same domain as
    Stage-1 ``rgb_object``; neither backscatter nor the clear head is included.

    Returns ``(object, alpha, expected_depth)`` with a leading camera
    dimension.  No clamp or display tone mapping is applied.
    """

    from threeduse.cuda._wrapper import rasterize_to_pixels_underwater

    required = (
        "means2d",
        "conics",
        "depths",
        "opacities",
        "isect_offsets",
        "flatten_ids",
    )
    missing = [name for name in required if name not in meta]
    if missing:
        raise ValueError(f"Underwater raster context is missing fields: {missing}")
    if colors.ndim != 2 or colors.shape[-1] != 3:
        raise ValueError(f"Expected Gaussian RGB [N,3], got {tuple(colors.shape)}")
    if beta_attn.shape != (height, width, 3):
        raise ValueError(f"Expected beta_attn [H,W,3], got {tuple(beta_attn.shape)}")
    if far_depths.shape != (height, width, 1):
        raise ValueError(f"Expected far_depths [H,W,1], got {tuple(far_depths.shape)}")
    if ray_depth_scales.shape != (height, width, 1):
        raise ValueError(
            f"Expected ray_depth_scales [H,W,1], got {tuple(ray_depth_scales.shape)}"
        )

    means2d = meta["means2d"]
    conics = meta["conics"]
    depths = meta["depths"]
    opacities = meta["opacities"]
    if means2d.ndim != 3 or means2d.shape[0] != 1:
        raise ValueError(f"Expected unpacked one-camera means2d [1,N,2], got {tuple(means2d.shape)}")
    if colors.shape[0] != means2d.shape[1]:
        raise ValueError(
            f"Color count {colors.shape[0]} does not match projected Gaussians {means2d.shape[1]}"
        )

    if geometry_grad_scale is None:
        geometry_grad_scale = 0.0 if detach_geometry else 1.0
    if opacity_grad_scale is None:
        opacity_grad_scale = 0.0 if detach_geometry else 1.0
    geometry_grad_scale = max(0.0, min(1.0, float(geometry_grad_scale)))
    opacity_grad_scale = max(0.0, min(1.0, float(opacity_grad_scale)))

    def scaled_gradient(tensor: Tensor, scale: float) -> Tensor:
        detached = tensor.detach()
        return detached + scale * (tensor - detached)

    means2d = scaled_gradient(means2d, geometry_grad_scale)
    conics = scaled_gradient(conics, geometry_grad_scale)
    depths = scaled_gradient(depths, geometry_grad_scale)
    opacities = scaled_gradient(opacities, opacity_grad_scale)
    projected_colors = colors.unsqueeze(0)
    frozen_attn = beta_attn.detach().unsqueeze(0)
    zeros = torch.zeros_like(frozen_attn)

    _, object_rgb, _, _, alpha, depth_accum = rasterize_to_pixels_underwater(
        means2d,
        conics,
        projected_colors,
        opacities,
        depths,
        zeros,
        zeros,
        frozen_attn,
        far_depths.detach().unsqueeze(0),
        ray_depth_scales.detach().unsqueeze(0),
        width,
        height,
        tile_size,
        meta["isect_offsets"],
        meta["flatten_ids"],
        packed=False,
        absgrad=False,
        optical_depth_grad_scale=0.0,
    )
    expected_depth = torch.where(
        alpha > 1e-6,
        depth_accum / alpha.clamp_min(1e-6),
        torch.zeros_like(depth_accum),
    )
    return object_rgb, alpha, expected_depth
