#pragma once

#include <cstdint>
#include <tuple>
#include <ATen/core/Tensor.h>

namespace gsplat {

// Rasterize RGB Gaussians with a homogeneous per-pixel underwater medium and
// an infinite residual water column. ``far_depths`` is retained as an ABI
// placeholder and must carry the negative sentinel enforced by the caller.
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
rasterize_to_pixels_3dgs_underwater_fwd(
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::Tensor depths,
    const at::Tensor medium_rgb,
    const at::Tensor beta_bs,
    const at::Tensor beta_attn,
    const at::Tensor far_depths,
    const at::Tensor ray_depth_scales,
    const at::optional<at::Tensor> masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids
);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
rasterize_to_pixels_3dgs_underwater_bwd(
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::Tensor depths,
    const at::Tensor medium_rgb,
    const at::Tensor beta_bs,
    const at::Tensor beta_attn,
    const at::Tensor far_depths,
    const at::Tensor ray_depth_scales,
    const at::optional<at::Tensor> masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    const at::Tensor render_alphas,
    const at::Tensor last_ids,
    const at::Tensor render_medium_weight,
    const at::Tensor v_render_rgb,
    const at::Tensor v_render_object,
    const at::Tensor v_render_clear,
    const at::Tensor v_render_medium,
    const at::Tensor v_render_alphas,
    const at::Tensor v_render_depth,
    const bool absgrad,
    const float optical_depth_grad_scale
);

} // namespace gsplat
