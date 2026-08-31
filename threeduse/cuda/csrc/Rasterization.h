#pragma once

#include <cstdint>
#include <ATen/core/Tensor.h>

namespace gsplat {

void launch_rasterize_to_pixels_3dgs_underwater_fwd_kernel(
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
    at::Tensor render_rgb,
    at::Tensor render_object,
    at::Tensor render_clear,
    at::Tensor render_medium,
    at::Tensor render_medium_weight,
    at::Tensor render_alphas,
    at::Tensor render_depth,
    at::Tensor last_ids
);

void launch_rasterize_to_pixels_3dgs_underwater_bwd_kernel(
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
    const float optical_depth_grad_scale,
    at::optional<at::Tensor> v_means2d_abs,
    at::Tensor v_means2d,
    at::Tensor v_conics,
    at::Tensor v_colors,
    at::Tensor v_opacities,
    at::Tensor v_depths,
    at::Tensor v_medium_rgb,
    at::Tensor v_beta_bs,
    at::Tensor v_beta_attn
);

} // namespace gsplat
