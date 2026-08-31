// Adapted from gsplat v1.5.3 under Apache-2.0 and modified for 3D-USE.
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"
#include "Rasterization.h"

namespace gsplat {

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
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(depths);
    CHECK_INPUT(medium_rgb);
    CHECK_INPUT(beta_bs);
    CHECK_INPUT(beta_attn);
    CHECK_INPUT(far_depths);
    CHECK_INPUT(ray_depth_scales);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(colors.size(-1) == 3, "underwater rasterizer expects RGB");
    TORCH_CHECK(
        means2d.scalar_type() == at::kFloat &&
            conics.scalar_type() == at::kFloat &&
            colors.scalar_type() == at::kFloat &&
            opacities.scalar_type() == at::kFloat &&
            depths.scalar_type() == at::kFloat &&
            medium_rgb.scalar_type() == at::kFloat &&
            beta_bs.scalar_type() == at::kFloat &&
            beta_attn.scalar_type() == at::kFloat &&
            far_depths.scalar_type() == at::kFloat &&
            ray_depth_scales.scalar_type() == at::kFloat,
        "underwater rasterizer currently supports float32 only"
    );

    const auto opt = means2d.options();
    at::DimVector image_dims(
        tile_offsets.sizes().slice(0, tile_offsets.dim() - 2)
    );

    at::DimVector rgb_dims(image_dims);
    rgb_dims.append({image_height, image_width, 3});
    at::Tensor render_rgb = at::empty(rgb_dims, opt);
    at::Tensor render_object = at::empty(rgb_dims, opt);
    at::Tensor render_clear = at::empty(rgb_dims, opt);
    at::Tensor render_medium = at::empty(rgb_dims, opt);
    at::Tensor render_medium_weight = at::empty(rgb_dims, opt);

    at::DimVector scalar_dims(image_dims);
    scalar_dims.append({image_height, image_width, 1});
    at::Tensor render_alphas = at::empty(scalar_dims, opt);
    at::Tensor render_depth = at::empty(scalar_dims, opt);

    at::DimVector id_dims(image_dims);
    id_dims.append({image_height, image_width});
    at::Tensor last_ids = at::empty(id_dims, opt.dtype(at::kInt));

    launch_rasterize_to_pixels_3dgs_underwater_fwd_kernel(
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
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids,
        render_rgb,
        render_object,
        render_clear,
        render_medium,
        render_medium_weight,
        render_alphas,
        render_depth,
        last_ids
    );

    return std::make_tuple(
        render_rgb,
        render_object,
        render_clear,
        render_medium,
        render_medium_weight,
        render_alphas,
        render_depth,
        last_ids
    );
}

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
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(depths);
    CHECK_INPUT(medium_rgb);
    CHECK_INPUT(beta_bs);
    CHECK_INPUT(beta_attn);
    CHECK_INPUT(far_depths);
    CHECK_INPUT(ray_depth_scales);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(render_medium_weight);
    CHECK_INPUT(v_render_rgb);
    CHECK_INPUT(v_render_object);
    CHECK_INPUT(v_render_clear);
    CHECK_INPUT(v_render_medium);
    CHECK_INPUT(v_render_alphas);
    CHECK_INPUT(v_render_depth);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(
        optical_depth_grad_scale >= 0.f,
        "optical_depth_grad_scale must be non-negative"
    );

    at::Tensor v_means2d = at::zeros_like(means2d);
    at::Tensor v_conics = at::zeros_like(conics);
    at::Tensor v_colors = at::zeros_like(colors);
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::Tensor v_depths = at::zeros_like(depths);
    at::Tensor v_medium_rgb = at::zeros_like(medium_rgb);
    at::Tensor v_beta_bs = at::zeros_like(beta_bs);
    at::Tensor v_beta_attn = at::zeros_like(beta_attn);
    at::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = at::zeros_like(means2d);
    }

    launch_rasterize_to_pixels_3dgs_underwater_bwd_kernel(
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
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids,
        render_alphas,
        last_ids,
        render_medium_weight,
        v_render_rgb,
        v_render_object,
        v_render_clear,
        v_render_medium,
        v_render_alphas,
        v_render_depth,
        optical_depth_grad_scale,
        absgrad ? c10::optional<at::Tensor>(v_means2d_abs) : c10::nullopt,
        v_means2d,
        v_conics,
        v_colors,
        v_opacities,
        v_depths,
        v_medium_rgb,
        v_beta_bs,
        v_beta_attn
    );

    return std::make_tuple(
        v_means2d_abs,
        v_means2d,
        v_conics,
        v_colors,
        v_opacities,
        v_depths,
        v_medium_rgb,
        v_beta_bs,
        v_beta_attn
    );
}

} // namespace gsplat
