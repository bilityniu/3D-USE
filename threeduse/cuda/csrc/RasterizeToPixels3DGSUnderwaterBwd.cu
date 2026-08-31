// Adapted from gsplat v1.5.3 under Apache-2.0 and modified for 3D-USE.
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

__global__ void rasterize_to_pixels_3dgs_underwater_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2 *__restrict__ means2d,
    const vec3 *__restrict__ conics,
    const float *__restrict__ colors,
    const float *__restrict__ opacities,
    const float *__restrict__ depths,
    const float *__restrict__ medium_rgb,
    const float *__restrict__ beta_bs,
    const float *__restrict__ beta_attn,
    const float *__restrict__ far_depths,
    const float *__restrict__ ray_depth_scales,
    const bool *__restrict__ masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    const float *__restrict__ render_alphas,
    const int32_t *__restrict__ last_ids,
    const float *__restrict__ render_medium_weight,
    const float *__restrict__ v_render_rgb,
    const float *__restrict__ v_render_object,
    const float *__restrict__ v_render_clear,
    const float *__restrict__ v_render_medium,
    const float *__restrict__ v_render_alphas,
    const float *__restrict__ v_render_depth,
    const float optical_depth_grad_scale,
    vec2 *__restrict__ v_means2d_abs,
    vec2 *__restrict__ v_means2d,
    vec3 *__restrict__ v_conics,
    float *__restrict__ v_colors,
    float *__restrict__ v_opacities,
    float *__restrict__ v_depths,
    float *__restrict__ v_medium_rgb,
    float *__restrict__ v_beta_bs,
    float *__restrict__ v_beta_attn
) {
    auto block = cg::this_thread_block();
    const uint32_t image_id = block.group_index().x;
    const uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i =
        block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j =
        block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += image_id * tile_height * tile_width;
    const uint32_t image_stride = image_height * image_width;
    render_alphas += image_id * image_stride;
    last_ids += image_id * image_stride;
    render_medium_weight += image_id * image_stride * 3;
    v_render_rgb += image_id * image_stride * 3;
    v_render_object += image_id * image_stride * 3;
    v_render_clear += image_id * image_stride * 3;
    v_render_medium += image_id * image_stride * 3;
    v_render_alphas += image_id * image_stride;
    v_render_depth += image_id * image_stride;
    medium_rgb += image_id * image_stride * 3;
    beta_bs += image_id * image_stride * 3;
    beta_attn += image_id * image_stride * 3;
    far_depths += image_id * image_stride;
    ray_depth_scales += image_id * image_stride;
    v_medium_rgb += image_id * image_stride * 3;
    v_beta_bs += image_id * image_stride * 3;
    v_beta_attn += image_id * image_stride * 3;
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    // Tile masks are block-uniform and must branch before synchronization.
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const bool inside = i < image_height && j < image_width;
    const float px = static_cast<float>(j) + 0.5f;
    const float py = static_cast<float>(i) + 0.5f;
    const int32_t pix_id = min(
        static_cast<int32_t>(i * image_width + j),
        static_cast<int32_t>(image_stride - 1)
    );

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? static_cast<int32_t>(n_isects)
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int shared_mem[];
    int32_t *id_batch = reinterpret_cast<int32_t *>(shared_mem);
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]);
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]);
    float *depth_batch = reinterpret_cast<float *>(&conic_batch[block_size]);
    float *colors_batch = &depth_batch[block_size];

    const float T_final = inside ? 1.f - render_alphas[pix_id] : 1.f;
    float T = T_final;
    const int32_t bin_final = inside ? last_ids[pix_id] : -1;

    float g_object[3] = {0.f, 0.f, 0.f};
    float g_medium[3] = {0.f, 0.f, 0.f};
    float g_clear[3] = {0.f, 0.f, 0.f};
    float pixel_medium[3] = {0.f, 0.f, 0.f};
    float pixel_bs[3] = {0.f, 0.f, 0.f};
    float pixel_attn[3] = {0.f, 0.f, 0.f};
    float path_depth_scale = 1.f;
    float g_alpha_out = 0.f;
    float g_depth_out = 0.f;
    if (inside) {
        path_depth_scale = ray_depth_scales[pix_id];
        g_alpha_out = v_render_alphas[pix_id];
        g_depth_out = v_render_depth[pix_id];
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            const int32_t p = pix_id * 3 + k;
            g_object[k] = v_render_rgb[p] + v_render_object[p];
            g_medium[k] = v_render_rgb[p] + v_render_medium[p];
            g_clear[k] = v_render_clear[p];
            pixel_medium[k] = medium_rgb[p];
            pixel_bs[k] = beta_bs[p];
            pixel_attn[k] = beta_attn[p];
        }
    }

    // r_next is dL/dT after the current Gaussian.  For the medium term we add
    // the same constant sum_k(gM_k B_k) to every weight derivative and to the
    // terminal derivative.  Because sum_i(w_i) + T_final == 1, this shift does
    // not change alpha gradients, but turns a near-dry subtraction of two
    // numbers close to -gM*B into the stable 1-exp(-beta*d) form.
    float r_next = -g_alpha_out;
#pragma unroll
    for (uint32_t k = 0; k < 3; ++k) {
        r_next += g_medium[k] * pixel_medium[k];
    }

    float bs_depth_moment[3] = {0.f, 0.f, 0.f};
    float beta_attn_grad[3] = {0.f, 0.f, 0.f};

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());

    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(
            static_cast<int32_t>(block_size), batch_end + 1 - range_start
        );
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
            conic_batch[tr] = conics[g];
            depth_batch[tr] = depths[g];
#pragma unroll
            for (uint32_t k = 0; k < 3; ++k) {
                colors_batch[tr * 3 + k] = colors[g * 3 + k];
            }
        }
        block.sync();

        for (uint32_t t = max(0, batch_end - warp_bin_final);
             t < static_cast<uint32_t>(batch_size);
             ++t) {
            bool valid = inside && (batch_end - static_cast<int32_t>(t) <= bin_final);
            float alpha = 0.f;
            float opacity = 0.f;
            float gaussian_vis = 0.f;
            vec2 delta = {0.f, 0.f};
            vec3 conic = {0.f, 0.f, 0.f};
            if (valid) {
                conic = conic_batch[t];
                const vec3 xy_opacity = xy_opacity_batch[t];
                opacity = xy_opacity.z;
                delta = {xy_opacity.x - px, xy_opacity.y - py};
                const float sigma =
                    0.5f * (conic.x * delta.x * delta.x +
                            conic.z * delta.y * delta.y) +
                    conic.y * delta.x * delta.y;
                gaussian_vis = __expf(-sigma);
                alpha = min(0.999f, opacity * gaussian_vis);
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                    valid = false;
                }
            }

            if (!warp.any(valid)) {
                continue;
            }

            float v_color_local[3] = {0.f, 0.f, 0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            float v_depth_local = 0.f;

            if (valid) {
                const float inv_one_minus_alpha = 1.f / (1.f - alpha);
                T *= inv_one_minus_alpha; // T is now T_i, before Gaussian i.
                const float weight = alpha * T;
                const float depth = depth_batch[t];
                const float path_depth = depth * path_depth_scale;

                float q_weight = g_depth_out * depth;
                float optical_depth_term = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < 3; ++k) {
                    const float color = colors_batch[t * 3 + k];
                    const float attn_exp =
                        __expf(-pixel_attn[k] * path_depth);
                    const float bs_exp =
                        __expf(-pixel_bs[k] * path_depth);

                    q_weight += g_object[k] * color * attn_exp;
                    q_weight += g_clear[k] * color;
                    const float one_minus_bs_exp =
                        -expm1f(-pixel_bs[k] * path_depth);
                    q_weight += g_medium[k] * pixel_medium[k] *
                        one_minus_bs_exp;

                    v_color_local[k] = weight *
                        (g_object[k] * attn_exp + g_clear[k]);
                    beta_attn_grad[k] += -g_object[k] * weight * color *
                        path_depth * attn_exp;
                    bs_depth_moment[k] += weight * path_depth * bs_exp;

                    optical_depth_term +=
                        -g_object[k] * pixel_attn[k] * color * attn_exp +
                        g_medium[k] * pixel_medium[k] * pixel_bs[k] * bs_exp;
                }

                const float v_alpha = T * (q_weight - r_next);
                r_next =
                    alpha * q_weight + (1.f - alpha) * r_next;
                v_depth_local = weight *
                    (g_depth_out +
                     optical_depth_grad_scale * path_depth_scale *
                         optical_depth_term);

                // The cap and threshold exactly match the forward path.
                if (opacity * gaussian_vis <= 0.999f) {
                    const float v_sigma =
                        -opacity * gaussian_vis * v_alpha;
                    v_conic_local = {
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    };
                    v_xy_local = {
                        v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma * (conic.y * delta.x + conic.z * delta.y)
                    };
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {
                            abs(v_xy_local.x), abs(v_xy_local.y)
                        };
                    }
                    v_opacity_local = gaussian_vis * v_alpha;
                }
            }

            warpSum<3>(v_color_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
            warpSum(v_depth_local, warp);

            if (warp.thread_rank() == 0) {
                const int32_t g = id_batch[t];
#pragma unroll
                for (uint32_t k = 0; k < 3; ++k) {
                    gpuAtomicAdd(v_colors + g * 3 + k, v_color_local[k]);
                }
                float *v_conic_ptr = reinterpret_cast<float *>(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);
                float *v_xy_ptr = reinterpret_cast<float *>(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);
                if (v_means2d_abs != nullptr) {
                    float *v_xy_abs_ptr =
                        reinterpret_cast<float *>(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }
                gpuAtomicAdd(v_opacities + g, v_opacity_local);
                gpuAtomicAdd(v_depths + g, v_depth_local);
            }
        }
    }

    if (inside) {
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            const int32_t p = pix_id * 3 + k;
            v_medium_rgb[p] =
                g_medium[k] * render_medium_weight[p];
            v_beta_bs[p] = g_medium[k] * pixel_medium[k] *
                bs_depth_moment[k];
            v_beta_attn[p] = beta_attn_grad[k];
        }
    }
}

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
) {
    const bool packed = means2d.dim() == 2;
    const uint32_t N = packed ? 0 : means2d.size(-2);
    const uint32_t I =
        render_alphas.numel() / (image_height * image_width);
    const uint32_t tile_height = tile_offsets.size(-2);
    const uint32_t tile_width = tile_offsets.size(-1);
    const uint32_t n_isects = flatten_ids.size(0);

    const dim3 threads = {tile_size, tile_size, 1};
    const dim3 grid = {I, tile_height, tile_width};
    const int64_t shmem_size = tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) +
         sizeof(float) + sizeof(float) * 3);

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_underwater_bwd_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set underwater backward shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    // Launch even for n_isects == 0: empty-water rays still contribute
    // infinite-tail medium and beta_bs gradients.
    rasterize_to_pixels_3dgs_underwater_bwd_kernel
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            I,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            depths.data_ptr<float>(),
            medium_rgb.data_ptr<float>(),
            beta_bs.data_ptr<float>(),
            beta_attn.data_ptr<float>(),
            far_depths.data_ptr<float>(),
            ray_depth_scales.data_ptr<float>(),
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            render_medium_weight.data_ptr<float>(),
            v_render_rgb.data_ptr<float>(),
            v_render_object.data_ptr<float>(),
            v_render_clear.data_ptr<float>(),
            v_render_medium.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_render_depth.data_ptr<float>(),
            optical_depth_grad_scale,
            v_means2d_abs.has_value()
                ? reinterpret_cast<vec2 *>(
                      v_means2d_abs.value().data_ptr<float>())
                : nullptr,
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>(),
            v_depths.data_ptr<float>(),
            v_medium_rgb.data_ptr<float>(),
            v_beta_bs.data_ptr<float>(),
            v_beta_attn.data_ptr<float>()
        );
}

} // namespace gsplat
