// Adapted from gsplat v1.5.3 under Apache-2.0 and modified for 3D-USE.
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"

namespace gsplat {

namespace cg = cooperative_groups;

// Ordered 3DGS compositing with a homogeneous, per-pixel underwater medium.
// Medium intervals are integrated between ordered Gaussians, followed by the
// residual Plenodium water column from the last contribution to infinity.
__global__ void rasterize_to_pixels_3dgs_underwater_fwd_kernel(
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
    float *__restrict__ render_rgb,
    float *__restrict__ render_object,
    float *__restrict__ render_clear,
    float *__restrict__ render_medium,
    float *__restrict__ render_medium_weight,
    float *__restrict__ render_alphas,
    float *__restrict__ render_depth,
    int32_t *__restrict__ last_ids
) {
    auto block = cg::this_thread_block();
    const int32_t image_id = block.group_index().x;
    const int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i =
        block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j =
        block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += image_id * tile_height * tile_width;
    const uint32_t image_stride = image_height * image_width;
    render_rgb += image_id * image_stride * 3;
    render_object += image_id * image_stride * 3;
    render_clear += image_id * image_stride * 3;
    render_medium += image_id * image_stride * 3;
    render_medium_weight += image_id * image_stride * 3;
    render_alphas += image_id * image_stride;
    render_depth += image_id * image_stride;
    last_ids += image_id * image_stride;
    medium_rgb += image_id * image_stride * 3;
    beta_bs += image_id * image_stride * 3;
    beta_attn += image_id * image_stride * 3;
    far_depths += image_id * image_stride;
    ray_depth_scales += image_id * image_stride;
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    const bool inside = i < image_height && j < image_width;
    const int32_t pix_id = static_cast<int32_t>(i * image_width + j);

    // Tile masks are block-uniform, so every thread must take the same branch
    // before any later block synchronization.
    if (masks != nullptr && !masks[tile_id]) {
        if (inside) {
#pragma unroll
            for (uint32_t k = 0; k < 3; ++k) {
                render_rgb[pix_id * 3 + k] = 0.f;
                render_object[pix_id * 3 + k] = 0.f;
                render_clear[pix_id * 3 + k] = 0.f;
                render_medium[pix_id * 3 + k] = 0.f;
                render_medium_weight[pix_id * 3 + k] = 0.f;
            }
            render_alphas[pix_id] = 0.f;
            render_depth[pix_id] = 0.f;
            last_ids[pix_id] = -1;
        }
        return;
    }

    bool done = !inside;
    const float px = static_cast<float>(j) + 0.5f;
    const float py = static_cast<float>(i) + 0.5f;

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

    float T = 1.f;
    int32_t cur_idx = -1;
    float clear_out[3] = {0.f, 0.f, 0.f};
    float object_out[3] = {0.f, 0.f, 0.f};
    float medium_weight[3] = {0.f, 0.f, 0.f};
    float previous_bs_exp[3] = {1.f, 1.f, 1.f};
    float previous_depth = 0.f;
    float depth_out = 0.f;

    const uint32_t tr = block.thread_rank();
    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        const uint32_t batch_start = range_start + block_size * b;
        const uint32_t idx = batch_start + tr;
        if (idx < static_cast<uint32_t>(range_end)) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
            conic_batch[tr] = conics[g];
            depth_batch[tr] = depths[g];
        }
        block.sync();

        const uint32_t batch_size =
            min(block_size, static_cast<uint32_t>(range_end) - batch_start);
        for (uint32_t t = 0; t < batch_size && !done; ++t) {
            const float depth = depth_batch[t];
            const float path_depth = depth * ray_depth_scales[pix_id];
            // The active Stage-1 path does not clip projected intersections;
            // object compositing remains identical to official gsplat.
            const vec3 conic = conic_batch[t];
            const vec3 xy_opac = xy_opacity_batch[t];
            const vec2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma =
                0.5f * (conic.x * delta.x * delta.x +
                        conic.z * delta.y * delta.y) +
                conic.y * delta.x * delta.y;
            const float alpha =
                min(0.999f, xy_opac.z * __expf(-sigma));
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }

            const float next_T = T * (1.f - alpha);
            // Match the official gsplat exclusive early-termination rule so
            // the dry path retains parity with v1.5.3.
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }

            const int32_t g = id_batch[t];
            const float delta_depth =
                fmaxf(0.f, path_depth - previous_depth);
            const float vis = alpha * T;
            const float *color = colors + g * 3;
            const float *pixel_bs = beta_bs + pix_id * 3;
            const float *pixel_attn = beta_attn + pix_id * 3;

#pragma unroll
            for (uint32_t k = 0; k < 3; ++k) {
                // exp(-beta*a) - exp(-beta*b), evaluated stably.
                const float interval = previous_bs_exp[k] *
                    (-expm1f(-pixel_bs[k] * delta_depth));
                medium_weight[k] += T * interval;
                // Use the same absolute-depth exponential as backward.  The
                // interval remains in stable expm1 form.
                previous_bs_exp[k] = __expf(-pixel_bs[k] * path_depth);

                clear_out[k] += vis * color[k];
                object_out[k] += vis * color[k] *
                    __expf(-pixel_attn[k] * path_depth);
            }
            depth_out += vis * depth;
            previous_depth = path_depth;
            cur_idx = static_cast<int32_t>(batch_start + t);
            T = next_T;
        }
    }

    if (inside) {
        const float *pixel_medium = medium_rgb + pix_id * 3;
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            // Legacy Plenodium integrates the residual water column from the
            // final contributing Gaussian to infinity.  This is the sole
            // Stage-1 paper semantics; finite-AABB tails are intentionally not
            // implemented in the active compositor.
            medium_weight[k] += T * previous_bs_exp[k];
            const float medium = pixel_medium[k] * medium_weight[k];
            render_clear[pix_id * 3 + k] = clear_out[k];
            render_object[pix_id * 3 + k] = object_out[k];
            render_medium[pix_id * 3 + k] = medium;
            render_medium_weight[pix_id * 3 + k] = medium_weight[k];
            render_rgb[pix_id * 3 + k] = object_out[k] + medium;
        }
        render_alphas[pix_id] = 1.f - T;
        render_depth[pix_id] = depth_out;
        last_ids[pix_id] = cur_idx;
    }
}

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
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float));

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_underwater_fwd_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set underwater forward shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_3dgs_underwater_fwd_kernel
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
            render_rgb.data_ptr<float>(),
            render_object.data_ptr<float>(),
            render_clear.data_ptr<float>(),
            render_medium.data_ptr<float>(),
            render_medium_weight.data_ptr<float>(),
            render_alphas.data_ptr<float>(),
            render_depth.data_ptr<float>(),
            last_ids.data_ptr<int32_t>()
        );
}

} // namespace gsplat
