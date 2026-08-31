// Adapted from Plenodium and gsplat under Apache-2.0; modified for 3D-USE.
#include <torch/extension.h>

#include "SphericalHarmonics.h"
#include "UnderwaterOps.h"

// Official gsplat owns projection, culling, sorting, and standard
// rasterization. This project extension contains only the two operations
// specific to 3D-USE: SH appearance decoding and underwater compositing.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_sh_forward", &compute_sh_forward_tensor);
    m.def("compute_sh_backward", &compute_sh_backward_tensor);
    m.def("sparse_compute_sh_forward", &sparse_compute_sh_forward_tensor);
    m.def("sparse_compute_sh_backward", &sparse_compute_sh_backward_tensor);
    m.def(
        "rasterize_to_pixels_3dgs_underwater_fwd",
        &gsplat::rasterize_to_pixels_3dgs_underwater_fwd
    );
    m.def(
        "rasterize_to_pixels_3dgs_underwater_bwd",
        &gsplat::rasterize_to_pixels_3dgs_underwater_bwd
    );
}
