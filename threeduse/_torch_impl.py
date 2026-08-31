# Adapted from gsplat v1.5.3 under Apache-2.0 and modified for 3D-USE.
"""Small PyTorch geometry helpers used by the final renderer."""

import torch
import torch.nn.functional as F
from torch import Tensor


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    """Convert normalized ``(w, x, y, z)`` quaternions to rotation matrices."""

    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    matrix = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return matrix.reshape(quat.shape[:-1] + (3, 3))


def quat_to_rotmat(quat: Tensor) -> Tensor:
    """Normalize quaternions and convert them to rotation matrices."""

    assert quat.shape[-1] == 4, quat.shape
    return normalized_quat_to_rotmat(F.normalize(quat, dim=-1))
