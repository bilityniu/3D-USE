"""Restricted checkpoint loading for the bundled Nerfstudio runtime."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.core.multiarray import scalar as numpy_scalar


_NUMPY_SAFE_GLOBALS = [
    numpy_scalar,
    np.dtype,
    type(np.dtype(np.float64)),
]


def safe_torch_load(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load tensor checkpoints without permitting arbitrary pickle objects."""

    with torch.serialization.safe_globals(_NUMPY_SAFE_GLOBALS):
        return torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
