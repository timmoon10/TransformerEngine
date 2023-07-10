"""
This file defines the aten functions for float8. Today, all of these functions
are emulated. In the future, they should be calling NVIDIA's float8 kernels.
"""

import torch
from torch.library import Library

from .float8_utils import (
    float32_to_float8,
    float8_to_float32,
    E4M3,
    E5M2,
    tensor_to_scale,
)

print(f"someone called API registrations")
#
# ATen op placeholders
#

# Register the aten level functions we need.
# These are mostly placeholder and might need to be implemented in c++ as needed
lib = Library("aten", "FRAGMENT")

# For now register on CPU,
# TODO(future) add GPU and test there
lib.define("float32_to_float8(Tensor t, int flavor) -> Tensor")
lib.impl("float32_to_float8", float32_to_float8, "CUDA")

lib.define("float8_to_float32(Tensor t, int flavor) -> Tensor")
lib.impl("float8_to_float32", float8_to_float32, "CUDA")
