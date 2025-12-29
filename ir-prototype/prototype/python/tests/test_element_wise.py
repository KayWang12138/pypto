# pylint: disable=missing-docstring
import os
import sys
from pathlib import Path

# Add parent directory to path for direct execution
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

import numpy as np
try:
    from .. import pypto
except ImportError:
    # Fallback for direct execution
    import pypto
import torch

SHAPE = (1, 2, 64, 128)

@pypto.frontend.jit
def element_wise_op(
    a: pypto.Tensor((1, 2, 64, 128), pypto.DT_FP32),
    b: pypto.Tensor((1, 2, 64, 128), pypto.DT_FP32),
) -> pypto.Tensor((1, 2, 64, 128), pypto.DT_FP32):
    # pypto.set_vec_tile_shapes(1, 1, 16, 32)
    c = pypto.sin(a) * pypto.sin(b)
    return c


def test_element_wise_op():
    # Prepare test data
    np.random.seed(0)
    a = torch.rand(SHAPE, dtype=torch.float32)
    b = torch.rand(SHAPE, dtype=torch.float32)

    # Execute kernel
    t3 = element_wise_op(a, b)

if __name__ == "__main__":
    test_element_wise_op()
