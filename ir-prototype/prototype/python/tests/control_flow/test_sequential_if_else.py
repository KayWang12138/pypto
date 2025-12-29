# pylint: disable=missing-docstring
import sys
from pathlib import Path

# Setup path for direct execution (when PYTHONPATH is not set)
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pypto

N = 1024
M = 1024
COND1 = True
COND2 = False


@pypto.frontend.jit()
def sequential_if_else(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32)
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
):
    """Test sequential if-else statements (not nested, but consecutive)."""
    c = pypto.Tensor((N, M), pypto.DT_FP32)
    c[:] = a

    # First if-else
    if pypto.cond(COND1):
        c[:] = pypto.add(c, b)
    else:
        c[:] = pypto.sub(c, b)

    # Second if-else (sequential, not nested)
    if pypto.cond(COND2):
        c[:] = pypto.mul(c, b)
    else:
        c[:] = pypto.div(c, b)

    return c


def test_sequential_if_else_run():
    n, m = 1024, 1024
    np.random.seed(0)
    a = torch.rand((n, m), dtype=torch.float32)
    b = torch.rand((n, m), dtype=torch.float32)

    c = sequential_if_else(a, b)
    print(f"Sequential if-else test completed. Output shape: {c.shape}")


if __name__ == "__main__":
    test_sequential_if_else_run()

