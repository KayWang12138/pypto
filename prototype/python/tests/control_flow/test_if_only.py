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
COND = True


@pypto.frontend.jit()
def if_only(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32)
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
):
    """Test if statement without else branch."""
    c = pypto.Tensor((N, M), pypto.DT_FP32)
    c[:] = a

    if pypto.cond(COND):
        # Only then branch, no else
        c[:] = pypto.add(c, b)

    return c


def test_if_only_run():
    n, m = 1024, 1024
    np.random.seed(0)
    a = torch.rand((n, m), dtype=torch.float32)
    b = torch.rand((n, m), dtype=torch.float32)

    c = if_only(a, b)
    print(f"If-only test completed. Output shape: {c.shape}")


if __name__ == "__main__":
    test_if_only_run()

