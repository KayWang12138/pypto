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
IS_ADD = True


@pypto.frontend.jit()
def multiple_vars(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32)
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
    pypto.Tensor((N, M), pypto.DT_FP32),
):
    """Test multiple variables modified in different branches."""
    c = pypto.Tensor((N, M), pypto.DT_FP32)
    d = pypto.Tensor((N, M), pypto.DT_FP32)

    if pypto.cond(IS_ADD):
        c[:] = pypto.add(a, b)
    else:
        d[:] = pypto.div(a, b)

    return c, d


def test_multiple_vars_run():
    n, m = 1024, 1024
    np.random.seed(0)
    a = torch.rand((n, m), dtype=torch.float32)
    b = torch.rand((n, m), dtype=torch.float32)

    c, d = multiple_vars(a, b)
    print(f"Multiple vars test completed. Output shapes: c={c.shape}, d={d.shape}")


if __name__ == "__main__":
    test_multiple_vars_run()

