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
LOOP_NUM = 10
IS_ADD = True


@pypto.frontend.jit()
def loop_in_if_else(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32)
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
):
    """Test loop inside if-else: different loops in different branches."""
    c = pypto.Tensor((N, M), pypto.DT_FP32)

    if pypto.cond(IS_ADD):
        # Then branch: accumulate with addition
        c[:] = a
        for _ in pypto.loop(LOOP_NUM):
            c[:] = pypto.add(c, b)
    else:
        # Else branch: accumulate with subtraction
        c[:] = a
        for _ in pypto.loop(LOOP_NUM):
            c[:] = pypto.sub(c, b)

    return c


def test_loop_in_if_else_run():
    n, m = 1024, 1024
    np.random.seed(0)
    a = torch.rand((n, m), dtype=torch.float32)
    b = torch.rand((n, m), dtype=torch.float32)

    c = loop_in_if_else(a, b)
    print(f"Loop in if-else test completed. Output shape: {c.shape}")


if __name__ == "__main__":
    test_loop_in_if_else_run()

