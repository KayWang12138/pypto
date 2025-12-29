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
LOOP_NUM = 5
COND1 = True
COND2 = True


@pypto.frontend.jit()
def deeply_nested(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32)
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
):
    """Test deeply nested control flow: if-else in loop, loop in if-else."""
    c = pypto.Tensor((N, M), pypto.DT_FP32)
    c[:] = a

    for _ in pypto.loop(LOOP_NUM):
        if pypto.cond(COND1):
            # Then branch in loop
            if pypto.cond(COND2):
                # Nested then branch
                c[:] = pypto.add(c, b)
            else:
                # Nested else branch
                c[:] = pypto.mul(c, b)
        else:
            # Else branch in loop
            for _ in pypto.loop(2):
                # Loop inside else branch
                c[:] = pypto.sub(c, b)

    return c


def test_deeply_nested_run():
    n, m = 1024, 1024
    np.random.seed(0)
    a = torch.rand((n, m), dtype=torch.float32)
    b = torch.rand((n, m), dtype=torch.float32)

    c = deeply_nested(a, b)
    print(f"Deeply nested test completed. Output shape: {c.shape}")


if __name__ == "__main__":
    test_deeply_nested_run()

