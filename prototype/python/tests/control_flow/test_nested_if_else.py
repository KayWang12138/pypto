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
COND2 = True


@pypto.frontend.jit()
def nested_if_else(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32)
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
):
    """Test nested if-else: if statement inside another if-else branch."""
    c = pypto.Tensor((N, M), pypto.DT_FP32)

    if pypto.cond(COND1):
        # Outer then branch
        if pypto.cond(COND2):
            # Inner then branch
            c[:] = pypto.add(a, b)
        else:
            # Inner else branch
            c[:] = pypto.mul(a, b)
    else:
        # Outer else branch
        if pypto.cond(COND2):
            # Inner then branch in outer else
            c[:] = pypto.sub(a, b)
        else:
            # Inner else branch in outer else
            c[:] = pypto.div(a, b)

    return c


def test_nested_if_else_run():
    n, m = 1024, 1024
    np.random.seed(0)
    a = torch.rand((n, m), dtype=torch.float32)
    b = torch.rand((n, m), dtype=torch.float32)

    c = nested_if_else(a, b)
    print(f"Nested if-else test completed. Output shape: {c.shape}")


if __name__ == "__main__":
    test_nested_if_else_run()

