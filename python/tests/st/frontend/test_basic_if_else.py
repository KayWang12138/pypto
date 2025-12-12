# pylint: disable=missing-docstring
import os
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose

N = 1024
M = 1024


@pypto.frontend.jit()
def basic_if_else(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32),
    is_add: bool,
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
):

    c = pypto.tensor((N, M), pypto.DT_FP32)

    pypto.set_vec_tile_shapes(32, 32)

    if is_add:
        c[:] = pypto.add(a, b)
    else:
        c[:] = pypto.sub(a, b)

    return c


def test_basic_if_else_run():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    n, m = 1024, 1024
    np.random.seed(0)
    a = torch.rand((n, m), dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.rand((n, m), dtype=torch.float32, device=f"npu:{device_id}")
    is_add = True

    c = basic_if_else(a, b, is_add)

    pypto.runtime._device_synchronize()

    if is_add:
        c_golden = (a + b).cpu()
    else:
        c_golden = (a - b).cpu()

    assert_allclose(c.cpu().flatten(), c_golden.flatten(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_basic_if_else_run()
