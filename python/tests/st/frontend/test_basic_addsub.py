# pylint: disable=missing-docstring
import os
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose

N = 1024
M = 1024


@pypto.frontend.jit()
def basic_addsub(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
    b: pypto.Tensor((N, M), pypto.DT_FP32),
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
    pypto.Tensor((N, M), pypto.DT_FP32),
):

    c = pypto.tensor((N, M), pypto.DT_FP32)
    d = pypto.tensor((N, M), pypto.DT_FP32)

    pypto.set_vec_tile_shapes(32, 32)

    c[:] = pypto.add(a, b)
    d[:] = pypto.sub(a, b)

    return c, d


def test_basic_addsub_run():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    n, m = 1024, 1024
    np.random.seed(0)
    a = torch.rand((n, m), dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.rand((n, m), dtype=torch.float32, device=f"npu:{device_id}")

    c, d = basic_addsub(a, b)

    pypto.runtime._device_synchronize()

    c_golden = (a + b).cpu()
    d_golden = (a - b).cpu()

    assert_allclose(c.cpu().flatten(), c_golden.flatten(), rtol=1e-5, atol=1e-5)
    assert_allclose(d.cpu().flatten(), d_golden.flatten(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_basic_addsub_run()
