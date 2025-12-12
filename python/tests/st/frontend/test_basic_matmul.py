# pylint: disable=missing-docstring
import os
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose

N = 128
M = 128


@pypto.frontend.jit()
def basic_matmul(
    a: pypto.Tensor((N, M), pypto.DT_FP16),
    b: pypto.Tensor((N, M), pypto.DT_FP16),
) -> (
    pypto.Tensor((N, M), pypto.DT_FP32),
    pypto.Tensor((N, M), pypto.DT_FP32),
):

    c = pypto.tensor((N, M), pypto.DT_FP32)
    d = pypto.tensor((N, M), pypto.DT_FP32)

    pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])

    c[:] = pypto.matmul(a, b, out_dtype=pypto.DT_FP32)
    d[:] = pypto.matmul(a, b, out_dtype=pypto.DT_FP32, b_trans=True)
    return c, d


def test_basic_matmul_run():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    n, m = 128, 128
    np.random.seed(0)
    a = torch.rand((n, m), dtype=torch.float16, device=f"npu:{device_id}")
    b = torch.rand((n, m), dtype=torch.float16, device=f"npu:{device_id}")

    c, d = basic_matmul(a, b)

    pypto.runtime._device_synchronize()

    c_golden = (a @ b).cpu()
    d_golden = (a @ b.T).cpu()
    assert_allclose(c.cpu().flatten(), c_golden.flatten(), rtol=1e-3, atol=1e-3)
    assert_allclose(d.cpu().flatten(), d_golden.flatten(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_basic_matmul_run()
