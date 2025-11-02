# pylint: disable=missing-docstring
import os

import numpy as np
import pypto
import torch
from numpy.testing import assert_allclose

SHAPE = (1, 2, 64, 128)

@pypto.frontend.jit
def element_wise_op(
    a: pypto.Tensor((1, 2, 64, 128), pypto.DT_FP32),
    b: pypto.Tensor((1, 2, 64, 128), pypto.DT_FP32),
) -> pypto.Tensor((1, 2, 64, 128), pypto.DT_FP32):
    pypto.set_vec_tile_shapes(1, 1, 16, 32)
    c = pypto.sin(a) * pypto.sin(b)
    return c


def test_element_wise_op():
    # Setup device
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    # Prepare test data
    np.random.seed(0)
    a = torch.rand(SHAPE, dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.rand(SHAPE, dtype=torch.float32, device=f"npu:{device_id}")

    # Execute kernel
    t3 = element_wise_op(a, b)
    pypto.runtime._device_synchronize()

    print("End to execute kernel")

    # PyTorch reference implementation
    ref = torch.sin(a) * torch.sin(b)

    assert_allclose(
        np.array(t3.cpu().flatten().tolist()),
        np.array(ref.cpu().flatten().tolist()),
        rtol=5e-3,
        atol=5e-3,
    )


if __name__ == "__main__":
    test_element_wise_op()
