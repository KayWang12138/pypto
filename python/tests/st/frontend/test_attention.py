# pylint: disable=missing-docstring
import os
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose

S1 = 32
S2 = 128
DQK = 64
DV = 128


def softmax(x):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = (x - x_max)
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    res = y / x_sum
    return res


@pypto.frontend.jit()
def test_attention(
    q: pypto.Tensor((S1, DQK), pypto.DT_FP16),
    k: pypto.Tensor((S2, DQK), pypto.DT_FP16),
    v: pypto.Tensor((S2, DV), pypto.DT_FP16),
) -> (
    pypto.Tensor((S1, DV), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])

    qk = pypto.cast(pypto.matmul(q, k, out_dtype=pypto.DT_FP32, b_trans=True), pypto.DT_FP16)
    qk_scale = np.reciprocal(np.sqrt(DQK))
    s = pypto.softmax(qk * qk_scale, -1)
    out = pypto.matmul(s, v, out_dtype=pypto.DT_FP32)
    
    attention = out
    return attention


def test_attention_run():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    s1 = S1
    s2 = S2
    dqk = DQK
    dv = DV

    q = np.random.uniform(-1, 1, (s1, dqk)).astype(np.float16)
    k = np.random.uniform(-1, 1, (s2, dqk)).astype(np.float16)
    v = np.random.uniform(-1, 1, (s2, dv)).astype(np.float16)

    attention_golden = (softmax(q @ k.T / np.sqrt(dqk)) @ v)
    
    q = torch.from_numpy(q).to(device=f"npu:{device_id}")
    k = torch.from_numpy(k).to(device=f"npu:{device_id}")
    v = torch.from_numpy(v).to(device=f"npu:{device_id}")

    attention = test_attention(q, k, v).cpu()

    pypto.runtime._device_synchronize()

    assert_allclose(
        attention,
        attention_golden,
        rtol=1e-3,
        atol=1e-3
    )


if __name__ == "__main__":
    test_attention_run()
