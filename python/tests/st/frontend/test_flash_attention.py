# pylint: disable=missing-docstring
import os
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose

S1 = 64
S2 = 1024
DQK = 128
DV = 128


SQ = 64
MAX_UNROLL_TIMES = 4


def softmax(x):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = (x - x_max)
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    res = y / x_sum
    return res


@pypto.frontend.jit()
def test_flash_attention(
    q: pypto.Tensor((S1, DQK), pypto.DT_FP16),
    k: pypto.Tensor((S2, DQK), pypto.DT_FP16),
    v: pypto.Tensor((S2, DV), pypto.DT_FP16),
) -> (
    pypto.Tensor((S1, DV), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 64)
    pypto.set_cube_tile_shapes([32, 64], [32, 64], [32, 64])

    softmax_scale = np.reciprocal(np.sqrt(DQK))

    qlen = S1
    seqlen = S2
    dk = DQK
    dv = DV

    sq = SQ

    maxUnrollTimes = MAX_UNROLL_TIMES

    oiUpdate = pypto.tensor((qlen, dv), pypto.DT_FP32)
    liUpdate = pypto.tensor((qlen, 1), pypto.DT_FP32)
    miUpdate = pypto.tensor((qlen, 1), pypto.DT_FP32)

    seqBlockNum = (seqlen + sq - 1) // sq

    for seqBlock in pypto.loop(seqBlockNum, unroll_List=[maxUnrollTimes]):
        curOffset = seqBlock * sq
        kj = pypto.view(k, (sq, dk), [curOffset, 0], valid_shape=[(seqlen - seqBlock * sq).min(sq), dk])
        vj = pypto.view(v, (sq, dv), [curOffset, 0], valid_shape=[(seqlen - seqBlock * sq).min(sq), dv])

        sij = pypto.matmul(q, kj, out_dtype=pypto.DT_FP32, b_trans=True)
        sij = sij * softmax_scale
        tildaMij = pypto.amax(sij, -1, True)
        tsub = sij - tildaMij
        tildaPij = pypto.exp(tsub)
        tildaLij = pypto.sum(tildaPij, -1, True)
        tildaPij = pypto.cast(tildaPij, pypto.DT_FP16)

        if seqBlock == 0:
            oiUpdate[:] = pypto.matmul(tildaPij, vj, out_dtype=pypto.DT_FP32)
            if (seqBlock == seqBlockNum - 1):
                attention = pypto.div(oiUpdate, tildaLij)
            liUpdate[:] = tildaLij
            miUpdate[:] = tildaMij
        else:
            mi = miUpdate
            miUpdate = pypto.maximum(mi, tildaMij)
            t1 = mi - miUpdate
            t2 = pypto.exp(t1)
            t3 = tildaMij - miUpdate
            t4 = pypto.exp(t3)
            t5 = t4 * tildaLij
            t6 = t2 * liUpdate
            liUpdate[:] = t6 + t5

            q3 = oiUpdate * t2
            q1 = pypto.matmul(tildaPij, vj, out_dtype=pypto.DT_FP32)
            q2 = q1 * t4
            oiUpdate[:] = q3 + q2
            if (seqBlock == seqBlockNum - 1):
                attention = pypto.div(oiUpdate, liUpdate)

    return attention


def test_flash_attention_run():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 8))
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

    attention = test_flash_attention(q, k, v).cpu()

    pypto.runtime._device_synchronize()

    print(f"Attention shape: {attention.shape}")
    print(f"Attention golden shape: {attention_golden.shape}")
    assert_allclose(
        attention,
        attention_golden,
        rtol=1e-3,
        atol=1e-3
    )


if __name__ == "__main__":
    test_flash_attention_run()
