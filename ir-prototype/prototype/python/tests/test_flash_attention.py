# pylint: disable=missing-docstring
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path for direct execution
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

try:
    from .. import pypto
except ImportError:
    # Fallback for direct execution
    import pypto

S1 = 64
S2 = 1024
DQK = 128
DV = 128
SQ = 64

@pypto.frontend.jit()
def test_flash_attention(
    q: pypto.Tensor((S1, DQK), pypto.DT_FP16),
    k: pypto.Tensor((S2, DQK), pypto.DT_FP16),
    v: pypto.Tensor((S2, DV), pypto.DT_FP16),
) -> (
    pypto.Tensor((S1, DV), pypto.DT_FP32),
):
    attention = pypto.Tensor((S1, DV), pypto.DT_FP32)
    softmax_scale = np.reciprocal(np.sqrt(DQK))

    qlen = S1
    seqlen = S2
    dk = DQK
    dv = DV

    sq = SQ

    oiUpdate = pypto.Tensor((qlen, dv), pypto.DT_FP32)
    liUpdate = pypto.Tensor((qlen, 1), pypto.DT_FP32)
    miUpdate = pypto.Tensor((qlen, 1), pypto.DT_FP32)

    seqBlockNum = (seqlen + sq - 1) // sq

    for seqBlock in pypto.loop(seqBlockNum):
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

    s1 = S1
    s2 = S2
    dqk = DQK
    dv = DV
    
    q = np.random.uniform(-1, 1, (s1, dqk)).astype(np.float16)
    k = np.random.uniform(-1, 1, (s2, dqk)).astype(np.float16)
    v = np.random.uniform(-1, 1, (s2, dv)).astype(np.float16)

    q = torch.from_numpy(q)
    k = torch.from_numpy(k)
    v = torch.from_numpy(v)

    attention = test_flash_attention(q, k, v)


if __name__ == "__main__":
    test_flash_attention_run()
