#!/usr/bin/env python3
"""Test with BT=32 to check if tiling boundary at BT/2 causes the error."""

import torch, torch_npu, pypto, math

device_id = 4
torch.npu.set_device(device_id)
npu = f'npu:{device_id}'

def to_npu(t):
    return t.clone().contiguous().to(npu).to(torch.float32)

def test_bt(BT, KD=128, VD=128):
    torch.manual_seed(42)
    qc = torch.randn(BT, KD) * 0.5
    kc = torch.randn(BT, KD) * 0.5
    gc_raw = torch.randn(BT) * 0.1
    doc = torch.randn(BT, VD) * 0.5
    scale = 1.0 / math.sqrt(KD)
    ones = torch.ones(BT, BT)
    c_cum = torch.tril(ones)
    m_le = torch.tril(ones)

    g_cum = c_cum @ gc_raw
    decay = torch.exp(g_cum[:, None] - g_cum[None, :])
    qk = qc @ kc.t()
    a_local = (qk * decay) * m_le
    dv0_golden = (a_local.t() @ doc) * scale

    @pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
    def kern(
        qc_in: pypto.Tensor([BT, KD], pypto.DT_FP32),
        kc_in: pypto.Tensor([BT, KD], pypto.DT_FP32),
        doc_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
        gc_in: pypto.Tensor([BT], pypto.DT_FP32),
        c_cum_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
        m_le_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
        scale_in: pypto.Tensor([1], pypto.DT_FP32),
        dv0_out: pypto.Tensor([BT, VD], pypto.DT_FP32),
    ):
        pypto.experimental.set_operation_options(combine_axis=True)
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        g_cum = pypto.matmul(c_cum_in, gc_in.reshape([BT, 1]), pypto.DT_FP32)
        pypto.set_vec_tile_shapes(128, 128)
        decay = pypto.exp(pypto.sub(g_cum, g_cum.reshape([1, BT])))
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        qk = pypto.matmul(qc_in, kc_in, pypto.DT_FP32, b_trans=True)
        pypto.set_vec_tile_shapes(128, 128)
        a_local = pypto.mul(pypto.mul(qk, decay), m_le_in)
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        dv0 = pypto.mul(pypto.matmul(a_local, doc_in, pypto.DT_FP32, a_trans=True), scale_in)
        dv0_out[:] = dv0

    dv0_out = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
    kern(to_npu(qc), to_npu(kc), to_npu(doc), to_npu(gc_raw),
         to_npu(c_cum), to_npu(m_le), to_npu(torch.tensor([scale])), dv0_out)
    torch_npu.npu.synchronize()

    diff = (dv0_golden - dv0_out.cpu()).abs()
    row_diffs = diff.max(dim=1).values
    
    # Find where error starts
    first_bad = -1
    for i in range(BT):
        if row_diffs[i] > 0.01:
            first_bad = i
            break
    
    print(f"BT={BT:3d}: max_diff={diff.max():.4e}, first_bad_row={first_bad}, "
          f"first_bad/BT={first_bad/BT if first_bad >= 0 else 'N/A'}")
    return diff.max()

test_bt(32)
test_bt(64)
test_bt(128)
