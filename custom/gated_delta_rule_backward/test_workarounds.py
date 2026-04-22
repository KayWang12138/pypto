#!/usr/bin/env python3
"""Test workarounds for matmul bug with computed tensors."""

import torch, torch_npu, pypto, math

device_id = 4
torch.npu.set_device(device_id)
npu = f'npu:{device_id}'
BT = 64; KD = 128; VD = 128

def to_npu(t):
    return t.clone().contiguous().to(npu).to(torch.float32)

torch.manual_seed(42)
qc = torch.randn(BT, KD) * 0.5
kc = torch.randn(BT, KD) * 0.5
gc_raw = torch.randn(BT) * 0.1
doc = torch.randn(BT, VD) * 0.5
scale = 1.0 / math.sqrt(KD)
ones = torch.ones(BT, BT); c_cum = torch.tril(ones); m_le = torch.tril(ones)
g_cum = c_cum @ gc_raw; decay = torch.exp(g_cum[:, None] - g_cum[None, :])
qk = qc @ kc.t(); a_local = (qk * decay) * m_le
dv0_golden = (a_local.t() @ doc) * scale

# Workaround 1: +0.0 trick (full + add)
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_plus_zero(
    qc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    kc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    doc_in: pypto.Tensor([BT,VD],pypto.DT_FP32),
    gc_in: pypto.Tensor([BT],pypto.DT_FP32),
    c_cum_in: pypto.Tensor([BT,BT],pypto.DT_FP32),
    m_le_in: pypto.Tensor([BT,BT],pypto.DT_FP32),
    scale_in: pypto.Tensor([1],pypto.DT_FP32),
    dv0_out: pypto.Tensor([BT,VD],pypto.DT_FP32),
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
    # +0.0 trick
    pypto.set_vec_tile_shapes(128, 128)
    zero_mat = pypto.full([BT, BT], 0.0, pypto.DT_FP32)
    a_local = pypto.add(a_local, zero_mat)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    dv0 = pypto.mul(pypto.matmul(a_local, doc_in, pypto.DT_FP32, a_trans=True), scale_in)
    dv0_out[:] = dv0

dv0_w1 = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
try:
    kern_plus_zero(to_npu(qc), to_npu(kc), to_npu(doc), to_npu(gc_raw),
                   to_npu(c_cum), to_npu(m_le), to_npu(torch.tensor([scale])), dv0_w1)
    torch_npu.npu.synchronize()
    d1 = (dv0_golden - dv0_w1.cpu()).abs().max()
    print(f"Workaround 1 (+0.0): max_diff={d1:.6e}")
except Exception as e:
    print(f"Workaround 1 (+0.0): FAILED - {e}")

# Workaround 2: Match tile shapes to actual dims [64, 128, 64]
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_small_tile(
    qc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    kc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    doc_in: pypto.Tensor([BT,VD],pypto.DT_FP32),
    gc_in: pypto.Tensor([BT],pypto.DT_FP32),
    c_cum_in: pypto.Tensor([BT,BT],pypto.DT_FP32),
    m_le_in: pypto.Tensor([BT,BT],pypto.DT_FP32),
    scale_in: pypto.Tensor([1],pypto.DT_FP32),
    dv0_out: pypto.Tensor([BT,VD],pypto.DT_FP32),
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
    # Use smaller tile shapes for the final matmul
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    dv0 = pypto.mul(pypto.matmul(a_local, doc_in, pypto.DT_FP32, a_trans=True), scale_in)
    dv0_out[:] = dv0

dv0_w2 = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
try:
    kern_small_tile(to_npu(qc), to_npu(kc), to_npu(doc), to_npu(gc_raw),
                    to_npu(c_cum), to_npu(m_le), to_npu(torch.tensor([scale])), dv0_w2)
    torch_npu.npu.synchronize()
    d2 = (dv0_golden - dv0_w2.cpu()).abs().max()
    print(f"Workaround 2 (small tile [64,64]): max_diff={d2:.6e}")
except Exception as e:
    print(f"Workaround 2 (small tile): FAILED - {e}")

# Workaround 3: Split the mul chain — store qk*decay first, then mul with m_le
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_split_mul(
    qc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    kc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    doc_in: pypto.Tensor([BT,VD],pypto.DT_FP32),
    gc_in: pypto.Tensor([BT],pypto.DT_FP32),
    c_cum_in: pypto.Tensor([BT,BT],pypto.DT_FP32),
    m_le_in: pypto.Tensor([BT,BT],pypto.DT_FP32),
    scale_in: pypto.Tensor([1],pypto.DT_FP32),
    dv0_out: pypto.Tensor([BT,VD],pypto.DT_FP32),
    temp_out: pypto.Tensor([BT,BT],pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    g_cum = pypto.matmul(c_cum_in, gc_in.reshape([BT, 1]), pypto.DT_FP32)
    pypto.set_vec_tile_shapes(128, 128)
    decay = pypto.exp(pypto.sub(g_cum, g_cum.reshape([1, BT])))
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    qk = pypto.matmul(qc_in, kc_in, pypto.DT_FP32, b_trans=True)
    pypto.set_vec_tile_shapes(128, 128)
    qkd = pypto.mul(qk, decay)
    # Write to temp output to force materialization
    temp_out[:] = qkd
    # Read back
    qkd_mat = temp_out
    pypto.set_vec_tile_shapes(128, 128)
    a_local = pypto.mul(qkd_mat, m_le_in)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    dv0 = pypto.mul(pypto.matmul(a_local, doc_in, pypto.DT_FP32, a_trans=True), scale_in)
    dv0_out[:] = dv0

dv0_w3 = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
temp_w3 = torch.zeros(BT, BT, device=npu, dtype=torch.float32)
try:
    kern_split_mul(to_npu(qc), to_npu(kc), to_npu(doc), to_npu(gc_raw),
                   to_npu(c_cum), to_npu(m_le), to_npu(torch.tensor([scale])),
                   dv0_w3, temp_w3)
    torch_npu.npu.synchronize()
    d3 = (dv0_golden - dv0_w3.cpu()).abs().max()
    print(f"Workaround 3 (split mul + temp output): max_diff={d3:.6e}")
except Exception as e:
    print(f"Workaround 3 (split mul): FAILED - {type(e).__name__}")
