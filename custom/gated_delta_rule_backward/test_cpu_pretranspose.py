#!/usr/bin/env python3
"""Test: pre-transpose a_local on CPU, then use regular matmul (no a_trans)."""

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

# Kernel A: compute a_local and write to output
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_a(
    qc_in: pypto.Tensor([BT, KD], pypto.DT_FP32),
    kc_in: pypto.Tensor([BT, KD], pypto.DT_FP32),
    gc_in: pypto.Tensor([BT], pypto.DT_FP32),
    c_cum_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
    m_le_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
    aloc_out: pypto.Tensor([BT, BT], pypto.DT_FP32),
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
    aloc_out[:] = a_local

# Kernel C: compute dv0 from PRE-TRANSPOSED a_local using regular matmul (no a_trans!)
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_c(
    aloc_t_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
    doc_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    scale_in: pypto.Tensor([1], pypto.DT_FP32),
    dv0_out: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    dv0_raw = pypto.matmul(aloc_t_in, doc_in, pypto.DT_FP32)
    pypto.set_vec_tile_shapes(128, 128)
    dv0 = pypto.mul(dv0_raw, scale_in)
    dv0_out[:] = dv0

# Step 1: Compute a_local on NPU
aloc_npu = torch.zeros(BT, BT, device=npu, dtype=torch.float32)
kern_a(to_npu(qc), to_npu(kc), to_npu(gc_raw), to_npu(c_cum), to_npu(m_le), aloc_npu)
torch_npu.npu.synchronize()

# Step 2: CPU round-trip + PRE-TRANSPOSE on CPU
aloc_cpu = aloc_npu.cpu()
aloc_t_cpu = aloc_cpu.t().contiguous()  # Transpose on CPU!
print(f"a_local^T check (vs golden): max_diff={(a_local.t() - aloc_t_cpu).abs().max():.6e}")

# Step 3: Pass PRE-TRANSPOSED a_local to kernel C (regular matmul, no a_trans!)
aloc_t_npu = to_npu(aloc_t_cpu)
dv0_pretransposed = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
kern_c(aloc_t_npu, to_npu(doc), to_npu(torch.tensor([scale])), dv0_pretransposed)
torch_npu.npu.synchronize()

diff_pretransposed = (dv0_golden - dv0_pretransposed.cpu()).abs().max()
print(f"dv0 (CPU pre-transpose + regular matmul): max_diff={diff_pretransposed:.6e}")

# Also test with the golden a_local^T as input (perfect values)
aloc_t_golden_npu = to_npu(a_local.t().contiguous())
dv0_golden_input = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
kern_c(aloc_t_golden_npu, to_npu(doc), to_npu(torch.tensor([scale])), dv0_golden_input)
torch_npu.npu.synchronize()

diff_golden_input = (dv0_golden - dv0_golden_input.cpu()).abs().max()
print(f"dv0 (golden a_local^T + regular matmul): max_diff={diff_golden_input:.6e}")
