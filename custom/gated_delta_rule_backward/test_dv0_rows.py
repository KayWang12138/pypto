#!/usr/bin/env python3
"""Identify which rows of dv0 are wrong."""

import torch, torch_npu, pypto, math
import sys, os

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

ones = torch.ones(BT, BT)
c_cum = torch.tril(ones)
m_le = torch.tril(ones)

g_cum = c_cum @ gc_raw
decay = torch.exp(g_cum[:, None] - g_cum[None, :])
qk = qc @ kc.t()
a_local = (qk * decay) * m_le
dv0_golden = (a_local.t() @ doc) * scale

@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def exact_dv0(
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
exact_dv0(to_npu(qc), to_npu(kc), to_npu(doc), to_npu(gc_raw),
          to_npu(c_cum), to_npu(m_le), to_npu(torch.tensor([scale])), dv0_out)
torch_npu.npu.synchronize()
dv0_impl = dv0_out.cpu()

# Per-row max diff
row_diffs = (dv0_golden - dv0_impl).abs().max(dim=1).values
worst_rows = row_diffs.argsort(descending=True)[:10]

print("Worst rows:")
for idx in worst_rows:
    print(f"  Row {idx.item():3d}: max_diff={row_diffs[idx]:.6e}, "
          f"golden_max={dv0_golden[idx].abs().max():.6f}, "
          f"impl_max={dv0_impl[idx].abs().max():.6f}, "
          f"g_cum={g_cum[idx]:.6f}")

# Check g_cum range
print(f"\ng_cum range: [{g_cum.min():.6f}, {g_cum.max():.6f}]")
print(f"decay range: [{decay.min():.6f}, {decay.max():.6f}]")
print(f"a_local range: [{a_local.min():.6f}, {a_local.max():.6f}]")

# Check: is the issue in specific columns of a_local?
# For worst row, compare a_local column sums (which contribute to dv0)
worst_row = worst_rows[0].item()
# dv0[worst_row] = sum_j a_local[j, worst_row] * doc[j] * scale
aloc_col = a_local[:, worst_row]  # [BT] — column of a_local for worst output row
print(f"\nWorst row {worst_row}: a_local column sum = {aloc_col.sum():.6f}")
print(f"  a_local[:, {worst_row}] max = {aloc_col.abs().max():.6f}")

# Check if g_cum pattern affects which rows are wrong
print(f"\nAll row diffs vs g_cum:")
for i in range(BT):
    print(f"  Row {i:3d}: diff={row_diffs[i]:.4e}, g_cum={g_cum[i]:.4f}")
