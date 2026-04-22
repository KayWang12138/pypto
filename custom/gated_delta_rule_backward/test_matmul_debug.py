#!/usr/bin/env python3
"""Investigate: what exactly is wrong with the matmul output?"""

import torch, torch_npu, pypto, math, numpy as np

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
a_local_t = a_local.t().contiguous()

# Golden reference
dv0_golden = a_local_t @ doc

# Simple kernel: just matmul(A, B) where A=a_local^T, B=doc
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_matmul(
    a_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
    b_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    c_out: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    c_out[:] = pypto.matmul(a_in, b_in, pypto.DT_FP32)

# Test with golden a_local^T
result = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
kern_matmul(to_npu(a_local_t), to_npu(doc), result)
torch_npu.npu.synchronize()
result_cpu = result.cpu()

diff = (dv0_golden - result_cpu).abs()
print("=== Per-row max diff ===")
for i in range(BT):
    row_diff = diff[i].max().item()
    golden_row_max = dv0_golden[i].abs().max().item()
    result_row_max = result_cpu[i].abs().max().item()
    if row_diff > 0.01:
        print(f"Row {i:3d}: max_diff={row_diff:.4f}, golden_max={golden_row_max:.4f}, result_max={result_row_max:.4f}")

print(f"\n=== Summary ===")
print(f"Overall max_diff: {diff.max():.4f}")
print(f"Overall mean_diff: {diff.mean():.4f}")
print(f"Golden value range: [{dv0_golden.min():.4f}, {dv0_golden.max():.4f}]")
print(f"Result value range: [{result_cpu.min():.4f}, {result_cpu.max():.4f}]")
print(f"Num zero rows in result: {(result_cpu.abs().sum(dim=1) == 0).sum()}")
print(f"Num NaN in result: {torch.isnan(result_cpu).sum()}")
print(f"Num Inf in result: {torch.isinf(result_cpu).sum()}")

# Also test with completely random inputs of same shape
print(f"\n=== Random [64,64]@[64,128] matmul test ===")
torch.manual_seed(99)
A_rand = torch.randn(BT, BT)
B_rand = torch.randn(BT, VD)
golden_rand = A_rand @ B_rand

result_rand = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
kern_matmul(to_npu(A_rand), to_npu(B_rand), result_rand)
torch_npu.npu.synchronize()
diff_rand = (golden_rand - result_rand.cpu()).abs()
print(f"Random matmul max_diff: {diff_rand.max():.6e}")
print(f"Random matmul mean_diff: {diff_rand.mean():.6e}")

# Test with a_local-like structure but simpler
print(f"\n=== Lower-triangular random [64,64]@[64,128] matmul test ===")
A_tril = torch.randn(BT, BT) * torch.tril(torch.ones(BT, BT))
golden_tril = A_tril @ B_rand

result_tril = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
kern_matmul(to_npu(A_tril), to_npu(B_rand), result_tril)
torch_npu.npu.synchronize()
diff_tril = (golden_tril - result_tril.cpu()).abs()
print(f"Lower-tri matmul max_diff: {diff_tril.max():.6e}")
