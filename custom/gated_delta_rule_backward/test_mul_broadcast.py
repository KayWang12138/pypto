#!/usr/bin/env python3
"""Test: is the mul by scalar [1] broken? Test with full-size scale tensor."""

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
raw_golden = a_local.t() @ doc  # unscaled

# Test A: mul with [1] scalar
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_mul_scalar(
    raw_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    scale_in: pypto.Tensor([1], pypto.DT_FP32),
    out: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    out[:] = pypto.mul(raw_in, scale_in)

# Test B: mul with [BT, VD] full tensor (no broadcast)
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_mul_full(
    raw_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    scale_full: pypto.Tensor([BT, VD], pypto.DT_FP32),
    out: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    out[:] = pypto.mul(raw_in, scale_full)

# Test C: mul with scalar via pypto.full (constant tensor)
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_mul_full_const(
    raw_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    out: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    s = pypto.full([BT, VD], scale, pypto.DT_FP32)
    out[:] = pypto.mul(raw_in, s)

# Test D: No mul at all, just copy (to verify input is correct)
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_copy(
    raw_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    out: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    out[:] = raw_in

# Run tests with perfect CPU input
raw_npu = to_npu(raw_golden)

print("=== Test D: Copy (verify input) ===")
res_copy = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
kern_copy(raw_npu, res_copy)
torch_npu.npu.synchronize()
diff_copy = (raw_golden - res_copy.cpu()).abs()
print(f"Copy max_diff: {diff_copy.max():.6e}")

print("\n=== Test A: Mul by scalar [1] ===")
res_scalar = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
kern_mul_scalar(raw_npu, to_npu(torch.tensor([scale])), res_scalar)
torch_npu.npu.synchronize()
diff_scalar = (dv0_golden - res_scalar.cpu()).abs()
print(f"Mul scalar [1] max_diff: {diff_scalar.max():.6e}")
print(f"Result range: [{res_scalar.cpu().min():.4f}, {res_scalar.cpu().max():.4f}]")
print(f"Golden range: [{dv0_golden.min():.4f}, {dv0_golden.max():.4f}]")

print("\n=== Test B: Mul by full [BT,VD] tensor ===")
scale_full = torch.full((BT, VD), scale)
res_full = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
kern_mul_full(raw_npu, to_npu(scale_full), res_full)
torch_npu.npu.synchronize()
diff_full = (dv0_golden - res_full.cpu()).abs()
print(f"Mul full [BT,VD] max_diff: {diff_full.max():.6e}")
print(f"Result range: [{res_full.cpu().min():.4f}, {res_full.cpu().max():.4f}]")

print("\n=== Test C: Mul by pypto.full constant ===")
res_const = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
kern_mul_full_const(raw_npu, res_const)
torch_npu.npu.synchronize()
diff_const = (dv0_golden - res_const.cpu()).abs()
print(f"Mul pypto.full max_diff: {diff_const.max():.6e}")
print(f"Result range: [{res_const.cpu().min():.4f}, {res_const.cpu().max():.4f}]")

# Extra sanity: simple mul with [1] on small random data
print("\n=== Test E: Simple mul [4,4] * [1] on random data ===")
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_mul_simple(
    a_in: pypto.Tensor([4, 4], pypto.DT_FP32),
    b_in: pypto.Tensor([1], pypto.DT_FP32),
    out: pypto.Tensor([4, 4], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    out[:] = pypto.mul(a_in, b_in)

torch.manual_seed(123)
a_simple = torch.randn(4, 4)
s_simple = torch.tensor([2.5])
golden_simple = a_simple * s_simple
res_simple = torch.zeros(4, 4, device=npu, dtype=torch.float32)
kern_mul_simple(to_npu(a_simple), to_npu(s_simple), res_simple)
torch_npu.npu.synchronize()
diff_simple = (golden_simple - res_simple.cpu()).abs()
print(f"Simple mul max_diff: {diff_simple.max():.6e}")
