#!/usr/bin/env python3
"""Test: separate matmul and mul into two kernels to isolate the bug."""

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

# Test 1: Combined kernel (matmul + mul in same kernel)
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_combined(
    a_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
    b_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    scale_in: pypto.Tensor([1], pypto.DT_FP32),
    out: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    raw = pypto.matmul(a_in, b_in, pypto.DT_FP32)
    pypto.set_vec_tile_shapes(128, 128)
    scaled = pypto.mul(raw, scale_in)
    out[:] = scaled

# Test 2: Matmul-only kernel
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_matmul_only(
    a_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
    b_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    out: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    out[:] = pypto.matmul(a_in, b_in, pypto.DT_FP32)

# Test 3: Mul-only kernel
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_mul_only(
    raw_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    scale_in: pypto.Tensor([1], pypto.DT_FP32),
    out: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    out[:] = pypto.mul(raw_in, scale_in)

# --- Run Test 1: Combined ---
print("=== Test 1: Combined matmul+mul kernel ===")
res_combined = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
kern_combined(to_npu(a_local.t().contiguous()), to_npu(doc), to_npu(torch.tensor([scale])), res_combined)
torch_npu.npu.synchronize()
diff_combined = (dv0_golden - res_combined.cpu()).abs()
print(f"max_diff={diff_combined.max():.6e}, mean_diff={diff_combined.mean():.6e}")

# --- Run Test 2: Separate matmul ---
print("\n=== Test 2: Matmul-only kernel ===")
res_matmul = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
kern_matmul_only(to_npu(a_local.t().contiguous()), to_npu(doc), res_matmul)
torch_npu.npu.synchronize()
diff_matmul = ((a_local.t() @ doc) - res_matmul.cpu()).abs()
print(f"matmul max_diff={diff_matmul.max():.6e}")

# --- Run Test 3: Mul on matmul result (separate kernel) ---
print("\n=== Test 3: Mul-only on matmul result (separate kernel) ===")
res_mul = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
kern_mul_only(res_matmul, to_npu(torch.tensor([scale])), res_mul)
torch_npu.npu.synchronize()
diff_separate = (dv0_golden - res_mul.cpu()).abs()
print(f"separate kernels max_diff={diff_separate.max():.6e}, mean_diff={diff_separate.mean():.6e}")

# --- Run Test 3b: Mul on CPU-computed matmul result ---
print("\n=== Test 3b: Mul on CPU matmul result ===")
cpu_matmul = a_local.t() @ doc
res_mul_cpu = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
kern_mul_only(to_npu(cpu_matmul), to_npu(torch.tensor([scale])), res_mul_cpu)
torch_npu.npu.synchronize()
diff_mul_cpu = (dv0_golden - res_mul_cpu.cpu()).abs()
print(f"CPU matmul → NPU mul max_diff={diff_mul_cpu.max():.6e}")

# --- Extra: What if scale is wrong? Check res_combined / res_matmul ---
print("\n=== Ratio check ===")
ratio = res_combined.cpu() / (res_matmul.cpu() + 1e-10)
print(f"ratio (combined/matmul): mean={ratio.mean():.6f}, min={ratio.min():.6f}, max={ratio.max():.6f}")
print(f"Expected scale: {scale:.6f}")
