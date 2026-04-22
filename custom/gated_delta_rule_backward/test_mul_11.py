#!/usr/bin/env python3
"""Test: is mul broadcasting from [1,1] to [K,V] also broken?"""

import torch, torch_npu, pypto, math

device_id = 4
torch.npu.set_device(device_id)
npu = f'npu:{device_id}'
K = 128; V = 128

def to_npu(t):
    return t.clone().contiguous().to(npu).to(torch.float32)

torch.manual_seed(42)
a = torch.randn(K, V) * 2.0
b_scalar = torch.tensor([[0.5]])  # [1, 1]
golden = a * b_scalar

@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_mul_11(
    a_in: pypto.Tensor([K, V], pypto.DT_FP32),
    b_in: pypto.Tensor([1, 1], pypto.DT_FP32),
    out: pypto.Tensor([K, V], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    out[:] = pypto.mul(a_in, b_in)

@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_mul_full_kv(
    a_in: pypto.Tensor([K, V], pypto.DT_FP32),
    b_full: pypto.Tensor([K, V], pypto.DT_FP32),
    out: pypto.Tensor([K, V], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    out[:] = pypto.mul(a_in, b_full)

# Test 1: mul [K,V] * [1,1]
print("=== Test 1: mul([128,128], [1,1]) ===")
res_11 = torch.zeros(K, V, device=npu, dtype=torch.float32)
kern_mul_11(to_npu(a), to_npu(b_scalar), res_11)
torch_npu.npu.synchronize()
diff_11 = (golden - res_11.cpu()).abs()
print(f"max_diff={diff_11.max():.6e}, mean_diff={diff_11.mean():.6e}")
# Check per-row errors
for i in range(0, K, 8):
    rd = diff_11[i].max().item()
    if rd > 0.001:
        print(f"  Row {i}: max_diff={rd:.4f}")

# Test 2: mul [K,V] * [K,V] full
print("\n=== Test 2: mul([128,128], [128,128]) full ===")
b_full = torch.full((K, V), 0.5)
res_full = torch.zeros(K, V, device=npu, dtype=torch.float32)
kern_mul_full_kv(to_npu(a), to_npu(b_full), res_full)
torch_npu.npu.synchronize()
diff_full = (golden - res_full.cpu()).abs()
print(f"max_diff={diff_full.max():.6e}")

# Test 3: mul [K,V] * pypto.full
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_mul_const(
    a_in: pypto.Tensor([K, V], pypto.DT_FP32),
    out: pypto.Tensor([K, V], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    out[:] = pypto.mul(a_in, pypto.full([K, V], 0.5, pypto.DT_FP32))

print("\n=== Test 3: mul([128,128], pypto.full([128,128], 0.5)) ===")
res_const = torch.zeros(K, V, device=npu, dtype=torch.float32)
kern_mul_const(to_npu(a), res_const)
torch_npu.npu.synchronize()
diff_const = (golden - res_const.cpu()).abs()
print(f"max_diff={diff_const.max():.6e}")
