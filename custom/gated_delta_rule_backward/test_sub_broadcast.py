#!/usr/bin/env python3
"""Test: is sub broadcasting from [1,1] to [BT,1] broken?"""

import torch, torch_npu, pypto, math

device_id = 4
torch.npu.set_device(device_id)
npu = f'npu:{device_id}'
BT = 64

def to_npu(t):
    return t.clone().contiguous().to(npu).to(torch.float32)

torch.manual_seed(42)
a_11 = torch.tensor([[3.0]])  # [1, 1]
b_bt1 = torch.randn(BT, 1) * 2.0
golden = a_11 - b_bt1  # [BT, 1]

@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_sub_11_bt1(
    a_in: pypto.Tensor([1, 1], pypto.DT_FP32),
    b_in: pypto.Tensor([BT, 1], pypto.DT_FP32),
    out: pypto.Tensor([BT, 1], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    out[:] = pypto.sub(a_in, b_in)

print("=== Test: sub([1,1], [BT,1]) ===")
res = torch.zeros(BT, 1, device=npu, dtype=torch.float32)
kern_sub_11_bt1(to_npu(a_11), to_npu(b_bt1), res)
torch_npu.npu.synchronize()
diff = (golden - res.cpu()).abs()
print(f"max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}")

# Also test exp(sub([1,1], [BT,1]))
golden_exp = torch.exp(a_11 - b_bt1)

@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_exp_sub(
    a_in: pypto.Tensor([1, 1], pypto.DT_FP32),
    b_in: pypto.Tensor([BT, 1], pypto.DT_FP32),
    out: pypto.Tensor([BT, 1], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    out[:] = pypto.exp(pypto.sub(a_in, b_in))

print("\n=== Test: exp(sub([1,1], [BT,1])) ===")
res2 = torch.zeros(BT, 1, device=npu, dtype=torch.float32)
kern_exp_sub(to_npu(a_11), to_npu(b_bt1), res2)
torch_npu.npu.synchronize()
diff2 = (golden_exp - res2.cpu()).abs()
print(f"max_diff={diff2.max():.6e}, mean_diff={diff2.mean():.6e}")

# Test mul([BT,V], [BT,1]) to verify per-row broadcast works
V = 128
torch.manual_seed(42)
a_btv = torch.randn(BT, V) * 2.0
b_bt1_v = torch.randn(BT, 1) * 0.5
golden_btv = a_btv * b_bt1_v

@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kern_mul_bt1(
    a_in: pypto.Tensor([BT, V], pypto.DT_FP32),
    b_in: pypto.Tensor([BT, 1], pypto.DT_FP32),
    out: pypto.Tensor([BT, V], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    out[:] = pypto.mul(a_in, b_in)

print("\n=== Test: mul([BT,V], [BT,1]) ===")
res3 = torch.zeros(BT, V, device=npu, dtype=torch.float32)
kern_mul_bt1(to_npu(a_btv), to_npu(b_bt1_v), res3)
torch_npu.npu.synchronize()
diff3 = (golden_btv - res3.cpu()).abs()
print(f"max_diff={diff3.max():.6e}, mean_diff={diff3.mean():.6e}")
