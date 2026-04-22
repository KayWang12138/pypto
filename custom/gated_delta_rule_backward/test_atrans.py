#!/usr/bin/env python3
"""Test if pypto.matmul with a_trans=True is correct."""

import torch, torch_npu, pypto, math
import sys, os

device_id = 4
torch.npu.set_device(device_id)
npu = f'npu:{device_id}'
BT = 64
VD = 128

def to_npu(t):
    return t.clone().contiguous().to(npu).to(torch.float32)

torch.manual_seed(42)
# Create test matrices
A = torch.randn(BT, BT) * 0.1   # a_local shape
B = torch.randn(BT, VD) * 0.1   # doc shape

# Golden: A^T @ B
golden = A.t() @ B

# Test 1: matmul(A, B, a_trans=True) — should be A^T @ B
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def test_atrans(
    A_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
    B_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    out1: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    result = pypto.matmul(A_in, B_in, pypto.DT_FP32, a_trans=True)
    out1[:] = result

out1 = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
test_atrans(to_npu(A), to_npu(B), out1)
torch_npu.npu.synchronize()
diff1 = (golden - out1.cpu()).abs()
print(f"Test 1: matmul(A, B, a_trans=True) — should be A^T @ B")
print(f"  max_diff={diff1.max():.6e}, mean={diff1.mean():.6e}")
print(f"  golden range: [{golden.min():.6f}, {golden.max():.6f}]")
print(f"  impl range: [{out1.cpu().min():.6f}, {out1.cpu().max():.6f}]")

# Test 2: matmul(A, B) without transpose — should be A @ B
golden2 = A @ B
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def test_notrans(
    A_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
    B_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    out2: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    result = pypto.matmul(A_in, B_in, pypto.DT_FP32)
    out2[:] = result

out2 = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
test_notrans(to_npu(A), to_npu(B), out2)
torch_npu.npu.synchronize()
diff2 = (golden2 - out2.cpu()).abs()
print(f"\nTest 2: matmul(A, B) — should be A @ B")
print(f"  max_diff={diff2.max():.6e}, mean={diff2.mean():.6e}")
print(f"  golden range: [{golden2.min():.6f}, {golden2.max():.6f}]")
print(f"  impl range: [{out2.cpu().min():.6f}, {out2.cpu().max():.6f}]")

# Cross-check: is impl1 (a_trans) equal to golden2 (no trans)?
cross = (out1.cpu() - golden2).abs()
print(f"\nCross: impl_a_trans vs golden_A@B (no trans): max_diff={cross.max():.6e}")

# Row 0 comparison
print(f"\nRow 0: golden_A^T@B={golden[0,:5].tolist()}")
print(f"       impl_a_trans ={out1.cpu()[0,:5].tolist()}")
print(f"       golden_A@B  ={golden2[0,:5].tolist()}")
print(f"       impl_notrans={out2.cpu()[0,:5].tolist()}")
