#!/usr/bin/env python3
"""Test: matmul(computed_tensor, B, a_trans=True) vs matmul(input_tensor, B, a_trans=True)."""

import torch, torch_npu, pypto
import sys, os

device_id = 4
torch.npu.set_device(device_id)
npu = f'npu:{device_id}'
BT = 64
VD = 128

def to_npu(t):
    return t.clone().contiguous().to(npu).to(torch.float32)

torch.manual_seed(42)
A = torch.randn(BT, BT) * 0.1
B = torch.randn(BT, VD) * 0.1
C = torch.randn(BT, BT) * 0.1  # for mul chain

golden_direct = A.t() @ B
golden_chain = (A * C).t() @ B

# Test 1: Direct input, a_trans=True (should work)
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def test_direct(
    A_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
    B_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    out: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    r = pypto.matmul(A_in, B_in, pypto.DT_FP32, a_trans=True)
    out[:] = r

out1 = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
test_direct(to_npu(A), to_npu(B), out1)
torch_npu.npu.synchronize()

# Test 2: Computed tensor (mul chain), a_trans=True
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def test_chain_atrans(
    A_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
    C_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
    B_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    out: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    computed = pypto.mul(A_in, C_in)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    r = pypto.matmul(computed, B_in, pypto.DT_FP32, a_trans=True)
    out[:] = r

out2 = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
test_chain_atrans(to_npu(A), to_npu(C), to_npu(B), out2)
torch_npu.npu.synchronize()

# Test 3: Computed tensor, explicit transpose then matmul
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def test_chain_transpose(
    A_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
    C_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
    B_in: pypto.Tensor([BT, VD], pypto.DT_FP32),
    out: pypto.Tensor([BT, VD], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    computed = pypto.mul(A_in, C_in)
    pypto.set_vec_tile_shapes(128, 128)
    computed_T = computed.transpose(0, 1)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    r = pypto.matmul(computed_T, B_in, pypto.DT_FP32)
    out[:] = r

out3 = torch.zeros(BT, VD, device=npu, dtype=torch.float32)
test_chain_transpose(to_npu(A), to_npu(C), to_npu(B), out3)
torch_npu.npu.synchronize()

# Results
print("=== matmul a_trans on computed tensor ===")
d1 = (golden_direct - out1.cpu()).abs().max()
d2 = (golden_chain - out2.cpu()).abs().max()
d3 = (golden_chain - out3.cpu()).abs().max()

print(f"  Test 1 (direct input, a_trans):     max_diff={d1:.6e}")
print(f"  Test 2 (computed mul, a_trans):     max_diff={d2:.6e}")
print(f"  Test 3 (computed mul, transpose()): max_diff={d3:.6e}")

print(f"\n  golden_chain range: [{golden_chain.min():.6f}, {golden_chain.max():.6f}]")
print(f"  impl_chain_atrans:  [{out2.cpu().min():.6f}, {out2.cpu().max():.6f}]")
print(f"  impl_chain_transp:  [{out3.cpu().min():.6f}, {out3.cpu().max():.6f}]")

# Row comparison
print(f"\n  Row 0: chain_golden={golden_chain[0,:5].tolist()}")
print(f"         chain_atrans ={out2.cpu()[0,:5].tolist()}")
print(f"         chain_transp ={out3.cpu()[0,:5].tolist()}")
