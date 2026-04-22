"""Test #7: Does pypto.view work on 4D tensors at all?"""
import os
os.environ['TILE_FWK_DEVICE_ID'] = '1'

import pypto
import torch
import torch_npu

K, H, BT = 128, 4, 64

# Test with 3D tensor (like forward)
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kernel_3d(
    q_3d: pypto.Tensor([pypto.DYNAMIC, H, K], pypto.DT_FP32),
    dq_3d: pypto.Tensor([pypto.DYNAMIC, H, K], pypto.DT_FP32),
):
    T_val = q_3d.shape[0]
    NT_val = T_val // BT

    for i_idx in pypto.loop(NT_val, name="LOOP", idx_name="i_idx", unroll_list=[16, 1]):
        t0 = i_idx * BT
        pypto.set_vec_tile_shapes(128, 128)
        qc = pypto.view(q_3d, [BT, 1, K], [t0, 0, 0])
        qc_2d = pypto.reshape(qc, [BT, K])
        dq_3d[t0:t0 + BT, 0, :] = qc_2d


# Test with 4D tensor (our case)
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kernel_4d(
    q_4d: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
    dq_4d: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
):
    B_val = q_4d.shape[0]
    T_val = q_4d.shape[1]
    NT_val = T_val // BT

    for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
        for i_idx in pypto.loop(NT_val, name="LOOP_C", idx_name="i_idx", unroll_list=[16, 1]):
            t0 = i_idx * BT
            pypto.set_vec_tile_shapes(128, 128)
            qc = pypto.view(q_4d, [1, BT, 1, K], [b_idx, t0, 0, 0])
            qc_2d = pypto.reshape(qc, [BT, K])
            dq_4d[b_idx, t0:t0 + BT, 0, :] = qc_2d


T = 128

print("=== Test 3D view ===")
try:
    q3 = torch.randn(T, H, K, dtype=torch.float32).npu()
    dq3 = torch.zeros(T, H, K, dtype=torch.float32).npu()
    kernel_3d(q3, dq3)
    torch.npu.synchronize()
    print("3D view: SUCCESS")
except Exception as e:
    print(f"3D view: FAILED - {str(e)[:200]}")

print("\n=== Test 4D view ===")
try:
    q4 = torch.randn(1, T, H, K, dtype=torch.float32).npu()
    dq4 = torch.zeros(1, T, H, K, dtype=torch.float32).npu()
    kernel_4d(q4, dq4)
    torch.npu.synchronize()
    print("4D view: SUCCESS")
except Exception as e:
    print(f"4D view: FAILED - {str(e)[:200]}")
