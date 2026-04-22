"""Minimal test #6: progressive simplification to find FFFFFF root cause."""
import os
os.environ['TILE_FWK_DEVICE_ID'] = '1'

import pypto
import torch
import torch_npu

K, V, H, BT = 128, 128, 4, 64

# Test A: NO combine_axis, NO in-place update
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kernel_a(
    q_norm: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
    k_norm: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
    dht:    pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
    dq_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
):
    B_val = q_norm.shape[0]
    T_val = q_norm.shape[1]
    NT_val = T_val // BT

    for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
        for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
            pypto.set_vec_tile_shapes(128, 128, 128, 128)
            d_s = dht[b_idx, h_idx]

            for i_idx in pypto.loop(NT_val, name="LOOP_CHUNK",
                                    idx_name="i_idx", unroll_list=[16, 1]):
                c = NT_val - 1 - i_idx
                t0 = c * BT

                pypto.set_vec_tile_shapes(128, 128)
                qc_4d = pypto.view(q_norm, [1, BT, 1, K], [b_idx, t0, h_idx, 0])
                qc = pypto.reshape(qc_4d, [BT, K])
                kc_4d = pypto.view(k_norm, [1, BT, 1, K], [b_idx, t0, h_idx, 0])
                kc = pypto.reshape(kc_4d, [BT, K])

                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                qk = pypto.matmul(qc, kc, pypto.DT_FP32, b_trans=True)

                # NO in-place update on d_s
                dq_out[b_idx, t0:t0 + BT, h_idx, :] = qk


# Test B: NO view, just output slice write
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kernel_b(
    q_norm: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
    dq_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
):
    B_val = q_norm.shape[0]
    T_val = q_norm.shape[1]
    NT_val = T_val // BT

    for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
        for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
            for i_idx in pypto.loop(NT_val, name="LOOP_CHUNK",
                                    idx_name="i_idx", unroll_list=[16, 1]):
                c = NT_val - 1 - i_idx
                t0 = c * BT

                pypto.set_vec_tile_shapes(128, 128)
                qc_4d = pypto.view(q_norm, [1, BT, 1, K], [b_idx, t0, h_idx, 0])
                qc = pypto.reshape(qc_4d, [BT, K])

                dq_out[b_idx, t0:t0 + BT, h_idx, :] = qc


B, T = 1, 128

print("=== Test A: view + matmul, NO combine_axis, NO in-place ===")
try:
    q = torch.randn(B, T, H, K, dtype=torch.float32).npu()
    k = torch.randn(B, T, H, K, dtype=torch.float32).npu()
    dht = torch.randn(B, H, K, V, dtype=torch.float32).npu()
    dq = torch.zeros(B, T, H, K, dtype=torch.float32).npu()
    kernel_a(q, k, dht, dq)
    torch.npu.synchronize()
    print("Test A: SUCCESS")
except Exception as e:
    print(f"Test A: FAILED - {str(e)[:200]}")

print("\n=== Test B: view + output write only ===")
try:
    q = torch.randn(B, T, H, K, dtype=torch.float32).npu()
    dq = torch.zeros(B, T, H, K, dtype=torch.float32).npu()
    kernel_b(q, dq)
    torch.npu.synchronize()
    print("Test B: SUCCESS")
except Exception as e:
    print(f"Test B: FAILED - {str(e)[:200]}")
