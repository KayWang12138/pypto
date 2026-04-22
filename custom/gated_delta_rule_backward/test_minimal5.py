"""Minimal test #5: test pypto.view on 4D + matmul + in-place update + output write."""
import os
os.environ['TILE_FWK_DEVICE_ID'] = '1'

import pypto
import torch
import torch_npu

K = 128
V = 128
H = 4
BT = 64

@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kernel(
    q_norm: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
    k_norm: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
    dht:    pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
    scale_val: pypto.Tensor([1], pypto.DT_FP32),
    dq_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
    dh0_out: pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
):
    B_val = q_norm.shape[0]
    T_val = q_norm.shape[1]
    NT_val = T_val // BT

    pypto.experimental.set_operation_options(combine_axis=True)

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
                dq_c = pypto.mul(qk, scale_val)

                pypto.set_vec_tile_shapes(128, 128)
                d_s_new = pypto.mul(d_s, d_s)
                d_s[:] = d_s_new

                pypto.set_vec_tile_shapes(128, 128, 128, 128)
                dq_out[b_idx, t0:t0 + BT, h_idx, :] = dq_c

            pypto.set_vec_tile_shapes(128, 128, 128, 128)
            dh0_out[b_idx, h_idx, :, :] = d_s


print("Testing view 4D + matmul + in-place update + output write...")
B, T = 1, 128
q = torch.randn(B, T, H, K, dtype=torch.float32).npu()
k = torch.randn(B, T, H, K, dtype=torch.float32).npu()
dht = torch.randn(B, H, K, V, dtype=torch.float32).npu()
scale = torch.tensor([0.1], dtype=torch.float32).npu()
dq = torch.zeros(B, T, H, K, dtype=torch.float32).npu()
dh0 = torch.zeros(B, H, K, V, dtype=torch.float32).npu()

kernel(q, k, dht, scale, dq, dh0)
torch.npu.synchronize()
print("SUCCESS!")
print(f"dq nan: {torch.isnan(dq.cpu()).any()}")
print(f"dh0 nan: {torch.isnan(dh0.cpu()).any()}")
