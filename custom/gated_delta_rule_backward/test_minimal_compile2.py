"""Minimal compile test #2: use rebind pattern like forward.

The forward uses:
    last_state = pypto.tensor([d, d], DT_FP32)  # outside loops
    last_state = states[b_idx, nv_idx]            # REBIND to input view
    last_state[:] = cur_state                     # in-place update

Our backward should use:
    d_s = dht[b_idx, h_idx]                       # REBIND to input view
    d_s[:] = d_s_new                              # in-place update
    dh0_out[b_idx, h_idx] = d_s                   # copy to output
"""
import os
os.environ['TILE_FWK_DEVICE_ID'] = '1'

import pypto
import torch
import torch_npu

def make_test_kernel(K, V, H, BT):
    """Minimal kernel using forward's rebind pattern."""
    @pypto.frontend.jit(
        runtime_options={"stitch_function_max_num": 128},
    )
    def kernel(
        q_norm: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        k_norm: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        dht:    pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
        dq_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        dh0_out: pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
    ):
        B_val = q_norm.shape[0]
        T_val = q_norm.shape[1]
        NT_val = T_val // BT

        for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
            for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
                # REBIND pattern like forward: d_s becomes a view of dht
                d_s = dht[b_idx, h_idx]

                for i_idx in pypto.loop(NT_val, name="LOOP_CHUNK",
                                        idx_name="i_idx",
                                        unroll_list=[16, 1]):
                    c = NT_val - 1 - i_idx
                    t0 = c * BT

                    pypto.set_vec_tile_shapes(128, 128)
                    qc_4d = pypto.view(q_norm, [1, BT, 1, K], [b_idx, t0, h_idx, 0])
                    qc = pypto.reshape(qc_4d, [BT, K])
                    kc_4d = pypto.view(k_norm, [1, BT, 1, K], [b_idx, t0, h_idx, 0])
                    kc = pypto.reshape(kc_4d, [BT, K])

                    # Use d_s in computation
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    dq_raw_c = pypto.matmul(qc, kc, pypto.DT_FP32, b_trans=True)

                    # Update d_s in-place (writes to dht through view)
                    pypto.set_vec_tile_shapes(128, 128)
                    d_s_new = pypto.mul(d_s, d_s)
                    d_s[:] = d_s_new

                    dq_out[b_idx, t0:t0 + BT, h_idx, :] = dq_raw_c

                # Copy final d_s to output
                dh0_out[b_idx, h_idx, :, :] = d_s
    return kernel


print("Creating test kernel (rebind pattern)...")
B, T, H, K, V, BT = 1, 128, 4, 128, 128, 64

kernel = make_test_kernel(K, V, H, BT)

print("Creating tensors...")
q = torch.randn(B, T, H, K, dtype=torch.float32).npu()
k = torch.randn(B, T, H, K, dtype=torch.float32).npu()
dht = torch.randn(B, H, K, V, dtype=torch.float32).npu()
dq = torch.zeros(B, T, H, K, dtype=torch.float32).npu()
dh0 = torch.zeros(B, H, K, V, dtype=torch.float32).npu()

print("Running kernel (compile + execute)...")
kernel(q, k, dht, dq, dh0)
torch.npu.synchronize()

print("SUCCESS: Minimal kernel compiled and ran!")
print(f"dq shape: {dq.shape}, has nan: {torch.isnan(dq.cpu()).any()}")
print(f"dh0 shape: {dh0.shape}")
