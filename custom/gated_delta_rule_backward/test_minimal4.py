"""Minimal compile test #4: set tile shapes before indexing."""
import os
os.environ['TILE_FWK_DEVICE_ID'] = '1'

import pypto
import torch
import torch_npu

K = 128
V = 128
H = 4

@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},
)
def kernel(
    q_norm: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
    dht:    pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
    dh0_out: pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
):
    B_val = q_norm.shape[0]

    for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
        for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
            pypto.set_vec_tile_shapes(128, 128)
            d_s = dht[b_idx, h_idx]
            dh0_out[b_idx, h_idx, :, :] = d_s

print("Testing with set_vec_tile_shapes before indexing...")
B, T = 1, 128
q = torch.randn(B, T, H, K, dtype=torch.float32).npu()
dht = torch.randn(B, H, K, V, dtype=torch.float32).npu()
dh0 = torch.zeros(B, H, K, V, dtype=torch.float32).npu()

kernel(q, dht, dh0)
torch.npu.synchronize()
print("SUCCESS: kernel compiled and ran!")
print(f"dh0 shape: {dh0.shape}, nan: {torch.isnan(dh0.cpu()).any()}")
