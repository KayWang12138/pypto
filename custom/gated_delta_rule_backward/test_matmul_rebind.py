#!/usr/bin/env python3
"""Test: rebind + matmul-based update (like forward)."""
import torch, torch_npu, pypto

DEVICE_ID = 4
torch.npu.set_device(DEVICE_ID)
torch.npu.empty_cache()

@pypto.frontend.jit
def test_matmul_rebind(
    dht: pypto.Tensor([pypto.DYNAMIC, 2, 32, 32], pypto.DT_FP32),
    x_cache: pypto.Tensor([pypto.DYNAMIC, 2, 2, 32, 32], pypto.DT_FP32),
    w_cache: pypto.Tensor([pypto.DYNAMIC, 2, 2, 32, 32], pypto.DT_FP32),
    dh0_out: pypto.Tensor([pypto.DYNAMIC, 2, 32, 32], pypto.DT_FP32),
):
    """Use matmul-based update like the forward's recurrent_state_attn_all."""
    pypto.experimental.set_operation_options(combine_axis=True)
    B_val = 1
    NT_val = 2
    for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
        for h_idx in pypto.loop(2, name="LOOP_H", idx_name="h_idx"):
            pypto.set_vec_tile_shapes(32, 32)
            d_s = dht[b_idx, h_idx]

            for i_idx in pypto.loop(NT_val, name="LOOP_C", idx_name="i_idx", unroll_list=[16, 1]):
                c = NT_val - 1 - i_idx
                x = x_cache[b_idx, h_idx, c]
                w = w_cache[b_idx, h_idx, c]

                # Forward-style: cur_state = f(d_s) via matmul
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                contrib = pypto.matmul(x, d_s, pypto.DT_FP32)

                pypto.set_vec_tile_shapes(32, 32)
                d_s_new = pypto.add(d_s, contrib)
                d_s_new = pypto.add(d_s_new, w)
                d_s[:] = d_s_new

            pypto.set_vec_tile_shapes(32, 32)
            dh0_out[b_idx, h_idx] = d_s

dht = torch.randn(1, 2, 32, 32, device=f"npu:{DEVICE_ID}", dtype=torch.float32)
x_cache = torch.randn(1, 2, 2, 32, 32, device=f"npu:{DEVICE_ID}", dtype=torch.float32)
w_cache = torch.randn(1, 2, 2, 32, 32, device=f"npu:{DEVICE_ID}", dtype=torch.float32)
dh0 = torch.zeros(1, 2, 32, 32, device=f"npu:{DEVICE_ID}", dtype=torch.float32)

test_matmul_rebind(dht, x_cache, w_cache, dh0)
torch.npu.synchronize()

import numpy as np
dht_np = dht[0, 0].cpu().numpy()
x_np = x_cache[0, 0, 1].cpu().numpy()
w_np = w_cache[0, 0, 1].cpu().numpy()
x2_np = x_cache[0, 0, 0].cpu().numpy()
w2_np = w_cache[0, 0, 0].cpu().numpy()

s1 = dht_np + x_np @ dht_np + w_np
expected = s1 + x2_np @ s1 + w2_np
actual = dh0[0, 0].cpu().numpy()
diff = np.abs(expected - actual).max()
print(f"diff: {diff}")
print("PASSED" if diff < 1e-3 else "FAILED")
