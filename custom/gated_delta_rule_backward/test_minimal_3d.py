#!/usr/bin/env python3
"""Minimal 3D view pattern test to verify the approach works."""
import torch
import torch_npu
import pypto

DEVICE_ID = 4
torch.npu.set_device(DEVICE_ID)
torch.npu.empty_cache()

@pypto.frontend.jit
def simple_3d_test(
    inp: pypto.Tensor([pypto.DYNAMIC, 4, 64], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, 4, 64], pypto.DT_FP32),
    act_seq_len: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
):
    B = 1
    H = 4
    D = 64
    BT = 16
    for b_idx in pypto.loop(B, name="LOOP_B", idx_name="b_idx"):
        s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        b_ofs = act_seq_len[b_idx]
        for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
            for s_idx in pypto.loop(0, s, BT, name="LOOP_S", idx_name="s_idx", unroll_list=[16, 1]):
                bs_ofs = b_ofs + s_idx
                pypto.set_vec_tile_shapes(BT, 1, D)
                v = pypto.view(inp, [BT, 1, D], [bs_ofs, h_idx, 0])
                v2d = pypto.reshape(v, [BT, D])
                pypto.set_vec_tile_shapes(BT, D)
                result = pypto.mul(v2d, v2d)
                out[bs_ofs:bs_ofs + BT, h_idx] = result

inp = torch.randn(128, 4, 64, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
out = torch.zeros(128, 4, 64, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
act_seq_len = torch.tensor([0, 128], dtype=torch.int32, device=f'npu:{DEVICE_ID}')

simple_3d_test(inp, out, act_seq_len)
torch.npu.synchronize()

expected = (inp ** 2).cpu()
actual = out.cpu()
max_diff = (expected - actual).abs().max().item()
print(f"Max diff: {max_diff}")
if max_diff < 1e-5:
    print("3D view test PASSED")
else:
    print("3D view test FAILED")
