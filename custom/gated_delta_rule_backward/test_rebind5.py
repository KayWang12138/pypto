#!/usr/bin/env python3
"""Test the absolute minimum rebind pattern."""
import torch, torch_npu, pypto

DEVICE_ID = 4
torch.npu.set_device(DEVICE_ID)
torch.npu.empty_cache()

@pypto.frontend.jit
def test_rebind_only(
    dht: pypto.Tensor([pypto.DYNAMIC, 2, 32, 32], pypto.DT_FP32),
    update: pypto.Tensor([pypto.DYNAMIC, 2, 2, 32, 32], pypto.DT_FP32),
    dh0_out: pypto.Tensor([pypto.DYNAMIC, 2, 32, 32], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    B_val = 1
    NT_val = 2
    for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
        for h_idx in pypto.loop(2, name="LOOP_H", idx_name="h_idx"):
            pypto.set_vec_tile_shapes(32, 32)
            d_s = dht[b_idx, h_idx]
            for i_idx in pypto.loop(NT_val, name="LOOP_C", idx_name="i_idx", unroll_list=[16, 1]):
                c = NT_val - 1 - i_idx
                u = update[b_idx, h_idx, c]
                pypto.set_vec_tile_shapes(32, 32)
                d_s_new = pypto.add(d_s, u)
                d_s[:] = d_s_new
            dh0_out[b_idx, h_idx] = d_s

dht = torch.randn(1, 2, 32, 32, device=f"npu:{DEVICE_ID}", dtype=torch.float32)
upd = torch.randn(1, 2, 2, 32, 32, device=f"npu:{DEVICE_ID}", dtype=torch.float32)
dh0 = torch.zeros(1, 2, 32, 32, device=f"npu:{DEVICE_ID}", dtype=torch.float32)

test_rebind_only(dht, upd, dh0)
torch.npu.synchronize()

expected = (dht[0, 0] + upd[0, 0, 1] + upd[0, 0, 0]).cpu()
actual = dh0[0, 0].cpu()
diff = (expected - actual).abs().max().item()
print(f"rebind_only diff: {diff}")
if diff < 1e-5:
    print("PASSED")
else:
    print("FAILED")
