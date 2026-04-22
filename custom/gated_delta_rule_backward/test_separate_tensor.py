#!/usr/bin/env python3
"""Test: separate tensor approach (no rebind)."""
import torch, torch_npu, pypto

DEVICE_ID = 4
torch.npu.set_device(DEVICE_ID)
torch.npu.empty_cache()

@pypto.frontend.jit
def test_separate_tensor(
    dht: pypto.Tensor([pypto.DYNAMIC, 2, 32, 32], pypto.DT_FP32),
    update: pypto.Tensor([pypto.DYNAMIC, 2, 2, 32, 32], pypto.DT_FP32),
    dh0_out: pypto.Tensor([pypto.DYNAMIC, 2, 32, 32], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    B_val = 1
    NT_val = 2
    for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
        for h_idx in pypto.loop(2, name="LOOP_H", idx_name="h_idx"):
            # Use DH0_OUT as scratch space, not rebind
            # Initialize: dh0_out[b,h] = dht[b,h]
            pypto.set_vec_tile_shapes(32, 32)
            dh0_out[b_idx, h_idx] = dht[b_idx, h_idx]

            for i_idx in pypto.loop(NT_val, name="LOOP_C", idx_name="i_idx", unroll_list=[16, 1]):
                c = NT_val - 1 - i_idx
                u = update[b_idx, h_idx, c]

                # Read current state from output
                pypto.set_vec_tile_shapes(32, 32)
                d_s_cur = dh0_out[b_idx, h_idx]
                d_s_new = pypto.add(d_s_cur, u)
                dh0_out[b_idx, h_idx] = d_s_new

dht = torch.randn(1, 2, 32, 32, device=f"npu:{DEVICE_ID}", dtype=torch.float32)
upd = torch.randn(1, 2, 2, 32, 32, device=f"npu:{DEVICE_ID}", dtype=torch.float32)
dh0 = torch.zeros(1, 2, 32, 32, device=f"npu:{DEVICE_ID}", dtype=torch.float32)

test_separate_tensor(dht, upd, dh0)
torch.npu.synchronize()

expected = (dht[0, 0] + upd[0, 0, 1] + upd[0, 0, 0]).cpu()
actual = dh0[0, 0].cpu()
diff = (expected - actual).abs().max().item()
print(f"diff: {diff}")
if diff < 1e-5:
    print("PASSED")
else:
    print("FAILED")
