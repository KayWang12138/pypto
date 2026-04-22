#!/usr/bin/env python3
"""Test: forward's loop(0, s, step) pattern with rebind."""
import torch, torch_npu, pypto

DEVICE_ID = 4
torch.npu.set_device(DEVICE_ID)
torch.npu.empty_cache()

BT = 32

@pypto.frontend.jit
def test_step_loop(
    dht: pypto.Tensor([pypto.DYNAMIC, 2, 32, 32], pypto.DT_FP32),
    update: pypto.Tensor([pypto.DYNAMIC, 2, 64, 32, 32], pypto.DT_FP32),
    act_seq_len: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    dh0_out: pypto.Tensor([pypto.DYNAMIC, 2, 32, 32], pypto.DT_FP32),
):
    """Use forward's loop(0, s, BT) pattern."""
    pypto.experimental.set_operation_options(combine_axis=True)
    B_val = 1
    for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
        s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        b_ofs = act_seq_len[b_idx]

        for h_idx in pypto.loop(2, name="LOOP_H", idx_name="h_idx"):
            pypto.set_vec_tile_shapes(32, 32)
            d_s = dht[b_idx, h_idx]

            for s_idx in pypto.loop(0, s, BT, name="LOOP_S", idx_name="s_idx",
                                    unroll_list=[16, 1]):
                # Read from update at position s_idx/BT
                chunk_idx = s_idx // BT
                # Since we iterate forward (s_idx=0, BT, 2*BT, ...)
                # But backward needs reverse order, so read from end
                # For this test, just read update[b, h, s_idx//BT]
                u = update[b_idx, h_idx, chunk_idx]

                pypto.set_vec_tile_shapes(32, 32)
                d_s_new = pypto.add(d_s, u)
                d_s[:] = d_s_new

            pypto.set_vec_tile_shapes(32, 32)
            dh0_out[b_idx, h_idx] = d_s

dht = torch.randn(1, 2, 32, 32, device=f"npu:{DEVICE_ID}", dtype=torch.float32)
upd = torch.randn(1, 2, 64 // BT, 32, 32, device=f"npu:{DEVICE_ID}", dtype=torch.float32)
act = torch.tensor([0, 64], dtype=torch.int32, device=f"npu:{DEVICE_ID}")
dh0 = torch.zeros(1, 2, 32, 32, device=f"npu:{DEVICE_ID}", dtype=torch.float32)

test_step_loop(dht, upd, act, dh0)
torch.npu.synchronize()

expected = (dht[0, 0] + upd[0, 0, 0] + upd[0, 0, 1]).cpu()
actual = dh0[0, 0].cpu()
diff = (expected - actual).abs().max().item()
print(f"diff: {diff}")
print("PASSED" if diff < 1e-5 else "FAILED")
