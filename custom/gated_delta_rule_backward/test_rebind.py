#!/usr/bin/env python3
"""Test d_s rebind vs explicit tensor pattern."""
import torch
import torch_npu
import pypto

DEVICE_ID = 4
torch.npu.set_device(DEVICE_ID)
torch.npu.empty_cache()

B, T, H, K, V, BT = 1, 64, 2, 32, 32, 32
NT = T // BT

def test_rebind():
    """Test d_s rebind pattern (like forward's last_state)."""
    print("Test: d_s rebind pattern...")

    @pypto.frontend.jit
    def kernel(
        dht:      pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
        kc_cache: pypto.Tensor([pypto.DYNAMIC, H, 128, BT, K], pypto.DT_FP32),
        doc_3d:   pypto.Tensor([pypto.DYNAMIC, H, V], pypto.DT_FP32),
        scale_val: pypto.Tensor([1], pypto.DT_FP32),
        act_seq_len: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
        dh0_out:  pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
    ):
        B_val = 1
        pypto.experimental.set_operation_options(combine_axis=True)

        for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
            s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
            b_ofs = act_seq_len[b_idx]
            NT_val = s // BT

            for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
                pypto.set_vec_tile_shapes(128, 128)
                d_s = dht[b_idx, h_idx]  # rebind [K, V]

                for i_idx in pypto.loop(NT_val, name="LOOP_CHUNK",
                                        idx_name="i_idx", unroll_list=[16, 1]):
                    c = NT_val - 1 - i_idx
                    s_ofs = b_ofs + c * BT

                    kc = kc_cache[b_idx, h_idx, c]  # [BT, K]

                    pypto.set_vec_tile_shapes(BT, 1, V)
                    doc_view = pypto.view(doc_3d, [BT, 1, V], [s_ofs, h_idx, 0])
                    doc = pypto.reshape(doc_view, [BT, V])

                    # d_s_new = d_s + kc^T @ doc * scale
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    d_s_update = pypto.mul(
                        pypto.matmul(kc, doc, pypto.DT_FP32, a_trans=True),
                        scale_val)

                    pypto.set_vec_tile_shapes(128, 128)
                    d_s_new = pypto.add(d_s, d_s_update)

                    pypto.set_vec_tile_shapes(K, V)
                    d_s[:] = d_s_new

                pypto.set_vec_tile_shapes(K, V)
                dh0_out[b_idx, h_idx] = d_s

    dht = torch.randn(B, H, K, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    kc_cache = torch.randn(B, H, 128, BT, K, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    doc_3d = torch.randn(B*T, H, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    scale = torch.tensor([1.0/K**0.5], device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    act = torch.tensor([0, T], dtype=torch.int32, device=f'npu:{DEVICE_ID}')
    dh0 = torch.zeros(B, H, K, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)

    kernel(dht, kc_cache, doc_3d, scale, act, dh0)
    torch.npu.synchronize()
    print("  PASSED")

try:
    test_rebind()
except Exception as e:
    err_str = str(e)
    if "FC4000" in err_str:
        print(f"  FAILED (FC4000 cube tile): {err_str[:200]}")
    elif "FC0000" in err_str:
        print(f"  FAILED (FC0000 shape): {err_str[:200]}")
    elif "F21004" in err_str:
        print(f"  FAILED (F21004 tile shape): {err_str[:200]}")
    elif "FFFFFF" in err_str:
        print(f"  FAILED (FFFFFF generic): {err_str[:200]}")
    else:
        print(f"  FAILED: {err_str[:200]}")

print("\nDone.")
