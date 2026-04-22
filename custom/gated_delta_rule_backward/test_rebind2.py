#!/usr/bin/env python3
"""Test d_s rebind WITHOUT 3D view — just cache + matmul + rebind."""
import torch
import torch_npu
import pypto

DEVICE_ID = 4
torch.npu.set_device(DEVICE_ID)
torch.npu.empty_cache()

B, H, K, V, BT = 1, 2, 32, 32, 32

def test_rebind_no_view():
    """Test: just cache indexing + rebind + matmul, no 3D view."""
    print("Test: rebind without 3D view...")

    @pypto.frontend.jit
    def kernel(
        dht:      pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
        kc_cache: pypto.Tensor([pypto.DYNAMIC, H, 2, BT, K], pypto.DT_FP32),
        doc_cache: pypto.Tensor([pypto.DYNAMIC, H, 2, BT, V], pypto.DT_FP32),
        scale_val: pypto.Tensor([1], pypto.DT_FP32),
        dh0_out:  pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
    ):
        B_val = 1
        NT_val = 2
        pypto.experimental.set_operation_options(combine_axis=True)

        for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
            for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
                pypto.set_vec_tile_shapes(128, 128)
                d_s = dht[b_idx, h_idx]  # rebind [K, V]

                for i_idx in pypto.loop(NT_val, name="LOOP_CHUNK",
                                        idx_name="i_idx", unroll_list=[16, 1]):
                    c = NT_val - 1 - i_idx
                    kc = kc_cache[b_idx, h_idx, c]  # [BT, K]
                    doc = doc_cache[b_idx, h_idx, c]  # [BT, V]

                    # d_s_new = d_s + kc^T @ doc * scale
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    d_s_update = pypto.mul(
                        pypto.matmul(kc, doc, pypto.DT_FP32, a_trans=True),
                        scale_val)

                    pypto.set_vec_tile_shapes(K, V)
                    d_s_new = pypto.add(d_s, d_s_update)
                    d_s[:] = d_s_new

                pypto.set_vec_tile_shapes(K, V)
                dh0_out[b_idx, h_idx] = d_s

    dht = torch.randn(B, H, K, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    kc_cache = torch.randn(B, H, 2, BT, K, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    doc_cache = torch.randn(B, H, 2, BT, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    scale = torch.tensor([1.0/K**0.5], device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    dh0 = torch.zeros(B, H, K, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)

    kernel(dht, kc_cache, doc_cache, scale, dh0)
    torch.npu.synchronize()
    print("  PASSED")

try:
    test_rebind_no_view()
except Exception as e:
    err_str = str(e)
    print(f"  FAILED: {err_str[:300]}")

print("Done.")
