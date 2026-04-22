#!/usr/bin/env python3
"""Test: isolate the exact pattern that fails."""
import torch
import torch_npu
import pypto

DEVICE_ID = 4
torch.npu.set_device(DEVICE_ID)
torch.npu.empty_cache()

B, H, K, V = 1, 2, 32, 32

def test_simple_rebind_update():
    """Test: simple rebind + update in loop."""
    print("Test: simple rebind + update...")

    @pypto.frontend.jit
    def kernel(
        dht:      pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
        kc_cache: pypto.Tensor([pypto.DYNAMIC, H, 2, K, V], pypto.DT_FP32),
        dh0_out:  pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
    ):
        B_val = 1
        NT_val = 2
        pypto.experimental.set_operation_options(combine_axis=True)

        for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
            for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
                pypto.set_vec_tile_shapes(K, V)
                d_s = dht[b_idx, h_idx]  # rebind [K, V]

                for i_idx in pypto.loop(NT_val, name="LOOP_CHUNK",
                                        idx_name="i_idx", unroll_list=[16, 1]):
                    c = NT_val - 1 - i_idx
                    update = kc_cache[b_idx, h_idx, c]  # [K, V]

                    pypto.set_vec_tile_shapes(K, V)
                    d_s_new = pypto.add(d_s, update)
                    d_s[:] = d_s_new

                pypto.set_vec_tile_shapes(K, V)
                dh0_out[b_idx, h_idx] = d_s

    dht = torch.randn(B, H, K, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    kc_cache = torch.randn(B, H, 2, K, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    dh0 = torch.zeros(B, H, K, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)

    kernel(dht, kc_cache, dh0)
    torch.npu.synchronize()

    # Verify: dht[b,h] + kc[b,h,1] + kc[b,h,0]
    expected = dht[0, 0].cpu() + kc_cache[0, 0, 1].cpu() + kc_cache[0, 0, 0].cpu()
    actual = dh0[0, 0].cpu()
    print(f"  Max diff: {(expected - actual).abs().max().item()}")
    print("  PASSED")

def test_simple_tensor_init():
    """Test: pypto.tensor init + update in loop (no rebind)."""
    print("Test: pypto.tensor init + update...")

    @pypto.frontend.jit
    def kernel(
        dht:      pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
        kc_cache: pypto.Tensor([pypto.DYNAMIC, H, 2, K, V], pypto.DT_FP32),
        dh0_out:  pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
    ):
        B_val = 1
        NT_val = 2
        pypto.experimental.set_operation_options(combine_axis=True)

        for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
            for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
                pypto.set_vec_tile_shapes(K, V)
                d_s = pypto.tensor([K, V], pypto.DT_FP32)
                d_s[:] = dht[b_idx, h_idx]

                for i_idx in pypto.loop(NT_val, name="LOOP_CHUNK",
                                        idx_name="i_idx", unroll_list=[16, 1]):
                    c = NT_val - 1 - i_idx
                    update = kc_cache[b_idx, h_idx, c]

                    pypto.set_vec_tile_shapes(K, V)
                    d_s_new = pypto.add(d_s, update)
                    d_s[:] = d_s_new

                pypto.set_vec_tile_shapes(K, V)
                dh0_out[b_idx, h_idx] = d_s

    dht = torch.randn(B, H, K, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    kc_cache = torch.randn(B, H, 2, K, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    dh0 = torch.zeros(B, H, K, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)

    kernel(dht, kc_cache, dh0)
    torch.npu.synchronize()

    expected = dht[0, 0].cpu() + kc_cache[0, 0, 1].cpu() + kc_cache[0, 0, 0].cpu()
    actual = dh0[0, 0].cpu()
    print(f"  Max diff: {(expected - actual).abs().max().item()}")
    print("  PASSED")

for test_fn in [test_simple_rebind_update, test_simple_tensor_init]:
    try:
        test_fn()
    except Exception as e:
        err_str = str(e)
        if "FFFFFF" in err_str:
            print(f"  FAILED (FFFFFF): compile error")
        else:
            print(f"  FAILED: {err_str[:200]}")

print("\nDone.")
