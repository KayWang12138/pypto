"""Minimal compile test #3: isolate the REGISTER_COPY issue.

Test with NO in-place update on d_s, NO pypto.view.
Just basic indexing + matmul + output write.
"""
import os
os.environ['TILE_FWK_DEVICE_ID'] = '1'

import pypto
import torch
import torch_npu

def make_test_kernel(K, V, H, BT):
    """Test 1: No pypto.view, no in-place update."""
    @pypto.frontend.jit(
        runtime_options={"stitch_function_max_num": 128},
    )
    def kernel(
        q_norm: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        dht:    pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
        dq_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        dh0_out: pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
    ):
        B_val = q_norm.shape[0]
        T_val = q_norm.shape[1]

        for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
            for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
                d_s = dht[b_idx, h_idx]
                # Just read d_s, no in-place update
                dh0_out[b_idx, h_idx, :, :] = d_s
    return kernel


print("=== Test 1: Simple index read + write ===")
B, T, H, K, V, BT = 1, 128, 4, 128, 128, 64
try:
    kernel1 = make_test_kernel(K, V, H, BT)
    q = torch.randn(B, T, H, K, dtype=torch.float32).npu()
    dht = torch.randn(B, H, K, V, dtype=torch.float32).npu()
    dq = torch.zeros(B, T, H, K, dtype=torch.float32).npu()
    dh0 = torch.zeros(B, H, K, V, dtype=torch.float32).npu()
    kernel1(q, dht, dq, dh0)
    torch.npu.synchronize()
    print("Test 1: SUCCESS")
except Exception as e:
    print(f"Test 1: FAILED - {str(e)[:300]}")


def make_test_kernel2(K, V, H, BT):
    """Test 2: Add in-place update on d_s via view of input."""
    @pypto.frontend.jit(
        runtime_options={"stitch_function_max_num": 128},
    )
    def kernel(
        q_norm: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        dht:    pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
        dq_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        dh0_out: pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
    ):
        B_val = q_norm.shape[0]
        NT_val = 2  # hardcode for simplicity

        for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
            for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
                d_s = dht[b_idx, h_idx]

                for i_idx in pypto.loop(NT_val, name="LOOP_CHUNK",
                                        idx_name="i_idx", unroll_list=[16, 1]):
                    # In-place update
                    d_s_new = pypto.mul(d_s, d_s)
                    d_s[:] = d_s_new

                dh0_out[b_idx, h_idx, :, :] = d_s
    return kernel


print("\n=== Test 2: In-place update on input view ===")
try:
    kernel2 = make_test_kernel2(K, V, H, BT)
    q = torch.randn(B, T, H, K, dtype=torch.float32).npu()
    dht = torch.randn(B, H, K, V, dtype=torch.float32).npu()
    dq = torch.zeros(B, T, H, K, dtype=torch.float32).npu()
    dh0 = torch.zeros(B, H, K, V, dtype=torch.float32).npu()
    kernel2(q, dht, dq, dh0)
    torch.npu.synchronize()
    print("Test 2: SUCCESS")
except Exception as e:
    print(f"Test 2: FAILED - {str(e)[:300]}")


def make_test_kernel3(K, V, H, BT):
    """Test 3: Use output buffer as mutable state (like dh0_out)."""
    @pypto.frontend.jit(
        runtime_options={"stitch_function_max_num": 128},
    )
    def kernel(
        q_norm: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        dht:    pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
        dq_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        dh0_out: pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
    ):
        B_val = q_norm.shape[0]
        NT_val = 2

        for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
            for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
                # Initialize output buffer from input
                dh0_out[b_idx, h_idx, :, :] = dht[b_idx, h_idx]
                d_s = dh0_out[b_idx, h_idx]

                for i_idx in pypto.loop(NT_val, name="LOOP_CHUNK",
                                        idx_name="i_idx", unroll_list=[16, 1]):
                    d_s_new = pypto.mul(d_s, d_s)
                    d_s[:] = d_s_new

                # dh0_out already has final value
    return kernel


print("\n=== Test 3: Output buffer as mutable state ===")
try:
    kernel3 = make_test_kernel3(K, V, H, BT)
    q = torch.randn(B, T, H, K, dtype=torch.float32).npu()
    dht = torch.randn(B, H, K, V, dtype=torch.float32).npu()
    dq = torch.zeros(B, T, H, K, dtype=torch.float32).npu()
    dh0 = torch.zeros(B, H, K, V, dtype=torch.float32).npu()
    kernel3(q, dht, dq, dh0)
    torch.npu.synchronize()
    print("Test 3: SUCCESS")
except Exception as e:
    print(f"Test 3: FAILED - {str(e)[:300]}")
