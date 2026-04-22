#!/usr/bin/env python3
"""Incremental test: build up the backward kernel piece by piece."""
import torch
import torch_npu
import pypto
import math
import sys, os

DEVICE_ID = 4
torch.npu.set_device(DEVICE_ID)
torch.npu.empty_cache()

# Test parameters
B, T, H, K, V, BT = 1, 128, 4, 128, 128, 64
NT = T // BT  # 2

def test_kernel_matmul_view():
    """Test: 3D view + reshape + matmul + output slice"""
    print("Test: 3D view + reshape + matmul + output slice...")

    @pypto.frontend.jit
    def kernel(
        q_norm:   pypto.Tensor([pypto.DYNAMIC, H, K], pypto.DT_FP32),
        k_norm:   pypto.Tensor([pypto.DYNAMIC, H, K], pypto.DT_FP32),
        act_seq_len: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
        scale_val: pypto.Tensor([1], pypto.DT_FP32),
        m_le:     pypto.Tensor([BT, BT], pypto.DT_FP32),
        c_cum:    pypto.Tensor([BT, BT], pypto.DT_FP32),
        dq_out:   pypto.Tensor([pypto.DYNAMIC, H, K], pypto.DT_FP32),
    ):
        B_val = 1
        pypto.experimental.set_operation_options(combine_axis=True)

        for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
            s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
            b_ofs = act_seq_len[b_idx]
            NT_val = s // BT

            for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
                for i_idx in pypto.loop(NT_val, name="LOOP_CHUNK",
                                        idx_name="i_idx", unroll_list=[16, 1]):
                    c = NT_val - 1 - i_idx
                    s_ofs = b_ofs + c * BT

                    pypto.set_vec_tile_shapes(BT, 1, K)
                    qc_view = pypto.view(q_norm, [BT, 1, K], [s_ofs, h_idx, 0])
                    qc = pypto.reshape(qc_view, [BT, K])

                    kc_view = pypto.view(k_norm, [BT, 1, K], [s_ofs, h_idx, 0])
                    kc = pypto.reshape(kc_view, [BT, K])

                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    qk = pypto.matmul(qc, kc, pypto.DT_FP32, b_trans=True)

                    pypto.set_vec_tile_shapes(128, 128)
                    result = pypto.mul(qk, scale_val)

                    dq_out[s_ofs:s_ofs + BT, h_idx] = result

    q = torch.randn(B*T, H, K, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    k = torch.randn(B*T, H, K, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    act = torch.tensor([0, T], dtype=torch.int32, device=f'npu:{DEVICE_ID}')
    scale = torch.tensor([1.0/math.sqrt(K)], device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    m_le = torch.tril(torch.ones(BT, BT, device=f'npu:{DEVICE_ID}', dtype=torch.float32))
    c_cum = torch.tril(torch.ones(BT, BT, device=f'npu:{DEVICE_ID}', dtype=torch.float32))
    out = torch.zeros(B*T, H, K, device=f'npu:{DEVICE_ID}', dtype=torch.float32)

    kernel(q, k, act, scale, m_le, c_cum, out)
    torch.npu.synchronize()
    print("  PASSED")

def test_kernel_with_cache():
    """Test: 3D view + cache indexing + matmul"""
    print("Test: 3D view + cache + matmul...")

    @pypto.frontend.jit
    def kernel(
        q_norm:   pypto.Tensor([pypto.DYNAMIC, H, K], pypto.DT_FP32),
        k_norm:   pypto.Tensor([pypto.DYNAMIC, H, K], pypto.DT_FP32),
        v:        pypto.Tensor([pypto.DYNAMIC, H, V], pypto.DT_FP32),
        do:       pypto.Tensor([pypto.DYNAMIC, H, V], pypto.DT_FP32),
        dht:      pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
        S_before: pypto.Tensor([pypto.DYNAMIC, H, 128, K, V], pypto.DT_FP32),
        scale_val: pypto.Tensor([1], pypto.DT_FP32),
        act_seq_len: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
        dq_out:   pypto.Tensor([pypto.DYNAMIC, H, K], pypto.DT_FP32),
        dh0_out:  pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
    ):
        B_val = 1
        pypto.experimental.set_operation_options(combine_axis=True)

        for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
            s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
            b_ofs = act_seq_len[b_idx]
            NT_val = s // BT

            for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
                d_s = dht[b_idx, h_idx]

                for i_idx in pypto.loop(NT_val, name="LOOP_CHUNK",
                                        idx_name="i_idx", unroll_list=[16, 1]):
                    c = NT_val - 1 - i_idx
                    s_ofs = b_ofs + c * BT

                    pypto.set_vec_tile_shapes(BT, 1, K)
                    qc_view = pypto.view(q_norm, [BT, 1, K], [s_ofs, h_idx, 0])
                    qc = pypto.reshape(qc_view, [BT, K])
                    kc_view = pypto.view(k_norm, [BT, 1, K], [s_ofs, h_idx, 0])
                    kc = pypto.reshape(kc_view, [BT, K])

                    pypto.set_vec_tile_shapes(BT, 1, V)
                    doc_view = pypto.view(do, [BT, 1, V], [s_ofs, h_idx, 0])
                    doc = pypto.reshape(doc_view, [BT, V])

                    s_before = S_before[b_idx, h_idx, c]

                    # dq = (doc @ s_before^T) * scale
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    dq_c = pypto.mul(
                        pypto.matmul(doc, s_before, pypto.DT_FP32, b_trans=True),
                        scale_val)

                    # d_s update
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    d_s_new = pypto.add(d_s, pypto.matmul(kc, d_s, pypto.DT_FP32))

                    pypto.set_vec_tile_shapes(128, 128)
                    d_s[:] = d_s_new

                    dq_out[s_ofs:s_ofs + BT, h_idx] = dq_c

                pypto.set_vec_tile_shapes(128, 128)
                dh0_out[b_idx, h_idx] = d_s

    q = torch.randn(B*T, H, K, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    k = torch.randn(B*T, H, K, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    v = torch.randn(B*T, H, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    do_t = torch.randn(B*T, H, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    dht = torch.randn(B, H, K, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    sb = torch.randn(B, H, 128, K, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    scale = torch.tensor([1.0/math.sqrt(K)], device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    act = torch.tensor([0, T], dtype=torch.int32, device=f'npu:{DEVICE_ID}')
    out = torch.zeros(B*T, H, K, device=f'npu:{DEVICE_ID}', dtype=torch.float32)
    dh0 = torch.zeros(B, H, K, V, device=f'npu:{DEVICE_ID}', dtype=torch.float32)

    kernel(q, k, v, do_t, dht, sb, scale, act, out, dh0)
    torch.npu.synchronize()
    print("  PASSED")

# Run tests
try:
    test_kernel_matmul_view()
except Exception as e:
    print(f"  FAILED: {e}")

try:
    test_kernel_with_cache()
except Exception as e:
    print(f"  FAILED: {e}")

print("\nAll incremental tests done.")
