#!/usr/bin/env python3
# coding: utf-8
"""
Flash Attention Forward Test with Dynamic Variable Length Sequences

Features:
- Input shape: [total_seq, num_heads * head_dim]
- Support large seq_lens with dynamic tiling
- Online softmax algorithm (Flash Attention)
- Q and K can have different sequence lengths
"""

import os
import sys
import argparse
import torch
import pypto
import numpy as np
import time

import logging

from numpy.testing import assert_allclose
BATCH = 2
NUM_HEADS = 8
HEAD_DIM = 64
HIDDEN_DIM = NUM_HEADS * HEAD_DIM
SCALE = 1.0 / (HEAD_DIM ** 0.5)

# Tile sizes for dynamic loops
Q_TILE_SIZE = 256  # Block size for Q sequence
K_TILE_SIZE = 512  # Block size for K/V sequence

SEQ_Q_SIZE = 4096  # Block size for K/V sequence
SEQ_K_SIZE = 4096  # Block size for K/V sequence


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set TILE_FWK_DEVICE_ID before running:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ['TILE_FWK_DEVICE_ID'])
    except ValueError:
        return None


@pypto.frontend.jit(
    debug_options={
        "runtime_debug_mode": 1,
        "compile_debug_mode": 1
                   },
    runtime_options={
        "device_sched_mode": 0,
        "stitch_function_max_num": 128,
        # "stitch_function_max_num": 256,
    },
    pass_options={
        "cube_l1_reuse_setting": {-1: 8},
        "vec_nbuffer_setting": {-1: 4},
        "cube_nbuffer_setting": {-1: 8},
        # "pg_upper_bound": 5000000,
    },
    # host_options={"compile_monitor_enable": True}
)
def flash_attention_varlen_forward_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, HIDDEN_DIM], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, HIDDEN_DIM], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, HIDDEN_DIM], pypto.DT_BF16),
    output: pypto.Tensor([pypto.DYNAMIC, HIDDEN_DIM], pypto.DT_BF16),
    l_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    m_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    cu_seqlens_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    cu_seqlens_k: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    batch_size: int,
):
    """
    Flash Attention Forward with online softmax.
    Outputs L (softmax sum) and M (softmax max) for backward pass.
    L and M are broadcast to [total_q, HIDDEN_DIM] for backward compatibility.
    """
    v1_tile = [8, 512]
    v2_tile = [256, 64]

    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 128])
    pypto.set_vec_tile_shapes(64, 256)

    for b_idx in pypto.loop(batch_size, name="batch_loop"):
        q_start = cu_seqlens_q[b_idx]
        q_end = cu_seqlens_q[b_idx + 1]
        seq_len_q = q_end - q_start
        seq_len_q.as_variable()

        k_start = cu_seqlens_k[b_idx]
        k_end = cu_seqlens_k[b_idx + 1]
        seq_len_k = k_end - k_start
        seq_len_k.as_variable()

        q_tile_count = (seq_len_q + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        k_tile_count = (seq_len_k + K_TILE_SIZE - 1) // K_TILE_SIZE
        # q_tile_count = 1
        # k_tile_count = 512

        oi_update = pypto.tensor([Q_TILE_SIZE, HEAD_DIM], pypto.DT_FP32, "oi_update")
        li_update = pypto.tensor([Q_TILE_SIZE, 1], pypto.DT_FP32, "li_update")
        mi_update = pypto.tensor([Q_TILE_SIZE, 1], pypto.DT_FP32, "mi_update")

        for h_idx in pypto.loop(NUM_HEADS, name="head_loop"):
            h_offset = h_idx * HEAD_DIM

            for q_tile_idx in pypto.loop(q_tile_count, name="q_tile_loop"):
                q_tile_start = q_tile_idx * Q_TILE_SIZE
                q_tile_end = pypto.min(q_tile_start + Q_TILE_SIZE, seq_len_q)
                q_tile_len = q_tile_end - q_tile_start

                for k_tile_idx in pypto.loop(0, k_tile_count, 1, name="k_tile_loop"):
                    k_tile_size = K_TILE_SIZE
                    k_tile_start = k_tile_idx * k_tile_size
                    k_tile_end = pypto.min(k_tile_start + k_tile_size, seq_len_k)
                    k_tile_len = k_tile_end - k_tile_start

                    q_tile = pypto.view(q, [Q_TILE_SIZE, HEAD_DIM],
                                        [q_start + q_tile_start, h_offset],
                                        valid_shape=[q_tile_len, HEAD_DIM])

                    k_tile = pypto.view(k, [k_tile_size, HEAD_DIM],
                                        [k_start + k_tile_start, h_offset],
                                        valid_shape=[k_tile_len, HEAD_DIM])
                    v_tile = pypto.view(v, [k_tile_size, HEAD_DIM],
                                        [k_start + k_tile_start, h_offset],
                                        valid_shape=[k_tile_len, HEAD_DIM])

                    pypto.set_cube_tile_shapes([64, 1024], [64, 64], [256, 1024])
                    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])

                    # print("k_tile   ", k_tile.shape)

                    pypto.set_pass_options(sg_set_scope=5001)
                    scores = pypto.matmul(q_tile, k_tile, out_dtype=pypto.DT_FP32, b_trans=True)
                    pypto.set_pass_options(sg_set_scope=-1)

                    # pypto.set_pass_options(sg_set_scope=1)
                    pypto.set_pass_options(sg_set_scope=5002)

                    scores_scaled = pypto.mul(scores, SCALE)

                    mij = pypto.amax(scores_scaled, dim=-1, keepdim=True)
                    s_shifted = pypto.sub(scores_scaled, mij)
                    pij = pypto.exp(s_shifted)
                    lij = pypto.sum(pij, dim=-1, keepdim=True)
                    pypto.set_cube_tile_shapes([128, 1024], [256, 1024], [64, 64])
                    if pypto.is_loop_begin(k_tile_idx):
                        if pypto.is_loop_end(k_tile_idx):
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            pij_div = pypto.div(pij, lij, precision_type=pypto.DivAlgorithm.INTRINSIC)
                            pij_bf16 = pypto.cast(pij_div, pypto.DT_BF16)
                            pypto.set_pass_options(sg_set_scope=-1)

                            pypto.set_pass_options(sg_set_scope=5003)
                            oij = pypto.matmul(pij_bf16, v_tile, out_dtype=pypto.DT_BF16)
                            pypto.set_pass_options(sg_set_scope=-1)

                            pypto.assemble(oij, [q_start + q_tile_start, h_offset], output)
                            pypto.assemble(lij, [q_start + q_tile_start, 0], l_output)
                            pypto.assemble(mij, [q_start + q_tile_start, 0], m_output)
                        else:
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            pij_bf16 = pypto.cast(pij, pypto.DT_BF16)
                            pypto.set_pass_options(sg_set_scope=-1)
                            
                            pypto.set_pass_options(sg_set_scope=5003)
                            oij = pypto.matmul(pij_bf16, v_tile, out_dtype=pypto.DT_FP32)
                            pypto.set_pass_options(sg_set_scope=-1)

                            oi_update[:] = oij
                            li_update[:] = lij
                            mi_update[:] = mij

                    elif pypto.is_loop_end(k_tile_idx):
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        pij_bf16 = pypto.cast(pij, pypto.DT_BF16)
                        pypto.set_pass_options(sg_set_scope=-1)

                        pypto.set_pass_options(sg_set_scope=5003)
                        oij = pypto.matmul(pij_bf16, v_tile, out_dtype=pypto.DT_FP32)
                        pypto.set_pass_options(sg_set_scope=-1)

                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        mi = mi_update
                        li = li_update
                        oi = oi_update

                        mi_new = pypto.maximum(mi, mij)
                        t1 = pypto.sub(mi, mi_new)
                        t2 = pypto.exp(t1)
                        t3 = pypto.sub(mij, mi_new)
                        t4 = pypto.exp(t3)

                        li_new = pypto.add(pypto.mul(t2, li), pypto.mul(t4, lij))
                        oi_tmp = pypto.add(pypto.mul(oi, t2), pypto.mul(oij, t4))

                        out_fp32 = pypto.div(oi_tmp, li_new, precision_type=pypto.DivAlgorithm.INTRINSIC)
                        out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)
                        pypto.assemble(out_bf16, [q_start + q_tile_start, h_offset], output)
                        pypto.assemble(li_new, [q_start + q_tile_start, 0], l_output)
                        pypto.assemble(mi_new, [q_start + q_tile_start, 0], m_output)
                    else:
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])

                        pij_bf16 = pypto.cast(pij, pypto.DT_BF16)
                        pypto.set_pass_options(sg_set_scope=-1)

                        pypto.set_pass_options(sg_set_scope=5003)
                        oij = pypto.matmul(pij_bf16, v_tile, out_dtype=pypto.DT_FP32)
                        pypto.set_pass_options(sg_set_scope=-1)

                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])

                        mi = mi_update
                        li = li_update
                        oi = oi_update

                        mi_new = pypto.maximum(mi, mij)
                        t1 = pypto.sub(mi, mi_new)
                        t2 = pypto.exp(t1)
                        t3 = pypto.sub(mij, mi_new)
                        t4 = pypto.exp(t3)

                        li_new = pypto.add(pypto.mul(t2, li), pypto.mul(t4, lij))
                        oi_tmp = pypto.add(pypto.mul(oi, t2), pypto.mul(oij, t4))

                        oi_update[:] = oi_tmp
                        li_update[:] = li_new
                        mi_update[:] = mi_new

def create_inputs(seq_lens_q, seq_lens_k, device):
    total_q = sum(seq_lens_q)
    total_k = sum(seq_lens_k)

    torch.manual_seed(42)
    q = torch.randn(total_q, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_k, HIDDEN_DIM, dtype=torch.bfloat16, device=device)

    cu_seqlens_q = torch.tensor([0] + list(np.cumsum(seq_lens_q)), dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0] + list(np.cumsum(seq_lens_k)), dtype=torch.int32, device=device)

    return q, k, v, cu_seqlens_q, cu_seqlens_k


def attention_golden(q, k, v):
    scores = torch.matmul(q.float(), k.float().T) * SCALE
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v.float())

def attention_forward_golden(q, k, v, scale):
    """
    Golden reference: 标准 attention 计算。

    Q: [s1_size, head_dim],  KV: [s2_size, head_dim]
    输入 q/k/v 均为 BF16。

    Args:
        q:     [s1_size, head_dim] BF16 — Q 切片
        k, v:  [s2_size, head_dim] BF16 — KV 切片
        scale: attention scale factor (1/sqrt(head_dim))
    Returns:
        o:     [s1_size, head_dim] FP32 — 输出 O
        m:     [s1_size, 1] FP32 —softmax 最大值 M
        l:     [s1_size, 1] FP32 — softmax 分母 L
    """
    scores = torch.matmul(q.float(), k.float().T) * scale
    m = scores.max(dim=-1, keepdim=True)[0]
    p = torch.softmax(scores, dim=-1)
    l = p.sum(dim=-1, keepdim=True)
    o = torch.matmul(p, v.float())
    return o, m, l


def test_forward(device, seq_lens_q, seq_lens_k, enable_perf=False):
    batch_size = len(seq_lens_q)
    total_q = sum(seq_lens_q)
    total_k = sum(seq_lens_k)

    print("=" * 60)
    print(f"Config: batch_size={batch_size}")
    print(f"        seq_lens_q={seq_lens_q}, total_q={total_q}")
    print(f"        seq_lens_k={seq_lens_k}, total_k={total_k}")
    print(f"        Q_tile={Q_TILE_SIZE}, K_tile={K_TILE_SIZE}")
    print("=" * 60)

    q, k, v, cu_seqlens_q, cu_seqlens_k = create_inputs(seq_lens_q, seq_lens_k, device)

    out = torch.empty(total_q, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    l_out = torch.empty(total_q, 1, dtype=torch.float, device=device)
    m_out = torch.empty(total_q, 1, dtype=torch.float, device=device)

    print("Testing forward...")
    start = time.time()
    flash_attention_varlen_forward_kernel(q, k, v, out, l_out, m_out, cu_seqlens_q, cu_seqlens_k, batch_size)

    elapsed = time.time() - start

    ###
    rtol = 0.0078125
    atol = 0.0001

    hidden_dim = NUM_HEADS * HEAD_DIM
    # ---- Golden 计算: 先完整计算所有 batch/head 的 golden O/M/L ----
    # golden O/M/L 与 kernel 输出同 shape: [total_q, ...], dtype与kernel输出一致
    out_golden = torch.empty(total_q, hidden_dim, dtype=torch.float32, device=device)
    m_golden = torch.empty(total_q, 1, dtype=torch.float32, device=device)
    l_golden = torch.empty(total_q, 1, dtype=torch.float32, device=device)

    q_off, k_off = 0, 0
    for b in range(batch_size):
        sq, sk = seq_lens_q[b], seq_lens_k[b]
        for h in range(NUM_HEADS):
            h_off = h * HEAD_DIM
            q_h = q[q_off:q_off + sq, h_off:h_off + HEAD_DIM]
            k_h = k[k_off:k_off + sk, h_off:h_off + HEAD_DIM]
            v_h = v[k_off:k_off + sk, h_off:h_off + HEAD_DIM]

            golden_o, golden_m, golden_l = attention_forward_golden(q_h, k_h, v_h, SCALE)
            # golden 返回 [sq, ...] FP32, 写入对应 [total_q, ...] 位置
            out_golden[q_off:q_off + sq, h_off:h_off + HEAD_DIM] = golden_o
            m_golden[q_off:q_off + sq, :] = golden_m
            l_golden[q_off:q_off + sq, :] = golden_l
        q_off += sq
        k_off += sk

    # ---- 调用 kernel ----
    # logging.info("  Running kernel...")
    # flash_attention_varlen_forward_kernel(
    #     q, k, v, out, l_out, m_out, cu_seqlens_q, cu_seqlens_k)

    # ---- 精度校验: kernel 输出 vs golden 输出 ----
    rtol = 0.0078125
    atol = 0.0001

    passed = True
    for name, npu_tensor, golden_tensor in [
        ("O", out, out_golden),
        ("M", m_out, m_golden),
        ("L", l_out, l_golden),
    ]:
        npu_np = npu_tensor.float().cpu().numpy()
        golden_np = golden_tensor.float().cpu().numpy()
        max_diff = np.abs(npu_np - golden_np).max()
        try:
            assert_allclose(npu_np, golden_np, rtol=rtol, atol=atol)
            logging.info(f"  {name}: PASSED (max_diff={max_diff:.6f}, rtol={rtol}, atol={atol})")
        except AssertionError as e:
            logging.info(f"  {name}: FAILED (max_diff={max_diff:.6f})")
            logging.info(f"    {e}")
            passed = False

    logging.info(f"  {'PASSED' if passed else 'FAILED'}")
    logging.info("")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--perf', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--seq_len', type=int, default=None, help='Test with specific sequence length')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Flash Attention Forward Test")
    print(f"Input shape: [total_seq, {HIDDEN_DIM}]")
    print(f"Tile sizes: Q={Q_TILE_SIZE}, K={K_TILE_SIZE}")
    print("=" * 60 + "\n")

    device_id = get_device_id()
    if device_id is None:
        return

    import torch_npu
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'
    # device="cpu"

    # # Test with specific sequence length
    # if args.seq_len:
    #     test_cases = [([args.seq_len], [args.seq_len])]
    # # Test cases with large seq_lens
    # elif args.all:
    test_cases = [
        # ([64, 64, 64, 64], [64, 64, 64, 64]),
        # ([32, 64, 32, 64], [64, 128, 64, 128]),
        # ([256, 256, 256], [256, 256, 256]),
        # ([512, 512], [512, 512]),
        # ([1024, 1024], [1024, 1024]),
        # ([320]*8, [320]*8),
        ([SEQ_Q_SIZE] * BATCH, [SEQ_K_SIZE] * BATCH),

        # ([512, 1024], [1024, 2048]),
    ]
    # else:
    #     test_cases = [
    #         ([64, 64, 64, 64], [64, 64, 64, 64]),
    #     ]

    try:
        all_passed = True
        for seq_q, seq_k in test_cases:
            if not test_forward(device, seq_q, seq_k, args.perf):
                all_passed = False
        print("=" * 60)
        if all_passed:
            print("All forward tests passed!")
        else:
            print("Some tests failed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
