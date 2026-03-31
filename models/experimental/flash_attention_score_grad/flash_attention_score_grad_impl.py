#!/usr/bin/env python3
# coding: utf-8
"""
FlashAttentionScoreGrad PyPTO Kernel 实现 (性能优化版)

优化项:
  1. S_TILE 64 → 128，减少循环迭代
  2. cube_tile_shapes 增大至 [128,128],[64,256],[128,128]
  3. pass_options: cube_l1_reuse_setting Q 常驻 L1
  4. runtime_options: stitch_function_max_num=128, device_sched_mode=1
  5. 内层 unroll_list=[8, 4, 2, 1]
"""

import pypto
import torch

# ============================================================
# 模块级常量
# ============================================================
NUM_HEADS = 8
HEAD_DIM = 64
S_TILE = 128  # 优化: 64 → 128


def compute_tile(q_i, k_j, v_j, dy_i, smax_i, ssum_i, D_i,
                 actual_s1, actual_s2, scale_value, c_tile, v_tile_s, v_tile_d, s_tile_size):
    """计算一个 (s1_tile, s2_tile) 块的 P_ij 和 dS_ij。"""
    # S_ij = Q_i @ K_j^T * scale
    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
    S_ij = pypto.matmul(q_i, k_j, pypto.DT_FP32, b_trans=True)
    S_ij = pypto.view(S_ij, [s_tile_size, s_tile_size], [0, 0],
                      valid_shape=[actual_s1, actual_s2])

    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
    S_ij = pypto.mul(S_ij, scale_value)
    P_ij = pypto.exp(pypto.sub(S_ij, smax_i))
    P_ij = pypto.div(P_ij, ssum_i)

    # dP_ij = dY_i @ V_j^T
    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
    dP_ij = pypto.matmul(dy_i, v_j, pypto.DT_FP32, b_trans=True)
    dP_ij = pypto.view(dP_ij, [s_tile_size, s_tile_size], [0, 0],
                       valid_shape=[actual_s1, actual_s2])

    # dS_ij = P_ij * (dP_ij - D_i)
    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
    dS_ij = pypto.mul(P_ij, pypto.sub(dP_ij, D_i))

    return P_ij, dS_ij


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128,   # 轮3回退: 256超时，保持128
        "device_sched_mode": 1,
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 8},
        "cube_nbuffer_setting": {0: 4},
    }
)
def flash_attention_score_grad_kernel(
    q:             pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    k:             pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    v:             pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dy:            pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    softmax_max:   pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    softmax_sum:   pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    attention_out: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dq:            pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dk:            pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dv:            pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    batch_size:    pypto.Tensor([pypto.DYN], pypto.DT_INT32),
    scale_value:   float,
):
    b = batch_size.shape[0]
    total = q.shape[0]
    s = total // b // NUM_HEADS

    q_2d = q
    k_2d = k
    v_2d = v
    dy_2d = dy
    ao_2d = attention_out
    sm_2d = softmax_max
    ss_2d = softmax_sum
    dq_2d = dq
    dk_2d = dk
    dv_2d = dv

    s_loop = s // S_TILE

    c_tile = [[S_TILE, S_TILE], [HEAD_DIM, 256], [S_TILE, S_TILE]]
    v_tile_s = [S_TILE, S_TILE]
    v_tile_d = [S_TILE, HEAD_DIM]

    for b_idx in pypto.loop(b, name="LOOP_b", idx_name="b_idx"):
        for n_idx in pypto.loop(NUM_HEADS, name="LOOP_n", idx_name="n_idx"):
            bn_base = (b_idx * NUM_HEADS + n_idx) * s

            # ===== 趟1: 计算 dQ =====
            for s1_idx in pypto.loop(s_loop, name="LOOP_s1_dq", idx_name="s1_idx"):
                s1_off = bn_base + s1_idx * S_TILE
                actual_s1 = (s - s1_idx * S_TILE).min(S_TILE)

                pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                q_i  = pypto.view(q_2d,  [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[actual_s1, HEAD_DIM])
                dy_i = pypto.view(dy_2d, [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[actual_s1, HEAD_DIM])
                ao_i = pypto.view(ao_2d, [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[actual_s1, HEAD_DIM])

                sm_i_8 = pypto.view(sm_2d, [S_TILE, 8], [s1_off, 0], valid_shape=[actual_s1, 8])
                ss_i_8 = pypto.view(ss_2d, [S_TILE, 8], [s1_off, 0], valid_shape=[actual_s1, 8])
                pypto.set_vec_tile_shapes(S_TILE, 8)
                smax_i = pypto.view(sm_i_8, [S_TILE, 1], [0, 0], valid_shape=[actual_s1, 1])
                ssum_i = pypto.view(ss_i_8, [S_TILE, 1], [0, 0], valid_shape=[actual_s1, 1])

                pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                dy_ao_fp32 = pypto.cast(pypto.mul(dy_i, ao_i), pypto.DT_FP32)
                D_i = pypto.sum(dy_ao_fp32, -1, keepdim=True)

                dQ_acc = pypto.tensor([S_TILE, HEAD_DIM], pypto.DT_FP32, "dQ_acc")

                for s2_idx in pypto.loop(s_loop, name="LOOP_s2_dq", idx_name="s2_idx",
                                         unroll_list=[8, 4, 2, 1]):
                    s2_off = bn_base + s2_idx * S_TILE
                    actual_s2 = (s - s2_idx * S_TILE).min(S_TILE)

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    k_j = pypto.view(k_2d, [S_TILE, HEAD_DIM], [s2_off, 0], valid_shape=[actual_s2, HEAD_DIM])
                    v_j = pypto.view(v_2d, [S_TILE, HEAD_DIM], [s2_off, 0], valid_shape=[actual_s2, HEAD_DIM])

                    _, dS_ij = compute_tile(q_i, k_j, v_j, dy_i, smax_i, ssum_i, D_i,
                                            actual_s1, actual_s2, scale_value,
                                            c_tile, v_tile_s, v_tile_d, S_TILE)

                    dS_bf16 = pypto.cast(dS_ij, pypto.DT_BF16)
                    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    dQ_tile = pypto.matmul(dS_bf16, k_j, pypto.DT_FP32)

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    if pypto.is_loop_begin(s2_idx):
                        dQ_acc[:] = dQ_tile
                    else:
                        dQ_acc[:] = dQ_acc + dQ_tile

                    if pypto.is_loop_end(s2_idx):
                        pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                        dQ_final = pypto.cast(pypto.mul(dQ_acc, scale_value), pypto.DT_BF16)
                        pypto.assemble(dQ_final, [s1_off, 0], dq_2d)

            # ===== 趟2: 计算 dK, dV =====
            for s2_idx in pypto.loop(s_loop, name="LOOP_s2_dkv", idx_name="s2_idx"):
                s2_off = bn_base + s2_idx * S_TILE
                actual_s2 = (s - s2_idx * S_TILE).min(S_TILE)

                pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                k_j = pypto.view(k_2d, [S_TILE, HEAD_DIM], [s2_off, 0], valid_shape=[actual_s2, HEAD_DIM])
                v_j = pypto.view(v_2d, [S_TILE, HEAD_DIM], [s2_off, 0], valid_shape=[actual_s2, HEAD_DIM])

                dK_acc = pypto.tensor([S_TILE, HEAD_DIM], pypto.DT_FP32, "dK_acc")
                dV_acc = pypto.tensor([S_TILE, HEAD_DIM], pypto.DT_FP32, "dV_acc")

                for s1_idx in pypto.loop(s_loop, name="LOOP_s1_dkv", idx_name="s1_idx",
                                         unroll_list=[8, 4, 2, 1]):
                    s1_off = bn_base + s1_idx * S_TILE
                    actual_s1 = (s - s1_idx * S_TILE).min(S_TILE)

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    q_i  = pypto.view(q_2d,  [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[actual_s1, HEAD_DIM])
                    dy_i = pypto.view(dy_2d, [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[actual_s1, HEAD_DIM])
                    ao_i = pypto.view(ao_2d, [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[actual_s1, HEAD_DIM])

                    sm_i_8 = pypto.view(sm_2d, [S_TILE, 8], [s1_off, 0], valid_shape=[actual_s1, 8])
                    ss_i_8 = pypto.view(ss_2d, [S_TILE, 8], [s1_off, 0], valid_shape=[actual_s1, 8])
                    pypto.set_vec_tile_shapes(S_TILE, 8)
                    smax_i = pypto.view(sm_i_8, [S_TILE, 1], [0, 0], valid_shape=[actual_s1, 1])
                    ssum_i = pypto.view(ss_i_8, [S_TILE, 1], [0, 0], valid_shape=[actual_s1, 1])

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    dy_ao_fp32 = pypto.cast(pypto.mul(dy_i, ao_i), pypto.DT_FP32)
                    D_i = pypto.sum(dy_ao_fp32, -1, keepdim=True)

                    P_ij, dS_ij = compute_tile(q_i, k_j, v_j, dy_i, smax_i, ssum_i, D_i,
                                               actual_s1, actual_s2, scale_value,
                                               c_tile, v_tile_s, v_tile_d, S_TILE)

                    dS_bf16 = pypto.cast(dS_ij, pypto.DT_BF16)
                    P_bf16 = pypto.cast(P_ij, pypto.DT_BF16)
                    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    dK_tile = pypto.matmul(dS_bf16, q_i, pypto.DT_FP32, a_trans=True)
                    dV_tile = pypto.matmul(P_bf16, dy_i, pypto.DT_FP32, a_trans=True)

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    if pypto.is_loop_begin(s1_idx):
                        dK_acc[:] = dK_tile
                        dV_acc[:] = dV_tile
                    else:
                        dK_acc[:] = dK_acc + dK_tile
                        dV_acc[:] = dV_acc + dV_tile

                    if pypto.is_loop_end(s1_idx):
                        pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                        dK_final = pypto.cast(pypto.mul(dK_acc, scale_value), pypto.DT_BF16)
                        dV_final = pypto.cast(dV_acc, pypto.DT_BF16)
                        pypto.assemble(dK_final, [s2_off, 0], dk_2d)
                        pypto.assemble(dV_final, [s2_off, 0], dv_2d)


def flash_attention_score_grad_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    attention_out: torch.Tensor,
    scale_value: float,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
):
    """算子 wrapper，供测试调用。"""
    B, N, S, D = query.shape
    assert N == num_heads and D == head_dim
    assert S % S_TILE == 0, f"S={S} must be multiple of S_TILE={S_TILE}"

    q_flat = query.reshape(-1, D).contiguous()
    k_flat = key.reshape(-1, D).contiguous()
    v_flat = value.reshape(-1, D).contiguous()
    dy_flat = dy.reshape(-1, D).contiguous()
    ao_flat = attention_out.reshape(-1, D).contiguous()
    sm_flat = softmax_max.reshape(-1, 8).contiguous()
    ss_flat = softmax_sum.reshape(-1, 8).contiguous()

    dq_flat = torch.empty_like(q_flat)
    dk_flat = torch.empty_like(k_flat)
    dv_flat = torch.empty_like(v_flat)

    batch_tensor = torch.zeros(B, dtype=torch.int32, device=query.device)

    flash_attention_score_grad_kernel(
        q_flat, k_flat, v_flat, dy_flat,
        sm_flat, ss_flat, ao_flat,
        dq_flat, dk_flat, dv_flat,
        batch_tensor,
        scale_value,
    )

    return (dq_flat.reshape(B, N, S, D),
            dk_flat.reshape(B, N, S, D),
            dv_flat.reshape(B, N, S, D))
