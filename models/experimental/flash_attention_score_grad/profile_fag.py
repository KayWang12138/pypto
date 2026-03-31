#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""采集性能数据 (生成泳道图)"""
import logging
import os
import sys

import torch
import pypto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_attention_score_grad_golden import generate_forward_data
from flash_attention_score_grad_impl import (
    NUM_HEADS, HEAD_DIM, S_TILE, compute_tile,
    flash_attention_score_grad_wrapper,
)

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]'
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)


# 重新定义带 debug_options 的 kernel (仅用于采集)
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1},
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 4},
    }
)
def fag_kernel_profile(
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
    s_loop = s // S_TILE

    c_tile = [[S_TILE, S_TILE], [HEAD_DIM, 256], [S_TILE, S_TILE]]
    v_tile_s = [S_TILE, S_TILE]
    v_tile_d = [S_TILE, HEAD_DIM]

    for b_idx in pypto.loop(b, name="LOOP_b", idx_name="b_idx"):
        for n_idx in pypto.loop(NUM_HEADS, name="LOOP_n", idx_name="n_idx"):
            bn_base = (b_idx * NUM_HEADS + n_idx) * s
            for s1_idx in pypto.loop(s_loop, name="LOOP_s1_dq", idx_name="s1_idx"):
                s1_off = bn_base + s1_idx * S_TILE
                actual_s1 = (s - s1_idx * S_TILE).min(S_TILE)
                pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                q_i  = pypto.view(q, [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[actual_s1, HEAD_DIM])
                dy_i = pypto.view(dy, [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[actual_s1, HEAD_DIM])
                ao_i = pypto.view(attention_out, [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[actual_s1, HEAD_DIM])
                sm_i_8 = pypto.view(softmax_max, [S_TILE, 8], [s1_off, 0], valid_shape=[actual_s1, 8])
                ss_i_8 = pypto.view(softmax_sum, [S_TILE, 8], [s1_off, 0], valid_shape=[actual_s1, 8])
                pypto.set_vec_tile_shapes(S_TILE, 8)
                smax_i = pypto.view(sm_i_8, [S_TILE, 1], [0, 0], valid_shape=[actual_s1, 1])
                ssum_i = pypto.view(ss_i_8, [S_TILE, 1], [0, 0], valid_shape=[actual_s1, 1])
                pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                dy_ao_fp32 = pypto.cast(pypto.mul(dy_i, ao_i), pypto.DT_FP32)
                D_i = pypto.sum(dy_ao_fp32, -1, keepdim=True)
                dQ_acc = pypto.tensor([S_TILE, HEAD_DIM], pypto.DT_FP32, "dQ_acc")
                for s2_idx in pypto.loop(s_loop, name="LOOP_s2_dq", idx_name="s2_idx", unroll_list=[8, 4, 2, 1]):
                    s2_off = bn_base + s2_idx * S_TILE
                    actual_s2 = (s - s2_idx * S_TILE).min(S_TILE)
                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    k_j = pypto.view(k, [S_TILE, HEAD_DIM], [s2_off, 0], valid_shape=[actual_s2, HEAD_DIM])
                    v_j = pypto.view(v, [S_TILE, HEAD_DIM], [s2_off, 0], valid_shape=[actual_s2, HEAD_DIM])
                    _, dS_ij = compute_tile(q_i, k_j, v_j, dy_i, smax_i, ssum_i, D_i,
                                            actual_s1, actual_s2, scale_value, c_tile, v_tile_s, v_tile_d, S_TILE)
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
                        pypto.assemble(dQ_final, [s1_off, 0], dq)
            for s2_idx in pypto.loop(s_loop, name="LOOP_s2_dkv", idx_name="s2_idx"):
                s2_off = bn_base + s2_idx * S_TILE
                actual_s2 = (s - s2_idx * S_TILE).min(S_TILE)
                pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                k_j = pypto.view(k, [S_TILE, HEAD_DIM], [s2_off, 0], valid_shape=[actual_s2, HEAD_DIM])
                v_j = pypto.view(v, [S_TILE, HEAD_DIM], [s2_off, 0], valid_shape=[actual_s2, HEAD_DIM])
                dK_acc = pypto.tensor([S_TILE, HEAD_DIM], pypto.DT_FP32, "dK_acc")
                dV_acc = pypto.tensor([S_TILE, HEAD_DIM], pypto.DT_FP32, "dV_acc")
                for s1_idx in pypto.loop(s_loop, name="LOOP_s1_dkv", idx_name="s1_idx", unroll_list=[8, 4, 2, 1]):
                    s1_off = bn_base + s1_idx * S_TILE
                    actual_s1 = (s - s1_idx * S_TILE).min(S_TILE)
                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    q_i  = pypto.view(q, [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[actual_s1, HEAD_DIM])
                    dy_i = pypto.view(dy, [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[actual_s1, HEAD_DIM])
                    ao_i = pypto.view(attention_out, [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[actual_s1, HEAD_DIM])
                    sm_i_8 = pypto.view(softmax_max, [S_TILE, 8], [s1_off, 0], valid_shape=[actual_s1, 8])
                    ss_i_8 = pypto.view(softmax_sum, [S_TILE, 8], [s1_off, 0], valid_shape=[actual_s1, 8])
                    pypto.set_vec_tile_shapes(S_TILE, 8)
                    smax_i = pypto.view(sm_i_8, [S_TILE, 1], [0, 0], valid_shape=[actual_s1, 1])
                    ssum_i = pypto.view(ss_i_8, [S_TILE, 1], [0, 0], valid_shape=[actual_s1, 1])
                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    dy_ao_fp32 = pypto.cast(pypto.mul(dy_i, ao_i), pypto.DT_FP32)
                    D_i = pypto.sum(dy_ao_fp32, -1, keepdim=True)
                    P_ij, dS_ij = compute_tile(q_i, k_j, v_j, dy_i, smax_i, ssum_i, D_i,
                                               actual_s1, actual_s2, scale_value, c_tile, v_tile_s, v_tile_d, S_TILE)
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
                        pypto.assemble(dK_final, [s2_off, 0], dk)
                        pypto.assemble(dV_final, [s2_off, 0], dv)


if __name__ == "__main__":
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    import torch_npu
    torch.npu.set_device(device_id)

    B, N, S, D = 2, 8, 1024, 64
    device = f"npu:{device_id}"
    q, k, v, dy_t, sm, ss, ao, scale = generate_forward_data(B, N, S, D, device=device)

    q_flat = q.reshape(-1, D).contiguous()
    k_flat = k.reshape(-1, D).contiguous()
    v_flat = v.reshape(-1, D).contiguous()
    dy_flat = dy_t.reshape(-1, D).contiguous()
    ao_flat = ao.reshape(-1, D).contiguous()
    sm_flat = sm.reshape(-1, 8).contiguous()
    ss_flat = ss.reshape(-1, 8).contiguous()
    dq_flat = torch.empty_like(q_flat)
    dk_flat = torch.empty_like(k_flat)
    dv_flat = torch.empty_like(v_flat)
    batch_tensor = torch.zeros(B, dtype=torch.int32, device=device)

    logger.info(f"Running profile: B={B}, N={N}, S={S}, D={D}")
    fag_kernel_profile(
        q_flat, k_flat, v_flat, dy_flat,
        sm_flat, ss_flat, ao_flat,
        dq_flat, dk_flat, dv_flat,
        batch_tensor, scale,
    )
    logger.info("Done. Check output/ for swimlane data.")
