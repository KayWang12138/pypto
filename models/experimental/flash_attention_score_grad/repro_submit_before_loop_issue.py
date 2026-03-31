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
"""
Minimal reproduction for submit_before_loop synchronization issue.

When outer loop has submit_before_loop=True and inner loop runs in parallel,
the "double-pointer" accumulation pattern (view tensor_in + add + assemble tensor_out,
where tensor_in and tensor_out share device memory) produces incorrect results
because inner parallel tasks from iteration i may not complete before iteration i+1
starts reading.

Expected: PASS for all shapes
Actual: PASS for S=128 (1 tile), FAIL for S>=256 (multiple tiles)
"""

import os
import sys
import logging
import torch
import pypto

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.handlers.clear()
logger.addHandler(handler)

TILE = 128
D = 64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_attention_score_grad_golden import generate_forward_data
from flash_attention_score_grad_impl import compute_tile, NUM_HEADS, HEAD_DIM, S_TILE


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128, "device_sched_mode": 1},
    pass_options={"cube_l1_reuse_setting": {0: 8}, "cube_nbuffer_setting": {0: 4}},
)
def fag_single_pass(
    q: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dy: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    softmax_max: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    softmax_sum: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    attention_out: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dq: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dk_in: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dk_out: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dv_in: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dv_out: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    batch_size: pypto.Tensor([pypto.DYN], pypto.DT_INT32),
    scale_value: float,
):
    b = batch_size.shape[0]
    s = q.shape[0] // b // NUM_HEADS
    s_loop = s // S_TILE
    c_tile = [[S_TILE, S_TILE], [HEAD_DIM, 256], [S_TILE, S_TILE]]
    v_s = [S_TILE, S_TILE]
    v_d = [S_TILE, HEAD_DIM]

    for b_idx in pypto.loop(b, name="LOOP_b", idx_name="b_idx"):
        for n_idx in pypto.loop(NUM_HEADS, name="LOOP_n", idx_name="n_idx"):
            bn = (b_idx * NUM_HEADS + n_idx) * s
            # BUG: only outer has submit_before_loop, inner runs parallel
            for s1_idx in pypto.loop(
                    s_loop, name="LOOP_s1", idx_name="s1_idx",
                    submit_before_loop=True):
                s1_off = bn + s1_idx * S_TILE
                a1 = (s - s1_idx * S_TILE).min(S_TILE)
                pypto.set_vec_tile_shapes(v_d[0], v_d[1])
                q_i = pypto.view(q, [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[a1, HEAD_DIM])
                dy_i = pypto.view(dy, [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[a1, HEAD_DIM])
                ao_i = pypto.view(attention_out, [S_TILE, HEAD_DIM], [s1_off, 0], valid_shape=[a1, HEAD_DIM])
                sm8 = pypto.view(softmax_max, [S_TILE, 8], [s1_off, 0], valid_shape=[a1, 8])
                ss8 = pypto.view(softmax_sum, [S_TILE, 8], [s1_off, 0], valid_shape=[a1, 8])
                pypto.set_vec_tile_shapes(S_TILE, 8)
                smax = pypto.view(sm8, [S_TILE, 1], [0, 0], valid_shape=[a1, 1])
                ssum = pypto.view(ss8, [S_TILE, 1], [0, 0], valid_shape=[a1, 1])
                pypto.set_vec_tile_shapes(v_d[0], v_d[1])
                d_i = pypto.sum(pypto.cast(pypto.mul(dy_i, ao_i), pypto.DT_FP32), -1, keepdim=True)
                dQ_acc = pypto.tensor([S_TILE, HEAD_DIM], pypto.DT_FP32, "dQ_acc")

                for s2_idx in pypto.loop(
                        s_loop, name="LOOP_s2", idx_name="s2_idx",
                        unroll_list=[8, 4, 2, 1]):
                    s2_off = bn + s2_idx * S_TILE
                    a2 = (s - s2_idx * S_TILE).min(S_TILE)
                    pypto.set_vec_tile_shapes(v_d[0], v_d[1])
                    k_j = pypto.view(k, [S_TILE, HEAD_DIM], [s2_off, 0], valid_shape=[a2, HEAD_DIM])
                    v_j = pypto.view(v, [S_TILE, HEAD_DIM], [s2_off, 0], valid_shape=[a2, HEAD_DIM])
                    p_ij, ds_ij = compute_tile(q_i, k_j, v_j, dy_i, smax, ssum, d_i,
                                               a1, a2, scale_value, c_tile, v_s, v_d, S_TILE)
                    ds_bf = pypto.cast(ds_ij, pypto.DT_BF16)
                    p_bf = pypto.cast(p_ij, pypto.DT_BF16)
                    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
                    pypto.set_vec_tile_shapes(v_d[0], v_d[1])
                    dQ_t = pypto.matmul(ds_bf, k_j, pypto.DT_FP32)
                    dK_t = pypto.matmul(ds_bf, q_i, pypto.DT_FP32, a_trans=True)
                    dV_t = pypto.matmul(p_bf, dy_i, pypto.DT_FP32, a_trans=True)
                    pypto.set_vec_tile_shapes(v_d[0], v_d[1])
                    if pypto.is_loop_begin(s2_idx):
                        dQ_acc[:] = dQ_t
                    else:
                        dQ_acc[:] = dQ_acc + dQ_t
                    dK_bf = pypto.cast(pypto.mul(dK_t, scale_value), pypto.DT_BF16)
                    dV_bf = pypto.cast(dV_t, pypto.DT_BF16)
                    dk_prev = pypto.view(dk_in, [S_TILE, HEAD_DIM], [s2_off, 0], valid_shape=[a2, HEAD_DIM])
                    dv_prev = pypto.view(dv_in, [S_TILE, HEAD_DIM], [s2_off, 0], valid_shape=[a2, HEAD_DIM])
                    pypto.assemble(pypto.add(dk_prev, dK_bf), [s2_off, 0], dk_out)
                    pypto.assemble(pypto.add(dv_prev, dV_bf), [s2_off, 0], dv_out)
                    if pypto.is_loop_end(s2_idx):
                        pypto.set_vec_tile_shapes(v_d[0], v_d[1])
                        pypto.assemble(
                            pypto.cast(pypto.mul(dQ_acc, scale_value), pypto.DT_BF16),
                            [s1_off, 0], dq)


def test_single_pass():
    """Test that exposes the synchronization issue."""
    import numpy as np
    from numpy.testing import assert_allclose
    from flash_attention_score_grad_golden import flash_attention_score_grad_golden

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    import torch_npu
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    for name, B, S in [("S=128 (1 tile)", 2, 128), ("S=256 (2 tiles)", 2, 256)]:
        N, D_val = NUM_HEADS, HEAD_DIM
        q, k_t, v_t, dy_t, sm, ss, ao, scale = generate_forward_data(B, N, S, D_val, device=device)
        q_f = q.reshape(-1, D_val).contiguous()
        k_f = k_t.reshape(-1, D_val).contiguous()
        v_f = v_t.reshape(-1, D_val).contiguous()
        dy_f = dy_t.reshape(-1, D_val).contiguous()
        ao_f = ao.reshape(-1, D_val).contiguous()
        sm_f = sm.reshape(-1, 8).contiguous()
        ss_f = ss.reshape(-1, 8).contiguous()
        dq_f = torch.empty_like(q_f)
        dk_f = torch.zeros_like(k_f)
        dv_f = torch.zeros_like(v_f)
        batch_t = torch.zeros(B, dtype=torch.int32, device=device)

        fag_single_pass(q_f, k_f, v_f, dy_f, sm_f, ss_f, ao_f,
                         dq_f, dk_f, dk_f, dv_f, dv_f, batch_t, scale)

        dq_g, dk_g, dv_g = flash_attention_score_grad_golden(q, k_t, v_t, dy_t, sm, ss, ao, scale)
        dk_diff = (dk_f.reshape(B, N, S, D_val).float() - dk_g.float()).abs().max().item()
        dv_diff = (dv_f.reshape(B, N, S, D_val).float() - dv_g.float()).abs().max().item()
        status = "PASS" if dk_diff < 0.02 and dv_diff < 0.02 else "FAIL"
        logger.info(f"{name}: dK diff={dk_diff:.6f}, dV diff={dv_diff:.6f} → {status}")


if __name__ == "__main__":
    test_single_pass()
