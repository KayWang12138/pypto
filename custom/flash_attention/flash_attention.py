#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import math
import torch
import torch_npu
import pypto
import numpy as np
from numpy.testing import assert_allclose


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("ERROR: TILE_FWK_DEVICE_ID not set. Please run:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def matmul_proxy(left, right):
    fp32 = torch.float32
    return torch.matmul(left.to(fp32), right.to(fp32))


def ifa_flash_torch(q, k, v, block_table, kv_act_seqs, out, is_fp32=False):
    fp32 = torch.float32
    dtype = q.dtype
    if is_fp32:
        q = q.to(fp32)
        k = k.to(fp32)
        v = v.to(fp32)
        dtype = fp32

    q_shape = q.shape
    bs1, n1, d = q_shape[0], q_shape[1], q_shape[2]
    b = kv_act_seqs.shape[0]
    s1 = bs1 // b
    k_shape = k.shape
    block_num, block_size, n2, _ = k_shape
    g = n1 // n2
    g_tile = g
    k_2d = k.reshape(-1, d).contiguous()
    v_2d = v.reshape(-1, d).contiguous()
    q_2d = q.reshape(-1, d).contiguous()
    v_2d = v.reshape(-1, d).contiguous()

    for b_idx in range(b):
        for s1_idx in range(s1):
            cur_seq = kv_act_seqs[b_idx] - (s1 - 1 - s1_idx)
            cur_seq = max(cur_seq.item(), 0)
            s2_loop = math.ceil(cur_seq / block_size)

            for n2_idx in range(n2):
                for g_idx in range(g // g_tile):
                    device = q.device
                    dtype = q.dtype
                    oi_upd = torch.zeros((g_tile, d), device=device, dtype=fp32)
                    li_upd = torch.zeros(g_tile, device=device, dtype=fp32)
                    mi_upd = torch.zeros(g_tile, device=device, dtype=fp32)

                    for s2_idx in range(s2_loop):
                        block_idx = block_table[b_idx][s2_idx].item()

                        bs_ofs = b_idx * s1 + s1_idx
                        n2g_ofs = n2_idx * g + g_idx * g_tile
                        actual_s2_tile = min(block_size, cur_seq - s2_idx * block_size)

                        qi_start = bs_ofs * n1 + n2g_ofs
                        qi_end = qi_start + g_tile
                        qi = q_2d[qi_start:qi_end, :]


                        kj_start = block_idx * block_size
                        kj_end = kj_start + actual_s2_tile
                        kj = k_2d[kj_start:kj_end, :]

                        vj = v_2d[kj_start:kj_end, :]

                        kj_t = kj.t()

                        mm1 = matmul_proxy(qi, kj.t()).to(fp32)
                        muls_res = mm1 * (d ** -0.5)

                        tilda_mij, _ = torch.max(muls_res, dim=-1, keepdim=True)

                        if s2_idx == 0:
                            tsub = muls_res - tilda_mij
                            tilda_pij = torch.exp(tsub)

                            tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
                            oi_tmp = matmul_proxy(tilda_pij.to(dtype), vj).to(fp32)
                            oi_upd = oi_tmp

                            li_upd = tilda_lij.squeeze(-1)
                            mi_upd = tilda_mij.squeeze(-1)
                        else:
                            mi = mi_upd.unsqueeze(-1)
                            max_new, _ = torch.max(torch.cat([mi, tilda_mij], dim=-1), dim=-1, keepdim=True)
                            tsub = muls_res - max_new

                            tilda_pij = torch.exp(tsub)
                            tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)

                            tsub2 = torch.sub(mi, max_new)
                            mi_upd = max_new.squeeze(-1)
                            update_mul = torch.exp(tsub2)
                            li = li_upd.unsqueeze(-1)
                            sum_new = li * update_mul + tilda_lij
                            li_upd = sum_new.squeeze(-1)
                            q1 = matmul_proxy(tilda_pij.to(dtype), vj).to(fp32)

                            oi_upd = oi_upd * update_mul + q1

                        if s2_idx == s2_loop - 1:
                            li = li_upd.unsqueeze(-1)
                            oi_final = oi_upd / li
                            oi_upd_3d = oi_final.unsqueeze(0)
                            attn_out_start_col = n2g_ofs
                            attn_out_end_col = n2g_ofs + g_tile
                            if attn_out_end_col > out.shape[1]:
                                attn_out_end_col = out.shape[1]
                                attn_out_start_col = attn_out_end_col - g_tile
                            out[bs_ofs:bs_ofs + 1, attn_out_start_col:attn_out_end_col, :] = oi_upd_3d.to(dtype)
    return out


def gen_block_table(actual_seq_len, block_size, block_table_shape):
    block_num_per_batch = []
    block_num = 0

    for actual_seq in actual_seq_len:
        block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
        block_num += math.ceil(actual_seq.item() / block_size)

    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]

    block_table = torch.full(block_table_shape, -1, dtype=torch.int32, device=actual_seq_len.device)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return block_table


def create_flash_attention_kernel(run_mode="npu"):
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")


    bs1 = pypto.frontend.dynamic("bs1")
    n1 = pypto.frontend.dynamic("n1")
    d = pypto.frontend.dynamic("d")
    block_num = pypto.frontend.dynamic("block_num")
    block_size = pypto.frontend.dynamic("block_size")
    n2 = pypto.frontend.dynamic("n2")
    b = pypto.frontend.dynamic("b")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def flash_attention_kernel(
        q: pypto.Tensor((bs1, n1, d), pypto.DT_BF16),
        k: pypto.Tensor((block_num, block_size, n2, d), pypto.DT_BF16),
        v: pypto.Tensor((block_num, block_size, n2, d), pypto.DT_BF16),
        block_table: pypto.Tensor((b, block_num), pypto.DT_INT32),
        kv_act_seqs: pypto.Tensor((b,), pypto.DT_INT32),
        scale: float,
        d_tile: int,
        g_tile: int,
        block_size_tile: int,
    ) -> pypto.Tensor((bs1, n1, d), pypto.DT_BF16):
        shape_q = q.shape
        bs1_val = shape_q[0]
        n1_val = shape_q[1]
        d_val = shape_q[2]

        shape_k = k.shape
        block_num_val = shape_k[0]
        block_size_val = shape_k[1]
        n2_val = shape_k[2]

        shape_act_seqs = kv_act_seqs.shape
        b_val = shape_act_seqs[0]

        g = n1_val // n2_val
        s1_val = bs1_val // b_val

        out = pypto.tensor((bs1_val, n1_val, d_val), pypto.DT_BF16)

        q_2d_shape = (bs1_val * n1_val, d_val)
        kv_2d_shape = (block_num_val * block_size_val * n2_val, d_val)

        q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
        k_2d = pypto.reshape(k, kv_2d_shape, inplace=True)
        v_2d = pypto.reshape(v, kv_2d_shape, inplace=True)

        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        pypto.set_vec_tile_shapes(1, d_tile)

        for b_idx in pypto.loop(b_val, name="batch_loop"):
            for s1_idx in pypto.loop(s1_val, name="seq_loop"):
                bs_ofs = b_idx * s1_val + s1_idx

                for n2_idx in pypto.loop(n2_val, name="head_loop"):
                    for g_idx in pypto.loop(g // g_tile, name="group_loop"):
                        n2g_ofs = n2_idx * g + g_idx * g_tile

                        oi_upd = pypto.tensor((g_tile, d_tile), pypto.DT_FP32)
                        li_upd = pypto.tensor((g_tile,), pypto.DT_FP32)
                        mi_upd = pypto.tensor((g_tile,), pypto.DT_FP32)

                        cur_seq_val = kv_act_seqs[b_idx] - (s1_val - 1 - s1_idx)
                        s2_loop = (cur_seq_val + block_size_val - 1) // block_size_val

                        for s2_idx in pypto.loop(s2_loop, name="block_loop", unroll_list=[8,4,2,1]):
                            block_idx_tensor = block_table[b_idx, s2_idx]

                            pypto.set_vec_tile_shapes(64, 64)
                            qi_offset = bs_ofs * n1_val + n2g_ofs
                            qi = pypto.view(q_2d, [g_tile, d_tile], [qi_offset, 0], valid_shape=[g_tile, d_val])

                            kj_offset = block_idx_tensor * block_size_val
                            actual_s2_tile = (cur_seq_val - s2_idx * block_size_val).min(block_size_val)

                            kj = pypto.view(k_2d, [block_size_tile, d_tile], [kj_offset, 0], valid_shape=[actual_s2_tile, d_val])
                            vj = pypto.view(v_2d, [block_size_tile, d_tile], [kj_offset, 0], valid_shape=[actual_s2_tile, d_val])

                            kj_t = pypto.transpose(kj, 0, 1)

                            mm1 = pypto.matmul(qi, kj_t, pypto.DT_FP32)
                            muls_res = mm1 * scale

                            tilda_mij = pypto.amax(muls_res, dim=-1, keepdim=True)

                            if pypto.is_loop_begin(s2_idx):

                                tsub = muls_res - tilda_mij
                                tilda_pij = pypto.exp(tsub)

                                tilda_lij = pypto.sum(tilda_pij, dim=-1, keepdim=True)

                                vj_fp32 = pypto.cast(vj, pypto.DT_FP32)
                                oi_tmp = pypto.matmul(tilda_pij, vj_fp32, out_dtype=pypto.DT_FP32)
                                oi_upd = oi_tmp

                                li_upd = pypto.reshape(tilda_lij, [g_tile])
                                mi_upd = pypto.reshape(tilda_mij, [g_tile])
                            else:
                                mi = pypto.reshape(mi_upd, [g_tile, 1])
                                max_new = pypto.amax(pypto.concat([mi, tilda_mij], dim=-1), dim=-1, keepdim=True)
                                tsub = muls_res - max_new

                                tilda_pij = pypto.exp(tsub)
                                tilda_lij = pypto.sum(tilda_pij, dim=-1, keepdim=True)

                                tsub2 = mi - max_new
                                update_mul = pypto.exp(tsub2)
                                li = pypto.reshape(li_upd, [g_tile, 1])
                                sum_new = li * update_mul + tilda_lij
                                li_upd = pypto.reshape(sum_new, [g_tile])
                                mi_upd = pypto.reshape(max_new, [g_tile])

                                vj_fp32 = pypto.cast(vj, pypto.DT_FP32)
                                q1 = pypto.matmul(tilda_pij, vj_fp32, out_dtype=pypto.DT_FP32)

                                oi_upd = oi_upd * update_mul + q1

                            if pypto.is_loop_end(s2_idx):
                                li = pypto.reshape(li_upd, [g_tile, 1])
                                oi_final = oi_upd / li
                                oi_final_bf16 = pypto.cast(oi_final, pypto.DT_BF16)
                                oi_upd_3d = pypto.reshape(oi_final_bf16, [1, g_tile, d_tile])

                                attn_out_start_col = n2g_ofs
                                attn_out_end_col = n2g_ofs + g_tile

                                out[bs_ofs:bs_ofs + 1, attn_out_start_col:attn_out_end_col, :] = oi_upd_3d

        return out

    return flash_attention_kernel


def test_flash_attention():
    print("=" * 60)
    print("Test: Flash Attention with Dynamic Axis")
    print("=" * 60)

    device_id = get_device_id()
    if device_id is None:
        return

    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'

    b = 4
    s1 = 2
    n1 = 8
    n2 = 2
    d = 16
    block_size = 32
    block_num = 4

    bs1 = b * s1
    g = n1 // n2
    g_tile = g

    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)

    q = torch.randn(bs1, n1, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(block_num, block_size, n2, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(block_num, block_size, n2, d, dtype=torch.bfloat16, device=device)

    actual_seq_len = torch.tensor([32, 16, 16, 16], dtype=torch.int32, device=device)
    block_table_shape = [b, block_num]
    block_table = gen_block_table(actual_seq_len, block_size, block_table_shape)

    out_golden = torch.empty(bs1, n1, d, dtype=torch.bfloat16, device=device)
    golden_out = ifa_flash_torch(q, k, v, block_table, actual_seq_len, out_golden)

    scale = 1.0 / (d ** 0.5)
    d_tile = d
    g_tile_param = g
    block_size_tile = block_size

    kernel = create_flash_attention_kernel(run_mode="npu")
    pypto_out = kernel(q, k, v, block_table, actual_seq_len, scale, d_tile, g_tile_param, block_size_tile)

    print(f"Q shape: {q.shape}")
    print(f"K shape: {k.shape}")
    print(f"V shape: {v.shape}")
    print(f"Output shape: {pypto_out.shape}")
    print(f"Block table shape: {block_table.shape}")
    print(f"Actual seq lengths: {actual_seq_len}")

    diff_output = (pypto_out.float() - golden_out.float()).abs().max().item()
    print(f"\nMax diff: {diff_output:.6f}")
    if torch.allclose(pypto_out.float(), golden_out.float(), rtol=5e-3, atol=5e-3):
        print("✓ Flash Attention output matches golden output")
    else:
        print("✗ Flash Attention output does not match golden output")


if __name__ == "__main__":
    test_flash_attention()
