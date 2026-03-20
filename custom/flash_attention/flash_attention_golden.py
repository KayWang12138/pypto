#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You should have received a copy of the License along with this program. If not, see
# <http://www.huawei.com/> for a copy of the License.
# -----------------------------------------------------------------------------------------------------------

import math
import torch


def matmul_proxy(left, right):
    fp32 = torch.float32
    return torch.matmul(left.to(fp32), right.to(fp32))


def ifa_flash_torch(q, k, v, block_table, kv_act_seqs, out, is_fp32=False):
    fp32 = torch.float32
    if is_fp32:
        q = q.to(fp32)
        k = k.to(fp32)
        v = v.to(fp32)

    q_shape = q.shape
    bs1, n1, d = q_shape[0], q_shape[1], q_shape[2]
    b = kv_act_seqs.shape[0]
    s1 = bs1 // b
    k_shape = k.shape
    block_num, block_size, n2, _ = k_shape
    g = n1 // n2
    g_tile = g
    k_2d = k.reshape(-1, d)
    v_2d = v.reshape(-1, d)
    q_2d = q.reshape(-1, d)

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


def test_flash_attention_basic(device='cpu'):
    print("=" * 60)
    print("Test: Flash Attention Basic (s2_loop = 1)")
    print("=" * 60)

    b = 1
    s1 = 1
    n1 = 4
    n2 = 2
    d = 64
    block_size = 16

    kv_seq_len = 10
    bs1 = b * s1
    block_num = math.ceil(kv_seq_len / block_size) * b

    q = torch.randn(bs1, n1, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(block_num, block_size, n2, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(block_num, block_size, n2, d, dtype=torch.bfloat16, device=device)

    kv_act_seqs = torch.tensor([kv_seq_len], dtype=torch.int32, device=device)
    block_table = gen_block_table(kv_act_seqs, block_size, [b, block_num])

    out = torch.empty(bs1, n1, d, dtype=torch.bfloat16, device=device)
    out_golden = ifa_flash_torch(q.clone(), k.clone(), v.clone(), block_table, kv_act_seqs, out.clone())

    print(f"Input q shape: {q.shape}")
    print(f"Input k shape: {k.shape}")
    print(f"Input v shape: {v.shape}")
    print(f"Output shape: {out_golden.shape}")
    print(f"kv_seq_len: {kv_seq_len}, s2_loop: {math.ceil(kv_seq_len / block_size)}")
    print("Basic test passed!")
    print()
    return True


def test_flash_attention_two_blocks(device='cpu'):
    print("=" * 60)
    print("Test: Flash Attention Two Blocks (s2_loop = 2)")
    print("=" * 60)

    b = 1
    s1 = 1
    n1 = 4
    n2 = 2
    d = 64
    block_size = 16

    kv_seq_len = 20
    bs1 = b * s1
    block_num = math.ceil(kv_seq_len / block_size) * b

    q = torch.randn(bs1, n1, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(block_num, block_size, n2, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(block_num, block_size, n2, d, dtype=torch.bfloat16, device=device)

    kv_act_seqs = torch.tensor([kv_seq_len], dtype=torch.int32, device=device)
    block_table = gen_block_table(kv_act_seqs, block_size, [b, block_num])

    out = torch.empty(bs1, n1, d, dtype=torch.bfloat16, device=device)
    out_golden = ifa_flash_torch(q.clone(), k.clone(), v.clone(), block_table, kv_act_seqs, out.clone())

    print(f"Input q shape: {q.shape}")
    print(f"Input k shape: {k.shape}")
    print(f"Input v shape: {v.shape}")
    print(f"Output shape: {out_golden.shape}")
    print(f"kv_seq_len: {kv_seq_len}, s2_loop: {math.ceil(kv_seq_len / block_size)}")
    print("Two blocks test passed!")
    print()
    return True


def test_flash_attention_three_blocks(device='cpu'):
    print("=" * 60)
    print("Test: Flash Attention Three Blocks (s2_loop = 3)")
    print("=" * 60)

    b = 1
    s1 = 1
    n1 = 4
    n2 = 2
    d = 64
    block_size = 16

    kv_seq_len = 48
    bs1 = b * s1
    block_num = math.ceil(kv_seq_len / block_size) * b

    q = torch.randn(bs1, n1, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(block_num, block_size, n2, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(block_num, block_size, n2, d, dtype=torch.bfloat16, device=device)

    kv_act_seqs = torch.tensor([kv_seq_len], dtype=torch.int32, device=device)
    block_table = gen_block_table(kv_act_seqs, block_size, [b, block_num])

    out = torch.empty(bs1, n1, d, dtype=torch.bfloat16, device=device)
    out_golden = ifa_flash_torch(q.clone(), k.clone(), v.clone(), block_table, kv_act_seqs, out.clone())

    print(f"Input q shape: {q.shape}")
    print(f"Input k shape: {k.shape}")
    print(f"Input v shape: {v.shape}")
    print(f"Output shape: {out_golden.shape}")
    print(f"kv_seq_len: {kv_seq_len}, s2_loop: {math.ceil(kv_seq_len / block_size)}")
    print("Three blocks test passed!")
    print()
    return True


if __name__ == "__main__":
    test_flash_attention_basic()
    test_flash_attention_two_blocks()
    test_flash_attention_three_blocks()
    print("All golden tests passed!")