#!/usr/bin/env python3
# coding: utf-8
import math
import torch
import torch_npu
import pypto
import numpy as np

B = 2
S1 = 4
N1 = 32
N2 = 2
D = 64
BLOCK_SIZE = 16
BLOCK_NUM = 16
MAX_BLOCKNUM_PERBATCH = 8

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


def ifa_flash_torch_golden(q, k, v, block_table, kv_act_seqs, out):
    fp32 = torch.float32
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

                        kj_start = block_idx * block_size * n2 + n2_idx * block_size
                        kj_end = kj_start + actual_s2_tile
                        kj = k_2d[kj_start:kj_end, :]

                        vj = v_2d[kj_start:kj_end, :]

                        mm1 = torch.matmul(qi.float(), kj.t().float())
                        muls_res = mm1 * (d ** -0.5)
                        tilda_mij, _ = torch.max(muls_res, dim=-1, keepdim=True)

                        if s2_idx == 0:
                            tsub = muls_res - tilda_mij
                            tilda_pij = torch.exp(tsub)
                            tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
                            oi_tmp = torch.matmul(tilda_pij.to(dtype), vj).float()
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
                            q1 = torch.matmul(tilda_pij.to(dtype), vj).float()

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


@pypto.frontend.jit
def flash_attention_kernel(
    q: pypto.Tensor((B * S1, N1, D), pypto.DT_BF16),
    k: pypto.Tensor((BLOCK_NUM, BLOCK_SIZE, N2, D), pypto.DT_BF16),
    v: pypto.Tensor((BLOCK_NUM, BLOCK_SIZE, N2, D), pypto.DT_BF16),
    block_table: pypto.Tensor((B, MAX_BLOCKNUM_PERBATCH), pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor((B,), pypto.DT_INT32),
    out: pypto.Tensor((B * S1, N1, D), pypto.DT_BF16),
):
    bs1 = B * S1
    n1 = N1
    n2 = N2
    d = D
    block_size = BLOCK_SIZE
    g = n1 // n2
    g_tile = g
    softmax_scale = d ** -0.5

    k_2d = pypto.reshape(k, [-1, d])
    v_2d = pypto.reshape(v, [-1, d])
    q_2d = pypto.reshape(q, [-1, d])

    pypto.set_cube_tile_shapes([g_tile, g_tile], [d, d], [block_size, block_size])
    pypto.set_vec_tile_shapes(1, 16, g_tile, d)

    for b_idx in pypto.loop(0, B, 1, name="LOOP_L0_bIdx", idx_name="bIdx"):
        for s1_idx in pypto.loop(0, S1, 1, name="LOOP_L1_s1Idx", idx_name="s1Idx"):
            cur_act_seq = kv_act_seqs[b_idx]
            cur_seq = cur_act_seq - (S1 - 1 - s1_idx)
            cur_seq = cur_seq.max(0)
            cur_seq.as_variable()
            s2_loop = (cur_seq + block_size - 1) // block_size

            for n2_idx in pypto.loop(0, n2, 1, name="LOOP_L2_n2Idx", idx_name="n2Idx"):
                for g_idx in pypto.loop(0, g // g_tile, 1, name="LOOP_L3_gIdx", idx_name="gIdx"):
                    oi_update = pypto.tensor([g_tile, d], pypto.DT_FP32, "oi_update")
                    li_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "li_update")
                    mi_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "mi_update")

                    bs_ofs = b_idx * S1 + s1_idx
                    n2g_ofs = n2_idx * g + g_idx * g_tile

                    for s2_idx, _ in pypto.loop_unroll(0, s2_loop, 1, name="LOOP_L4_s2Idx", idx_name="s2Idx", unroll_list={1}):
                        actual_s2_tile = (cur_seq - s2_idx * block_size).min(block_size)

                        block_idx = block_table[b_idx, s2_idx]

                        qi_start = bs_ofs * n1 + n2g_ofs
                        qi = pypto.view(q_2d, [g_tile, d], [qi_start, 0])

                        kj_start = block_idx * block_size * n2 + n2_idx * block_size
                        kj = pypto.view(k_2d, [block_size, d], [kj_start, 0], 
                                       valid_shape=[actual_s2_tile, d])

                        vj = pypto.view(v_2d, [block_size, d], [kj_start, 0],
                                       valid_shape=[actual_s2_tile, d])

                        # 第一个 matmul: Q @ K^T, M=g_tile=12, K=d=128, N=block_size=128
                        # 使用对齐值 128，matmul 会自动处理 M 轴的尾块（12 < 128）
                        pypto.set_cube_tile_shapes([128, 128], [d, d], [block_size, block_size])
                        sij = pypto.matmul(qi, kj, pypto.DT_FP32, a_trans=False, b_trans=True)

                        pypto.set_vec_tile_shapes(g_tile, block_size)
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

                        t_sub = pypto.sub(sij_scale, tilda_mij)
                        tilda_pij = pypto.exp(t_sub)
                        tilda_pij_bf16 = pypto.cast(tilda_pij, pypto.DT_BF16)
                        tilda_lij = pypto.sum(tilda_pij, dim=-1, keepdim=True)

                        # 第二个 matmul: P @ V, M=g_tile=12, K=block_size=128, N=d=128
                        pypto.set_cube_tile_shapes([128, 128], [block_size, block_size], [d, d])
                        q1 = pypto.matmul(tilda_pij_bf16, vj, pypto.DT_FP32)

                        if pypto.is_loop_begin(s2_idx):
                            if pypto.is_loop_end(s2_idx):
                                oi_final = pypto.div(q1, tilda_lij)
                                oi_final_bf16 = pypto.cast(oi_final, pypto.DT_BF16)
                                oi_final_3d = pypto.reshape(oi_final_bf16, [1, g_tile, d])
                                pypto.assemble(oi_final_3d, [bs_ofs, n2g_ofs, 0], out)
                            else:
                                oi_update[:] = q1
                            li_update[:] = tilda_lij
                            mi_update[:] = tilda_mij
                        else:
                            pypto.set_vec_tile_shapes(g_tile, 1)
                            mi_new = pypto.maximum(mi_update, tilda_mij)
                            t1 = pypto.sub(mi_update, mi_new)
                            t2 = pypto.exp(t1)
                            t3 = pypto.sub(tilda_mij, mi_new)
                            t4 = pypto.exp(t3)
                            t5 = pypto.mul(t4, tilda_lij)
                            t6 = pypto.mul(t2, li_update)
                            li_new = pypto.add(t6, t5)
                            q3 = pypto.mul(oi_update, t2)
                            q2 = pypto.mul(q1, t4)
                            oi_tmp = pypto.add(q3, q2)
                            if pypto.is_loop_end(s2_idx):
                                oi_final = pypto.div(oi_tmp, li_new)
                                oi_final_bf16 = pypto.cast(oi_final, pypto.DT_BF16)
                                oi_final_3d = pypto.reshape(oi_final_bf16, [1, g_tile, d])
                                pypto.assemble(oi_final_3d, [bs_ofs, n2g_ofs, 0], out)
                            else:
                                oi_update[:] = oi_tmp
                            li_update[:] = li_new
                            mi_update[:] = mi_new


def test_flash_attention(run_mode="npu", actual_seq_lens=None):
    print("=" * 60)
    print("Test: Flash Attention")
    print("=" * 60)

    device = 'npu:0' if run_mode == "npu" else 'cpu'

    if actual_seq_lens is None:
        actual_seq_lens = torch.tensor([32, 48], dtype=torch.int32, device=device)
    else:
        actual_seq_lens = torch.tensor(actual_seq_lens, dtype=torch.int32, device=device)
    
    block_table = gen_block_table(actual_seq_lens, BLOCK_SIZE, (B, MAX_BLOCKNUM_PERBATCH))

    q = torch.randn(B * S1, N1, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(BLOCK_NUM, BLOCK_SIZE, N2, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(BLOCK_NUM, BLOCK_SIZE, N2, D, dtype=torch.bfloat16, device=device)
    out = torch.empty(B * S1, N1, D, dtype=torch.bfloat16, device=device)

    flash_attention_kernel(q, k, v, block_table, actual_seq_lens, out)

    golden_out = torch.empty(B * S1, N1, D, dtype=torch.bfloat16, device=device)
    golden_out = ifa_flash_torch_golden(q, k, v, block_table, actual_seq_lens, golden_out)

    print(f"Input shape: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"Output shape: {out.shape}")
    print(f"actual_seq_lens: {actual_seq_lens.tolist()}")

    if run_mode == "npu":
        max_diff = (out - golden_out).abs().max().item()
        is_close = torch.allclose(out, golden_out, rtol=0.01, atol=0.01)
        print(f"Max difference: {max_diff:.6f}")
        print(f"All close: {is_close}")
    
    print("✓ Flash Attention test completed")
    print()


if __name__ == "__main__":
    import os
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Warning: TILE_FWK_DEVICE_ID not set, using default 0")
        os.environ['TILE_FWK_DEVICE_ID'] = '0'
    
    # Case 1: 小规模测试
    print("\n[Case 1] Small scale test (s2_loop ~ 2-3)")
    test_flash_attention(run_mode="npu", actual_seq_lens=[32, 48])
    
    # Case 2: 增大规模，s2_loop >= 5
    print("\n[Case 2] Large scale test (s2_loop >= 5)")
    test_flash_attention(run_mode="npu", actual_seq_lens=[96, 112])


# ============== GLM风格配置（原始参数）==============
GLM_B = 8
GLM_S1 = 1
GLM_N1 = 12
GLM_N2 = 1
GLM_D = 128
GLM_BLOCK_SIZE = 128
GLM_BLOCK_NUM = 1024
GLM_MAX_BLOCKNUM_PERBATCH = 128


@pypto.frontend.jit
def flash_attention_glm_kernel(
    q: pypto.Tensor((GLM_B * GLM_S1, GLM_N1, GLM_D), pypto.DT_BF16),
    k: pypto.Tensor((GLM_BLOCK_NUM, GLM_BLOCK_SIZE, GLM_N2, GLM_D), pypto.DT_BF16),
    v: pypto.Tensor((GLM_BLOCK_NUM, GLM_BLOCK_SIZE, GLM_N2, GLM_D), pypto.DT_BF16),
    block_table: pypto.Tensor((GLM_B, GLM_MAX_BLOCKNUM_PERBATCH), pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor((GLM_B,), pypto.DT_INT32),
    out: pypto.Tensor((GLM_B * GLM_S1, GLM_N1, GLM_D), pypto.DT_BF16),
):
    bs1 = GLM_B * GLM_S1
    n1 = GLM_N1
    n2 = GLM_N2
    d = GLM_D
    block_size = GLM_BLOCK_SIZE
    g = n1 // n2
    g_tile = g
    softmax_scale = d ** -0.5

    k_2d = pypto.reshape(k, [-1, d])
    v_2d = pypto.reshape(v, [-1, d])
    q_2d = pypto.reshape(q, [-1, d])

    pypto.set_cube_tile_shapes([128, 128], [d, d], [block_size, block_size])
    pypto.set_vec_tile_shapes(g_tile, block_size)

    for b_idx in pypto.loop(0, GLM_B, 1, name="GLM_LOOP_L0_bIdx", idx_name="bIdx"):
        for s1_idx in pypto.loop(0, GLM_S1, 1, name="GLM_LOOP_L1_s1Idx", idx_name="s1Idx"):
            cur_act_seq = kv_act_seqs[b_idx]
            cur_seq = cur_act_seq - (GLM_S1 - 1 - s1_idx)
            cur_seq = cur_seq.max(0)
            cur_seq.as_variable()
            s2_loop = (cur_seq + block_size - 1) // block_size

            for n2_idx in pypto.loop(0, n2, 1, name="GLM_LOOP_L2_n2Idx", idx_name="n2Idx"):
                for g_idx in pypto.loop(0, g // g_tile, 1, name="GLM_LOOP_L3_gIdx", idx_name="gIdx"):
                    oi_update = pypto.tensor([g_tile, d], pypto.DT_FP32, "oi_update")
                    li_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "li_update")
                    mi_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "mi_update")

                    bs_ofs = b_idx * GLM_S1 + s1_idx
                    n2g_ofs = n2_idx * g + g_idx * g_tile

                    for s2_idx, _ in pypto.loop_unroll(0, s2_loop, 1, name="GLM_LOOP_L4_s2Idx", idx_name="s2Idx", unroll_list={1}):
                        actual_s2_tile = (cur_seq - s2_idx * block_size).min(block_size)

                        block_idx = block_table[b_idx, s2_idx]

                        qi_start = bs_ofs * n1 + n2g_ofs
                        qi = pypto.view(q_2d, [g_tile, d], [qi_start, 0])

                        kj_start = block_idx * block_size * n2 + n2_idx * block_size
                        kj = pypto.view(k_2d, [block_size, d], [kj_start, 0], 
                                       valid_shape=[actual_s2_tile, d])

                        vj = pypto.view(v_2d, [block_size, d], [kj_start, 0],
                                       valid_shape=[actual_s2_tile, d])

                        # 第一个 matmul: Q @ K^T, M=g_tile=12, K=d=128, N=block_size=128
                        # 使用对齐值 128，matmul 会自动处理 M 轴的尾块（12 < 128）
                        pypto.set_cube_tile_shapes([128, 128], [d, d], [block_size, block_size])
                        sij = pypto.matmul(qi, kj, pypto.DT_FP32, a_trans=False, b_trans=True)

                        pypto.set_vec_tile_shapes(g_tile, block_size)
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

                        t_sub = pypto.sub(sij_scale, tilda_mij)
                        tilda_pij = pypto.exp(t_sub)
                        tilda_pij_bf16 = pypto.cast(tilda_pij, pypto.DT_BF16)
                        tilda_lij = pypto.sum(tilda_pij, dim=-1, keepdim=True)

                        # 第二个 matmul: P @ V, M=g_tile=12, K=block_size=128, N=d=128
                        pypto.set_cube_tile_shapes([128, 128], [block_size, block_size], [d, d])
                        q1 = pypto.matmul(tilda_pij_bf16, vj, pypto.DT_FP32)

                        if pypto.is_loop_begin(s2_idx):
                            if pypto.is_loop_end(s2_idx):
                                oi_final = pypto.div(q1, tilda_lij)
                                oi_final_bf16 = pypto.cast(oi_final, pypto.DT_BF16)
                                oi_final_3d = pypto.reshape(oi_final_bf16, [1, g_tile, d])
                                pypto.assemble(oi_final_3d, [bs_ofs, n2g_ofs, 0], out)
                            else:
                                oi_update[:] = q1
                            li_update[:] = tilda_lij
                            mi_update[:] = tilda_mij
                        else:
                            pypto.set_vec_tile_shapes(g_tile, 1)
                            mi_new = pypto.maximum(mi_update, tilda_mij)
                            t1 = pypto.sub(mi_update, mi_new)
                            t2 = pypto.exp(t1)
                            t3 = pypto.sub(tilda_mij, mi_new)
                            t4 = pypto.exp(t3)
                            t5 = pypto.mul(t4, tilda_lij)
                            t6 = pypto.mul(t2, li_update)
                            li_new = pypto.add(t6, t5)
                            q3 = pypto.mul(oi_update, t2)
                            q2 = pypto.mul(q1, t4)
                            oi_tmp = pypto.add(q3, q2)
                            if pypto.is_loop_end(s2_idx):
                                oi_final = pypto.div(oi_tmp, li_new)
                                oi_final_bf16 = pypto.cast(oi_final, pypto.DT_BF16)
                                oi_final_3d = pypto.reshape(oi_final_bf16, [1, g_tile, d])
                                pypto.assemble(oi_final_3d, [bs_ofs, n2g_ofs, 0], out)
                            else:
                                oi_update[:] = oi_tmp
                            li_update[:] = li_new
                            mi_update[:] = mi_new


def test_glm_style_flash_attention(run_mode="npu", actual_seq_lens=None):
    print("=" * 60)
    print("Test: GLM Style Flash Attention")
    print("=" * 60)

    device = 'npu:0' if run_mode == "npu" else 'cpu'

    if actual_seq_lens is None:
        actual_seq_lens = torch.tensor([1024, 2048, 3072, 4096], dtype=torch.int32, device=device)
    else:
        actual_seq_lens = torch.tensor(actual_seq_lens, dtype=torch.int32, device=device)
    
    block_table = gen_block_table(actual_seq_lens, GLM_BLOCK_SIZE, (GLM_B, GLM_MAX_BLOCKNUM_PERBATCH))

    q = torch.randn(GLM_B * GLM_S1, GLM_N1, GLM_D, dtype=torch.bfloat16, device=device)
    k = torch.randn(GLM_BLOCK_NUM, GLM_BLOCK_SIZE, GLM_N2, GLM_D, dtype=torch.bfloat16, device=device)
    v = torch.randn(GLM_BLOCK_NUM, GLM_BLOCK_SIZE, GLM_N2, GLM_D, dtype=torch.bfloat16, device=device)
    out = torch.empty(GLM_B * GLM_S1, GLM_N1, GLM_D, dtype=torch.bfloat16, device=device)

    flash_attention_glm_kernel(q, k, v, block_table, actual_seq_lens, out)

    golden_out = torch.empty(GLM_B * GLM_S1, GLM_N1, GLM_D, dtype=torch.bfloat16, device=device)
    golden_out = ifa_flash_torch_golden(q, k, v, block_table, actual_seq_lens, golden_out)

    print(f"Input shape: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"Output shape: {out.shape}")
    print(f"actual_seq_lens: {actual_seq_lens.tolist()}")
    
    s2_loops = [(s.item() + GLM_BLOCK_SIZE - 1) // GLM_BLOCK_SIZE for s in actual_seq_lens]
    print(f"s2_loop counts: {s2_loops}")

    if run_mode == "npu":
        max_diff = (out - golden_out).abs().max().item()
        is_close = torch.allclose(out, golden_out, rtol=0.01, atol=0.01)
        print(f"Max difference: {max_diff:.6f}")
        print(f"All close: {is_close}")
    
    print("✓ GLM Style Flash Attention test completed")
    print()


if __name__ == "__main__":
    import os
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Warning: TILE_FWK_DEVICE_ID not set, using default 0")
        os.environ['TILE_FWK_DEVICE_ID'] = '0'
    
    # Case 1: 小规模测试
    print("\n[Case 1] Small scale test (s2_loop ~ 2-3)")
    test_flash_attention(run_mode="npu", actual_seq_lens=[32, 48])
    
    # Case 2: 增大规模，s2_loop >= 5
    print("\n[Case 2] Large scale test (s2_loop >= 5)")
    test_flash_attention(run_mode="npu", actual_seq_lens=[96, 112])

    # Case 3: GLM原始配置测试 (decode阶段, s1=1, s2=16384, nq=12, nkv=1)
    print("\n[Case 3] GLM original config (s1=1, s2=16384, nq=12, nkv=1)")
    test_glm_style_flash_attention(run_mode="npu", actual_seq_lens=[16384] * 8)