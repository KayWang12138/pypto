#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
""" mla_prolog_quant_v32 子图 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import math
import time
import logging
from pathlib import Path
from typing import List

import torch
import numpy as np
from bfloat16 import bfloat16


if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../cmake").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister

fp32 = np.float32


def rms_norm(x, gamma, eps):
    x_dtype = x.dtype
    mean_coff = 1.0 / x.shape[-1]

    x_f32 = x.astype(fp32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff

    reduce_sum = np.sum(mean_res, axis=-1, keepdims=True) + eps
    reduce_sqrt = np.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt

    res = res_div * gamma

    if x_dtype != fp32:
        res = res.astype(x_dtype)
    return res


def scatter_update_bnsd(inputs, axis):
    # inputs: cache, key_states, indices
    # cache shape: [b, 1, s2, d]
    # key_states shape: [b, 1, s1, d]
    # indices shape: [b, s1]
    cache, key_states, indices = inputs
    b, n2, s2, d = cache.shape  # n2=1
    s1 = indices.shape[1]
    res = cache

    if axis == -2:
        for b_i in range(b):
            for s2_i in range(s2):
                for s1_i in range(s1):
                    index_value = indices[b_i][s1_i]
                    if s2_i == index_value:
                        logging.debug("find the index value and to replace!")
                        res[b_i][0][s2_i][:] = key_states[b_i][0][s1_i][:]

    return res


def scatter_update_pa_bsnd(inputs, axis):
    # inputs: cache, key_states, indices
    # cache shape: [block_number,block_size,n2,d], n2=1
    # key_states shape: [b*s1*1, d]
    # indices shape: [b, s1], s1=1
    cache, key_states, indices = inputs
    block_number, block_size, n2, d = cache.shape
    res = cache.reshape(block_number * block_size * n2, d)
    b, s1 = indices.shape

    if axis == -2:
        for b_i in range(b):
            for s1_i in range(s1):
                index_value = indices[b_i][s1_i]
                res[index_value][:] = key_states[b_i * s1 + s1_i][:]

    return res.reshape(block_number, block_size, n2, d)


def scatter_update(inputs, axis, cache_mode="BNSD"):
    if cache_mode != "BNSD":
        return scatter_update_pa_bsnd(inputs, axis)
    else:
        return scatter_update_bnsd(inputs, axis)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return np.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb_v2(q, k, cos, sin, unsqueeze_dim=2):
    input_dtype = q.dtype
    if input_dtype != fp32:
        q = q.astype(fp32)
        k = k.astype(fp32)
    if cos.dtype != fp32:
        cos = cos.astype(fp32)
        sin = sin.astype(fp32)

    cos = np.expand_dims(cos, axis=unsqueeze_dim)  # [b,s,1,qk_d]
    sin = np.expand_dims(sin, axis=unsqueeze_dim)  # [b,s,1,qk_d]
    logging.debug("expand sin.shape: %s", sin.shape)
    logging.debug("expand cos.shape: %s", cos.shape)

    b, s, h, d = q.shape
    q = q.reshape(b, s, h, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, s, h, d)  # [b,s,n,qk_d]

    b, s, h, d = k.shape
    k = k.reshape(b, s, h, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, s, h, d)  # [b,s,1,qk_d]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    if input_dtype != fp32:
        q_embed, k_embed = q_embed.astype(input_dtype), k_embed.astype(input_dtype)
    return q_embed, k_embed


def quant(input_t, is_pertoken: bool = True, has_smooth=False, smooth_cq=None):
    input_fp32 = input_t.astype(fp32)
    if has_smooth:
        input_fp32 = input_fp32 * smooth_cq
    abs_res = np.abs(input_fp32)
    reduce_idx = -1
    if not is_pertoken:
        reduce_idx = -2
        logging.debug("This PerChannel Quant!!")

    max_value = np.max(abs_res, axis=reduce_idx, keepdims=True)
    scale_quant = 127 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = np.rint(out_fp32).astype(np.int32)
    out_fp16 = out_int32.astype(np.float16)
    out_int8 = np.trunc(out_fp16).astype(np.int8)
    scale_dequant = 1 / scale_quant

    return out_int8, scale_dequant


def to_file(data, dir, name):
    bin_path = Path(dir, f'{name}')
    data.tofile(bin_path)


def mla_prolog_quant_v32_compute(inputs):
    dtype = inputs.get("dtype")
    is_quant_a = inputs.get("is_quant_a")
    is_quant_b = inputs.get("is_quant_b")
    has_smooth = inputs.get("has_smooth")
    cache_mode = inputs.get("cache_mode")
    gamma_cq = inputs.get("gamma_cq")
    gamma_ckv = inputs.get("gamma_ckv")
    epsilon = inputs.get("epsilon")
    x = inputs.get("x")
    w_dq = inputs.get("w_dq")
    w_uqqr = inputs.get("w_uqqr")
    w_uk = inputs.get("w_uk")
    w_dkvkr = inputs.get("w_dkvkr")
    cos = inputs.get("cos")
    sin = inputs.get("sin")
    kv_cache = inputs.get("kv_cache")
    kr_cache = inputs.get("kr_cache")
    kv_quant_scale_cache = None
    if is_quant_b:
        kv_quant_scale_cache = inputs.get("kv_quant_scale_cache")
    cache_index = inputs.get("cache_index")
    if is_quant_a:
        w_qa_scale = inputs.get("w_qa_scale")
        w_kva_scale = inputs.get("w_kva_scale")
    if is_quant_b:
        w_qb_scale = inputs.get("w_qb_scale")
        if has_smooth:
            smooth_cq = inputs.get("smooth_cq")

    b, s, h = x.shape
    qk_rope_head_dim = cos.shape[2]
    n, qk_nope_head_dim, kv_lora_rank = w_uk.shape
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    """ q """
    x_2d = x.reshape(b * s, h)
    # shape is: [b * s, h] @ [h, q_lora_rank] -> [b * s, q_lora_rank]
    if is_quant_a:
        # no smooth
        x_2d_quant, x_2d_scale_dequant = quant(x_2d, True)
        q_a_proj = np.matmul(x_2d_quant.astype(np.int32), w_dq.astype(np.int32))

        """ dequant """
        q_a_proj_fp32 = q_a_proj.astype(fp32)
        q_a_proj_fp32_dequant = q_a_proj_fp32 * x_2d_scale_dequant
        q_a_proj = q_a_proj_fp32_dequant * w_qa_scale
    else:
        q_a_proj = np.matmul(x_2d.astype(fp32), w_dq.astype(fp32))  # [b * s, q_lora_rank]

    q_a_proj = q_a_proj.astype(dtype)

    q_a_layernorm = rms_norm(q_a_proj, gamma_cq, epsilon)
    logging.debug("q_a_layernorm.shape: %s %s", q_a_layernorm.shape, q_a_layernorm.dtype)

    # shape is: [b * s, q_lora_rank] @ [q_lora_rank, n * q_head_dim] -> [b * s, n * q_head_dim]
    q_a_layernorm_scale_dequant = None
    if is_quant_b:
        if has_smooth:
            q_a_layernorm, q_a_layernorm_scale_dequant = quant(q_a_layernorm, True, True, smooth_cq)
        else:
            q_a_layernorm, q_a_layernorm_scale_dequant = quant(q_a_layernorm, True)  # scale: [b*s,1]
        q_b_proj = np.matmul(q_a_layernorm.astype(np.int32), w_uqqr.astype(np.int32))  # q_b_proj

        """ dequant """
        q_b_proj_fp32 = q_b_proj.astype(fp32)
        q_b_proj_fp32_dequant = q_b_proj_fp32 * q_a_layernorm_scale_dequant
        q_b_proj = q_b_proj_fp32_dequant * w_qb_scale
    else:
        q_b_proj = np.matmul(q_a_layernorm.astype(fp32), w_uqqr.astype(fp32))  # [b * s, n * q_head_dim]

    q_b_proj = q_b_proj.astype(dtype)
    logging.debug("q_b_proj.shape: %s %s", q_b_proj.shape, q_b_proj.dtype)

    q_reshape = q_b_proj.reshape(b, s, n, q_head_dim)
    logging.debug("q_reshape.shape: %s %s", q_reshape.shape, q_reshape.dtype)

    q_nope = q_reshape[:, :, :, 0:qk_nope_head_dim]  # [b, s, n, qk_nope_head_dim]
    q_nope_r = q_nope.reshape(b * s, n, qk_nope_head_dim)
    q_nope_t = q_nope_r.transpose(1, 0, 2)  # [n, b*s, qk_nope_head_dim]
    # shape is: [n, b*s, qk_nope_head_dim] @ [n, qk_nope_head_dim, kv_lora_rank] -> [n, b*s, kv_lora_rank]
    q_nope_new = np.matmul(q_nope_t.astype(fp32), w_uk.astype(fp32))
    q_nope_new = q_nope_new.astype(dtype)
    q_nope_new_t = q_nope_new.transpose(1, 0, 2)  # [b*s, n, kv_lora_rank]
    q_out = q_nope_new_t.reshape(b, s, n, kv_lora_rank)  # [b, s, n, kv_lora_rank]

    """ kv """
    # shape is: [b*s, h] @ [h, kv_lora_rank + qk_rope_head_dim] -> [b*s, kv_lora_rank + qk_rope_head_dim]
    if is_quant_a:
        # no smooth
        x_2d_quant, x_2d_scale_dequant = quant(x_2d, True)
        kv_a_proj = np.matmul(x_2d_quant.astype(np.int32), w_dkvkr.astype(np.int32))
        """ dequant """
        kv_a_proj_fp32 = kv_a_proj.astype(fp32)
        kv_a_proj_fp32_dequant = kv_a_proj_fp32 * x_2d_scale_dequant
        kv_a_proj = kv_a_proj_fp32_dequant * w_kva_scale
    else:
        kv_a_proj = np.matmul(x_2d.astype(fp32), w_dkvkr.astype(fp32))  # [b*s, kv_lora_rank + qk_rope_head_dim]

    kv_a_proj = kv_a_proj.astype(dtype)
    logging.debug("kv_a_proj.shape: %s %s", kv_a_proj.shape, kv_a_proj.dtype)
    kv_reshape = kv_a_proj.reshape(b, s, kv_lora_rank + qk_rope_head_dim)
    logging.debug("kv_reshape.shape: %s %s", kv_reshape.shape, kv_reshape.dtype)

    compressed_kv = kv_reshape[:, :, 0:kv_lora_rank]  # [b, s, kv_lora_rank]
    compressed_kv_norm = rms_norm(compressed_kv, gamma_ckv, epsilon)
    compressed_kv_quant_scale = None
    if is_quant_b:
        compressed_kv_norm_split = compressed_kv_norm.reshape(b*s, 4, kv_lora_rank//4)
        compressed_kv_norm, compressed_kv_quant_scale = quant(compressed_kv_norm_split, True)
        compressed_kv_quant_scale = compressed_kv_quant_scale.reshape(b, s, 1, 4)
    compressed_kv_r = compressed_kv_norm.reshape(b, s, 1, kv_lora_rank)
    if cache_mode != "BNSD":
        k_nope = compressed_kv_r.reshape(b * s * 1, kv_lora_rank)
    else:
        k_nope = compressed_kv_r.transpose(0, 2, 1, 3)  # [b, 1, s, kv_lora_rank]

    """ RoPE """
    q_pe = q_reshape[:, :, :, qk_nope_head_dim:]  # [b, s, n, qk_rope_head_dim]

    k_pe = kv_reshape[:, :, kv_lora_rank:]  # [b, s, qk_rope_head_dim]
    k_pe_r = k_pe.reshape(b, s, 1, qk_rope_head_dim)

    # q_embed: [b, s, n, qk_rope_head_dim], k_embed: [b, s, 1, qk_rope_head_dim]
    q_embed, k_embed = apply_rotary_pos_emb_v2(q_pe, k_pe_r, cos, sin, 2)
    if cache_mode != "BNSD":
        k_embed_r = k_embed.reshape(b * 1 * s, qk_rope_head_dim)
    else:
        k_embed_r = k_embed.reshape(b, 1, s, qk_rope_head_dim)

    """ kv_cache output, [b,1,s2,kv_lora_rank] """
    kv_cache_out = scatter_update([kv_cache, k_nope, cache_index], -2, cache_mode)

    """ kr_cache output, [b,1,s2,qk_rope_head_dim] """
    kr_cache_out = scatter_update([kr_cache, k_embed_r, cache_index], -2, cache_mode)

    if is_quant_b:
        compressed_kv_quant_scale = compressed_kv_quant_scale.reshape(-1, 4)
        kv_quant_scale_cache_out = scatter_update([kv_quant_scale_cache, compressed_kv_quant_scale, cache_index], -2, cache_mode)
    else:
        kv_quant_scale_cache_out = None

    return q_out, q_embed, q_a_layernorm, q_a_layernorm_scale_dequant, kv_cache_out, kr_cache_out, kv_quant_scale_cache_out


def gen_block_table(act_seq, block_size, s1, need_indices=False):
    b = act_seq.shape[0]
    block_num = 0
    block_num_each = []
    max_kv = max(act_seq)
    for cur_s in act_seq:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    block_table_shape = [b, math.ceil(max_kv / block_size)]
    block_idx_list = torch.arange(0, block_num, 1)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))].to(torch.int32)

    block_table = -torch.ones(block_table_shape, dtype=torch.int32)

    block_idx = 0
    block_table_bidx = 0
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx, j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    if need_indices:
        cache_index = -torch.ones((b, s1), dtype=torch.int64)
        for i in range(b):
            cur_act = act_seq[i]
            for j in range(s1):
                pos = cur_act - s1 + j
                block_idx_in_seq = pos // block_size
                global_block_id = block_table[i, block_idx_in_seq]

                offset_in_block = pos % block_size
                global_index = global_block_id * block_size + offset_in_block
                cache_index[i, j] = global_index
    else:
        cache_index = None

    if need_indices:
        return block_num, block_table.numpy(), cache_index.numpy()
    else:
        return block_num, block_table.numpy()

def gen_mla_prolog_quant_v32_input_data(params, dtypes,  actual_seq,  output_dir: Path, is_quant=(False, False), is_nz=False,
                                        has_smooth=False, block_size=128, cache_mode="BNSD"):
    dtype, w_dtype = dtypes
    logging.debug(f"gen_mla_prolog_quant_v32_input_data  dtype:{dtype}, w_dtype:{w_dtype}")
    is_quant_a, is_quant_b = is_quant
    b = params.get("b")
    s = params.get("s")  # s=1 or 2
    s1 = params.get("s1")  # s2=4k
    s2 = params.get("s2")  # s2=4k
    h = params.get("h")
    n = params.get("num_heads")
    q_lora_rank = params.get("q_lora_rank")
    qk_nope_head_dim = params.get("qk_nope_head_dim")
    qk_rope_head_dim = params.get("qk_rope_head_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    v_head_dim = params.get("v_head_dim")
    block_num, block_table, cache_index = gen_block_table(torch.from_numpy(actual_seq) , block_size, s1, need_indices=True)

    skv_max = actual_seq.max()
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    NzFrac = 16 if dtype != float else 8
    x_shape = [b, s, h]
    w_qa_shape = [h, q_lora_rank]
    w_qb_shape = [q_lora_rank, n * q_head_dim]
    w_kv_a_shape = [h, kv_lora_rank + qk_rope_head_dim]
    w_kv_b_k_shape = [n, qk_nope_head_dim, kv_lora_rank]
    gamma_cq_shape = [q_lora_rank]
    gamma_ckv_shape = [kv_lora_rank]
    cos_shape = [b, s, qk_rope_head_dim]
    kv_cache_shape = [b, 1, s2, kv_lora_rank]
    kr_cache_shape = [b, 1, s2, qk_rope_head_dim]
    kv_quant_scale_cache_shape =  [b, 1, s2, 4]
    if cache_mode != "BNSD":
        kv_bsnd_shape = [b, skv_max, 1, kv_lora_rank + qk_rope_head_dim]
        kv_cache_shape = [block_num, block_size, 1, kv_lora_rank]
        kr_cache_shape = [block_num, block_size, 1, qk_rope_head_dim]
        kv_quant_scale_cache_shape =  [block_num, block_size, 1, 4]
        index_value_max = block_num * block_size
    smooth_cq_shape = [1, q_lora_rank]
    logging.debug("x shape is %s", x_shape)
    logging.debug("w_dq shape is %s", w_qa_shape)
    logging.debug("w_uqqr shape is %s", w_qb_shape)
    logging.debug("w_dkvkr shape is %s", w_kv_a_shape)
    logging.debug("w_uk shape is %s", w_kv_b_k_shape)
    logging.debug("cos sin shape is %s", cos_shape)
    logging.debug("cgamma_cq shape is %s", gamma_cq_shape)
    logging.debug("cgamma_ckv shape is %s", gamma_ckv_shape)
    logging.debug("kv_len shape is %s", cache_index.shape)
    logging.debug("kv_cache shape is %s", kv_cache_shape)
    logging.debug("kr_cache shape is %s", kr_cache_shape)
    logging.debug("block_num is %s", block_num)
    logging.debug("block_table shape is %s", block_table.shape)
    logging.debug("actual_seq is %s", actual_seq)
    if is_quant_b:
        logging.debug("kv_quant_scale_cache shape is %s", kv_quant_scale_cache_shape)

    x_path = Path(output_dir, 'x.bin')
    w_dq_path = Path(output_dir, 'wDq.bin')
    w_qa_scale_path = Path(output_dir, 'w_qa_scale.bin')
    w_uqqr_path = Path(output_dir, 'wUqQr.bin')
    w_qb_scale_path = Path(output_dir, 'w_qb_scale.bin')
    w_dkvkr_path = Path(output_dir, 'wDkvKr.bin')
    w_kva_scale_path = Path(output_dir, 'w_kva_scale.bin')
    w_uk_path = Path(output_dir, 'wUk.bin')  # kv_b_proj_w_k
    gamma_cq_path = Path(output_dir, 'gamma_cq.bin')
    gamma_ckv_path = Path(output_dir, 'gamma_ckv.bin')
    cos_path = Path(output_dir, 'cos.bin')
    sin_path = Path(output_dir, 'sin.bin')
    kv_len_path = Path(output_dir, 'kv_len.bin')
    kv_cache_path = Path(output_dir, 'kv_cache.bin')
    kr_cache_path = Path(output_dir, 'kr_cache.bin')
    kv_quant_scale_cache_path = Path(output_dir, 'kv_quant_scale_cache.bin')
    smooth_cq_path = Path(output_dir, 'smooth_cq.bin')

    res = [None] * 17
    x = np.random.uniform(-1, 1, x_shape).astype(dtype)
    x.tofile(x_path)
    res[0] = x
    w_dq = np.random.uniform(-0.1, 0.1, w_qa_shape).astype(w_dtype)
    w_uqqr = np.random.uniform(-0.1, 0.1, w_qb_shape).astype(w_dtype)
    w_dkvkr = np.random.uniform(-0.1, 0.1, w_kv_a_shape).astype(w_dtype)
    res[4] = dict()

    if is_quant_a:
        w_dq, w_qa_scale = quant(w_dq, False)
        w_dkvkr, w_kva_scale = quant(w_dkvkr, False)
        w_qa_scale.tofile(w_qa_scale_path)
        w_kva_scale.tofile(w_kva_scale_path)
        res[4]["w_dq"] = w_qa_scale
        res[4]["w_dkvkr"] = w_kva_scale
        if is_nz:
            w_dq.reshape(h, q_lora_rank // 32, 32).transpose(1, 0, 2).tofile(w_dq_path)
            w_dkvkr.reshape(h, (kv_lora_rank + qk_rope_head_dim) // 32, 32).transpose(1, 0, 2).tofile(w_dkvkr_path)
        else:
            w_dq.tofile(w_dq_path)
            w_dkvkr.tofile(w_dkvkr_path)
    else:
        if is_nz:
            w_dq.reshape(h, q_lora_rank // 16, 16).transpose(1, 0, 2).tofile(w_dq_path)
            w_dkvkr.reshape(h, (kv_lora_rank + qk_rope_head_dim) // 16, 16).transpose(1, 0, 2).tofile(w_dkvkr_path)
        else:
            w_dq.tofile(w_dq_path)
            w_dkvkr.tofile(w_dkvkr_path)

    if is_quant_b:
        w_uqqr, w_qb_scale = quant(w_uqqr, False)
        w_qb_scale.tofile(w_qb_scale_path)
        res[4]["w_uqqr"] = w_qb_scale
        # smooth_data
        if has_smooth:
            smooth_cq = np.random.uniform(-1, 1, smooth_cq_shape).astype(np.float32)
            smooth_cq.tofile(smooth_cq_path)
            res[3] = smooth_cq
        if is_nz:
            w_uqqr.reshape(q_lora_rank, n * q_head_dim // 32, 32).transpose(1, 0, 2).tofile(w_uqqr_path)
        else:
            w_uqqr.tofile(w_uqqr_path)
    else:
        if is_nz:
            w_uqqr.reshape(q_lora_rank, n * q_head_dim // 16, 16).transpose(1, 0, 2).tofile(w_uqqr_path)
        else:
            w_uqqr.tofile(w_uqqr_path)

    res[1] = w_dq
    res[2] = w_uqqr
    res[5] = w_dkvkr

    w_uk = np.random.uniform(-0.1, 0.1, w_kv_b_k_shape).astype(w_dtype)
    w_uk.tofile(w_uk_path)
    res[6] = w_uk
    gamma_cq = np.random.uniform(-1, 1, gamma_cq_shape).astype(dtype)  # [q_lora_rank]
    gamma_ckv = np.random.uniform(-1, 1, gamma_ckv_shape).astype(dtype)  # [kv_lora_rank]
    gamma_cq.tofile(gamma_cq_path)
    gamma_ckv.tofile(gamma_ckv_path)
    res[7] = gamma_cq
    res[8] = gamma_ckv
    cos = np.random.uniform(-0.1, 0.1, cos_shape).astype(dtype)  # [b, s, qk_rope_head_dim]
    sin = np.random.uniform(-0.1, 0.1, cos_shape).astype(dtype)  # [b, s, qk_rope_head_dim]
    cos.tofile(cos_path)
    sin.tofile(sin_path)
    res[9] = cos
    res[10] = sin
    cache_index.tofile(kv_len_path)
    res[11] = cache_index
    kv_cache = np.random.uniform(-1, 1, kv_cache_shape).astype(dtype)
    kr_cache = np.random.uniform(-1, 1, kr_cache_shape).astype(dtype)
    k_bsnd = np.random.uniform(-1, 1, kv_bsnd_shape).astype(dtype)
    v_bsnd = k_bsnd[:, :, :, : kv_lora_rank]
    # kv paddIng
    per_batch_max_num = math.ceil(skv_max / block_size)
    k_tensor_bsnd = np.zeros((b, per_batch_max_num * block_size, 1, kv_lora_rank + qk_rope_head_dim)).astype(dtype)
    k_tensor_bsnd[:, :k_bsnd.shape[1], :, :] = k_bsnd[:, :, :, :]
    # kv_cache
    k_cache_tensor = np.zeros([block_num, block_size, 1, kv_lora_rank + qk_rope_head_dim]).astype(dtype)
    for b_idx in range(b):
        for block_i, kv_cache_blk_id in enumerate(block_table[b_idx]):
            block_offset = block_i * block_size
            if kv_cache_blk_id == -1:
                continue
            else:
                k_cache_tensor[kv_cache_blk_id, 0:block_size, :, :] = k_tensor_bsnd[
                    b_idx, block_offset:(block_offset + block_size), :, :]
    kv_cache = k_cache_tensor[:, :, :, : kv_lora_rank]
    kr_cache = k_cache_tensor[:, :, :, kv_lora_rank :]
    kv_quant_scale_cache = None
    if is_quant_b:
        kv_cache_split = kv_cache.reshape(-1, 4, kv_lora_rank//4)
        kv_cache, kv_quant_scale_cache = quant(kv_cache_split, True)
        kv_cache = kv_cache.reshape(kv_cache_shape)
        kv_quant_scale_cache = kv_quant_scale_cache.reshape(kv_quant_scale_cache_shape)
        kv_quant_scale_cache.tofile(kv_quant_scale_cache_path)
    if cache_mode == "PA_NZ":
        kr_cache.reshape((block_num, block_size, qk_rope_head_dim // NzFrac, NzFrac)).transpose(0, 2, 1, 3).tofile(kr_cache_path)
        if is_quant_b:
            kv_cache.reshape((block_num, block_size, kv_lora_rank // 32, 32)).transpose(0, 2, 1, 3).tofile(kv_cache_path)
        else:
            kv_cache.reshape((block_num, block_size, kv_lora_rank // NzFrac, NzFrac)).transpose(0, 2, 1, 3).tofile(kv_cache_path)
    else:
        kr_cache.tofile(kr_cache_path)  # kr_cache in
        kv_cache.tofile(kv_cache_path)  # kv_cache in
    res[12] = kv_cache
    res[13] = kr_cache
    res[14] = kv_quant_scale_cache
    res[15] = block_num
    res[16] = block_table

    return res

def gen_mla_prolog_quant_v32_data(params, dtypes, actual_seq, epsilon, output_dir: Path, is_quant=(False, False), is_nz=False,
                                  has_smooth=False, block_size=128, cache_mode="BNSD"):
    np.random.seed(int(time.time()))
    dtype, w_dtype = dtypes
    logging.debug(f"gen_mla_prolog_quant_v32_data  dtype:{dtype}, w_dtype:{w_dtype}")
    x, w_dq, w_uqqr, smooth_cq, scale_data, w_dkvkr, w_uk, gamma_cq, gamma_ckv, cos, sin, kv_len, kv_cache, kr_cache,  kv_quant_scale_cache, block_num, block_table= \
        gen_mla_prolog_quant_v32_input_data(params, dtypes, actual_seq, output_dir, is_quant, is_nz, has_smooth, block_size, cache_mode)
    is_quant_a, is_quant_b = is_quant
    b = params.get("b")
    s2 = params.get("s2")  # s2=4k
    qk_rope_head_dim = params.get("qk_rope_head_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    NzFrac = 16 if dtype != float else 8

    # output
    q_golden_path = Path(output_dir, 'q_golden.bin')
    q_rope_golden_path = Path(output_dir, 'q_rope_golden.bin')
    rms_norm_golden_path = Path(output_dir, 'rms_norm_golden.bin')
    rms_norm_scale_golden_path = Path(output_dir, 'rms_norm_scale_golden.bin')
    kv_golden_path = Path(output_dir, 'kv_cache_golden.bin')
    kr_golden_path = Path(output_dir, 'kr_cache_golden.bin')
    kv_quant_scale_cache_golden_path = Path(output_dir, 'kv_quant_scale_cache_golden.bin')

    inputs = {"dtype": dtype, "is_quant_a": is_quant_a, "is_quant_b": is_quant_b, "has_smooth": has_smooth}
    inputs["cache_mode"] = cache_mode
    inputs["gamma_cq"] = gamma_cq
    inputs["gamma_ckv"] = gamma_ckv
    inputs["epsilon"] = epsilon
    inputs["x"] = x
    inputs["w_dq"] = w_dq
    inputs["w_uqqr"] = w_uqqr
    inputs["w_uk"] = w_uk
    inputs["w_dkvkr"] = w_dkvkr
    inputs["cos"] = cos
    inputs["sin"] = sin
    inputs["kv_cache"] = kv_cache
    inputs["kr_cache"] = kr_cache
    inputs["kv_quant_scale_cache"] = kv_quant_scale_cache
    inputs["cache_index"] = kv_len
    if is_quant_a:
        inputs["w_qa_scale"] = scale_data["w_dq"]
        inputs["w_kva_scale"] = scale_data["w_dkvkr"]
    if is_quant_b:
        inputs["w_qb_scale"] = scale_data["w_uqqr"]
        if has_smooth:
            inputs["smooth_cq"] = smooth_cq

    q_out, q_embed, rms_norm, rms_norm_scale, kv_cache_out, kr_cache_out, kv_quant_scale_cache_out = mla_prolog_quant_v32_compute(inputs)

    q_out.tofile(q_golden_path)  # [b,s,n,kv_lora_rank]
    q_embed.tofile(q_rope_golden_path)  # [b,s,n,qk_rope_head_dim]

    if cache_mode == "PA_NZ":
        kr_cache_out.reshape((block_num, block_size, qk_rope_head_dim // NzFrac, NzFrac)).transpose(0, 2, 1, 3).tofile(kr_golden_path)
        if is_quant_b:
            kv_cache_out.reshape((block_num, block_size, kv_lora_rank // 32, 32)).transpose(0, 2, 1, 3).tofile(kv_golden_path)
        else:
            kv_cache_out.reshape((block_num, block_size, kv_lora_rank // NzFrac, NzFrac)).transpose(0, 2, 1, 3).tofile(kv_golden_path)
        kr_cache_out = kr_cache_out.transpose(0, 2, 1, 3)
        kv_cache_out = kv_cache_out.transpose(0, 2, 1, 3)
    else:
        kr_cache_out.tofile(kr_golden_path)
        kv_cache_out.tofile(kv_golden_path)
    if kv_quant_scale_cache_out is not None:
        kv_quant_scale_cache_out.tofile(kv_quant_scale_cache_golden_path)
    rms_norm.tofile(rms_norm_golden_path)
    if rms_norm_scale is not None:
        rms_norm_scale.tofile(rms_norm_scale_golden_path)
    return q_out, q_embed, kv_cache_out, kr_cache_out

def gen_mla_prolog_v32_quantB_test(dtypes, bn1s1s2, epsilon, output_dir: Path, is_quant=False, is_nz=False,
                                   is_smooth=False, block_size=128, cache_mode="BNSD"):
    b, n, s1, s2 = bn1s1s2
    quant_choice = (False, is_quant)
    params = {
        "b": b,
        "s": s1,
        "s1": s1,
        "s2": s2,
        "h": 7168,
        "num_heads": n,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    actual_seq  = torch.tensor([s2] * b, dtype = torch.int32).unsqueeze(-1).numpy()
    gen_mla_prolog_quant_v32_data(params, dtypes, actual_seq, epsilon, output_dir, quant_choice, is_nz, is_smooth, block_size, cache_mode)

def gen_mla_prolog_quant_v32_data_wrap(case_name: str, output: Path):
    # fp16, quant, weight nd, "PA_BSND"
    if case_name == "MlaPrologQuantV32STest.b1_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((np.float16, np.float16), (1, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b4_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((np.float16, np.float16), (4, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b8_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((np.float16, np.float16), (8, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b16_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((np.float16, np.float16), (16, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((np.float16, np.float16), (32, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b64_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((np.float16, np.float16), (64, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b128_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((np.float16, np.float16), (128, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    # quant
    elif case_name == "MlaPrologQuantV32STest.b32_s64k1_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((np.float16, np.float16), (32, 128, 1, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s64k4_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((np.float16, np.float16), (32, 128, 4, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    # quant
    elif case_name == "MlaPrologQuantV32STest.b32_s4k4_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((np.float16, np.float16), (32, 128, 4, 4 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s16k4_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((np.float16, np.float16), (32, 128, 4, 16 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s128k4_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((np.float16, np.float16), (32, 128, 4, 128 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    # quant small shape
    elif case_name == "MlaPrologQuantV32STest.b1_s11_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((np.float16, np.float16), (1, 128, 1, 1), 1e-5, output, True, False, False, 128, "PA_BSND")

    # bf16, quant, weight nd, "PA_BSND"
    if case_name == "MlaPrologQuantV32STest.b1_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((bfloat16, bfloat16), (1, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b4_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((bfloat16, bfloat16), (4, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b8_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((bfloat16, bfloat16), (8, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b16_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((bfloat16, bfloat16), (16, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((bfloat16, bfloat16), (32, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b64_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((bfloat16, bfloat16), (64, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b128_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((bfloat16, bfloat16), (128, 128, 2, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    # quant
    elif case_name == "MlaPrologQuantV32STest.b32_s64k1_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((bfloat16, bfloat16), (32, 128, 1, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s64k4_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((bfloat16, bfloat16), (32, 128, 4, 64 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    # quant
    elif case_name == "MlaPrologQuantV32STest.b32_s4k4_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((bfloat16, bfloat16), (32, 128, 4, 4 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s16k4_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((bfloat16, bfloat16), (32, 128, 4, 16 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s128k4_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((bfloat16, bfloat16), (32, 128, 4, 128 * 1024), 1e-5, output, True, False, False, 128, "PA_BSND")
    # quant small shape
    elif case_name == "MlaPrologQuantV32STest.b1_s11_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((bfloat16, bfloat16), (1, 128, 1, 1), 1e-5, output, True, False, False, 128, "PA_BSND")
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    return True


# MlaPrologQuantV32
case_names = [
    # fp16, quant, weight nd, "PA_BSND"
    "MlaPrologQuantV32STest.b1_s64k2_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b4_s64k2_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b8_s64k2_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b16_s64k2_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b32_s64k2_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b64_s64k2_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b128_s64k2_pa_nd_fp16_quantB",
    #
    "MlaPrologQuantV32STest.b32_s64k1_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b32_s64k4_pa_nd_fp16_quantB",
    #
    "MlaPrologQuantV32STest.b32_s4k4_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b32_s16k4_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b32_s128k4_pa_nd_fp16_quantB",
    # small shape, 有bug
    #"MlaPrologQuantV32STest.b1_s11_pa_nd_fp16_quantB",

    # bf16, quant, weight nd, "PA_BSND"
    "MlaPrologQuantV32STest.b1_s64k2_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b4_s64k2_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b8_s64k2_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b16_s64k2_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b32_s64k2_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b64_s64k2_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b128_s64k2_pa_nd_bf16_quantB",
    #
    "MlaPrologQuantV32STest.b32_s64k1_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b32_s64k4_pa_nd_bf16_quantB",
    #
    "MlaPrologQuantV32STest.b32_s4k4_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b32_s16k4_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b32_s128k4_pa_nd_bf16_quantB",
    # small shape, 有bug
    #"MlaPrologQuantV32STest.b1_s11_pa_nd_bf16_quantB",
]


@GoldenRegister.reg_golden_func(
    case_names = [
        *case_names
    ]
)
def gen_mla_prolog_quant_v32_date_one_case(case_name: str, output: Path) -> bool:
    if case_name in case_names:
        gen_mla_prolog_quant_v32_data_wrap(case_name, output)
        return True
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False

def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "MlaPrologQuantV32STest.b4_s64k2_pa_nd_fp16_quantB"
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "../../build/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_mla_prolog_quant_v32_date_one_case(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    # 只有当脚本作为主程序执行时，才会调用 main()
    # TestTranspose_DEBUG_ABC_BAC(1,1,128)
    # TestTranspose_DEBUG_BNDS_BNSD(2,1,1,128)
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    exit(0 if main() else 1)
