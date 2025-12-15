#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""

import numpy as np
import os
import torch
import torch_npu
import pytest
import pypto
from dataclasses import dataclass

from glm_attention_pre import add_rms_norm_npu_golden, rms_norm_npu_golden, \
    apply_rotary_pos_emb_v2, rms_norm_bias, rope_data
from glm_scatter import scatter_update_golden	
from glm_attention import gen_block_table, kv_cache_concat_bsnd, softmax, \
    AttentionConfig, AttentionTileConfig
np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})


@dataclass
class AttentionInputs:
    hidden_states: torch.Tensor  # shape:[bs, hidden_size]
    residual: torch.Tensor  # shape:[bs, hidden_size]
    # LayerNorm 相关
    input_layernorm_weight: torch.Tensor  # shape:[hidden_size]
    input_layernorm_bias: torch.Tensor  # shape:[hidden_size]
    # QKV 投影相关
    qkv_proj_scale: torch.Tensor  # shape:[hidden_size]
    qkv_proj_offset: torch.Tensor  # shape:[hidden_size]
    qkv_proj_weight: torch.Tensor  # shape:[hidden_size]
    qkv_proj_quant_bias: torch.Tensor  # shape:[total_head_size]
    qkv_proj_deq_scale: torch.Tensor  # shape:[total_head_size]
    # Q/K 归一化相关
    q_norm_weight: torch.Tensor  # shape:[head_size]
    q_norm_bias: torch.Tensor  # shape:[head_size]
    k_norm_weight: torch.Tensor  # shape:[head_size]
    k_norm_bias: torch.Tensor  # shape:[head_size]
    # 旋转位置编码相关
    cos: torch.Tensor  # shape:[bs, 1, half_rotary_dim]
    sin: torch.Tensor  # shape:[bs, 1, half_rotary_dim]
    # KV 缓存相关
    key_cache: torch.Tensor  # shape:[kv_num_blocks, block_size, n2, d]
    value_cache: torch.Tensor  # shape:[kv_num_blocks, block_size, n2, d]
    block_tables: torch.Tensor  # shape:[block_table_batch, max_num_blocks_per_query]
    actual_seq_lens: torch.Tensor  # shape:[b]
    slot_mapping: torch.Tensor  # shape:[b]
    eps: torch.float32  # rms_norm eps
    enable_residual: torch.bool
    num_decode_tokens: torch.int32 = 0  # 默认值保持原逻辑
    

def _attention(
        tensor_inputs: AttentionInputs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 从数据类中解构需要的参数（保持原逻辑不变）
    dtype, device = tensor_inputs.hidden_states.dtype, tensor_inputs.hidden_states.device
    bs = tensor_inputs.hidden_states.shape[0]
    total_head_size = tensor_inputs.qkv_proj_weight.shape[1]
    head_size = tensor_inputs.q_norm_weight.shape[0]
    n1 = total_head_size // head_size - 2
    q_shape = (bs, n1, head_size)
    out_torch = torch.full(q_shape, 9, dtype=dtype, device=device)
    q_tmp = torch.zeros((bs, n1 * head_size), dtype=dtype, device=device)
    k_tmp = torch.zeros((bs, head_size), dtype=dtype, device=device)
    v_tmp = torch.zeros((bs, head_size), dtype=dtype, device=device)
    hidden_size = tensor_inputs.qkv_proj_scale.shape[0]
    residual_tmp = torch.zeros((bs, hidden_size), dtype=dtype, device=device)

    # 构造 inputs/outputs 字典（使用封装后的 tensor_inputs 解构）
    inputs = {
        tensor_inputs.key_cache: [0],
        tensor_inputs.value_cache: [0],
        tensor_inputs.block_tables: [],
        tensor_inputs.actual_seq_lens: [0],
        tensor_inputs.slot_mapping: [0],
        tensor_inputs.hidden_states: [0],
        tensor_inputs.residual: [0],
        tensor_inputs.input_layernorm_weight: [],
        tensor_inputs.input_layernorm_bias: [],
        tensor_inputs.qkv_proj_scale: [],
        tensor_inputs.qkv_proj_offset: [],
        tensor_inputs.qkv_proj_weight: [],
        tensor_inputs.qkv_proj_quant_bias: [],
        tensor_inputs.qkv_proj_deq_scale: [],
        tensor_inputs.q_norm_weight: [],
        tensor_inputs.q_norm_bias: [],
        tensor_inputs.k_norm_weight: [],
        tensor_inputs.k_norm_bias: [],
        tensor_inputs.cos: [0],
        tensor_inputs.sin: [0],
    }
    outputs = {
        out_torch: [],
        q_tmp: [0],
        k_tmp: [0],
        v_tmp: [0],
        residual_tmp: [0]
    }

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

    # 调用 kernel（使用 config 中的配置参数）
    ifa_func(pto_inputs, pto_outputs, tensor_inputs.enable_residual, tensor_inputs.eps, tensor_inputs.num_decode_tokens)
    pypto.runtime._device_synchronize()
    return out_torch, q_tmp, k_tmp, v_tmp, residual_tmp


def get_qwen_common_config(device="cpu"):
    b = 8
    s1 = 1
    s2 = 4096
    q_d = 128
    n1 = 12
    n2 = 1
    kv_layout = "PA_BSND"
    softmax_scale = q_d ** -0.5
    block_table_batch = 256
    kv_num_blocks = 100

    # 创建 torch tensor 类型的 actual_seq
    actual_seq_values = [8, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6]
    actual_seq_values = [8, 6] * 4
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)
    atten_cfg = AttentionConfig(b=b, s1=s1, s2=s2, n1=n1, n2=n2, softmax_scale=softmax_scale, kv_layout=kv_layout,
                                q_d=q_d, kv_d=q_d, block_table_batch=block_table_batch, kv_num_blocks=kv_num_blocks,
                                actual_seq=actual_seq_tensor)  # 传入 tensor
    atten_cfg.max_num_blocks_per_query = 32
    cube_tile = 128
    vector_tile = 128
    s2_tile = 128
    tile_cfg = AttentionTileConfig(
        n1,
        s2_tile,
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [vector_tile, vector_tile],
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [vector_tile, vector_tile])
    return atten_cfg, tile_cfg


@pypto.jit
def ifa_func(inputs, outputs, enable_residual=True, eps=1e-05, num_decode_tokens=0):
    # 1. 添加支持动态的config
    pypto.set_codegen_options(support_dynamic_unaligned=True)
    pypto.set_host_options(only_codegen=True)
    pypto.set_option('profile_enable', True)

    # 2. 从入参拿到输入和输出tensor
    key_cache, value_cache, block_table, kv_act_seqs, index, x, residual_input, x_gamma, \
    x_bias, x_scale, x_offset, weight, quant_bias, deq_scale, q_gamma, q_bias, k_gamma, \
    k_bias, cos, sin = inputs
    bs_tile = 8
    atten_out = outputs[0]
    q_tmp, k_tmp, v_tmp, residual = outputs[1:]

    # 4. 得到动态tensor的shape
    bs = x.shape[0]
    hidden_size = x.shape[1]
    total_head_size = weight.shape[1]

    head_size = 128
    x_mean_coff = 1.0 / x.shape[-1]
    qk_mean_coff = 1.0 / head_size
    half_rotary_dim = cos.shape[-1]
    rotary_dim = cos.shape[-1] * 2
    stay_dim = head_size - rotary_dim

    q_size = q_tmp.shape[-1]
    kv_size = k_tmp.shape[-1]
    q_num_head = q_size // head_size
    kv_num_head = kv_size // head_size
    kv_index = q_num_head + kv_num_head

    bs_loop = (bs + bs_tile - 1) // bs_tile
    calc_dtype = pypto.DT_FP32
    input_dtype = x.dtype
    tiling_value = 128
    vec_tile_value = 5120
    q_batch_tile = 4

    # 4. 定义动态函数
    shape_k = key_cache.shape
    shape_act_seqs = kv_act_seqs.shape
    shape_hidden_states = x.shape
    weigh_shape = weight.shape
    q_gamma_shape = q_gamma.shape

    atten_cfg, tile_cfg = get_qwen_common_config()
    softmax_scale = atten_cfg.softmax_scale

    bs_scalar = shape_hidden_states[0]
    n1 = weigh_shape[1] // q_gamma_shape[0] - 2
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    n2 = shape_k[2]
    dn = shape_k[3]
    b_scalar = shape_act_seqs[0]

    dtype = key_cache.dtype
    group = n1 // n2

    g_tile = tile_cfg.g_tile
    s2_tile = tile_cfg.s2_tile
    c1_tile = tile_cfg.c1_tile_shape
    v1_tile = tile_cfg.v1_tile_shape
    c2_tile = tile_cfg.c2_tile_shape
    v2_tile = tile_cfg.v2_tile_shape

    # 5. 得到动态tensor的shape
    s1_scalar = bs_scalar // b_scalar
    g = n1 // n2
    g_loop = g // g_tile

    q_2d_shape = (b_scalar * s1_scalar * n1, dn)
    kv_cache_2d_shape = (block_num_scalar * block_size, n2 * dn)

    b_tile = 1
    d = dn
    b_loop = (b_scalar + b_tile - 1) // b_tile

    for _ in pypto.loop(1, name="LOOP_RESHAPE_INPLACE", idx_name="tmp_idx"):
        pypto.set_vec_tile_shapes(5120)
        x_gamma_2d = pypto.reshape(x_gamma, [1, 5120], inplace=True)
        x_bias_2d = pypto.reshape(x_bias, [1, 5120], inplace=True)
        x_scale_2d = pypto.reshape(x_scale, [1, 5120], inplace=True)
        x_offset_2d = pypto.reshape(x_offset, [1, 5120], inplace=True)
        quant_bias_2d = pypto.reshape(quant_bias, [1, 1792], inplace=True)
        deq_scale_2d = pypto.reshape(deq_scale, [1, 1792], inplace=True)

    # 6. 实现kernel逻辑，循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_ATT_PRE_L0", idx_name="bs_idx"):

        x_tile = pypto.view(x, [bs_tile, hidden_size], [bs_idx * bs_tile, 0])
        # init
        pypto.set_vec_tile_shapes(1, vec_tile_value)
        x_tile_fp32 = pypto.cast(x_tile, calc_dtype)
        # add
        if enable_residual:
            residual_input_tile = pypto.view(residual_input, [bs_tile, hidden_size], [bs_idx * bs_tile, 0])
            residual_input_tile_fp32 = pypto.cast(residual_input_tile, calc_dtype)
            x_f32 = pypto.add(residual_input_tile_fp32, x_tile_fp32)  # tile_x
        else:
            x_f32 = x_tile_fp32

        # rms norm
        square = pypto.mul(x_f32, x_f32)  # square
        mean_res = pypto.mul(square, x_mean_coff)  # mean_res = square * mean_coff
        reduce_asum = pypto.sum(mean_res, -1, keepdim=True)  # reduce_asum = mean_res.sum(dim=-1, keepdim=True)
        reduce_sum = pypto.add(reduce_asum, eps)  # reduce_sum = reduce_asum + eps
        reduce_sqrt = pypto.sqrt(reduce_sum)  # reduce_sqrt = torch.sqrt(reduce_sum)
        res_div = pypto.div(x_f32, reduce_sqrt)  # res_div = x_f32 / reduce_sqrt

        x_int8 = pypto.tensor([bs_tile, 5120], pypto.DT_INT8, "x_int8")
        residual_bf16 = pypto.cast(x_f32, input_dtype)

        for tmp_idx in range(bs_tile):
            pypto.set_vec_tile_shapes(1, vec_tile_value)
            x_gamma_2d_fp32 = pypto.cast(x_gamma_2d, calc_dtype)
            x_bias_2d_fp32 = pypto.cast(x_bias_2d, calc_dtype)
            x_scale_2d_fp32 = pypto.cast(x_scale_2d, calc_dtype)
            x_offset_2d_fp32 = pypto.cast(x_offset_2d, calc_dtype)

            res_div_single = pypto.view(res_div, [1, hidden_size], [tmp_idx, 0])

            res = pypto.mul(res_div_single, x_gamma_2d_fp32)  # res = res_div * weight
            res_add = pypto.add(res, x_bias_2d_fp32)
            x_norm = pypto.cast(res_add, input_dtype)

            # x quant
            pypto.set_vec_tile_shapes(1, vec_tile_value)
            x_norm_fp32 = pypto.cast(x_norm, calc_dtype)  # bf16 -> fp32
            x_mul = pypto.mul(x_norm_fp32, x_scale_2d_fp32)
            x_add = pypto.add(x_mul, x_offset_2d_fp32)
            x_int32 = pypto.cast(x_add, pypto.DT_INT32, pypto.CastMode.CAST_RINT)  # Align ascendC
            x_fp16 = pypto.cast(x_int32, pypto.DT_FP16)
            x_int8[tmp_idx:tmp_idx + 1, 0:] = pypto.cast(x_fp16, pypto.DT_INT8)

        pypto.set_vec_tile_shapes(32, 256)
        extend_params = {'bias_tensor': quant_bias_2d}
        pypto.set_cube_tile_shapes([tiling_value, tiling_value], [tiling_value, tiling_value],
                                   [tiling_value, tiling_value])
        mm_add = pypto.matmul(x_int8, weight, pypto.DT_INT32, a_trans=False, b_trans=False, extend_params=extend_params)

        pypto.set_vec_tile_shapes(8, 1792)
        mm_fp32 = pypto.cast(mm_add, calc_dtype)  # int32 -> fp32
        mm_deq_scale = pypto.mul(mm_fp32, deq_scale_2d)
        mm_bf16 = pypto.cast(mm_deq_scale, input_dtype)  # fp32 -> bf16

        pypto.set_vec_tile_shapes(tiling_value, tiling_value, head_size)
        mm_3d = pypto.reshape(mm_bf16, [bs_tile, total_head_size // head_size, head_size], inplace=True)

        # split
        q_tile = pypto.view(mm_3d, [bs_tile, q_num_head, head_size], [0, 0, 0])
        k_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, q_num_head, 0])
        v_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, kv_index, 0])

        # rms norm
        q_norm = rms_norm_bias(q_tile, q_gamma, q_bias, qk_mean_coff, eps, [q_batch_tile, q_num_head, head_size])
        k_norm = rms_norm_bias(k_tile, k_gamma, k_bias, qk_mean_coff, eps, [q_batch_tile, kv_num_head, head_size])

        q_rot = pypto.view(q_norm, [bs_tile, q_num_head, rotary_dim], [0, 0, 0])
        q_pass = pypto.view(q_norm, [bs_tile, q_num_head, stay_dim], [0, 0, rotary_dim])

        k_rot = pypto.view(k_norm, [bs_tile, kv_num_head, rotary_dim], [0, 0, 0])
        k_pass = pypto.view(k_norm, [bs_tile, kv_num_head, stay_dim], [0, 0, rotary_dim])

        # apply rope
        # cast
        pypto.set_vec_tile_shapes(q_batch_tile, q_num_head, head_size)
        cos_tile = pypto.view(cos, [bs_tile, 1, half_rotary_dim], [bs_idx * bs_tile, 0, 0])
        sin_tile = pypto.view(sin, [bs_tile, 1, half_rotary_dim], [bs_idx * bs_tile, 0, 0])
        q_fp32 = pypto.cast(q_rot, calc_dtype)
        k_fp32 = pypto.cast(k_rot, calc_dtype)
        cos_fp32 = pypto.cast(cos_tile, calc_dtype)
        sin_fp32 = pypto.cast(sin_tile, calc_dtype)

        # q split
        q1 = pypto.view(q_fp32, [bs_tile, q_num_head, half_rotary_dim], [0, 0, 0])
        q2 = pypto.view(q_fp32, [bs_tile, q_num_head, half_rotary_dim], [0, 0, half_rotary_dim])

        # rope data
        q_rope = rope_data(q1, q2, cos_fp32, sin_fp32, [q_batch_tile, q_num_head, half_rotary_dim])
        q_cat = pypto.concat([q_rope, q_pass], 2)

        # k split
        k1 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_rotary_dim], [0, 0, 0])
        k2 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_rotary_dim], [0, 0, half_rotary_dim])

        # rope data
        k_rope = rope_data(k1, k2, cos_fp32, sin_fp32, [q_batch_tile, q_num_head, half_rotary_dim])
        k_cat = pypto.concat([k_rope, k_pass], 2)

        # post process
        q_res = pypto.reshape(q_cat, [bs_tile, q_size])
        k_res = pypto.reshape(k_cat, [bs_tile, kv_size])
        v_res = pypto.reshape(v_tile, [bs_tile, kv_size])

        # # 9. 将结果搬运到输出tensor上
        # # update output
        q_tmp[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = q_res
        k_tmp[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = k_res
        v_tmp[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = v_res
        residual[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = residual_bf16

    for _ in pypto.loop(1, name="LOOP_RESHAPE_INPLACE", idx_name="tmp_idx"):
        key_cache_2d = pypto.reshape(key_cache, kv_cache_2d_shape, inplace=True)
        value_cache_2d = pypto.reshape(value_cache, kv_cache_2d_shape, inplace=True)

        q_2d = pypto.reshape(q_tmp, q_2d_shape, inplace=True)

    for b_idx in pypto.loop(b_loop, name="LOOP_SCATTER_UPDATE", idx_name="b_idx", submit_before_loop=True):
        pypto.set_vec_tile_shapes(16, 16, 16)
        b_ofs = b_idx * b_tile
        b_valid = (b_scalar - b_idx * b_tile).min(b_tile)
        key_view = pypto.view(k_tmp, [b_tile, n2 * d], [b_ofs, 0], valid_shape=[b_valid, n2 * d])
        value_view = pypto.view(v_tmp, [b_tile, n2 * d], [b_ofs, 0], valid_shape=[b_valid, n2 * d])
        index_view = pypto.view(index, [b_tile], [b_ofs], valid_shape=[b_valid])
        index_view = pypto.reshape(index_view, [b_tile, 1], valid_shape=[b_valid, 1])
        pypto.set_vec_tile_shapes(16, 128)
        key_cache.move(pypto.scatter_update(key_cache_2d, -2, index_view, key_view))
        value_cache.move(pypto.scatter_update(value_cache_2d, -2, index_view, value_view))

        # 6. 实现kernel逻辑，循环展开B动态轴
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for n2_idx in pypto.loop(n2, name="LOOP_n2", idx_name="n2_idx"):
                for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                    oi_upd = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_upd")
                    li_upd = pypto.tensor([g_tile, 1], pypto.DT_FP32, "li_upd")
                    mi_upd = pypto.tensor([g_tile, 1], pypto.DT_FP32, "mi_upd")
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx"):
                        block_idx = block_table[b_idx, s2_idx]
                        bs_ofs = b_idx * s1_scalar + s1_idx
                        n1g_ofs = n2_idx * group + g_idx * g_tile
                        actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                        oi_ofs = [bs_ofs, n1g_ofs, 0]
                        # 7. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                        pypto.set_vec_tile_shapes(16, v1_tile[0], v1_tile[1])
                        qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * n1 + n1g_ofs, 0])
                        kj = pypto.view(key_cache_2d, [block_size, dn], [block_idx * block_size, 0],
                                        valid_shape=[actual_s2_tile, dn])
                        vj = pypto.view(value_cache_2d, [block_size, dn], [block_idx * block_size, 0],
                                        valid_shape=[actual_s2_tile, dn])
                        # c1
                        # 9. 下面是flash attention的计算逻辑
                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        sij = pypto.matmul(qi, kj, pypto.DT_FP32, a_trans=False, b_trans=True)
                        sij = pypto.reshape(sij, [g_tile, s2_tile], valid_shape=[g_tile, actual_s2_tile])
                        # v1
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij = pypto.amax(sij_scale, -1, True)
                        tsub = pypto.sub(sij_scale, tilda_mij)
                        tilda_pij = pypto.exp(tsub)
                        tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                        tilda_lij = pypto.sum(tilda_pij, -1, True)
                        pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                        oi_upd_3d = pypto.cast(pypto.reshape(tsub, [1, 1, g_tile * dn]), dtype)
                        if pypto.cond(pypto.is_loop_begin(s2_idx)):
                            # c2
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(tilda_pij_fp16, vj, pypto.DT_FP32)
                            oi_upd[:] = pypto.tensor(oi_tmp.shape, pypto.DT_FP32, "oi_upd")
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            if pypto.cond(pypto.is_loop_end(s2_idx)):
                                oi_upd[:] = pypto.div(oi_tmp, tilda_lij)
                                pypto.set_vec_tile_shapes(16, 16, v2_tile[0], v2_tile[1])
                                oi_upd_3d = pypto.cast(pypto.reshape(oi_upd, [1, g_tile, dn]),
                                                       dtype)
                                # 10. 将结果搬运到输出tensor上
                                pypto.assemble(oi_upd_3d, oi_ofs, atten_out)
                            else:
                                oi_upd[:] = oi_tmp

                            li_upd[:] = tilda_lij
                            mi_upd[:] = tilda_mij

                        else:
                            oi = oi_upd
                            li = li_upd
                            mi = mi_upd

                            mi_new = pypto.maximum(mi, tilda_mij)
                            t1 = pypto.sub(mi, mi_new)
                            t2 = pypto.exp(t1)
                            t3 = pypto.sub(tilda_mij, mi_new)
                            t4 = pypto.exp(t3)
                            t5 = pypto.mul(t4, tilda_lij)
                            t6 = pypto.mul(t2, li)
                            li_new = pypto.add(t6, t5)

                            q3 = pypto.mul(oi, t2)
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            q1 = pypto.matmul(tilda_pij_fp16, vj, pypto.DT_FP32)
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            q2 = pypto.mul(q1, t4)
                            oi_tmp = pypto.add(q3, q2)
                            if pypto.cond(pypto.is_loop_end(s2_idx)):
                                oi_upd[:] = pypto.div(oi_tmp, li_new)
                                pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                                oi_upd_3d = pypto.cast(pypto.reshape(oi_upd, [1, g_tile, dn]),
                                                       dtype)
                                # 11. 将结果搬运到输出tensor上
                                pypto.assemble(oi_upd_3d, oi_ofs, atten_out)
                            else:
                                oi_upd[:] = oi_tmp
                            li_upd[:] = li_new
                            mi_upd[:] = mi_new


def ifa(atten_cfg, device_id):
    # 使用 torch 生成数据
    device = f'npu:{device_id}'
    torch_dtype = torch.bfloat16
    b = atten_cfg.b
    s1 = atten_cfg.s1
    d = atten_cfg.q_d
    n1 = atten_cfg.n1
    n2 = atten_cfg.n2
    bs = b * s1
    hidden_size = 5120
    total_head_size = 1792
    head_size = 128
    q_size = 1536
    kv_size = 128
    rotary_dim = 64
    half_rotary_dim = rotary_dim // 2
    eps = 1e-05

    block_num = atten_cfg.kv_num_blocks
    block_size = atten_cfg.block_size
    max_num_blocks_per_query = atten_cfg.max_num_blocks_per_query

    # 获取 torch tensor 类型的 actual_seq
    kv_cache_actual_seq = atten_cfg.actual_seq

    q_shape = [b * s1, n1, d]
    kv_cache_shape = [atten_cfg.kv_num_blocks, block_size, n2, d]
    block_table_shape = [atten_cfg.block_table_batch, max_num_blocks_per_query]

    slot_mapping = torch.randperm(block_num * block_size)[:b].to(device=device)

    key_cache = torch.empty(kv_cache_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    value_cache = torch.empty(kv_cache_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)

    hidden_states = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    residual = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    input_layernorm_weight = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    input_layernorm_bias = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    qkv_proj_scale = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    qkv_proj_offset = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

    qkv_proj_weight = torch.randint(0, 255, size=(hidden_size, total_head_size), dtype=torch.int8,
                                    device=f'npu:{device_id}')
    qkv_proj_quant_bias = torch.randint(0, 255, size=(total_head_size,), dtype=torch.int32, device=f'npu:{device_id}')
    qkv_proj_deq_scale = torch.rand(total_head_size, dtype=torch.float32, device=f'npu:{device_id}')
    q_norm_weight = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    q_norm_bias = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    k_norm_weight = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    k_norm_bias = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    cos = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=f'npu:{device_id}')
    sin = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=f'npu:{device_id}')

    # outputs
    q_tmp = torch.zeros((bs, q_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
    k_tmp = torch.zeros((bs, kv_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
    v_tmp = torch.zeros((bs, kv_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
    residual_tmp = torch.zeros((bs, hidden_size), dtype=torch.bfloat16, device=f'npu:{device_id}')
    enable_residual = True
    if torch.all(residual == 0):
        residual = torch.zeros((bs, hidden_states.shape[1]), dtype=hidden_states.dtype,
                               device=f'{hidden_states.device}')
        enable_residual = False
    # 5. 与PyTorch参考实现对比
    # add rms norm
    x_g, residual_g = add_rms_norm_npu_golden(hidden_states, residual, input_layernorm_weight, input_layernorm_bias,
                                              eps)

    # matmul
    x_quant = torch_npu.npu_quantize(x_g, qkv_proj_scale, qkv_proj_offset, torch.qint8, -1, False)
    mm_golden = torch_npu.npu_quant_matmul(x_quant, qkv_proj_weight, qkv_proj_deq_scale, bias=qkv_proj_quant_bias,
                                           output_dtype=torch.bfloat16)

    # split
    q_g, k_g, v_g = mm_golden.split([q_size, kv_size, kv_size], dim=-1)
    # nms norm
    q_by_head = q_g.view(*q_g.shape[:-1], q_g.shape[-1] // head_size, head_size)
    q_by_head = rms_norm_npu_golden(q_by_head, q_norm_weight, q_norm_bias, eps)

    k_by_head = k_g.view(*k_g.shape[:-1], k_g.shape[-1] // head_size, head_size)
    k_by_head = rms_norm_npu_golden(k_by_head, k_norm_weight, k_norm_bias, eps)

    # apply rope
    q_rot = q_by_head[..., :rotary_dim]
    q_pass = q_by_head[..., rotary_dim:]
    k_rot = k_by_head[..., :rotary_dim]
    k_pass = k_by_head[..., rotary_dim:]
    q_r, k_r = apply_rotary_pos_emb_v2(q_rot, k_rot, cos, sin)
    q_cat = torch.cat((q_r, q_pass), dim=-1)
    k_cat = torch.cat((k_r, k_pass), dim=-1)
    # post process
    q_r = q_cat.view(bs, q_size)
    k_r = k_cat.view(bs, kv_size)

    key_cache_clone = key_cache.clone()
    value_cache_clone = value_cache.clone()
    key_scatter = k_r.view(b, n2, d)
    value_scatter = v_g.view(b, n2, d)
    scatter_update_golden(key_scatter, key_cache_clone, slot_mapping)
    scatter_update_golden(value_scatter, value_cache_clone, slot_mapping)
    q_atten = q_r.view(q_shape)

    attention_output = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    # 2. 生成block table - 传入 torch tensor
    block_tables = gen_block_table(kv_cache_actual_seq, block_size, block_table_shape)

    # 3. 根据block table 将pa格式的数据转换成
    k_cache_bsnd, v_cache_bsnd = kv_cache_concat_bsnd(key_cache_clone, value_cache_clone, block_tables, atten_cfg)

    for i in range(b):
        for j in range(s1):
            for n2_idx in range(n2):
                # 从 torch tensor 获取值
                kv_seq_len = kv_cache_actual_seq[i].item()  # 使用 .item() 获取标量值
                seq_len = kv_seq_len - s1 + 1 + j
                q_bs = q_atten[i * s1 + j]
                k_bs = k_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                v_bs = v_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                # MM1: 矩阵乘法
                qk_bmm_res = torch.matmul(q_bs, k_bs.transpose(1, 0))  # 1,n1, d  -> n_q,d @ d, s2_actual_len
                qk_ele_res = qk_bmm_res * atten_cfg.softmax_scale
                # Softmax计算
                softmax_res, _, _ = softmax(qk_ele_res, True)

                # MM2: 矩阵乘法
                bmm2_res = torch.matmul(softmax_res, v_bs)

                # 存储结果
                attention_output[i * s1 + j] = bmm2_res

    # 4. 准备测试数据 - 直接使用 torch 张量
    block_tables = block_tables.to(dtype=torch.int32, device=device)
    actual_seq_lens = kv_cache_actual_seq.to(dtype=torch.int32, device=device)  # 直接使用已有的 tensor

    # 5. 执行kernel并获取结果
    
    tensor_inputs = AttentionInputs(
        hidden_states=hidden_states,
        residual=residual,
        input_layernorm_weight=input_layernorm_weight,
        input_layernorm_bias=input_layernorm_bias,
        qkv_proj_scale=qkv_proj_scale,
        qkv_proj_offset=qkv_proj_offset,
        qkv_proj_weight=qkv_proj_weight,
        qkv_proj_quant_bias=qkv_proj_quant_bias,
        qkv_proj_deq_scale=qkv_proj_deq_scale,
        q_norm_weight=q_norm_weight,
        q_norm_bias=q_norm_bias,
        k_norm_weight=k_norm_weight,
        k_norm_bias=k_norm_bias,
        cos=cos,
        sin=sin,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        actual_seq_lens=actual_seq_lens,
        slot_mapping=slot_mapping,
        eps=1e-6,
        enable_residual=True,
        num_decode_tokens=0        
    )
    
    output, q_tmp, k_tmp, v_tmp, residual_tmp = _attention(tensor_inputs)    
    
    from utils.np_compare import detailed_allclose_manual
    # compare result
    detailed_allclose_manual(np.array(residual_g.cpu().flatten().tolist()), np.array(residual_tmp.flatten().tolist()),
                             "residual_g", rtol=0.001, atol=0.001)
    detailed_allclose_manual(np.array(q_r.cpu().flatten().tolist()), np.array(q_tmp.flatten().tolist()), "q_r",
                             rtol=0.001, atol=0.001)
    detailed_allclose_manual(np.array(k_r.cpu().flatten().tolist()), np.array(k_tmp.flatten().tolist()), "k_r",
                             rtol=0.01, atol=0.01)
    detailed_allclose_manual(np.array(v_g.cpu().flatten().tolist()), np.array(v_tmp.flatten().tolist()), "v_g",
                             rtol=0.001, atol=0.001)

    y_data = output.cpu()
    # 6. 与PyTorch参考实现对比
    detailed_allclose_manual(np.array(attention_output.flatten().tolist()), np.array(y_data.flatten().tolist()),
                             "attention", rtol=0.003, atol=0.003)

def test_super_attention():
    # 1. 设置参数
    device_id = os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0)
    device = f'npu:{device_id}'
    torch.npu.set_device(int(device_id))
    atten_cfg, _ = get_qwen_common_config(device=device)
    # 检查 B 的大小和 actual_seq 长度是否相等
    assert atten_cfg.b == len(
        atten_cfg.actual_seq), f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等'
    # 检查所有值是否都小于 s2
    if atten_cfg.actual_seq.device.type != 'cpu':
        actual_seq_cpu = atten_cfg.actual_seq.cpu()
    else:
        actual_seq_cpu = atten_cfg.actual_seq
    assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2"
    ifa(atten_cfg, device_id)


if __name__ == "__main__":
    test_super_attention()
