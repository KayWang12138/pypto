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
"""
"""
from dataclasses import dataclass
from typing import List
import logging
import math
import pto

NUM_1 = 1
NUM_2 = 2
NUM_16 = 16
NUM_32 = 32
NUM_64 = 64
NUM_128 = 128
NUM_512 = 512
NUM_1536 = 1536
NUM_7168 = 7168
NUM_65536 = 65536
KEY_SUPPORT_DYNAMIC_UNALIGNED = "support_dynamic_unaligned"


def ceil_div(a, b):
    return (a + b - 1) // b


@dataclass
class NSASimpleParams:
    b: int = 32
    s1: int = 1
    s2: int = 65536
    n1: int = 128
    n2: int = 1
    h: int = 7168
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    q_head_dim: int = 64 + 128  # 直接使用数值，因为不能引用其他字段
    rope_dim: int = 64
    cmp_block_size: int = 32
    cmp_stride: int = 16
    slc_block_size: int = 64
    front: int = 1
    near: int = 2
    topk: int = 16
    cache_mode: str = "BSND"
    block_size: int = 128
    win_size: int = 512
    v_head_dim: int = 128
    eps: float = 1e-5


@dataclass
class SATileShapeConfig:
    kv_slc_v0_tile_shape: List[int]
    g_tile: int
    s_kv_tile: int
    c1_tile_shape: List[int]
    v1_tile_shape: List[int]
    c2_tile_shape: List[int]
    v2_tile_shape: List[int]


def selected_attention_compute(**kwargs):
    top_k_indices = kwargs.get("top_k_indices")
    kv_nope_cache = kwargs.get("kv_nope_cache")
    k_rope_cache = kwargs.get("k_rope_cache")
    kv_act_seqs = kwargs.get("kv_act_seqs")
    block_table = kwargs.get("block_table")
    q_nope = kwargs.get("q_nope")
    q_rope = kwargs.get("q_rope")
    attention_out = kwargs.get("attention_out")
    n_q = kwargs.get("n_q")
    n_kv = kwargs.get("n_kv")
    softmax_scale = kwargs.get("softmax_scale")
    front = kwargs.get("front")
    near = kwargs.get("near")
    topk = kwargs.get("topk")
    block_size = kwargs.get("block_size")
    cmp_block_size = kwargs.get("cmp_block_size")
    slc_block_size = kwargs.get("slc_block_size")
    sa_tile_config = kwargs.get("sa_tile_config")
    debug = kwargs.get("debug")

    dtype = q_nope.get_dtype()
    d_n = q_nope.shape[1]
    d_r = q_rope.shape[1]
    group = n_q // n_kv

    # 接口？
    v0_tile = sa_tile_config.kv_slc_v0_tile_shape
    g_tile = sa_tile_config.g_tile
    c1_tile = sa_tile_config.c1_tile_shape
    v1_tile = sa_tile_config.v1_tile_shape
    c2_tile = sa_tile_config.c2_tile_shape
    v2_tile = sa_tile_config.v2_tile_shape

    batch_size_sym = top_k_indices.shape[0]
    s1_n2_g_sym = q_nope.shape[0] // batch_size_sym
    s1_sym = s1_n2_g_sym // n_q
    g_loop_sym = group // g_tile
    n2_sym = n_kv

    pto.set_codegen_config(KEY_SUPPORT_DYNAMIC_UNALIGNED, True)

    with pto.loop_function("LOOP_L0_b_SA", "b_idx", pto.loop_range(0, batch_size_sym, 1), set(), True) as b_idx_loop:
        for b_idx in b_idx_loop:
            def inside_b_loop_sa(b_idx):
                # nonlocal sa_out
                cur_act_seq = pto.get_tensor_data(kv_act_seqs, [b_idx])
                cur_act_seq.as_intermediate_variable()
                with pto.loop_function("LOOP_L1_s1_SA", "s1_idx", pto.loop_range(0, s1_sym, 1)) as s1_idx_loop:
                    for s1_idx in s1_idx_loop:
                        with pto.loop_function("LOOP_L2_n2_SA", "n2_idx", 
                            pto.loop_range(0, n2_sym, 1)) as n2_idx_loop:
                            for n2_idx in n2_idx_loop:
                                with pto.loop_function("LOOP_L3_g_SA", "g_idx", 
                                    pto.loop_range(0, g_loop_sym, 1)) as g_idx_loop:
                                    for g_idx in g_idx_loop:
                                        def inside_g_loop_sa(b_idx, s1_idx, n2_idx, g_idx):
                                            # nonlocal sa_out
                                            # nonlocal g_tile
                                            cur_g_tile = g_tile
                                            cur_offset = (b_idx * s1_n2_g_sym + s1_idx * n_q 
                                                        + n2_idx * group + g_idx * cur_g_tile)
                                            oi_offset = [b_idx, s1_idx, n2_idx * group + g_idx * cur_g_tile, 0]
                                            
                                            with pto.loop_function("LOOP_L4_s2_SA", "s2_idx", pto.loop_range(0, 1, 1), 
                                                pto.powers_of_2(1)) as s2_idx_loop:
                                                for s2_idx in s2_idx_loop:
                                                    def inside_s2_loop_sa(b_idx, s1_idx, n2_idx, s2_idx, topk):
                                                        cur_s2_tile = pto.symbolic_scalar(topk * slc_block_size)
                                                        pto.set_semantic_label("kv_slc")
                                                        k_slc = pto.tensor([topk * slc_block_size, 
                                                            d_n + d_r], dtype, "k_slc")
                                                        cur_kv_slc_seq = pto.symbolic_scalar(0)
                                                        s_slc = (cur_act_seq - s1_sym + 1 + s1_idx - 
                                                            cmp_block_size + slc_block_size) // slc_block_size
                                                        s_slc.as_intermediate_variable()
                                                        positions = pto.symbolic_scalar(0)
                                                        for top_k_idx in range(topk):
                                                            if top_k_idx < front:
                                                                # 获取到topk的position
                                                                # 头部的front个
                                                                positions = top_k_idx * slc_block_size
                                                            elif top_k_idx > (topk - near - front):
                                                                # 尾部的near个
                                                                positions = (s_slc - near + (top_k_idx - (
                                                                    topk - front - near)) - 1) * slc_block_size
                                                            else:
                                                                # 中间的topk-front-near个
                                                                top_k_index = None
                                                                if debug:
                                                                    pto.set_vec_tile_shapes(1, 1, NUM_16)
                                                                    top_k_index = pto.get_tensor_data(top_k_indices, 
                                                                        [b_idx, s1_idx, pto.symbolic_scalar(top_k_idx 
                                                                                                            - front)])
                                                                else:
                                                                    top_k_index = pto.get_tensor_data(top_k_indices, 
                                                                        [b_idx, s1_idx, pto.symbolic_scalar(top_k_idx 
                                                                                                            - front)])
                                                                positions = top_k_index * slc_block_size 
                                                            cur_kv_slc_seq = cur_kv_slc_seq + min(slc_block_size, 
                                                                cur_act_seq - positions)
                                                            block_idx_in_batch = positions // pto.symbolic_scalar(
                                                                                                            block_size)
                                                            tail = positions % block_size
                                                            slc_block_idx = pto.get_tensor_data(block_table, 
                                                                                        [b_idx, block_idx_in_batch])
                                                            pto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
                                                            kv_slc_block = pto.view(kv_nope_cache, 
                                                                                        [slc_block_size, d_n], 
                                                                [slc_block_idx * block_size + tail, n2_idx * d_n])
                                                            kr_slc_block = pto.view(k_rope_cache, [slc_block_size, d_r], 
                                                                [slc_block_idx * block_size + tail, n2_idx * d_r])

                                                            pto.set_semantic_label("kv_slc_cast_fp32")
                                                            pto.set_vec_tile_shapes(v0_tile[0], v1_tile[1])
                                                            kv_slc_block_fp32 = pto.cast(kv_slc_block, 
                                                                                        pto.DataType.DT_FP32)
                                                            kr_slc_block_fp32 = pto.cast(kr_slc_block, 
                                                                                        pto.DataType.DT_FP32)
                                                            pto.set_semantic_label("kv_slc_cast")
                                                            pto.set_vec_tile_shapes(v0_tile[0], v1_tile[1])
                                                            kv_slc_block_fp16 = pto.cast(kv_slc_block_fp32, 
                                                                                        k_slc.get_dtype())
                                                            kr_slc_block_fp16 = pto.cast(kr_slc_block_fp32, 
                                                                                        k_slc.get_dtype())
                                                            pto.set_vec_tile_shapes(v0_tile[0], v1_tile[1])

                                                            slc_out_s_offset = (top_k_idx 
                                                                * slc_block_size)
                                                            pto.assemble(kv_slc_block_fp16, 
                                                                [slc_out_s_offset, 0], k_slc)
                                                            pto.assemble(kr_slc_block_fp16, 
                                                                [slc_out_s_offset, d_n], k_slc)

                                                        # qAssemble
                                                        pto.set_semantic_label("Sa")
                                                        # View, 临时规避改成 View
                                                        qn = pto.view(q_nope, [cur_g_tile, d_n], [cur_g_tile, d_n], 
                                                                    [cur_offset, 0])
                                                        qr = pto.view(q_rope, [cur_g_tile, d_r], [cur_g_tile, d_r], 
                                                                    [cur_offset, 0])
                                                        qi = pto.tensor([cur_g_tile, d_n + d_r], dtype, "qi")
                                                        pto.assemble(qn, [0, 0], qi)
                                                        pto.assemble(qr, [0, d_n], qi)

                                                        # slc_attn
                                                        cur_seq = pto.symbolic_scalar(max(
                                                                        cur_kv_slc_seq - s1_sym + 1 + s1_idx, 0))
                                                        cur_seq.as_intermediate_variable()
                                                        kj = pto.view(k_slc, [cur_s2_tile, d_n + d_r], 
                                                            [(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), 
                                                            d_n + d_r], [s2_idx * cur_s2_tile, 0])
                                                        vj = pto.view(k_slc, [cur_s2_tile, d_n], 
                                                            [(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), d_n],
                                                                [s2_idx * cur_s2_tile, 0])

                                                        # C1
                                                        pto.set_semantic_label("Sa_QkMM")
                                                        pto.set_cube_tile_shapes(
                                                            [c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]], 
                                                            [c1_tile[4], c1_tile[5]], True)
                                                        pto.set_matrix_size([qi.shape[0], 0, kj.shape[0]])
                                                        sij = pto.matmul(pto.DataType.DT_FP32, qi, kj, False, True)

                                                        # V1
                                                        pto.set_semantic_label("Sa_Qkvec1")
                                                        pto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                                        sij_scale = pto.mul_s(sij, pto.element(sij.get_dtype(), 
                                                                            softmax_scale))
                                                        tilda_mij = pto.row_max_single(sij_scale)  
                                                        tsub = pto.sub(sij_scale, tilda_mij)  
                                                        tilda_pij = pto.exp(tsub) 
                                                        tilda_lij = pto.row_sum_single(tilda_pij)
                                                        t_softmax = pto.div(tilda_pij, tilda_lij)
                                                        tilda_pij_f16 = pto.cast(t_softmax, dtype)

                                                        # C2
                                                        pto.set_semantic_label("Sa_KvMm")
                                                        pto.set_cube_tile_shapes(
                                                            [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], 
                                                            [c2_tile[4], c2_tile[5]], True)
                                                        pto.set_matrix_size(
                                                            [tilda_pij_f16.shape[0], tilda_pij_f16.shape[1], 
                                                            vj.shape[1]])
                                                        oi = pto.matmul(pto.DataType.DT_FP32, 
                                                            tilda_pij_f16, vj, False, False)

                                                        # V2
                                                        pto.set_semantic_label("Sa_KvVec2")
                                                        pto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                                        oi_4_dim = pto.add_s(pto.reshape(oi, [1, 1, cur_g_tile, 
                                                            d_n]),
                                                            pto.element(oi.get_dtype(), float(0)))
                                                        pto.assemble(oi_4_dim, oi_offset, attention_out)
                                                    inside_s2_loop_sa(b_idx, s1_idx, n2_idx, s2_idx, topk)
                                        inside_g_loop_sa(b_idx, s1_idx, n2_idx, g_idx)
            inside_b_loop_sa(b_idx)                        


def selected_attention(**kwargs):
    top_k_indices = kwargs.get("top_k_indices")
    kv_nope_cache = kwargs.get("kv_nope_cache")
    k_rope_cache = kwargs.get("k_rope_cache")
    kv_act_seqs = kwargs.get("kv_act_seqs")
    block_table = kwargs.get("block_table")
    q_nope = kwargs.get("q_nope")
    q_rope = kwargs.get("q_rope")
    attention_out = kwargs.get("attention_out")
    n_q = kwargs.get("n_q")
    n_kv = kwargs.get("n_kv")
    softmax_scale = kwargs.get("softmax_scale")
    front = kwargs.get("front")
    near = kwargs.get("near")
    topk = kwargs.get("topk")
    block_size = kwargs.get("block_size")
    cmp_block_size = kwargs.get("cmp_block_size")
    slc_block_size = kwargs.get("slc_block_size")
    sa_tile_config = kwargs.get("sa_tile_config")

    with pto.dyn_function("SA_MAIN", 
    [top_k_indices, kv_nope_cache, k_rope_cache, kv_act_seqs, block_table, q_nope, q_rope], [attention_out]):
        def inside_main_function():
            selected_attention_compute(
                top_k_indices=top_k_indices,
                kv_nope_cache=kv_nope_cache,
                k_rope_cache=k_rope_cache,
                kv_act_seqs=kv_act_seqs,
                block_table=block_table,
                q_nope=q_nope,
                q_rope=q_rope,
                attention_out=attention_out,
                n_q=n_q,
                n_kv=n_kv,
                softmax_scale=softmax_scale,
                front=front,
                near=near,
                topk=topk,
                block_size=block_size,
                cmp_block_size=cmp_block_size,
                slc_block_size=slc_block_size,
                sa_tile_config=sa_tile_config
            )
        inside_main_function()


def test_kv_slc_attn(params, sa_tile_config):
    b = params.b
    s1 = params.s1
    s2 = params.s2
    n1 = params.n1
    n2 = params.n2
    v_dim = params.kv_lora_rank
    dn = v_dim
    dr = params.rope_dim
    softmax_scale = 1.0 / math.sqrt((dn + dr))
    block_size = params.block_size
    cmp_block_size = params.cmp_block_size
    slc_block_size = params.slc_block_size
    front = params.front
    near = params.near
    topk = params.topk
    smax = params.topk * params.slc_block_size

    kv_cache_act_seq_vec = [s2] * b
    block_num = 0
    for seq_item in kv_cache_act_seq_vec:
        block_num += ceil_div(seq_item, block_size)
    max_seq_all_batch = max(kv_cache_act_seq_vec)
    max_block_num_per_batch = ceil_div(max_seq_all_batch, block_size)

    d_type = pto.DataType.DT_FP16

    # 1. 设置shape
    topk_indices_shape = [b, s1, topk - front - near]
    topk_tensor_shape_shape = [b, s1]
    kv_nope_cache_shape = [int(block_num * block_size), n2 * dn]
    k_rope_cache_shape = [int(block_num * block_size), n2 * dr]
    kv_cache_act_seq_shape = [b]
    block_table_shape = [b, max_block_num_per_batch]
    slc_act_seqs_shape = [b, s1]

    q_nope_shape = [b * s1 * n1, dn]
    q_rope_shape = [b * s1 * n1, dr]
    k_slc_shape = [b * s1 * n2 * smax, dn + dr]
    v_slc_shape = [b * s1 * n2 * smax, dn]

    shape_sel_atten = [b, s1, n1, v_dim]

    # 2. 构造tensor
    topk_indices = pto.tensor(topk_indices_shape, pto.DataType.DT_INT32, "topk_tensor")
    topk_tensor_shape = pto.tensor(topk_tensor_shape_shape, pto.DataType.DT_INT32, "topk_tensor_shape")
    kv_nope_cache = pto.tensor(kv_nope_cache_shape, d_type, "k_nope_cache")
    k_rope_cache = pto.tensor(k_rope_cache_shape, d_type, "v_nope_cache")
    kv_cache_act_seq = pto.tensor(kv_cache_act_seq_shape, pto.DataType.DT_INT32, "kv_cache_act_seq")
    block_table = pto.tensor(block_table_shape, pto.DataType.DT_INT32, "block_table")
    slc_act_seqs = pto.tensor(slc_act_seqs_shape, pto.DataType.DT_INT32, "slc_act_seqs")

    q_nope = pto.tensor(q_nope_shape, d_type, "q_nope")
    q_rope = pto.tensor(q_rope_shape, d_type, "q_rope")

    atten_out = pto.tensor(shape_sel_atten, pto.DataType.DT_FP32, "atten_out")

    selected_attention(
        top_k_indices=topk_indices,
        kv_nope_cache=kv_nope_cache,
        k_rope_cache=k_rope_cache,
        kv_act_seqs=kv_cache_act_seq, 
        block_table=block_table,
        q_nope=q_nope,
        q_rope=q_rope,
        attention_out=atten_out,
        n_q=n1,
        n_kv=n2,
        softmax_scale=softmax_scale,
        front=front,
        near=near,
        topk=topk,
        block_size=block_size,
        cmp_block_size=cmp_block_size,
        slc_block_size=slc_block_size,
        sa_tile_config=sa_tile_config
    )


def main():
    params: NSASimpleParams = NSASimpleParams()

    input_params = [16, 1, 8192, 128, 1, 0, 0]

    params.b = input_params[0]  # 16
    params.s1 = input_params[1]
    params.s2 = input_params[2]
    params.n1 = input_params[3]
    params.n2 = input_params[4]

    sa_tile_config = SATileShapeConfig(
        kv_slc_v0_tile_shape=[64, 256],  # slc_block_size=64
        g_tile=128,
        s_kv_tile=1024,
        c1_tile_shape=[128, 128, 64, 64, 256, 256],  # (n1, dn+dr) @ (s2_tile, dn+dr) -> (n1, s2_tile)
        v1_tile_shape=[16, 256],  # (n1, s2_tile)
        c2_tile_shape=[128, 128, 128, 128, 128, 128],  # (n1, s2_tile) @ (s2_tile, dn) -> (n1, d)
        v2_tile_shape=[64, 128],  # (n1, d)
    )
    
    test_kv_slc_attn(params, sa_tile_config)  # 假设使用np.float16类型
    logging.info("finished")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    main()