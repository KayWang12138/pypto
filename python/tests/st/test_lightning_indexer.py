#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
from pypto.pypto_impl import ir
from pypto.blockgraph.builder_helper import BlockBuilderHelper
from pypto.blockgraph.block_call import BlockCallHelper
import pypto
import math

def fd_sort(dst, src, topk_num) -> None:
    """
    归并排序，单队列长度2048
    """
    
    shepe = src.shape
    for s1_idx in range(0, shape[SHAPE_DIM0]):
        view = pypto.view(Src, [1, shape[SHAPE_DIM1]], [s1_idx, 0])
        pypto.mrgsort(view, src_list, params)
        dst_topk = pypto.view(tmp, [topk_num * 2])
        pypto.assemble(dst_topk, [s1_idx, 0], dst)

def _get_topk_input_shape():
    B = ir.Scalar(ir.DataType.uint32, None, "B")
    N1 = ir.Scalar(ir.DataType.uint32, None, "N1")
    N2 = ir.Scalar(ir.DataType.uint32, 1, "N1")
    S1 = ir.Scalar(ir.DataType.int64, None, "S1")
    S2 = ir.Scalar(ir.DataType.uint32, None, "S2")
    D = ir.Scalar(ir.DataType.uint32, None, "D")
    return B, N1, N2, S1, S2, D
 

def lightning_indexer_mm1(args):
    """
    query, key, act_seq_len_query, act_seq_len_key, sparse_count, bn2_idx, gs1_idx, s2_idx, res_mm1
    """
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)
    sig = ir.FunctionSignature(args)
    func = block.create_function("li_mm1", ir.FunctionKind.Block, sig)
    with block.function_scope(func):
        constant0 = block.const(0, "constant0")
        constant8 = block.const(8, "constant8")
        constant2k = block.const(2048, "constant2k")
        constant8k = block.const(8192, "constant8k")
        # 核内切分搬运参数
        m_basic_block = block.const(256, "m_basic_block")
        d_basic_block = block.const(128, "d_basic_block")
        s2_basic_block = block.const(256, "s2_basic_block")
        m_basic_block_l0 = block.const(128, "m_basic_block_l0")
        d_basic_block_l0 = block.const(128, "d_basic_block_l0")
        s2_basic_block_l0 = block.const(128, "s2_basic_block_l0")

        # 基本块参数
        s2_single_size = block.const(2048, "s2_single_size")
        head_dim = block.const(128, "head_dim")
        k_head_num = block.const(1, "k_head_num")
        q_head_num = block.scalar(ir.DataType.uint32, "q_head_num")
        g_size = block.scalar(ir.DataType.uint32, "g_size")
        block.call_1(args[0].shape[2], q_head_num, "")
        block.call_1(args[0].shape[2], g_size, "")

        s1_single_size = ir.Scalar(ir.DataType.uint32, "s1SingleSize")

        cond1 = block.scalar(ir.DataType.int32, "cond1")
        block.sub(args[4], constant2k, cond1)
        ifs1 = block.if_node(cond1)
        with block.if_then_scope(ifs1):
            is_sparse_count_over_2k = block.scalar(ir.DataType.int32, "isSparseCountOver2k")
            block.call_1(constant1, is_sparse_count_over_2k, "")
            block.divs(constant8k, args[4], s1_single_size)
        with block.if_else_scope(ifs1):
            is_sparse_count_over_2k = block.scalar(ir.DataType.int32, "isSparseCountOver2k")
            block.call_1(constant0, is_sparse_count_over_2k, "")
            block.call_1(constant8, s1_single_size, "")
        block.exit_if(ifs1)

        m_single_size = block.scalar(ir.DataType.uint32, "m_single_size")
        block.mul(s1_single_size, g_size, m_single_size)

        # s1size、s2size
        b_idx = block.scalar(ir.DataType.uint32, "b_idx")
        block.divs(bn2_idx, k_head_num, b_idx)
        n2_idx =  block.scalar(ir.DataType.uint32, "bn2_idx")
        # TODO:求余，待适配
        block.mod(bn2_idx, k_head_num, n2_idx)
        act_s1_size = block.scalar(ir.DataType.uint32, "act_s1_size")
        act_s2_size = block.scalar(ir.DataType.uint32, "act_s2_size")
        block.call_1(args[2][b_idx], act_s1_size, "")
        block.call_1(args[3][b_idx], act_s2_size, "")

        # 处理基本块大小
        m_single_size_tail = block.scalar(ir.DataType.uint32, "m_single_size_tail")
        s2_single_size_tail = block.scalar(ir.DataType.uint32, "s2_base_size_tail")
        s1g_process_size = block.scalar(ir.DataType.uint32, "s1g_process_size")
        s2_process_size = block.scalar(ir.DataType.uint32, "s2_process_size")
        
        cond2 = block.scalar(ir.DataType.int32, "cond2")
        m_size = block.scalar(ir.DataType.uint32, "m_size")
        block.muls(act_s1_size, g_size, m_size)
        block.mod(m_size, m_single_size, cond2)
        ifs2 = block.if_node(cond2)
        with block.if_then_scope(ifs4):
            block.mod(m_size, m_single_size, m_single_size_tail)
        with block.if_else_scope(ifs4):
            block.call_1(m_single_size, m_base_size_tail, "GET_COA")
        block.exit_if(ifs4)

        cond3 = block.scalar(ir.DataType.int32, "cond3")
        block.mod(act_s2_size, s2_single_size, cond3)
        ifs3 = block.if_node(cond3)
        with block.if_then_scope(ifs4):
            block.mod(act_s2_size, s2_single_size, s2_single_size_tail)
        with block.if_else_scope(ifs4):
            block.call_1(s2_single_size, s2_single_size_tail, "GET_COA")
        block.exit_if(ifs4)
        
        # if act_s2_size * g_size - gs1_idx * m_single_sizes <= m_single_size
        cond4 = block.scalar(ir.DataType.int32, "cond4")
        temp = block.scalar(ir.DataType.int64, "temp")
        block.muls(args[6], m_single_size, temp)
        block.subs(m_size, temp, temp)
        block.subs(temp, m_single_size, cond4)
        ifs4 = block.if_node(cond4)
        with block.if_then_scope(ifs4):
            block.call_1(m_single_size, s1g_process_size, "GET_COA")
        with block.if_else_scope(ifs4):
            block.call_1(m_base_size_tail, s1g_process_size, "GET_COA")
        block.exit_if(ifs4)

        # if s2_idx == ceil_div(act_s2_size, s2_single_size) -1
        # 除向上取整，待适配
        cond5 = block.scalar(ir.DataType.int32, "cond5")
        block.ceil(act_s2_size, s2_single_size, temp)
        block.subs(temp, constant1, temp)
        block.subs(temp, s2_idx, cond5)
        ifs5 = block.if_node(cond5)
        with block.if_then_scope(ifs5):
            block.call_1(s2_single_size, s2_process_size, "GET_COA")
        with block.if_else_scope(ifs5):
            block.call_1(s2_single_size_tail, s2_process_size, "GET_COA")
        block.exit_if(ifs5)

        # 256 × 128
        s2_gm_offset = block.scalar(ir.DataType.int32, "s2_gm_offset")
        fs_s2_gm = block.for_node(s2_gm_offset, constant0, s2_process_size, s2_basic_block, unroll = 4)
        with block.for_scope(fs_s2_gm):
            # s2_l1_real_size = s2_process_size - s2_gm_offset if s2_gm_offset + s2_basic_block > s2_process_size else s2_basic_block
            cond6= block.scalar(ir.DataType.int32, "cond6")
            block.adds(s2_gm_offset, s2_basic_block, cond6)
            block.subs(cond6, s2_process_size, cond6)
            ifs6 = block.if_node(cond6)
            with block.if_then_scope(ifs6):
                s2_l1_real_size = block.scalar(ir.DataType.int32, "s2_l1_real_size")
                block.subs(s2_process_size, s2_gm_offset, s2_l1_real_size)
            with block.if_else_scope(ifs6):
                block.call_1(s2_basic_block, s2_l1_real_size, "GET_COA")
            block.exit_if(ifs6)

            # GM load to L1   s2_basic_block × D (256 × 128)
            tile_shape = [s2_basic_block, head_dim]
            s2_basic_block_l1 = block.tile(tile_shape, ir.DataType.float32, "s2_basic_block_l1")
            s2_basic_block_l1.set_valid_shape(tile_shape) 
            # Q:内存偏移
            s2_basic_block_l1.set_memory_param(0x10000, ir.MemSpaceKind.L1, 0x0)
            # Q:索引怎么定
            block.matmul_load(args[1], [b_idx, args[6], n2_idx, 0], s2_basic_block_l1)

            # 256 × 128
            s1g_gm_offset = block.scalar(ir.DataType.int32, "s1g_gm_offset")
            fs_s1_gm = block.for_node(s1g_gm_offset, constant0, s1g_process_size, m_basic_block, unroll = 4)
            with block.for_scope(fs_s1_gm):
                # s1g_l1_real_size = s1g_process_size - s1g_gm_offset if s1g_gm_offset + m_basic_block > s1g_process_size else m_basic_block
                cond7 = block.scalar(ir.DataType.int32, "cond7")
                block.adds(s1g_gm_offset, m_basic_block, cond7)
                block.subs(cond7, s1g_process_size, cond7)
                ifs7 = block.if_node(cond7)
                with block.if_then_scope(ifs7):
                    s1g_l1_real_size = block.scalar(ir.DataType.int32, "s1g_l1_real_size")
                    block.sub(s1g_process_size, s1g_gm_offset, s1g_l1_real_size)
                with block.if_else_scope(ifs7):
                    block.call_1(m_basic_block, s1g_l1_real_size, "GET_COA")
                block.exit_if(ifs7)
    
                # GM load to L1 m_basic_block × D (256 × 128)
                tile_shape = [m_basic_block, head_dim]
                s1_basic_block_l1 = block.tile(tile_shape, ir.DataType.float32, "s1_basic_block_l1")
                s1_basic_block_l1.set_valid_shape(tile_shape) 
                s1_basic_block_l1.set_memory_param(0x10000, ir.MemSpaceKind.UB, 0x0)
                # Q:索引怎么定
                block.matmul_load(args[0], [b_idx, args[5], 0, 0], s1_basic_block_l1)

                # 128 × 128
                s2_l1_offset = block.scalar(ir.DataType.int32, "s2_l1_offset")
                fs_s2_l1 = block.for_node(s2_l1_offset, constant0, s2_l1_real_size, s2_basic_block_l0, unroll = 4)
                with block.for_scope(fs_s2_l1):
                    # s2_l0_real_size = s2_l1_real_size - s2_l1_offset if s2_l1_offset + s2_basic_block_l0 > s2_l1_real_size else s2_basic_block_l0
                    cond8 = block.scalar(ir.DataType.int32, "cond8")
                    block.adds(s2_l1_offset, s2_basic_block_l0, cond8)
                    block.subs(cond8, s2_l1_real_size, cond8)
                    ifs8 = block.if_node(cond8)
                    with block.if_then_scope(ifs8):
                        s2_l0_real_size = block.scalar(ir.DataType.int32, "s2_l0_real_size")
                        block.sub(s2_l1_real_size, s2_l1_offset, s2_l0_real_size)
                    with block.if_else_scope(ifs8):
                        block.call_1(s2_basic_block_l0, s2_l0_real_size, "GET_COA")
                    block.exit_if(ifs8)

                    # 128 × 128
                    s1g_l1_offset = block.scalar(ir.DataType.int32, "s1g_l1_offset")
                    fs_s1g_l1 = block.for_node(s1g_l1_offset, constant0, s1g_l1_real_size , m_basic_block_l0, unroll = )
                    with block.for_scope(fs_s1g_l1):
                        # s1g_l0_real_size = s1g_l1_real_size - s1g_l1_offset if s1g_l1_offset + m_basic_block_l0 > s1g_l1_real_size else m_basic_block_l0
                        cond9 = block.scalar(ir.DataType.int32, "cond9")
                        block.adds(s1g_l1_offset, m_basic_block_l0, cond9)
                        block.subs(cond9, s1g_l1_real_size, cond9)
                        ifs9 = block.if_node(cond9)
                        with block.if_then_scope(ifs9):
                            s1g_l0_real_size = block.scalar(ir.DataType.int32, "s1g_l0_real_size")
                            block.sub(s1g_l1_real_size, s1g_l1_offset, s1g_l0_real_size)
                        with block.if_else_scope(ifs9):
                            block.call_1(m_basic_block_l0, s1g_l0_real_size, "GET_COA")
                        block.exit_if(ifs9)

                        # L1 load to L0A
                        tile_shape = [m_basic_block_l0, head_dim]
                        cur_query = block.tile(tile_shape, ir.DataType.float32, "cur_query")
                        cur_query.set_valid_shape(tile_shape)
                        cur_query.set_memory_param(0x10000, ir.MemSpaceKind.L0A, 0x0)
                        block.matmul_extract(s1_basic_block_l1, [s1g_l1_offset, 0], cur_query)

                        # L1 load to L0B
                        tile_shape = [s2_basic_block_l0, head_dim]
                        cur_key = block.tile(tile_shape, ir.DataType.float32, "cur_query")
                        cur_key.set_valid_shape(tile_shape)
                        cur_key.set_memory_param(0x10000, ir.MemSpaceKind.L0A, 0x0)
                        block.matmul_extract_l0b(s2_basic_block_l1, [s2_l1_offset, 0], cur_key)

                        # L0C
                        # matmul
                        tile_shape_mm1 = [m_basic_block_l0, s2_basic_block_l0]
                        mm1_res_block = block.tile(tile_shape_mm1, cir.DataType.float, "mm1_res_block")
                        block.matmul_mmad(cur_query, cur_key, mm1_res_block)

                        # GM store from L0C
                        # Q:索引位置
                        block.matmul_store(mm1_res_block, {}, args[8])

        block.create_return([constant0])
    return func



def lightning_indexer_vec(query, key, act_seq_len_query, act_seq_len_key, weights, res_mm1, sparse_count, sparse_mode,
                          bn2_idx, gs1_idx, s2_idx, aiv_idx, topk_score):
    """
    query, key, act_seq_len_query, act_seq_len_key, weights, res_mm1, sparse_count, sparse_mode, 
    bn2_idx, gs1_idx, s2_idx, aiv_idx, topk_score
    """
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)
    sig = ir.FunctionSignature(args)
    func = block.create_function("li_vec", ir.FunctionKind.Block, sig)
    with block.function_scope(func):
        constant0 = block.const(0, "constant0")
        constant2 = block.const(2, "constant2")
        constant2 = block.const(4, "constant4")
        constant8 = block.const(8, "constant8")
        constant1536 = block.scalar(1536, "constant1536")
        constant8k = block.const(8192, "constant8k")
        constantN1 = block.const(-1, "constantN1")

        # 基本块
        s2_single_size = block.const(2048, "s2_single_size")
        s2_base_size = block.const(512, "s2_base_size")
        head_dim = block.const(128, "head_dim")
        k_head_num = block.const(1, "k_head_num")
        q_head_num = block.scalar(ir.DataType.uint32, "q_head_num")
        g_size = block.scalar(ir.DataType.uint32, "g_size")
        block.call_1(args[0].shape[2], q_head_num, "")
        block.call_1(args[0].shape[2], g_size, "")
        
        s1_single_size = ir.Scalar(ir.DataType.uint32, "s1SingleSize")
        cond1 = block.scalar(ir.DataType.int32, "cond1")
        block.sub(sparse_count, constant2k, cond)
        ifs1 = block.if_node(cond1)
        with block.if_then_scope(ifs1):
            is_sparse_count_over_2k = block.scalar(ir.DataType.int32, "isSparseCountOver2k")
            block.call_1(constant1, is_sparse_count_over_2k, "")
            block.divs(constant8k, sparse_count, s1_single_size)
        with block.if_else_scope(ifs1):
            is_sparse_count_over_2k = block.scalar(ir.DataType.int32, "isSparseCountOver2k")
            block.call_1(constant0, is_sparse_count_over_2k, "")
            block.call_1(constant8, s1_single_size, "")
        block.exit_if

        m_single_size = block.scalar(ir.DataType.uint32, "m_single_size")
        block.mul(s1_single_size, g_size, m_single_size)

        # s1size、s2size
        b_idx = block.scalar(ir.DataType.uint32, "b_idx")
        block.div(bn2_idx, k_head_num, b_idx)
        n2_idx =  block.scalar(ir.DataType.uint32, "bn2_idx")
        # TODO:求余
        block.mod(bn2_idx, k_head_num, n2_idx)
        act_s1_size = block.scalar(ir.DataType.uint32, "act_s1_size")
        act_s2_size = block.scalar(ir.DataType.uint32, "act_s2_size")


        # 当前基本块起始偏移
        cu_s1_begin_idx = block.scalar(ir.DataType.int32, "cu_s1_begin_idx")
        cu_s2_begin_idx = block.scalar(ir.DataType.int32, "cu_begin_s2_idx")
        block.muls(args[9], s1_single_size, cu_s1_begin_idx)
        block.muls(args[10], s2_single_size, cu_begin_s2_idx)

        # cu_s1_proc_num = act_s1_size % s1_single_size if cu_s1_begin_idx + s1_single_size > act_s1_size else s1_single_size
        # cu_s1_proc_num: s1实际长度
        cu_s1_proc_num = block.scalar(ir.DataType.int32, "cu_s1_proc_num")
        cond2 = block.scalar(ir.DataType.int32, "cond2")
        block.adds(cu_s1_begin_idx, s1_single_size, cond2)
        block.subs(cond2, act_s1_size. cond2)
        ifs2 = block.if_node(cond2)
        with block.if_then_scope(ifs2):
            block.mod(act_s1_size, s1_single_size, cu_s1_proc_num)
        with block.if_else_scope(ifs2):
            block.call_1(s1_single_size, cu_s1_proc_num, "GET_COA")
        block.exit_if(ifs2)
        
        ifs3 = block.if_node(args[11])
        with block.if_then_scope(ifs3):
            block.ceil(cu_s1_proc_num, constant2, cu_s1_proc_num)
        with block.if_else_scope(ifs3):
            block.divs(cu_s1_proc_num, constant2)
            aiv1_s1_proc_num = block.scalar(ir.DataType.int32, "aiv1_s1_proc_num")
            block.ceil(cu_s1_proc_num, constant2, aiv1_s1_proc_num)
            block.adds(cu_s1_begin_idx, aiv1_s1_proc_num, cu_s1_begin_idx)
        block.exit_if(ifs3)

        # g轴切分
        group_inner = block.const(16, "group_inner")
        outer_g = block.scalar(ir.DataType.int32, "outer_g")
        block.ceil(g_size, group_inner, outer_g)

        # atten_mask_flag
        atten_mask_flag = block.scalar(ir.DataType.bool, "atten_mask_flag")
        cond4 = block.scalar(ir.DataType.int32, "cond4")
        block.subs(constant3, args[7], cond4)
        ifs4 = block.if_node(cond4)
        with block.if_then_scope(ifs4):
            block.call_1(constant0, atten_mask_flag, "GET_COA")
        with block.if_else_scope(ifs4):
            block.call_1(constant1, atten_mask_flag, "GET_COA")
        block.exit_if(ifs4)

        # cu_real_ac_seq:当前s1基本块对应的s2Seq
        # cu_real_ac_seq = act_s2_size - (act_s1_size - cu_s1_begin_idx) if atten_mask_flag else act_s2_size
        # if atten_mask_flag: cu_real_ac_seq += 1
        cu_real_ac_seq = block.scalar(ir.DataType.int32, "cu_real_ac_seq")
        ifs5 = block.if_node(atten_mask_flag)
        with block.if_then_scope(ifs5):
            block.adds(act_s2_size, cu_s1_begin_idx, cu_real_ac_seq)
            block.subs(cu_real_ac_seq, act_s1_size, cu_real_ac_seq)
            block.adds(cu_real_ac_seq, constant1, cu_real_ac_seq)
        with block.if_else_scope(ifs5):
            block.call_1(act_s2_size, cu_real_ac_seq, "GET_COA")
        block.exit_if(ifs5)

        inner_s1_idx = block.scalar(ir.DataType.int32, "inner_s1_idx")
        fs_inner_s1 = block.for_node(inner_s1_idx, constant0, cu_s1_proc_num, constant1, unroll = 4)
        with block.for_scope(fs_inner_s1):
            # score, index
            constantTopkDouble = block.const(4096, "constantTopkDouble")
            topk_shape = [constantTopkDouble]
            # 每个s1输出的最后结果[1, 4096]
            global_topk_score_ub = block.tile(topk_shape, ir.DataType.int32, "global_topk_score_ub")
            cur_s1_topk_score = block.tile(topk_shape, ir.DataType.int32, "cur_s1_topk_score")
            
            # 当前基本块s2,2048
            # cur_seq_len = cu_real_ac_seq - cu_begin_s2_idx if cu_begin_s2_idx + s2_single_size >= cu_real_ac_seq else s2_single_size
            cur_seq_len =  block.scalar(ir.DataType.int32, "cur_seq_len")
            cond6 = block.scalar(ir.DataType.int32, "cond6")
            block.adds(cu_s2_begin_idx, s2_single_size, cond6)
            block.subs(cond6, cu_real_ac_seq, cond6)
            ifs6 = block.if_node(cond6)
            with block.if_then_scope(ifs6):
                block.subs(cu_real_ac_seq, cu_begin_s2_idx, cur_seq_len)
            with block.if_else_scope(ifs6):
                block.call_1(s2_single_size, cur_seq_len, "")
            block.exit_if(ifs6)
            # s1索引
            cu_s1_idx = block.scalar(ir.DataType.int32, "cu_s1_idx")
            block.adds(cu_s1_begin_idx, inner_s1_idx, cu_s1_idx)

            # s2切分，512
            inner_s2_idx = block.scalar(ir.DataType.int32, "inner_s2_idx")
            fs_inner_s2 = block.for_node(inner_s2_idx, constant0, cur_seq_len, s2_base_size, unroll = 4)
            with block.for_scope(fs_inner_s2):
                # cu_s2_len = cur_seq_len % s2_base_size if inner_s2_idx + s2_base_size > cur_seq_len eles s2_base_size
                cu_s2_len = block.scalar(ir.DataType.int32, "cu_s2_len")
                cond7 =  block.scalar(ir.DataType.int32, "cond7")
                block.adds(inner_s2_idx, s2_base_size, cond7)
                block.subs(cond7, cur_seq_len, cond7)
                ifs7 = block.if_node(cond7)
                with block.if_then_scope(ifs7):
                    block.mod(cur_seq_len, s2_base_size, cu_s2_len)
                with block.if_else_scope(ifs7):
                    block.call_1(s2_base_size, cu_s2_len, "")
                block.exit_if(ifs7)
                
                # if cu_real_ac_seq > 0 and cu_s2_len > 0
                cond8 = block.scalar(ir.DataType.int32, "cond8")
                block.muls(cu_real_ac_seq, cu_s2_len, cond8)
                ifs8 = block.if_node(cond8)
                with block.if_then_scope(ifs8):
                    cu_s2_len_vec_align = block.scalar(ir.DataType.int32, "cu_s2_len_vec_align")
                    # TODO:align待适配
                    block.align(cu_s2_len, s2_base_size, cu_s2_len_vec_align)

                    g_red_cnt = block.scalar(ir.DataType.int32, "g_red_cnt")
                    block.min(group_inner, g_size, g_red_cnt)

                    reduce_cache = block.tile([g_red_cnt, s2_base_size], "reduce_cache")

                    outer_g_idx = blokc.scalar(ir.DataType.int32, "outer_g_idx")
                    fs_outer_g = block.for_node(outer_g_idx, constant0, outer_g, constant1, unroll = 4)
                    with block.for_scope(fs_outer_g):
                        proc_g_num = block.scalar(ir.DataType.int32, "proc_g_num")
                        cond9 = block.scalar(ir.DataType.int32, "cond9")
                        block.subs(outer_g, outer_g_idx, cond9)
                        block.subs(cond9, constant1, cond9)
                        ifs9 = block.if_node(cond9)
                        with if_then_scope(ifs9):
                            block.call_1(group_inner, proc_g_num, "")
                        with if_else_scope(ifs9):
                            block.mod(g_size, group_inner, proc_g_num)
                        block.exit_if(ifs9)

                        proc_g_idx = block.scalar(ir.DataType.int32, "proc_g_idx")
                        block.muls(outer_g_idx, group_inner, proc_g_idx)
                        s1g_idx = block.scalar(ir.DataType.int32, "s1g_idx")
                        block.muls(inner_s1_idx, g_size, s1g_idx)

                        # GM load to UB
                        weights_in_ub = block.tile([constant1, proc_g_num], ir.DataType.float)
                        weights_in_ub.set_valid_shape([constant1, proc_g_num])
                        weights_in_ub.set_memory_param()

                        mm1_res_in_ub = block.tile([proc_g_num, s2_base_size], ir.DataType.float)
                        mm1_res_in_ub.set_valid_shape([proc_g_num, s2_base_size])
                        mm1_res_in_ub.set_memory_param()

                        # copy输入
                        # Q:偏移确定
                        block.ub_copy_in(args[4], {b_idx, cu_s1_idx, proc_g_idx}, weights_in_ub)
                        block.ub_copy_in(args[5], {}, mm1_res_in_ub)

                        # DoScale
                        # TODO:Brob:广播, trans转置
                        weight_brob = block.tile([s2_base_size, proc_g_num], in.DataType.float)
                        weight_trans = block.tile([proc_g_num, s2_base_size], in.DataType.float)
                        block.brcb(weights_in_ub, weight_brob)
                        block.trans(weight_brob, weight_trans)

                        # mul
                        ifs10 = block.if_node(outer_g_idx)
                        with block.if_then_scope(ifs10):
                            mul_res = block.tile([proc_g_num, s2_base_size])
                            block.mul(mm1_res_in_ub, weight_trans, mul_res)
                            # add
                            block.add(reduce_cache, mul_res, reduce_cache)
                        with block.if_else_scope(ifs10):
                            block.mul(mm1_res_in_ub, weight_trans, reduce_cache)
                        block.exit_if(ifs10)

                    # [g_red_cnt, s2_base_size] -> [1, s2_base_size]
                    # DoReduce
                    reduce_out_inner = block.tile([s2_base_size])
                    cond11 = block.scalar(ir.DataType.int32, "cond11")
                    block.subs(g_red_cnt, constant1, cond11)
                    ifs12 = block.if_node(cond11)
                    # g_red_cnt == 1
                    with block.if_then_scope(ifs11):
                        block.adds(reduce_cache, constant0, reduce_out_inner)
                    with block.if_else_scope(ifs11):
                        # TODO:Q:在g轴相加,怎么实现类似view功能，或者直接类似索引a[]
                        for idx in range(0, g_red_cnt):
                            view_single_g = pypto.view([cu_s2_len_vec_align], dtype = pypto.DT_INT32, "sort_score_ub")
                            reduce_out_inner = pypto.add(reduce_out_inner, view_single_g)
                    block.exit_if(ifs11)

                    # sort结果:score、index
                    sort_score_ub = block.tile([cu_s2_len_vec_align], ir.DataType.int32)
                    sort_indices_ub =  block.tile([cu_s2_len_vec_align], ir.DataType.int32)

                    # 填充、赋值
                    # TODO:Duplicate:填充指定值, create_vec_dup_op,这里该填充负无穷
                    block.dup(constantmin, sort_score_ub)
                    # dst = src + scalar
                    block.adds(reduce_out_inner, constant0, sort_score_ub)

                    # if cu_s2_len_vec_align != cu_s2_len, 索引填充-1
                    cond12 = block.scalar(ir.DataType.int32, "cond12")
                    block.subs(cu_s2_len_vec_align, cu_s2_len, cond12)
                    ifs12 = block.if_node(cond12)
                    with if_then_scope(cond12):
                        block.dup(constantN1, sort_indices_ub)
                    block.exit_if(ifs12)
                    
                    # 索引赋值
                    # TODO:global_topk_indices赋值，从cu_base_s2_idx开始的cu_s2_len个索引
                    sort_indices_ub = block.adds(global_topk_indices, cu_base_s2_idx, sort_indices_ub)

                    # 排序
                    # if act_s1_size > 4 or is_sparse_count_over_2k
                    cond13 = block.scalar(ir.DataType.int32, "cond13")
                    cond14 = block.scalar(ir.DataType.int32, "cond14")
                    block.subs(act_s1_size, constant4, cond13)
                    ifs13 = block.if_then_scope(cond13):
                    with block.if_then_scope(ifs13):
                        block.call_1(constant1, cond14, "")
                    with block.if_else_scope(ifs13):
                        block.call_1(constant0, cond14, "")
                    block.exit_if(ifs13)
                    block.adds(cond13, is_sparse_count_over_2k, cond14)
                    ifs14 = block.if_node(cond14)
                    with block.if_then_scope(ifs14):
                        sort_ub_len = block.scalar(ir.DataType.int32, "sort_ub_len")
                        # 1024
                        block.muls(cu_s2_len_vec_align, constant2, sort_ub_len)
                        sort_ub_32 = block.tile([sort_ub_len], ir.DataType.int32)
                        # TODO:TBitSort,每32个数排序
                        block.tbitsort(sort_score_ub, cu_s2_begin_idx, sort_ub_32)
                        
                        ifs15 = block.if_node(inner_s2_idx)
                        # 其余循环先精排1024，再和global_topk_score_ub二路归并排序
                        with block.if_then_scope(ifs15):
                            # TODO:TMrgSort,精排, 1024
                            sort_ub_block = block.tile([sort_ub_len], ir.DataType.int32)
                            block.TMrgSort(sort_ub_32, sort_ub_block)
                            block.TTiledMrgSort(sort_ub_block, global_topk_score_ub, global_topk_score_ub)

                        # 第一块1024直接覆盖到global_topk_score_ub，不需要二路归并
                        with block.if_else_scope(ifs15):
                            # TODO:TMrgSort,精排, 1024
                            block.TMrgSort(sort_ub_32, global_topk_score_ub)
                        block.exit_if(ifs15)
                    

                    with block.if_else_scope(ifs14):
                        # sort
                        sort_ub_len = block.scalar(ir.DataType.int32, "sort_ub_len")
                        block.muls(cu_s2_len_vec_align, constant2, sort_ub_len)
                        sort_ub_32 = block.tile([sort_ub_len], ir.DataType.int32)
                        # TODO:TBitSort,每32个数排序
                        block.tbitsort(sort_score_ub, cu_s2_begin_idx, sort_ub_32)
                        # TODO:TMrgSort,精排->1024
                        sort_ub_block = block.tile([sort_ub_len], ir.DataType.int32)
                        block.TMrgSort(sort_score_ub, sort_ub_block)

                        # 四个有序512块，拼接为2048-->四路排序
                        # or 四个有序512块-->四路归并排序有序2048
                        block.concat(sort_ub_block, cur_s1_topk_score)
                        
                        # is_s2_end = cu_begin_s2_idx + inner_s2_idx + s2_base_size >= cu_real_ac_seq
                        is_s2_end = block.scalar(ir.DataType.int32, "is_s2_end")
                        cur_s2_idx = block.scalar(ir.DataType.int32, "cur_s2_idx")
                        block.adds(cu_begin_s2_idx, inner_s2_idx, cur_s2_idx)
                        block.adds(cur_s2_idx, s2_base_size, is_s2_end)
                        block.subs(is_s2_end, cu_real_ac_seq, is_s2_end)
                        block.adds(is_s2_end, constant1, is_s2_end)
                        ifs16 = block.if_node(is_s2_end)
                        with block.if_then_scope(ifs16):
                            block.call_1(constant1, is_s2_end, "")
                        with block.if_else_scope(ifs16):
                            block.call_1(constant0, is_s2_end, "")
                        block.exit_if(ifs16)

                        # if inner_s2_idx == 1536, 第四块512
                        is_fourth_block = block.scalar(ir.DataType.int32, "is_fourth_block")
                        block.subs(constant1536, inner_s2_idx, is_fourth_block)
                        ifs17 = block.if_node(is_fourth_block)
                        with block.if_then_scope(ifs17):
                            block.call_1(constant0, is_fourth_block, "")
                        with block.if_else_scope(ifs17):
                            block.call_1(constant1, is_fourth_block, "")
                        block.exit_if(ifs17)

                        # 缓存四块 or s2结束，进行精排
                        cond18 = block.scalar(ir.DataType.int32, "cond18")
                        block.add(is_s2_end, is_fourth_block, cond18)
                        ifs18 = block.if_node(cond18)
                        with block.if_then_scope(ifs18):
                            # TODO:TTiledMrgSort，归并排序
                            block.TTiledMrgSort(cur_s1_topk_score, global_topk_score_ub)
                        block.exit_if(ifs18)
                    block.exit_if(ifs14)
                block.exit_if(ifs8)

            
            # 每次s1循环，得到一个[4096]，形如（score，index）的2048有序输出global_topk_score_ub
            # copy到gm
            block.ub_copy_out(global_topk_score_ub, {b_idx, cu_s1_idx, 0, cu_begin_s2_idx}, args[12])
        block.create_return([constant0])
    return func

            


def lightning_indexer(query, key, weights, act_seq_len_query, act_seq_len_query, sparse_count, sparse_mode,
                           sparse_indices, sparse_value):
    """
    query, key, weights, act_seq_len_query, act_seq_len_query, sparse_count, sparse_mode, sparse_indices, sparse_value
    """
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)
    sig = ir.FunctionSignature(args)
    func = block.create_function("lightning_indexer", ir.FunctionKind.Block, sig)
    with block.function_scope(func):
        constant0 = block.const(0, "constant0")
        constant1 = block.const(1, "constant1")
        constant8 - block.const(8, "constant8")
        constant2k = block.const(2048, "constant2k")
        batch_size = block.scalar(ir.DataType.int, "batch_size")
        s1 = block.scalar(ir.DataType.int64, "s1")
        block.call_1(args[0].shape[0], s1, "")
        s2 = block.scalar(ir.DataType.int64, "s2")
        block.call_1(args[1].shape[0], s2, "")
        q_head_num = block.scalar(ir.DataType.int32, "q_head_num")
        block.call_1(args[0].shape[2], q_head_num, "")
        k_head_num = block.const(1, "k_head_num")
        g_size = block.scalar(ir.DataType.int32, "g_size")
        block.call_1(q_head_num, g_size, "")
        s2_single_size = block.const(2048, "s2SingleSize")

        cond1 = block.scalar(ir.DataType.int32, "cond1")
        block.sub(args[5], constant2k, cond1)
        ifs1 = block.if_node(cond1)
        with block.if_then_scope(ifs1):
            is_sparse_count_over_2k = block.scalar(ir.DataType.int32, "isSparseCountOver2k")
            block.call_1(constant1, is_sparse_count_over_2k, "")
            block.divs(constant8k, args[5], s1_single_size)
        with block.if_else_scope(ifs1):
            is_sparse_count_over_2k = block.scalar(ir.DataType.int32, "isSparseCountOver2k")
            block.call_1(constant0, is_sparse_count_over_2k, "")
            block.call_1(constant8, s1_single_size, "")
        block.exit_if(ifs1)

        m_single_size = block.scalar(ir.DataType.int32, "m_single_size")
        
        # atten_mask_flag
        atten_mask_flag = block.scalar(ir.DataType.bool, "atten_mask_flag")
        cond2 = block.scalar(ir.DataType.int32, "cond2")
        block.subs(constant3, args[6], cond2)
        ifs4 = block.if_node(cond2)
        with block.if_then_scope(ifs2):
            block.call_1(constant0, atten_mask_flag, "GET_COA")
        with block.if_else_scope(ifs2):
            block.call_1(constant1, atten_mask_flag, "GET_COA")
        block.exit_if(ifs2)
        

        bn2 = block.scalar(ir.DataType.int32, "bn2")
        block.muls(batch_size, k_head_num, bn2)
        bn2_idx = block.scalar(ir.DataType.int32, "bn2_idx")
        fs_bn2_idx = block.for_node(bn2_idx, constant0, bn2, constant1, unroll = 4)
        with block.for_scope(fs_bn2_idx):
            # s1、s2实际大小
            b_idx = block.scalar(ir.DataType.uint32, "b_idx")
            block.divs(bn2_idx, k_head_num, b_idx)
            n2_idx =  block.scalar(ir.DataType.uint32, "bn2_idx")
            block.mod(bn2_idx, k_head_num, n2_idx)
            act_s1_size = block.scalar(ir.DataType.uint32, "act_s1_size")
            act_s2_size = block.scalar(ir.DataType.uint32, "act_s2_size")
            block.call_1(args[2][b_idx], act_s1_size, "")
            block.call_1(args[3][b_idx], act_s2_size, "")

            cond3 = block.scalar(ir.DataType.int32, "cond3")
            block.muls(act_s1_size, act_s2_size, cond3)
            ifs3 = block.if_node(cond3)
            #if act_s1_size == 0 or act_s2_size == 0:
            #    continue
            with block.if_then_scope(ifs3):    
                gs1_split_num = block.scalar(ir.DataType.int32, "gs1_split_num")
                gs1 = block.scalar(ir.DataType.int64, "gs1")
                block.muls(act_s1_size, g_size, gs1)
                block.ceil(gs1, m_single_size, gs1_split_num)
                
                gs1_idx = block.scalar(ir.DataType.int32, "gs1_idx")
                fs_gs1_idx = blcok.for_node(gs1_idx, constant0, gs1_split_num, constant1, unroll = 4)
                with block.for_scope(fs_gs1_idx):
                    s2_block_num = block.scalar(ir.DataType.int32, "s2BlockNum")
                    ifs4 = block.if_node(atten_mask_flag)
                    with block.if_then_scope(ifs4):   
                        s2_block_num = get_s2_base_block_num(gs1_idx, act_s1_size, act_s2_size, s1_base_size, s2_single_size)
                    with block.if_then_scope(ifs4):
                        block.ceil(act_s2_size, s2_single_size, s2_block_num)
                    block.exit_if(ifs4)

                    s2_idx = block.scalar(ir.DataType.int32, "s2_idx")
                    fs_s2_idx = blcok.for_node(s2_idx, constant0, s2_block_num, constant1, unroll = 4)
                    with block.for_scope(fs_s2_idx):
                        # cube
                        # 512 × 2048
                        res_mm1 = ir.Tensor([m_base_size, s2_single_size], ir.DataType.float32, "resMM1", ir.Format.ND)
                        # Q:函数定义、调用方式？
                        lightning_indexer_mm1(query, key, act_seq_len_query, act_seq_len_key, bn2_idx, gs1_idx, s2_idx, res_mm1)

                        # aic:aiv = 1:2
                        aiv_idx = block.scalar(ir.DataType.int32, "aiv_idx")
                        fs_aiv_idx = blcok.for_node(aiv_idx, constant0, constant2, constant1, unroll = 4)
                        with block.for_scope(fs_aiv_idx):
                            # vector
                            # 256 × 4096
                            topk_score = pypto.tensor([m_base_size // 2, s2_single_size * VALUE_AND_INDEX_NUM], dtype = pypto.DT_INT32, "topk_score")
                            lightning_indexer_vec(query, key, act_seq_len_query, act_seq_len_query, weights, res_mm1, sparse_count, sparse_mode,
                                                bn2_idx, gs1_idx, s2_idx, aiv_idx, topk_score)

                        pypto.assemble(topk_score, [0, s2_idx * s2_single_size * VALUE_AND_INDEX_NUM], output_s2)
                    
                    pypto.assemble(output_s2, [gs1 * m_base_size, 0], output_s1)
                
                # fd_sort:以2048为粒度进行归并排序，输出前sparse_count位
                fd_sort(output_s1, sparse_count)

                # 间隔copy
                sparse_value_b_idx = pypto.copy(output_s2, [], [])
                sparse_indices_b_idx = pypto.copy(output_s2, [], [])

                pypto.assemble(sparse_value_b_idx, [b_idx, 0, 0], sparse_value)
                pypto.assemble(sparse_indices_b_idx, [b_idx, 0, 0], sparse_indices)

            block.exit_if(ifs3)


                