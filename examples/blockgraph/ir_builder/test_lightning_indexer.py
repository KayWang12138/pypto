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
 	 
 	 import pypto
 	 import math
 	 
 	 SHAPE_DIM0 = 0
 	 SHAPE_DIM1 = 1
 	 SHAPE_DIM2 = 2
 	 SHAPE_DIM3 = 3
 	 
 	 NEG_INF = 0xFF8000000
 	 BUFFER_SIZE_BYTE_32B = 32
 	 B32_BLOCK_ALIGN_NUM = 8
 	 SPARSE_COUNT_8K = 8192
 	 BASE_TOPK = 2048
 	 BLOCK_BYTES = 32
 	 VIR_TOPK = 2048
 	 VEC_REPEAT_BYTES = 256
 	 VALUE_AND_INDEX_NUM = 2
 	 
 	 def ceil_div(num, rnd) -> int:
 	     return 0 if rnd == 0 else (num + rnd -1) // rnd
 	 
 	 def align(num, rnd) -> int:
 	     return 0 if rnd == 0 else ((num + rnd -1) // rnd) * rnd
 	 
 	 def get_s2_base_block_num(gs1_idx, act_s1_size, act_s2_size, s1_single_size, s2_single_size) -> int:
 	     """
 	     sparse_mode = 3 时的s2切分块数
 	     """
 	     if act_s2_size == 0:
 	         return 0
 	 
 	     s1_offset = gs1_idx * s1_single_size
 	     vaild_s2_len = act_s2_size - act_s1_size + s1_offset + s1_single_size
 	     vaild_s2_len = min(vaild_s2_len, act_s2_size)
 	     vaild_s2_len = max(vaild_s2_len, 1)
 	     return (vaild_s2_len + s2_single_size -1) // s2_single_size
 	 
 	 def sort_all(dst1, src1, src2, logits_num) -> None:
 	     """
 	     AscendC::Sort32
 	     AscendC::MrgSort
 	     src1:分数
 	     src2:索引
 	     logits_num:排序的元素个数
 	     """
 	     sort32_repeats = logits_num // BLOCK_BYTES
 	     # TODO:AscendC::Sort32
 	     sort_32 = pypto.sort32(src1, src2, sort32_repeats)
 	 
 	     # 归并排序
 	     # TODO:AscendC::MrgSort
 	     pypto.mrgsort(dst1, src_list, params)
 	 
 	 
 	 def merge_sort(dst1, dst_len, src1, srclen) -> None:
 	     """
 	     dst1和src1归并排序,输出最大的dst_len位
 	     """
 	     pypto.mrgsort(dst1, src_list, params)
 	 
 	     dst1 = pypto.view(dst1, [dst_len * VALUE_AND_INDEX_NUM], [0])
 	 
 	 
 	 def sort(dst1, src1, src2, logits_num) -> None:
 	     """
 	     AscendC::Sort32
 	     """
 	     dst1 = pypto.sort32(src1, src2, logits_num // BLOCK_BYTES)
 	 
 	 def mrg_basic_block(dst1, src1, block_num, basic_block_size):
 	     """
 	     AscendC::MrgSort
 	     src1:(score, index)
 	     blcok_num:归并路数
 	     basic_block_size:单队列长度
 	     """
 	     pypto.mrgsort(dst1, src_list, params)
 	 
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
 	     
 	 
 	 @pypto.block
 	 def lightning_indexer_mm1(query, key, act_seq_len_query, act_seq_len_key, bn2_idx, gs1_idx, s2_idx, res_mm1):
 	     if ASCEND_IS_AIC:
 	         # 切分参数
 	         m_basic_block = 256
 	         d_basic_block = 128
 	         s2_basic_block = 256
 	         m_basic_block_l0 = 128
 	         d_basic_block_l0 = 128
 	         s2_basic_block_l0 = 128
 	 
 	         g_size = query.shape[SHAPE_DIM2]
 	         s2_single_size = 2048
 	         is_saprse_count_over_2k = True if sparse_count > 2048 else False
 	         s1_base_size = SPARSE_COUNT_8K // sparse_count if is_saprse_count_over_2k else 8
 	         m_base_size = s1_base_size * g_size
 	         head_dim = 128
 	         k_head_num = 1
 	         qk_dtype = query.dtype
 	 
 	         # s1size、s2size
 	         b_idx = bn2_idx // k_head_num
 	         n2_idx = bn2_idx % k_head_num
 	         act_s1_size = act_seq_len_query[b_idx]
 	         act_s2_size = act_seq_len_key[b_idx]
 	 
 	         # 处理基本块大小
 	         m_base_size_tail = m_base_size if act_s1_size * g_size % m_base_size == 0 else act_s1_size * g_size % m_base_size
 	         s2_base_size_tail = s2_single_size if act_s2_size % s2_single_size ==0 else act_s2_size % s2_single_size
 	         s1g_process_size = m_base_size_tail if act_s1_size * g_size - gs1_idx * m_base_size <= m_base_size else m_base_size
 	         s2_process_size = s2_base_size_tail if s2_idx == ceil_div(act_s2_size, s2_single_size) -1 else s2_single_size
 	 
 	         # 256 × 128
 	         for s2_gm_offset in pypto.loop(0, s2_process_size, s2_basic_block, name = "INDEX_LOOP_S2GM", idx_name = "s2GmOffset"):
 	             s2_l1_real_size = s2_process_size - s2_gm_offset if s2_gm_offset + s2_basic_block > s2_process_size else s2_basic_block
 	 
 	             # TODO:GM load to L1  256 × 128
 	             s2_basic_block_l1 = pypto.gm_load_to_l1(key, [1, s2_basic_block, 1, head_dim], [b_idx, s2_idx, n2_idx, 0])
 	 
 	             # 256 × 128
 	             for s1g_gm_offset in pypto.loop(0, s1g_process_size, m_basic_block, name = "INDEX_LOOP_S1GGM", idx_name = "s1gGmOffset"):
 	                 s1g_l1_real_size = s1g_process_size -s1g_gm_offset if s1g_gm_offset + m_basic_block > s1g_process_size else m_basic_block
 	 
 	                 # GM load to L1 128 × 128
 	                 s1_basic_block_l1 = pypto.gm_load_to_l1(query, [1, m_basic_block, 1, head_dim], [b_idx, gs1_idx, 0, 0])
 	 
 	                 # 128 × 128
 	                 for s2_l1_offset in pypto.loop(0, s2_l1_real_size, s2_basic_block_l0, name = "INDEX_LOOP_S2L1", idx_name = "s2L1Offset"):
 	                     # L0B
 	                     s2_l0_real_size = s2_l1_real_size - s2_l1_offset if s2_l1_offset + s2_basic_block_l0 > s2_l1_real_size else s2_basic_block_l0
 	 
 	                     # 128
 	                     for s1g_l1_offdset in pypto(0 , s1g_l1_real_size , m_basic_block_l0, name = "INDEX_LOOP_S1GL1", idx_name = "s1gL1Offset"):
 	                         # L0A
 	                         s1g_l0_real_szie = s1g_l1_real_size - s1g_l1_offdset if s1g_l1_offdset + m_basic_block_l0 > s1g_l1_real_size else m_basic_block_l0
 	 
 	                         # TODO:L1 load to L0A
 	                         # L0A
 	                         cur_query = pypto.l1_copy_in_l0a(s1_basic_block_l1, [s1g_l0_real_szie, head_dim], [s1g_l1_offdset, 0])
 	 
 	                         # TODO:L1 load to L0B
 	                         # L0B
 	                         cur_key = pypto.l1_copy_in_l0b(s2_basic_block_l1, [s2_l0_real_size, head_dim], [s2_l1_offset, 0])
 	 
 	                         # L0C
 	                         # matmul
 	                         mm1_res_block = pypto.matmul(cur_query, cur_key, qk_dtype, b_trans = true)
 	 
 	                         # TODO:GM store from L0C
 	                         pypto.l0c_copy_out_gm(mm1_res_block, [s1g_l0_real_szie, s2_l0_real_size], res_mm1)
 	 
 	 
 	 
 	 
 	 @pypto.block
 	 def lightning_indexer_vec(query, key, act_seq_len_query, act_seq_len_query, weights, res_mm1, sparse_count, sparse_mode,
 	                           bn2_idx, gs1_idx, s2_idx, topk_score):
 	 
 	     if ASCEND_IS_AIV:
 	         # TODO:获取核idx
 	         block_idx = pypto.get_block_idx()
 	 
 	         # 基本块
 	         g_size = query.shape[SHAPE_DIM2]
 	         s2_single_size = 2048
 	         s2_base_size = 512
 	         is_saprse_count_over_2k = True if sparse_count > 2048 else False
 	         s1_base_size = SPARSE_COUNT_8K // sparse_count if is_saprse_count_over_2k else 8
 	         m_base_size = s1_base_size * g_size
 	         head_dim = 128
 	         k_head_num = 1
 	         qk_dtype = query.dtype
 	 
 	         # s1size、s2size
 	         b_idx = bn2_idx // k_head_num
 	         n2_idx = bn2_idx % k_head_num
 	         act_s1_size = act_seq_len_query[b_idx]
 	         act_s2_size = act_seq_len_key[b_idx]
 	 
 	         # 当前基本块idx便宜
 	         cu_base_s1_idx = gs1_idx * s1_base_size
 	         cu_base_s2_idx = s2_idx * s2_single_size
 	 
 	         # cu_s1_begin_idx:AIV的起始偏移
 	         cu_s1_begin_idx = cu_base_s1_idx
 	         cu_s1_proc_num = act_s1_size % s1_base_size if cu_base_s1_idx + s1_base_size > act_s1_size else s1_base_size
 	         # cu_s1_proc_num_per_aiv:单个AIV的s1计算量
 	         cu_s1_proc_num_per_aiv = ceil_div(cu_s1_proc_num, 2) if block_idx % 2 ==0 else cu_s1_proc_num // 2
 	         cu_s1_begin_idx += (block_idx % 2) * ceil_div(cu_s1_proc_num, 2)
 	 
 	         # g轴切分
 	         group_inner = 16
 	         outer_g = ceil_div(g_size, group_inner)
 	 
 	         # cu_real_ac_seq:当前s1基本块对应的s2Seq
 	         atten_mask_flag = sparse_mode == 3
 	         cu_real_ac_seq = act_s2_size - (act_s1_size - cu_s1_begin_idx) if atten_mask_flag else act_s2_size
 	         
 	         for inner_s1_idx in pypto.loop(0, cu_s1_proc_num, 1, name = "INDEX_LOOP_VEC_S1", idx_name = "innerS1Idx"):
 	             if atten_mask_flag:
 	                 cu_real_ac_seq += 1
 	 
 	             global_topk_score_ub = pypto.tensor([BASE_TOPK * VALUE_AND_INDEX_NUM], dtype = pypto.DT_INT32, "global_topk_ub")
 	             cur_s1_topk_score = pypto.tensor([s2_base_size * VALUE_AND_INDEX_NUM], dtype = pypto.DT_INT32, "cur_s1_topk_score")
 	 
 	             for inner_s2_idx in pypto.loop(0, 4, 1, name = "INDEX_LOOP_VEC_S2", idx_name = "innnerS2Idx"):
 	                 cu_s2_len = cu_real_ac_seq - cu_base_s2_idx - inner_s2_idx * s2_base_size if cu_base_s2_idx + (inner_s2_idx +1) * s2_base_size >= cu_real_ac_seq else s2_base_size
 	                 cu_s1_idx = cu_s1_begin_idx + inner_s1_idx
 	                 
 	                 if cu_real_ac_seq > 0 and cu_s2_len > 0:
 	                     cu_s2_len_vec_align = align(cu_s2_len, s2_base_size)
 	                     g_red_cnt = min(group_inner, g_size)
 	                     reduce_cache = pypto.tensor([g_red_cnt, s2_base_size], dtype = pypto.DT_INT32, "reduce_cache")
 	 
 	                     for outer_g_idx in pypto.loop(0, outer_g, 1, name = "INDEX_LOOP_VEC_G", idx_name = "outterGIdx"):
 	                         proc_g_num = group_inner if outer_g_idx == outer_g - 1 else g_size - outer_g_idx * group_inner
 	 
 	                         # TODO:GM load to UB
 	                         weights_in_ub = pypto.gm_load_to_ub(weights, [1, 1, proc_g_num, 1], [b_idx, cu_s1_idx, outer_g_idx * group_inner, 0])
 	                         mm1_res_in_ub = pypto.gm_load_to_ub(res_mm1, [1, 1, proc_g_num, s2_base_size], [b_idx, s1_idx,outer_g_idx * group_inner, s2_idx])
 	 
 	                         # DoScale
 	                         weights_ub_float = pypto.case(weights_in_ub, pypto.DT_FP32)
 	 
 	                         # TODO:Brob:广播
 	                         tmp_weight = pypto.brcb(weights_ub_float, [proc_g_num, s2_base_size])
 	 
 	                         # mul
 	                         tmp_mul = pypto.mul(mm_out_ub, tmp_weight)
 	 
 	                         # add
 	                         pypto.add(reduce_cache, tmp_mul)
 	                     
 	                     is_s2_end = cu_base_s2_idx + s2_single_size >= cu_real_ac_seq
 	                     #DoReduce
 	                     if g_red_cnt == 1:
 	                         # copy
 	                         reduce_out_inner = pypto.add(reduce_cache)
 	                     else:
 	                         # 在g轴相加
 	                         for idx in range(0, g_red_cnt):
 	                             view_single_g = pypto.view([cu_s2_len_vec_align], dtype = pypto.DT_INT32, "sort_score_ub")
 	                             reduce_out_inner = pypto.add(reduce_out_inner, view_single_g)
 	 
 	                     # sort 结果：score、index
 	                     sort_score_ub = pypto.tensor([cu_s2_len_vec_align], dtype = pypto.DT_INT32, "sort_score_ub")
 	                     sort_indices_ub = pypto.tensor([cu_s2_len_vec_align], dtype = pypto.DT_INT32, "sort_indices_ub")
 	 
 	                     # 填充、赋值
 	                     # TODO:Duplicate：指定位置填充指定值
 	                     pypto.duplicate(sort_score_ub, NEG_INF, [cu_s2_len_vec_align], [0])
 	                     # TODO:add不主动广播，实际reduce_out_inner为cu_s2_len,支持指定长度
 	                     # dst = src + scalar
 	                     sort_score_ub = pypto.add(reduce_out_inner, 0, [cu_s2_len], [0])
 	 
 	                     # 索引填充-1
 	                     if cu_s2_len_vec_align != cu_s2_len:
 	                         pypto.duplicate(sort_indices_ub, -1, [cu_s2_len_vec_align]. [0])
 	                     
 	                     # 索引赋值
 	                     # TODO: global_topk_indices赋值，从cu_base_s2_idx开始的cu_s2_len个索引
 	                     sort_indices_ub = pypto.add(global_topk_indices, cu_base_s2_idx, [cu_s2_len], [0])
 	 
 	                     # 排序
 	                     # false
 	                     if act_s1_size > 4 or is_saprse_count_over_2k:
 	                         # sort_all:先每32个数排序，再归并排序
 	                         # TODO:AscendC::Sort32、AscendC::MrgSort
 	                         sort_ub = pypto.tensor([cu_s2_len_vec_align * VALUE_AND_INDEX_NUM], dtype = pypto.DT_INT32, "sort_ub")
 	                         sort_all(sort_ub, sort_score_ub, sort_indices_ub, cu_s2_len_vec_align)
 	                         
 	                         # merge_sort:全剧topk和当前s1的score结果归并排序，保留前VIP_TOPK位
 	                         merge_sort(global_topk_score_ub, VIR_TOPK, sort_ub, cu_s2_len_vec_align)
 	                     else:
 	                         # sort
 	                         # AscendC::Sort32
 	                         if inner_s1_idx == 0:
 	                             sort(cur_s1_topk_score, sort_score_ub, sort_indices_ub, cu_s2_len_vec_align)
 	                         else:
 	                             sort_ub = pypto.tensor([cu_s2_len_vec_align * VALUE_AND_INDEX_NUM], dtype = pypto.DT_INT32, "sort_ub")
 	                             sort(sort_ub, sort_score_ub, sort_indices_ub, cu_s2_len_vec_align)
 	                             cur_s1_topk_score = pypto.concat([cur_s1_topk_score, sort_ub])
 	                         
 	                         # 缓存四块 512 * 4，或者s2结束，进行精排
 	                         if inner_s2_idx == 3 or is_s2_end:
 	                             # 多路归并排序
 	                             mrg_basic_block(global_topk_score_ub, cur_s1_topk_score, inner_s2_idx + 1, s2_base_size)
 	 
 	             pypto.assemble(global_topk_score_ub, [inner_s1_idx, 0], topk_score)
 	 
 	 
 	 
 	 def lightning_indexer_main(query, key, act_seq_len_query, act_seq_len_query, weights, sparse_count, sparse_mode,
 	                            sparse_indices, sparse_value):
 	 
 	     batch_size = query.shape[SHAPE_DIM0]
 	     s1 = query.shape[SHAPE_DIM1]
 	     s2 = key.shape[SHAPE_DIM1]
 	     q_head_num = g_size = query.shape[SHAPE_DIM2]
 	     k_head_num = 1
 	 
 	     atten_mask_flag = sparse_mode == 3
 	     is_saprse_count_over_2k = False
 	     if sparse_count > 2048:
 	         is_saprse_count_over_2k = True
 	     s2_single_size = 2048
 	     s1_base_size = SPARSE_COUNT_8K // sparse_count if is_saprse_count_over_2k else 8
 	     m_base_size = s1_base_size * g_size
 	     qk_dtype = query.dtype
 	     res_mm1 = pypto.tensor([batch_size, q_head_num,s1, s2])
 	 
 	     for bn2_idx in pypto.loop(0, batch_size * k_head_num, 1, name = "INDEX_LOOP_BATCH", idx_name = "bn2Idx"):
 	         # s1、s2实际大小
 	         b_idx = bn2_idx // k_head_num
 	         n2_idx = bn2_idx % k_head_num
 	         act_s1_size = act_seq_len_query[b_idx]
 	         act_s2_size = act_seq_len_key[b_idx]
 	         if act_s1_size == 0 or act_s2_size ==0:
 	             continue
 	         gs1_split_num = ceil_div(act_s1_size * g_size, m_base_size)
 	 
 	         outpuit_s1 = pypto.tensor([act_s1_size * g_size, act_s2_size * VALUE_AND_INDEX_NUM], dtype = pypto.DT_INT32, "output_s1")
 	 
 	         for gs1_idx in pypto.loop(0, gs1_split_num, 1, name ="INDEX_LOOP_GS1", idx_name = "gs1Idx"):
 	             if atten_mask_flag:
 	                 s2_block_num = get_s2_base_block_num(gs1_idx, act_s1_size, act_s2_size, s1_base_size, s2_single_size)
 	             else:
 	                 s2_block_num = ceil_div(act_s2_size, s2_single_size)
 	 
 	             output_s2 = pypto.tensor([m_base_size, act_s2_size * VALUE_AND_INDEX_NUM])
 	 
 	             for s2_idx in pypto.loop(0, s2_block_num, 1, name = "INDEX_LOOP_S2", idx_name = "s2Idx"):
 	                 # cube
 	                 # 512 × 2048
 	                 res_mm1= pypto.tensor([m_base_size, s2_single_size], dtype = pypto.DT_INT32, "res_mm1")
 	                 lightning_indexer_mm1(query, key, act_seq_len_query, act_seq_len_key, bn2_idx, gs1_idx, s2_idx, res_mm1)
 	 
 	                 # vector
 	                 # 512 × 4096
 	                 topk_score = pypto.tensor([m_base_size, s2_single_size * VALUE_AND_INDEX_NUM], dtype = pypto.DT_INT32, "topk_score")
 	                 lightning_indexer_vec(query, key, act_seq_len_query, act_seq_len_query, weights, res_mm1, sparse_count, sparse_mode,
 	                                       bn2_idx, gs1_idx, s2_idx, topk_score)
 	 
 	                 pypto.assemble(topk_score, [0, s2_idx * s2_single_size * VALUE_AND_INDEX_NUM], output_s2)
 	             
 	             pypto.assemble(output_s2, [gs1 * m_base_size, 0], output_s1)
 	         
 	         # fd_sort:以2048为粒度进行归并排序，输出前sparse_count位
 	         fd_sort(output_s1, sparse_count)
 	 
 	         # 间隔copy
 	         sparse_value_b_idx = pypto.copy(output_s2, [], [])
 	         sparse_indices_b_idx = pypto.copy(output_s2, [], [])
 	 
 	         pypto.assemble(sparse_value_b_idx, [b_idx, 0, 0], sparse_value)
 	         pypto.assemble(sparse_indices_b_idx, [b_idx, 0, 0], sparse_indices)