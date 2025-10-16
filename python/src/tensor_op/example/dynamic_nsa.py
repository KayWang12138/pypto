#!/usr/bin/env python3TopK
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
from dataclasses import dataclass, field
from typing import List
import pto

NUM_2 = 2
NUM_3 = 3
NUM_4 = 4
NUM_8 = 8
NUM_13 = 13
NUM_16 = 16
NUM_32 = 32
NUM_64 = 64
NUM_128 = 128
NUM_256 = 256
NUM_512 = 512
NUM_1536 = 1536
NUM_4096 = 4096
NUM_6144 = 6144
NUM_7168 = 7168
NUM_8192 = 8192

KEY_SUPPORT_DYNAMIC_UNALIGNED = "SUPPORT_DYNAMIC_UNALIGNED"


@dataclass
class SimpleParams:
    b: int = 0
    s: int = 0
    s2: int = 0
    d: int = 0
    m: int = 0
    k: int = 0
    n: int = 0
    n2: int = 0
    right: int = 0
    h: int = 0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 0
    q_head_dim: int = 0
    cache_mode: str = ""
    block_size: int = 0
    vec_tile: List[int] = field(default_factory=list)
    cube_m_tile: List[int] = field(default_factory=list)
    cube_k_tile: List[int] = field(default_factory=list)
    cube_n_tile: List[int] = field(default_factory=list)
    tile_b: int = 0


def get_common_params():
    params = SimpleParams()
    params.n2 = 1
    params.s = 1
    params.h = NUM_7168
    params.q_lora_rank = NUM_1536
    params.kv_lora_rank = NUM_512
    params.qk_rope_head_dim = NUM_64
    params.qk_nope_head_dim = NUM_128
    params.q_head_dim = params.qk_rope_head_dim + params.qk_nope_head_dim
    params.cache_mode = "BNSD"
    params.block_size = NUM_128
    return params


def get_low_params():
    params = get_common_params()
    params.b = NUM_4
    params.n = NUM_32
    params.s2 = NUM_256
    return params


def get_high_params():
    params = get_common_params()
    params.b = NUM_32
    params.n = NUM_128
    params.s2 = NUM_4096
    return params


def ceil_div(a, b):
    return (a + b - 1) // b


def gen_topk_indices(tmp_out: pto.tensor, s_slc: int, actual_topk: int, valid_size: pto.SymbolicScalar, is_dyn: bool):
    res = []
    pto.set_vec_tile_shapes(1, s_slc)
    view0 = pto.view(tmp_out, [1, 128], [1, valid_size], [0, 1])
    if not is_dyn:
        view0 = pto.view(tmp_out, [1, valid_size], [0, 1])
    pto.set_vec_tile_shapes(1, s_slc)
    topk_idx = pto.topk(view0, NUM_16, -1, True)[1]
    topk_idx = pto.cast(topk_idx, pto.data_type.DT_FP32)
    topk_idx = pto.add_s(topk_idx, pto.element(pto.data_type.DT_FP32, 1.0))
    res.append(topk_idx)

    topk_idx_tmp = pto.view(topk_idx, [1, NUM_16], [1, actual_topk], [0, 0])
    if not is_dyn:
        topk_idx_tmp = pto.view(topk_idx_tmp, [1, actual_topk], [0, 0])
    out32 = pto.topk(topk_idx_tmp, NUM_16, -1, False)[0]
    res.append(out32)
    return res


def single_topk(tmp_out: pto.tensor, actual_value: int):
    res = []
    pto.set_vec_tile_shapes(1, NUM_128)
    view0 = pto.view(tmp_out, [1, NUM_128], [1, actual_value], [0, 1])
    pto.set_vec_tile_shapes(1, NUM_128)
    topk_idx = pto.topk(view0, NUM_16, -1, True)[1]
    topk_idx = pto.cast(topk_idx, pto.data_type.DT_FP32)
    res.append(topk_idx)
    return res


def gen_slc(**kwargs):
    x = kwargs.get("x")
    trans0_res = kwargs.get("trans0_res")
    reduce0_res = kwargs.get("reduce0_res")
    trans1_res = kwargs.get("trans1_res")
    reduce1_res = kwargs.get("reduce1_res")
    topk_ind = kwargs.get("topk_ind")
    topk_val = kwargs.get("topk_val")
    out = kwargs.get("out")
    actual_len = kwargs.get("actual_len")
    l_prime = kwargs.get("l_prime", NUM_64)
    d = kwargs.get("d", NUM_16)
    front = kwargs.get("front", 1)
    near = kwargs.get("near", NUM_2)
    topk = kwargs.get("topk", NUM_16)

    n2 = x.shape[0]
    if n2 != 1:
        raise AssertionError("n2 should be 1")
    g = x.shape[1]
    s_cmp = x.shape[NUM_2]
    s_slc = (s_cmp + NUM_3) // NUM_4
    loop = s_slc
    out_loop = l_prime // d
    actual_topk = topk - (front + near)
    actual_valid_len = actual_len - (front + near)

    tile_s2 = s_cmp
    s_loop = pto.SymbolicScalar(ceil_div(s_cmp, tile_s2))
    tmp_out = pto.tensor([1, g], pto.data_type.DT_FP32, "tmpout")
    tmp_out1 = pto.tensor([1, NUM_16], pto.data_type.DT_FP32, "tmpout1")
    tmp_trans2 = pto.tensor([1, s_cmp, NUM_128], pto.data_type.DT_FP32, "trans1")

    with pto.dyn_function("main", [x], [trans0_res, reduce0_res, trans1_res, reduce1_res, topk_ind, topk_val, out]):
        for s_idx in pto.loop(0, s_loop, 1, name="LOOP_L0_sIdx", idx_name="s_idx", submit_before_loop=True):
            def inside_s_idx_loop(s_idx):
                s_ofs = s_idx * tile_s2
                pto.set_vec_tile_shapes(1, NUM_4, s_cmp)
                viewer = pto.view(x, [n2, g, s_cmp], [0, 0, s_ofs])
                input32 = pto.cast(viewer, pto.data_type.DT_FP32)
                tmp_trans = pto.transpose(input32, [1, NUM_2])
                pto.assemble(tmp_trans, [0, 0, 0], tmp_trans2)
                pto.set_vec_tile_shapes(1, NUM_16, g)
                trans0_res[:] = pto.cast(tmp_trans2, pto.data_type.DT_FP16)
                abc = pto.tensor([n2, loop, g], pto.data_type.DT_FP16, "reduce0")
                for i in range(loop):
                    max_len0 = min(out_loop, s_cmp - i * out_loop)
                    view0 = pto.view(tmp_trans, [1, max_len0, g], [0, i * out_loop, 0])
                    max_len1 = min(out_loop, s_cmp - i * out_loop - 1)
                    pto.set_vec_tile_shapes(1, NUM_8, g)
                    reduce0 = pto.row_sum_single(view0, 1)
                    if max_len1 > 0:
                        view1 = pto.view(tmp_trans, [1, max_len1, g], [0, i * out_loop + 1, 0])
                        reduce1 = pto.row_sum_single(view1, 1)
                        reduce_sum = pto.add(reduce0, reduce1)
                        sum_tmp = pto.cast(reduce_sum, pto.data_type.DT_FP16)
                        pto.assemble(sum_tmp, [0, i, 0], abc)
                    else:
                        reduce_tmp = pto.cast(reduce0, pto.data_type.DT_FP16)
                        pto.assemble(reduce_tmp, [0, i, 0], abc)
                trans1 = pto.transpose(pto.cast(abc, pto.data_type.DT_FP32), [1, 2])
                reduce0_res[:] = abc
                trans1_res[:] = pto.cast(trans1, pto.data_type.DT_FP16)
                pto.set_vec_tile_shapes(1, g, NUM_8)
                reduce2 = pto.row_sum_single(trans1, 1)
                tmp_out[:] = pto.reshape(reduce2, [1, NUM_128])
                reduce1_res[:] = pto.cast(reduce2, pto.data_type.DT_FP16)
            inside_s_idx_loop(s_idx)
        for _ in pto.loop(0, 1, 1, name="LOOP_topk1", idx_name="s_idx", submit_before_loop=True):
            pto.set_codegen_config(KEY_SUPPORT_DYNAMIC_UNALIGNED, True)
            res = gen_topk_indices(tmp_out, s_slc, actual_topk, actual_valid_len, True)
            out[:] = res[1]
            topk_ind[:] = res[0]


def gen_slc_v2(**kwargs):
    x = kwargs.get("x")
    out = kwargs.get("out")
    valid_size = kwargs.get("valid_size")
    l_prime = kwargs.get("l_prime", NUM_64)
    d = kwargs.get("d", NUM_16)
    front = kwargs.get("front", 1)
    near = kwargs.get("near", NUM_2)
    topk = kwargs.get("topk", NUM_16)
    
    n = x.shape[0]
    s_cmp = x.shape[1]
    s_slc = (s_cmp + NUM_3) // NUM_4
    loop = s_slc
    out_loop = l_prime // d
    actual_topk = topk - (front + near)
    actual_valid_len = valid_size - (front + near)
    tmp_out = pto.tensor([1, s_slc], pto.data_type.DT_FP32, "tmpout")

    with pto.dyn_function("main", [x], [out]):
        for s_idx in pto.loop(0, 1, 1, name="LOOP_L0_sIdx", idx_name="s_idx", submit_before_loop=True):
            def inside_s_idx_loop(s_idx):
                pto.set_vec_tile_shapes(NUM_4, s_cmp)
                viewer = pto.view(x, [n, s_cmp], [0, 0])
                input32 = pto.cast(viewer, pto.data_type.DT_FP32)
                tmp_trans = pto.transpose(input32, [0, 1])
                pto.set_vec_tile_shapes(NUM_16, n)
                abc = pto.tensor([loop, n], pto.data_type.DT_FP16, "reduce0")
                for i in range(loop):
                    max_len0 = min(out_loop, s_cmp - i * out_loop)
                    view0 = pto.view(tmp_trans, [max_len0, n], [i * out_loop, 0])
                    max_len1 = min(out_loop, s_cmp - i * out_loop - 1)
                    pto.set_vec_tile_shapes(NUM_8, n)
                    reduce0 = pto.row_sum_single(view0, 0)
                    if max_len1 > 0:
                        view1 = pto.view(tmp_trans, [max_len1, n], [i * out_loop + 1, 0])
                        reduce1 = pto.row_sum_single(view1, 0)
                        reduce_sum = pto.add(reduce0, reduce1)
                        sum_tmp = pto.cast(reduce_sum, pto.data_type.DT_FP16)
                        pto.assemble(sum_tmp, [i, 0], abc)
                    else:
                        reduce_tmp = pto.cast(reduce0, pto.data_type.DT_FP16)
                        pto.assemble(reduce_tmp, [i, 0], abc)
                trans1 = pto.transpose(pto.cast(abc, pto.data_type.DT_FP32), [0, 1])
                pto.set_vec_tile_shapes(n, NUM_8)
                reduce2 = pto.row_sum_single(trans1, 0)
                tmp_out[:] = pto.reshape(reduce2, [1, s_slc])
            inside_s_idx_loop(s_idx)
        for _ in pto.loop(0, 1, 1, name="LOOP_topk1", idx_name="s_idx", submit_before_loop=True):
            pto.set_codegen_config(KEY_SUPPORT_DYNAMIC_UNALIGNED, True)
            res = gen_topk_indices(tmp_out, s_slc, actual_topk, actual_valid_len, True)
            out[:] = res[1]


def gen_topk_indices_fun(**kwargs):
    x = kwargs.get("x")
    trans0_res = kwargs.get("trans0_res")
    reduce0_res = kwargs.get("reduce0_res")
    trans1_res = kwargs.get("trans1_res")
    reduce1_res = kwargs.get("reduce1_res")
    topk_ind = kwargs.get("topk_ind")
    topk_val = kwargs.get("topk_val")
    out = kwargs.get("out")
    actual_len = kwargs.get("actual_len")
    front = kwargs.get("front", 1)
    near = kwargs.get("near", NUM_2)

    s_slc = x.shape[1]
    actual_valid_len = actual_len - (front + near)
    tmp_out = pto.tensor([1, s_slc], pto.data_type.DT_FP32, "tmpout")
    tmp_out1 = pto.tensor([1, NUM_16], pto.data_type.DT_FP32, "tmpout1")

    with pto.dyn_function("main", [x], [trans0_res, reduce0_res, trans1_res, reduce1_res, topk_ind, topk_val, out]):
        for _ in pto.loop(0, 1, 1, name="LOOP_topk0", idx_name="s_idx", submit_before_loop=True):
            pto.set_vec_tile_shapes(1, s_slc)
            tmp_out[:] = pto.cast(x, pto.data_type.DT_FP32)
        for _ in pto.loop(0, 1, 1, name="LOOP_topk1", idx_name="s_idx", submit_before_loop=True):
            res = single_topk(tmp_out, actual_valid_len)
            topk_ind[:] = res[0]


def test_gen_slc(d_type, params, topk_actual_len, is_gen_slc):
    params = get_high_params()
    params.b = 1
    params.s2 = NUM_4096 * NUM_2
    params.n2 = 1
    d_type = pto.data_type.DT_FP16
    topk_actual_len = params.s2
    is_gen_slc = True
    n2 = params.n2
    n = params.n
    g = n // n2
    s2 = params.s2
    d = NUM_16
    w = NUM_32
    l_prime = NUM_64
    s_cmp = (s2 - w) // d + 1
    out_loop = l_prime // d
    s_slc = (s_cmp + out_loop - 1) // out_loop
    tmp_s_scmp = (topk_actual_len - NUM_32) // NUM_16 + 1
    tmp_s_slc = (tmp_s_scmp + NUM_3) // NUM_4

    x_shape = [n2, g, s_cmp]
    if not is_gen_slc:
        x_shape = [1, s_slc]
    x = pto.tensor(x_shape, d_type, "x")
    trans0 = pto.tensor([n2, s_cmp, g], d_type, "trans0")
    reduce0 = pto.tensor([n2, s_slc, g], d_type, "reduce0")
    trans1 = pto.tensor([n2, g, s_slc], d_type, "trans1")
    reduce1 = pto.tensor([n2, 1, s_slc], d_type, "reduce1")
    topk_ind = pto.tensor([1, NUM_16], pto.data_type.DT_FP32, "topkInd")
    topk_val = pto.tensor([1, NUM_16], pto.data_type.DT_FP32, "topkVal")
    res = pto.tensor([1, NUM_16], pto.data_type.DT_FP32, "res")

    if is_gen_slc:
        gen_slc(x=x,
                trans0_res=trans0,
                reduce0_res=reduce0,
                trans1_res=trans1,
                reduce1_res=reduce1,
                topk_ind=topk_ind,
                topk_val=topk_val,
                out=res,
                actual_len=tmp_s_slc)
    else:
        gen_topk_indices_fun(x=x,
                            trans0_res=trans0,
                            reduce0_res=reduce0,
                            trans1_res=trans1,
                            reduce1_res=reduce1,
                            topk_ind=topk_ind,
                            topk_val=topk_val,
                            out=res,
                            actual_len=tmp_s_slc)


def test_gen_slc_v2():
    params = get_high_params()
    params.b = 1
    params.s2 = NUM_8192
    params.n2 = 1
    d_type = pto.data_type.DT_FP16
    topk_actual_len = NUM_6144 + 1
    n = params.n
    s2 = params.s2
    window_stride = NUM_16
    window_size = NUM_32
    s_cmp = (s2 - window_size) // window_stride + 1
    s_cmp_valid = (topk_actual_len - window_size) // window_stride + 1
    valid_size = (s_cmp_valid + NUM_3) // 4

    x = pto.tensor([n, s_cmp], d_type, "x")
    res = pto.tensor([1, NUM_13], pto.data_type.DT_FP32, "res")

    gen_slc_v2(x=x, out=res, valid_size=valid_size)