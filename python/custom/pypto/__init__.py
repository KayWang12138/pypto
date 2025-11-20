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
from .stub_fun import mul, add, sub, div, maximum, sigmoid, cast, rms_norm, matmul, \
    batch_matmul, topk, row_max_single, row_sum_single, scatter_update, tensor_index, \
    softmax, softmax_new, assemble, arg_sort, scatter_element, gather_element, index_put, \
    reduce, quant, dassemble, get_mask_row, less_than_zero_inv, sum, max, min, \
    inplace_muls, inplace_mul, inplace_adds, inplace_add, inplace_subs, inplace_sub, \
    inplace_divs, inplace_div, lt, le, gt, ge, add_comment, data_type, tensor_to_aggregation_vec, \
    is_not_null_ptr, continue_loop, break_loop, call_module_function, zeros, ones, \
    var_min, var_max, var_ceil_div, view, dview, dview_pad, shape_size, shape_emplace_back, \
    reshape, unsqueeze, transpose, concat, expand, vec_size, vec_emplace_back, \
    update_record_tile_op, set_vec_tile_shapes, set_tile_shape, set_cube_tile_shapes, \
    set_c1_cube_config, set_c2_cube_config, set_matrix_size, exp, sqrt, log, abs, reciprocal, \
    muls, adds, divs
from .utils import AggregationVec, CodeHelper, ConfigMap, CustStruct, DATATYPE, \
    ReduceMode, CastMode, Instruction, Shape, Tuple, Vector, TensorMap, \
    get_obj_dtype, get_vartype_str, Var
from .utils.tensor import Tensor
from .module import AscppModule
from .flowcontrol import Loop, If, Else, Elif
from .funcsim.simulator import Simulator
from . import context

context.sim = Simulator()
