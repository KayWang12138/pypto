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

from .util import get_var_str, get_curlybrace_list, get_scalar_dtype
from ..utils import CodeHelper, Instruction, Tensor, Var, Shape, Tuple, Vector


def rms_norm(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    if len(inst.src) == 1:
        h(f'auto tsr{inst.dst.idx} = RmsNorm(tsr{inst.src[0].idx});')
    else:
        if not isinstance(inst.src[1], Tensor):
            raise Exception()
        if not isinstance(inst.src[2], (Var, int, float)):
            raise Exception()
        c = get_var_str(inst.src[2])
        h(f'auto tsr{inst.dst.idx} = RmsNorm(tsr{inst.src[0].idx}, tsr{inst.src[1].idx}, {c});')


def cast(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (str, Var)):
        raise Exception()
    if not isinstance(inst.src[2], (type(None), str)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    var_str1 = get_var_str(inst.src[1])
    if inst.src[2] is None:
        h(f'auto tsr{inst.dst.idx} = Cast(tsr{inst.src[0].idx}, {var_str1});')
    else:
        h(f'auto tsr{inst.dst.idx} = Cast(tsr{inst.src[0].idx}, {var_str1}, {inst.src[2]});')


def row_max_single(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    if len(inst.src) == 1:
        h(f'auto tsr{inst.dst.idx} = RowMaxSingle(tsr{inst.src[0].idx});')
    else:
        if not isinstance(inst.src[1], (int, float, Var)):
            raise Exception()
        axis_str = get_var_str(inst.src[1])
        h(f'auto tsr{inst.dst.idx} = RowMaxSingle(tsr{inst.src[0].idx}, {axis_str});')


def row_sum_single(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    if len(inst.src) == 1:
        h(f'auto tsr{inst.dst.idx} = RowSumSingle(tsr{inst.src[0].idx});')
    else:
        if not isinstance(inst.src[1], (int, float, Var)):
            raise Exception()
        axis_str = get_var_str(inst.src[1])
        h(f'auto tsr{inst.dst.idx} = RowSumSingle(tsr{inst.src[0].idx}, {axis_str});')


def sum(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (int, Var)):
        raise Exception()
    if len(inst.src) == 1:
        h(f'auto tsr{inst.dst.idx} = Sum(tsr{inst.src[0].idx});')
    else:
        if not isinstance(inst.src[1], (int, float, Var)):
            raise Exception()
        axis_str = get_var_str(inst.src[1])
        h(f'auto tsr{inst.dst.idx} = Sum(tsr{inst.src[0].idx}, {axis_str});')


def max(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (int, Var)):
        raise Exception()
    if len(inst.src) == 1:
        h(f'auto tsr{inst.dst.idx} = Max(tsr{inst.src[0].idx});')
    else:
        if not isinstance(inst.src[1], (int, float, Var)):
            raise Exception()
        axis_str = get_var_str(inst.src[1])
        h(f'auto tsr{inst.dst.idx} = Max(tsr{inst.src[0].idx}, {axis_str});')


def min(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (int, Var)):
        raise Exception()
    if len(inst.src) == 1:
        h(f'auto tsr{inst.dst.idx} = Min(tsr{inst.src[0].idx});')
    else:
        if not isinstance(inst.src[1], (int, float, Var)):
            raise Exception()
        axis_str = get_var_str(inst.src[1])
        h(f'auto tsr{inst.dst.idx} = Min(tsr{inst.src[0].idx}, {axis_str});')


def check_matmul_inst(inst: Instruction):
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.src[2], Tensor):
        raise Exception()
    if not isinstance(inst.src[3], bool):
        raise Exception()
    if not isinstance(inst.src[4], bool):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()


def matmul(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (str, Var)):
        raise Exception()
    check_matmul_inst(inst)

    if len(inst.src) == 6:
        if not isinstance(inst.src[5], Tensor):
            raise Exception()
        dtype_str = get_var_str(inst.src[0])
        h(f'auto tsr{inst.dst.idx} = Matrix::Matmul({dtype_str}, tsr{inst.src[1].idx}, \
            tsr{inst.src[2].idx}, tsr{inst.src[5].idx});')
    else:
        dtype_str = get_var_str(inst.src[0])
        trans_a = 'true' if inst.src[3] else 'false'
        trans_b = 'true' if inst.src[4] else 'false'

        h(f'auto tsr{inst.dst.idx} = Matrix::Matmul<{trans_a},{trans_b}>({dtype_str}, \
            tsr{inst.src[1].idx}, tsr{inst.src[2].idx});')


def batch_matmul(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, str)):
        raise Exception()
    check_matmul_inst(inst)

    dtype_str = get_var_str(inst.src[0])
    trans_a = 'true' if inst.src[3] else 'false'
    trans_b = 'true' if inst.src[4] else 'false'
    h(f'auto tsr{inst.dst.idx} = Matrix::BatchMatmul<{trans_a},{trans_b}>({dtype_str}, \
        tsr{inst.src[1].idx}, tsr{inst.src[2].idx});')


def scatter_update(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.src[2], Tensor):
        raise Exception()
    if not isinstance(inst.src[3], (int, Var)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    axis_str = get_var_str(inst.src[3])
    h(f'auto tsr{inst.dst.idx} = ScatterUpdate(tsr{inst.src[0].idx}, tsr{inst.src[1].idx}, \
        tsr{inst.src[2].idx}, {axis_str});')


def scatter_element(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.src[2], (int, float, Var)):
        raise Exception()
    if not isinstance(inst.src[3], (int, Var)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    scalar_str = get_scalar_dtype(inst.src[2]), + ', ' + get_var_str(inst.src[2])
    axis_str = get_var_str(inst.src[3])
    h(f'auto tsr{inst.dst.idx} = ScatterElement(tsr{inst.src[0].idx}, tsr{inst.src[1].idx}, \
        Element({scalar_str}), {axis_str});')


def gather(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.src[2], (int, Var)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    axis_str = get_var_str(inst.src[2])
    h(f'auto tsr{inst.dst.idx} = Gather(tsr{inst.src[0].idx}, tsr{inst.src[1].idx}, {axis_str});')


def gather_element(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.src[2], (int, Var)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    axis_str = get_var_str(inst.src[2])
    h(f'auto tsr{inst.dst.idx} = GatherElement(tsr{inst.src[0].idx}, tsr{inst.src[1].idx}, {axis_str});')


def tensor_index(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    h(f'auto tsr{inst.dst.idx} = TensorIndex(tsr{inst.src[0].idx}, tsr{inst.src[1].idx});')


def softmax(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    h(f'auto tsr{inst.dst.idx} = Softmax(tsr{inst.src[0].idx});')


def softmax_new(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    h(f'auto tsr{inst.dst.idx} = SoftmaxNew(tsr{inst.src[0].idx});')


def sigmoid(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    h(f'auto tsr{inst.dst.idx} = Sigmoid(tsr{inst.src[0].idx});')


def arg_sort(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    if len(inst.src) == 1:
        h(f'auto tsr{inst.dst.idx} = ArgSort(tsr{inst.src[0].idx});')
    else:
        if not isinstance(inst.src[1], (int, Var)):
            raise Exception()
        axis_str = get_var_str(inst.src[1])

        if len(inst.src) == 2:
            h(f'auto tsr{inst.dst.idx} = ArgSort(tsr{inst.src[0].idx}, {axis_str});')
        else:
            if not isinstance(inst.src[2], bool):
                raise Exception()
            is_largest_str = 'true' if inst.src[2] else 'false'
            h(f'auto tsr{inst.dst.idx} = ArgSort(tsr{inst.src[0].idx}, {axis_str}, {is_largest_str});')


def index_put(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], list):
        raise Exception()
    if not isinstance(inst.src[2], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()

    tensor_str = get_curlybrace_list(inst.src[1])

    h(f'auto tsr{inst.dst.idx} = IndexPut(tsr{inst.src[0].idx}, {tensor_str}, tsr{inst.src[2].idx});')


def topk(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.src[1], (Var, int)):
        raise Exception()
    if not isinstance(inst.src[2], (Var, int)):
        raise Exception()
    if not isinstance(inst.src[3], bool):
        raise Exception()
    if not isinstance(inst.dst, Tuple):
        raise Exception()
    if not inst.dst.dtypes == ["Tensor", "Tensor"]:
        raise Exception()

    operand = f"tsr{inst.src[0].idx}"
    k = get_var_str(inst.src[1])
    axis = get_var_str(inst.src[2])
    is_largest = str(inst.src[3]).lower()

    h(f'{inst.dst.get_type()} {get_var_str(inst.dst)} = TopK({operand}, {k}, {axis}, {is_largest});')


def is_not_null_ptr(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Var):
        raise Exception()
    h(f'bool v{inst.dst.idx} = tsr{inst.src[0].idx}.storage != nullptr;')


def quant(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Tensor):
        raise Exception()
    if not isinstance(inst.dst, Tuple):
        raise Exception()
    if len(inst.src) == 1:
        h(f'{inst.dst.get_type()} {get_var_str(inst.dst)} = Quant(tsr{inst.src[0].idx});')
    else:
        if not isinstance(inst.src[1], (Var, bool)):
            raise Exception()
        if not isinstance(inst.src[2], (Var, bool)):
            raise Exception()
        if not isinstance(inst.src[3], Tensor):
            raise Exception()
        b = get_var_str(inst.src[1])
        c = get_var_str(inst.src[2])
        h(f'{inst.dst.get_type()} {get_var_str(inst.dst)} = Quant(tsr{inst.src[0].idx}, {b}, {c}, \
            tsr{inst.src[3].idx});')


def vector_duplicate(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], (Var, str)):
        raise Exception()
    if not isinstance(inst.src[1], (Var, Shape, list)):
        raise Exception()
    if not isinstance(inst.src[2], (Var, str)):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    
    h(f'auto {get_var_str(inst.dst)} = VectorDuplicate(Element(\
        {get_scalar_dtype(inst.src[0]) + ", " + inst.src[0]}), \
        {get_var_str(inst.src[2])}, {get_curlybrace_list(inst.src[1])});')


def reduce(h: CodeHelper, inst: Instruction):
    if not isinstance(inst.src[0], Vector):
        raise Exception()
    if not isinstance(inst.src[1], str):
        raise Exception()
    if not isinstance(inst.dst, Tensor):
        raise Exception()
    h(f'auto tsr{inst.dst.idx} = ascend::Reduce(vec{inst.src[0].idx}, {inst.src[1]});')
