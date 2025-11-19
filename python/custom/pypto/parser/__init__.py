#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
from collections.abc import Callable
from typing import Union, Optional, Sequence, TYPE_CHECKING
from ..utils import Instruction, Tensor, CustStruct, Var, Shape, TensorMap, AggregationVec, Vector, Tuple, \
    ConfigMap, get_vartype_str, CodeHelper

# get the implementations
from .misc import get_attr, get_attr_tensor, get_attr_vector, get_attr_config, call_func, make_mod_class, \
    new_class, add_comment, call_module_function
from .map import new_map, map_set, map_ret
from .var_basic import var_declare, var_assign
from .var_logic import var_and, var_inv, var_or, var_compare_eq, var_compare_ge, var_compare_gt, \
    var_compare_neq
from .var_arithmetic import var_min, var_max, var_add, var_inplace_add, var_sub, var_inplace_sub, var_mul, \
    var_inplace_mul, var_true_div, var_inplace_true_div, var_floor_div, var_inplace_floor_div, var_mod, \
    var_inplace_mod, var_pow, var_inplace_pow
from .tensor_basic import new_tensor, declare, assign, view, dview, dview_pad, dassemble, reshape, unsqueeze, \
    get_data_type, transpose, expand, concat, less_than_zero_inv, pad
from .tensor_unary import abs, exp, log, sqrt, reciprocal, round, logical_not, sin, cos, rotate_half
from .tensor_unary_scalar import muls, adds, sub, divs
from .tensor_inplace import inplace_add, inplace_div, inplace_mul, inplace_sub, inplace_muls, inplace_divs, \
    inplace_subs, inplace_adds
from .tensor_binary import add, sub, mul, div, maximum
from .tensor_highlevel import rms_norm, cast, matmul, batch_matmul, row_max_single, row_sum_single, \
    scatter_element, scatter_update, gather_element, tensor_index, softmax, softmax_new, sigmoid, arg_sort, \
    index_put, topk, is_not_null_ptr, quant, vector_duplicate, reduce, gather, sum, max
from .std_vec import vec_declare, vec_declare_len, vec_declare_list, make_mod_vec, \
    make_mod_vec_len, make_mod_vec_list, get_vec_size, get_vec_index, set_vec_index, vec_emplace_back
from .std_tuple import tuple_get
from .shape import get_shape, get_shape_dim, get_shape_size, set_shape, new_shape, shape_emplace_back
from .aggrvec import new_aggregation_vec, tensor_to_aggregation_vec, assemble
from .flowcontrol import if_control, elif_control, else_control, close_bracket, start_loop, end_loop, \
    continue_loop, break_loop
from .tiling import update_record_tile_op, set_vec_tile_shapes, set_tile_shape, set_cube_tile_shapes, \
    set_c1_cube_config, \
    set_c2_cube_config, set_matrix_size

if TYPE_CHECKING:
    from ..module import AscppModule


INST_MAPPING: dict[str, Callable[[CodeHelper, Instruction], None]] = {
    'get_attr_tensor': get_attr_tensor,
    'get_attr_config': get_attr_config,
    'call_func': call_func,
    'make_mod_class': make_mod_class,
    'new_class': new_class,
    'add_comment': add_comment,

    # Cust struct
    'get_attr': get_attr,
    'get_attr_vector': get_attr_vector,

    # Map
    'new_map': new_map,
    'map_set': map_set,
    'map_ret': map_ret,

    # tensor operations
    'new_tensor': new_tensor,
    'declare': declare,
    'assign': assign,
    'add': add,
    'sub': sub,
    'mul': mul,
    'div': div,
    'abs': abs,
    'inplace_add': inplace_add,
    'inplace_mul': inplace_mul,
    'inplace_sub': inplace_sub,
    'inplace_div': inplace_div,
    'muls': muls,
    'inplace_muls': inplace_muls,
    'adds': adds,

    'inplace_adds': inplace_adds,
    'inplace_subs': inplace_subs,
    'divs': divs,
    'inplace_divs': inplace_divs,
    'exp': exp,
    'log': log,
    'sqrt': sqrt,
    'reciprocal': reciprocal,
    'view': view,
    'dview': dview,
    'dview_pad': dview_pad,
    'dassemble': dassemble,
    'reshape': reshape,
    'unsqueeze': unsqueeze,
    'get_data_type': get_data_type,
    'transpose': transpose,
    'expand': expand,
    'concat': concat,
    'less_than_zero_inv': less_than_zero_inv,
    'round': round,
    'logical_not': logical_not,
    'sin': sin,
    'cos': cos,
    'pad': pad,
    'rotate_half': rotate_half,

    # var operations
    'var_assign': var_assign,
    'var_declare': var_declare,
    'var_or': var_or,
    'var_and': var_and,
    'var_inv': var_inv,
    'var_compare_eq': var_compare_eq,
    'var_compare_neq': var_compare_neq,
    'var_compare_gt': var_compare_gt,
    'var_compare_ge': var_compare_ge,
    'var_min': var_min,
    'var_max': var_max,
    'var_add': var_add,
    'var_inplace_add': var_inplace_add,
    'var_sub': var_sub,
    'var_inplace_sub': var_inplace_sub,
    'var_mul': var_mul,
    'var_inplace_mul': var_inplace_mul,
    'var_true_div': var_true_div,
    'var_inplace_true_div': var_inplace_true_div,
    'var_floor_div': var_floor_div,
    'var_inplace_floor_div': var_inplace_floor_div,
    'var_mod': var_mod,
    'var_inplace_mod': var_inplace_mod,
    'var_pow': var_pow,
    'var_inplace_pow': var_inplace_pow,

    # high-level tensor ops
    'rms_norm': rms_norm,
    'cast': cast,
    'matmul': matmul,
    'batch_matmul': batch_matmul,
    'row_max_single': row_max_single,
    'row_sum_single': row_sum_single,
    'maximum': maximum,
    'scatter_update': scatter_update,
    'scatter_element': scatter_element,
    'gather_element': gather_element,
    'tensor_index': tensor_index,
    'softmax': softmax,
    'softmax_new': softmax_new,
    'arg_sort': arg_sort,
    'index_put': index_put,
    'sigmoid': sigmoid,
    'topk': topk,
    'is_not_null_ptr': is_not_null_ptr,
    'quant': quant,
    'vector_duplicate': vector_duplicate,
    'reduce': reduce,
    'gather': gather,
    'sum': sum,
    'max': max,

    # shape operations
    'get_shape': get_shape,
    'get_shape_dim': get_shape_dim,
    'get_shape_size': get_shape_size,
    'set_shape': set_shape,
    'new_shape': new_shape,
    'shape_emplace_back': shape_emplace_back,

    # vector operations
    'vec_declare': vec_declare,
    'vec_declare_len': vec_declare_len,
    'vec_declare_list': vec_declare_list,
    'make_mod_vec': make_mod_vec,
    'make_mod_vec_len': make_mod_vec_len,
    'make_mod_vec_list': make_mod_vec_list,
    'get_vec_size': get_vec_size,
    'get_vec_index': get_vec_index,
    'set_vec_index': set_vec_index,
    'vec_emplace_back': vec_emplace_back,

    # tuple
    'tuple_get': tuple_get,

    # flow control operations
    'if': if_control,
    'elif': elif_control,
    'else': else_control,
    'close_bracket': close_bracket,
    'start_loop': start_loop,
    'end_loop': end_loop,
    'continue_loop': continue_loop,
    'break_loop': break_loop,

    # aggregation vector
    'new_aggregation_vec': new_aggregation_vec,
    'tensor_to_aggregation_vec': tensor_to_aggregation_vec,
    'assemble': assemble,

    # AscendProgram
    'update_record_tile_op': update_record_tile_op,
    'set_vec_tile_shapes': set_vec_tile_shapes,
    'set_tile_shape': set_tile_shape,
    'set_cube_tile_shapes': set_cube_tile_shapes,
    'set_c1_cube_config': set_c1_cube_config,
    'set_c2_cube_config': set_c2_cube_config,
    'set_matrix_size': set_matrix_size,

    # General Ascend Module function
    'call_module_function': call_module_function,
}


def get_args_str(args: Sequence[Union[Tensor, CustStruct, ConfigMap, Var, Vector]]) -> str:
    args_str = []
    for i in args:
        if isinstance(i, Tensor):
            args_str.append(f'Tensor &tsr{i.idx}')
        elif isinstance(i, (CustStruct, ConfigMap)):
            args_str.append(f'{i.get_type()} &stru{i.idx}')
        elif isinstance(i, Var):
            args_str.append(f'{i.dtype} v{i.idx}')
        elif isinstance(i, Vector):
            args_str.append(f'{i.dtype_str} &vec{i.idx}')
        else:
            raise NotImplementedError()
    args_str = ', '.join(args_str)
    return args_str


def get_return_type(res: Union[None, Tensor, Sequence[Tensor]]) -> str:
    if res is None:
        return_type = 'void'
    elif isinstance(res, Tensor):
        return_type = 'Tensor'
    elif isinstance(res, (Sequence, Vector)):
        return_type = 'std::vector<Tensor>'
    return return_type


def parse_init(class_name: str, inst_list: Sequence[Instruction],
               args: Sequence[Union[Tensor, CustStruct, ConfigMap, Var, Vector]], \
               variables: list[Union[Tensor, CustStruct, ConfigMap, Var, Vector, 'AscppModule']]):
    helper = CodeHelper()
    helper(f'class {class_name} {{')

    helper(f'private:')
    helper.ir()

    from ..module import AscppModule

    for v in variables:
        if isinstance(v, Var):
            helper(f'{v.dtype} v{v.idx};')
        elif isinstance(v, Tensor):
            helper(f'Tensor tsr{v.idx};')
        elif isinstance(v, (CustStruct, ConfigMap)):
            helper(f'{v.get_type()} stru{v.idx};')
        elif isinstance(v, Vector):
            helper(f'{v.dtype_str} vec{v.idx};')
        elif isinstance(v, AscppModule):
            helper(f'{type(v).__name__} cls{v.idx};')
        else:
            raise NotImplementedError()
    helper.il()

    helper('')
    helper(f'public:')
    helper.ir()

    if len(args) > 0:
        helper(f'{class_name}() {{}}')

    args_str = get_args_str(args)

    helper(f'{class_name}({args_str}) {{')
    helper.ir()

    for i in inst_list:
        if i.inst in INST_MAPPING:
            INST_MAPPING[i.inst](helper, i)
        else:
            raise Exception(f'Unsupported instruction {i.inst}')

    helper.il()
    helper('}')

    helper.il()
    helper('};')

    return helper.res


def get_signature(args: Sequence[Union[Tensor, CustStruct, ConfigMap, Var, Vector]],
                  res: Union[None, Tensor, Sequence[Tensor]]) -> str:
    args_str = get_args_str(args)
    return_type = get_return_type(res)

    return f'{return_type} forward({args_str});'


def parse(func_name: str, inst_list: Sequence[Instruction], \
          args: Sequence[Union[Tensor, CustStruct, ConfigMap, Var, Vector]], \
          res: Union[None, Tensor, tuple[Tensor, ...], list[Tensor]]) -> str:
    args_str = get_args_str(args)
    return_type = get_return_type(res)

    helper = CodeHelper()
    helper(f'{return_type} {func_name}::forward({args_str}){{')
    helper.ir()
    for i in inst_list:
        if i.inst in INST_MAPPING:
            INST_MAPPING[i.inst](helper, i)
        else:
            raise Exception(f'Unsupported instruction {i.inst}')
    if isinstance(res, Tensor):
        helper(f'return tsr{res.idx};')
    elif isinstance(res, Sequence):
        tmp = []
        for i in res:
            tmp.append(f'tsr{i.idx}')
        tmp = ', '.join(tmp)
        helper(f'return {{{tmp}}};')
    elif isinstance(res, Vector):
        helper(f'return vec{res.idx};')

    helper.il()
    helper('}')

    return helper.res


def get_custstruct_code(custstruct_obj: CustStruct) -> str:
    helper = CodeHelper()

    class_name = custstruct_obj.__class__.__name__
    helper(f"struct {class_name} {{")
    helper.ir()

    for var_name, type_ in custstruct_obj.get_configs().items():
        if type_ == Tensor:
            type_str = 'Tensor'
        elif type_ == Vector:
            type_str = "std::vector<int>"
        elif isinstance(type_, Vector):  # because Vector[dtype] returns Vector object
            type_str = type_.dtype_str
        else:
            type_str = get_vartype_str(type_)
        helper(f"{type_str} {var_name};")
    helper.il()
    helper("};")

    return helper.res