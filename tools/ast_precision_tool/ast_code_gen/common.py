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
import sys
import struct
from pprint import pprint


_ast_gen_conf = {
    'version_gen_torch': '1.0.5',
    'seed': 87654321,
    'device_str': 'cpu',
    'data_in_gen_mode': 'randn',
    'check_mode': 0,
    'golden_filapath': [],
    'code_file_path': '',
    'dump_data_file_path': '',
    'golden_data_file_path': '',
}


_gen_dtype_list = [
    'torch.int4',
    'torch.int8',
    'torch.int16',
    'torch.int32',
    'torch.int64',
    '',
    'torch.float16',
    'torch.float32',
    'torch.bfloat16',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    ''
]


def init_gen_conf(meta_conf):
    for k in _ast_gen_conf.keys():
        if k in meta_conf:
            _ast_gen_conf[k] = meta_conf[k]


def get_gen_conf(key):
    value = _ast_gen_conf.get(key, None)
    return value


def get_dtype(ast_dtype):
    dtype = _gen_dtype_list[ast_dtype]
    error_checker(dtype, f'unsupported dtype: {ast_dtype}')
    return dtype


def gen_node_comment(meta_op):
    return f"# g{meta_op['subgraphid']}f{meta_op['meta_fn']['funcmagic']}n{meta_op['opmagic']} {meta_op['opcode']}"


def gen_err_comment(info, meta_op=None):
    op_info = f'  {gen_node_comment(meta_op)}' if meta_op is not None else ''
    return f'# Error: {info}{op_info}'


def error_checker(cond, info='', meta_op=None):
    comment = gen_err_comment(info, meta_op)
    assert cond, comment


def gen_build_code_lines(code_lines_raw, indent_level=0):
    error_checker(indent_level in range(100), 'indent level too large')
    indent_str = ' ' * indent_level
    code_lines = []
    if not code_lines_raw:
        return code_lines
    if type(code_lines_raw) not in [list, tuple]:
        code_lines_raw = [code_lines_raw]
    for line in code_lines_raw:
        code_lines.append(f'{indent_str}{line}')
    return code_lines


def syms_op_get_info(op, sym_list, sym_list_readable):
    if op in [0, 1, 2]:
        return 1, f'{syms_op_name_dict[op]}'
    elif op in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        return 2, f'{syms_op_name_dict[op]}'
    elif op in [16]:
        arg_num = sym_list.pop(0)
        sym_list_readable.append(int(arg_num))
        return int(arg_num), f'{syms_op_name_dict[op]}'
    else:
        assert False, f'op={op}'


def ssop_call(fn, *args):
    return fn(*args)


syms_sym_name_dict = {
        0: 'S_IMM',
        1: 'S_SYM',
        2: 'S_EXPR',
        }


syms_op_name_dict = {
        0: 'ssop_pos',
        1: 'ssop_neg',
        2: 'ssop_not',
        3: 'ssop_add',
        4: 'ssop_sub',
        5: 'ssop_mul',
        6: 'ssop_div',
        7: 'ssop_mod',
        8: 'ssop_eq',
        9: 'ssop_ne',
        10: 'ssop_lt',
        11: 'ssop_le',
        12: 'ssop_gt',
        13: 'ssop_ge',
        14: 'ssop_min',
        15: 'ssop_max',
        16: ssop_call.__name__,
        }


def ssop_pos(d):
    return d


def ssop_neg(d):
    return -1 * d


def ssop_not(d):
    return not d


def ssop_add(a, b):
    return a + b


def ssop_sub(a, b):
    return a - b


def ssop_mul(a, b):
    return a * b


def ssop_div(a, b):
    return a // b


def ssop_mod(a, b):
    return a % b


def ssop_eq(a, b):
    return a == b


def ssop_ne(a, b):
    return a != b


def ssop_lt(a, b):
    return a < b


def ssop_le(a, b):
    return a <= b


def ssop_gt(a, b):
    return a > b


def ssop_ge(a, b):
    return a >= b


def ssop_min(a, b):
    return min(a, b)


def ssop_max(a, b):
    return max(a, b)


# note: recursive call been used, modify with care
def syms_sym_get_expr(sym_list, sym_list_readable):
    sym = int(sym_list.pop(0))
    sym_list_readable.append(syms_sym_name_dict[sym])
    arg_str = sym_list.pop(0)
    if sym in [0]:
        arg_str = arg_str    # val:int
        sym_list_readable.append(int(arg_str))
    elif sym in [1]:
        arg_str = arg_str    # label:str
        sym_list_readable.append(arg_str)
    elif sym in [2]:
        op = int(arg_str)
        sym_list_readable.append(syms_op_name_dict[op])
        arg_num, syms_op_name = syms_op_get_info(op, sym_list, sym_list_readable)
        arg_list = []
        for _ in range(arg_num):
            arg_list.append(str(syms_sym_get_expr(sym_list, sym_list_readable)))
        arg_str = f'{syms_op_name}(' + ', '.join(arg_list) + ')'
    else:
        assert False, f'sym={sym}'
    return arg_str


def gen_symbol_scalar_expr(line_in_csv):
    sym_list = list(line_in_csv)
    sym_list_readable = []
    expr_out = syms_sym_get_expr(sym_list, sym_list_readable)
    return expr_out


def gen_func_head(func_name, meta_op=None, meta_g=None):
    code_line = ""
    code_line_in = f"{func_name}("
    code_line_out = ""
    for id, v in enumerate(meta_op['data_i_o_magic_dict']['ioperands']):
        if (id == len(meta_op['data_i_o_magic_dict']['ioperands']) - 1):
            code_line_in += f"{v})"
        else:
            code_line_in += f"{v}, "
    for id, v in enumerate(meta_op['data_i_o_magic_dict']['ooperands']):
        if (id == len(meta_op['data_i_o_magic_dict']['ooperands']) - 1):
            code_line_out += f"{v}"
        else:
            code_line_out += f"{v}, "
    code_line = f'{code_line_out} = {code_line_in}'
    return code_line


def gen_def_func_head(func_name, meta_fn=None, meta_g=None):
    code_line_in = f"{func_name}("
    if not meta_fn:
        code_line_in += f")"
        return code_line_in
    functype = meta_fn['functype']
    meta_t = list(meta_g['meta_fn_dict'][functype].values())[0]['meta_t']
    t_format = 0
    for id, v in enumerate(meta_fn['state_']['fn_i_o_magic_dict']['incasts']):
        t_magic = int(v.replace("t", ""))
        if functype in [2, 3, 4]:
            t_format = meta_t[t_magic]['format']
        if (id == len(meta_fn['state_']['fn_i_o_magic_dict']['incasts']) - 1):
            if t_format == 1 and functype in [2, 3, 4]:
                meta_fn['state_']['data_dump_done_set'].add(f'{v}_nz')
                code_line_in += f"{v}_nz)"
            else:
                code_line_in += f"{v})"
        else:
            if t_format == 1 and functype in [2, 3, 4]:
                code_line_in += f"{v}_nz, "
                meta_fn['state_']['data_dump_done_set'].add(f'{v}_nz')
            else:
                code_line_in += f"{v}, "
        meta_fn['state_']['data_dump_done_set'].add(f'{v}')

    if functype != 4:
        return code_line_in

    tile_fn = list(meta_g['meta_fn_dict'][3].values())[0]
    for _, v in enumerate(meta_fn['state_']['fn_i_o_magic_dict']['incasts']):
        t_magic = int(v.replace("t", ""))
        next_op_tuple = tile_fn['consumers_mapping'][t_magic]
        for _, tup in enumerate(next_op_tuple):
            next_op = tile_fn['meta_op_dict'].get(tup)
            if next_op['opcode'] == 'RESHAPE' and meta_t[t_magic]['format'] == 1:
                do0_name = f"{next_op['data_o_list'][0]['t_name']}_nd_to_nz"
                meta_fn['state_']['data_dump_done_set'].add(f'{do0_name}')
    return code_line_in


def gen_def_head(func_name, meta_fn=None, meta_g=None):
    code_line = f"def "
    code_line += gen_def_func_head(func_name, meta_fn, meta_g)
    code_line += f":"
    code_lines = gen_build_code_lines(code_line)
    return code_lines	


def get_functype(meta_app, funchash):
    meta_func = meta_app['meta_g']['meta_fn_dict']
    for functype in meta_func:
        if funchash in meta_func[functype]:
            return functype
    return None