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
from pprint import pprint
from ast_code_gen.common import get_gen_conf, get_dtype, gen_node_comment, get_functype
from ast_code_gen.common import gen_build_code_lines, gen_def_func_head, gen_def_head, gen_symbol_scalar_expr
from ast_code_gen.gen_op_code import get_op_func


def gen_code_head(comment_str=None):
    dev_str = get_gen_conf('device_str')
    code_lines_raw = []
    code_lines_raw.append(f"# gen_torch_v1:v{get_gen_conf('version_gen_torch')}")
    if comment_str:
        code_lines_raw.append(f'#')
        code_lines_raw.append(f'# {comment_str}')

    code_lines_raw.append('')
    code_lines_raw.append('import os')
    code_lines_raw.append('import random')
    code_lines_raw.append('import numpy as np')
    code_lines_raw.append('import hashlib')
    code_lines_raw.append('import torch')
    code_lines_raw.append(f"seed = {get_gen_conf('seed')}")
    code_lines_raw.append('random.seed(seed)')
    code_lines_raw.append('os.environ["PYTHONHASHSEED"] = str(seed)')
    code_lines_raw.append('np.random.seed(seed)')
    code_lines_raw.append('torch.manual_seed(seed)')
    if dev_str.startswith('cuda'):
        code_lines_raw.append('os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"')
        code_lines_raw.append('torch.cuda.manual_seed_all(seed)')
        code_lines_raw.append('torch.cuda.manual_seed(seed)')
        code_lines_raw.append('torch.backends.cudnn.enable = False')
        code_lines_raw.append('torch.backends.cudnn.deterministic = True')
        code_lines_raw.append('torch.backends.cudnn.benchmark = False')
    if dev_str.startswith('npu'):
        code_lines_raw.append('import torch_npu')
        code_lines_raw.append('os.environ["HCCL_DETERMINISTIC"] = str(True)')
        code_lines_raw.append('torch.npu.manual_seed_all(seed)')
        code_lines_raw.append('torch.npu.manual_seed(seed)')
    code_lines_raw.append('torch.use_deterministic_algorithms(True)')
    code_lines_raw.append('SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def _buf_to_t(buf, dtype, shape=None):')
    code_lines_raw.append("    t_storage = torch.UntypedStorage.from_buffer(buf, dtype=dtype, byte_order='native')")
    code_lines_raw.append('    t = torch.tensor(t_storage, dtype=dtype)')
    code_lines_raw.append('    if shape:')
    code_lines_raw.append('        t = t.reshape(shape)')
    code_lines_raw.append('    return t')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def file_bin_to_pt(filename_in, dtype, filename_out=None, shape=None):')
    code_lines_raw.append("    with open(filename_in, 'rb') as f:")
    code_lines_raw.append('        bin_buf = f.read()')
    code_lines_raw.append('        t = _buf_to_t(bin_buf, dtype, shape)')
    code_lines_raw.append('        if filename_out:')
    code_lines_raw.append('            torch.save(t, filename_out)')
    code_lines_raw.append('    return t')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def set_value(tensor, coords, value):')
    code_lines_raw.append('    if len(coords) == 1:')
    code_lines_raw.append('        tensor[coords[0]] = value')
    code_lines_raw.append('    else:')
    code_lines_raw.append('        set_value(tensor[coords[0]], coords[1:], value)')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def get_value(tensor, coords):')
    code_lines_raw.append('    if len(coords) == 1:')
    code_lines_raw.append('        return tensor[coords[0]]')
    code_lines_raw.append('    else:')
    code_lines_raw.append('        return get_value(tensor[coords[0]], coords[1:])')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def get_coordinates(shape):')
    code_lines_raw.append('    coordinates = []')
    code_lines_raw.append('    def helper(current_indices, dim):')
    code_lines_raw.append('        if dim == len(shape):')
    code_lines_raw.append('            coordinates.append(tuple(current_indices))')
    code_lines_raw.append('            return')
    code_lines_raw.append('        for i in range(shape[dim]):')
    code_lines_raw.append('            helper(current_indices + [i], dim + 1)')
    code_lines_raw.append('')
    code_lines_raw.append('    helper([], 0)')
    code_lines_raw.append('    return coordinates')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def scatter_update(dst, src, idx, axis):')
    code_lines_raw.append('    if axis < 0:')
    code_lines_raw.append('        axis = axis + len(dst.shape)')
    code_lines_raw.append('    assert (axis == len(dst.shape) - 2)')
    code_lines_raw.append('    assert (len(idx.shape) == 2)')
    code_lines_raw.append('    assert (dst.shape[0] * dst.shape[1] == idx.shape[0])')
    code_lines_raw.append('    assert (dst.shape[1] == 1)')
    code_lines_raw.append('    for b in range(dst.shape[0]):')
    code_lines_raw.append('        for s in range(dst.shape[axis]):')
    code_lines_raw.append('            for index in range(idx.shape[1]):')
    code_lines_raw.append('                idxVal = idx[b][index]')
    code_lines_raw.append('                if s == idxVal:')
    code_lines_raw.append('                    dst[b][0][idxVal][:] = src[b][0][index][:]')
    code_lines_raw.append('    return dst')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def gather_v2(params, indices, indices_ori_shape, axis):')
    code_lines_raw.append('    ori_shape = indices_ori_shape')
    code_lines_raw.append('    if (len(indices_ori_shape) == 0):')
    code_lines_raw.append('        ori_shape = indices.shape')
    code_lines_raw.append('    indices = torch.zeros(ori_shape).to(torch.int)')
    code_lines_raw.append('    if axis < 0:')
    code_lines_raw.append('        axis = axis + len(params.shape)')
    code_lines_raw.append('    dst_shape = list(ori_shape)')
    code_lines_raw.append('    dst_shape.extend(list(params.shape)[axis + 1:])')
    code_lines_raw.append('    dst = torch.zeros(dst_shape)')
    code_lines_raw.append('    coords = get_coordinates(list(ori_shape))')
    code_lines_raw.append('    for i in range(len(coords)):')
    code_lines_raw.append('        set_value(dst, coords[i],params[get_value(indices,coords[i])])')
    code_lines_raw.append('    return dst')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def bitsort(src, axis, isLargest=True):')
    code_lines_raw.append('    assert (len(src.shape) == 2)')
    code_lines_raw.append('    assert (src.shape[1] % 32 == 0)')
    code_lines_raw.append('    dst = torch.randn(src.shape[0], src.shape[1] * 4)')
    code_lines_raw.append('    unit_size = int(src.shape[1] / 32)')
    code_lines_raw.append('    for i in range(src.shape[0]):')
    code_lines_raw.append('        for j in range(unit_size):')
    code_lines_raw.append('            list_to_sort = [[32 * j + k, src[i][32 * j + k]] for k in range(32)]')
    code_lines_raw.append('            sorted_list = sorted(list_to_sort, key=lambda x: x[1], reverse=not isLargest)')
    code_lines_raw.append('            for k in range(len(sorted_list)):')
    code_lines_raw.append('                dst[i][2 * (32 * j + k)] = sorted_list[k][0]')
    code_lines_raw.append('                dst[i][2 * (32 * j + k) + 1] = sorted_list[k][1]')
    code_lines_raw.append('    return dst')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def mrgsort(src, axis, isLargest, k):')
    code_lines_raw.append('    assert (len(src.shape) == 2)')
    code_lines_raw.append('    assert (src.shape[1] % 4 == 0)')
    code_lines_raw.append('    dst = torch.randn(src.shape[0], k * 2)')
    code_lines_raw.append('    item_size = int(src.shape[1] / 4)')
    code_lines_raw.append('    for i in range(src.shape[0]):')
    code_lines_raw.append('        list_to_sort = []')
    code_lines_raw.append('        for j in range(item_size):')
    code_lines_raw.append('            list_to_sort.append([src[i][2 * j], src[i][2 * j + 1]])')
    code_lines_raw.append('        sorted_list = sorted(list_to_sort, key=lambda x: x[1], reverse=not isLargest)')
    code_lines_raw.append('        for k_idx in range(k):')
    code_lines_raw.append('            dst[i][2 * k_idx] = sorted_list[k_idx][0]')
    code_lines_raw.append('            dst[i][2 * k_idx + 1] = sorted_list[k_idx][1]')
    code_lines_raw.append('    return dst')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def extract(src, mod, isLargest=True):')
    code_lines_raw.append('    dst = torch.randn(src.shape[0], int(src.shape[1]/2))')
    code_lines_raw.append('    for i in range(src.shape[0]):')
    code_lines_raw.append('        for j in range(int(src.shape[1] / 2)):')
    code_lines_raw.append('            if mod == 0:')
    code_lines_raw.append('                dst[i][j] = src[i][2 * j + 1]')
    code_lines_raw.append('            else:')
    code_lines_raw.append('                dst[i][j] = src[i][2 * j]')
    code_lines_raw.append('    return dst')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def nz_to_nd_2dim(tensor_nz, block):')
    code_lines_raw.append('    s = tensor_nz.shape[0]')
    code_lines_raw.append('    h = tensor_nz.shape[1]')
    code_lines_raw.append('    tensor_nd_tmp = torch.reshape(tensor_nz, (h // block, s, block)).permute(1, 0, 2)')
    code_lines_raw.append('    tensor_nd = torch.reshape(tensor_nd_tmp, (s, h))')
    code_lines_raw.append('    return tensor_nd')
    code_lines_raw.append('')
    code_lines_raw.append('')
    code_lines_raw.append('def nz_to_nd_3dim_MLA(tensor_nz, block):')
    code_lines_raw.append('    b = tensor_nz.shape[0]')
    code_lines_raw.append('    s = tensor_nz.shape[1]')
    code_lines_raw.append('    h = tensor_nz.shape[2]')
    code_lines_raw.append('    tensor_nd_tmp = torch.reshape(tensor_nz, (h // block, b * s, block)).permute(1, 0, 2)')
    code_lines_raw.append('    tensor_nd = torch.reshape(tensor_nd_tmp, (b, s, h))')
    code_lines_raw.append('    return tensor_nd')
    code_lines_raw.append('')
    code_lines_raw.append('')
    code_lines_raw.append('def nz_to_nd_3dim_PA(tensor_nz, block):')
    code_lines_raw.append('    b = tensor_nz.shape[0]')
    code_lines_raw.append('    s = tensor_nz.shape[1]')
    code_lines_raw.append('    h = tensor_nz.shape[2]')
    code_lines_raw.append('    tensor_nd_tmp = torch.reshape(tensor_nz, (b, h // block, s, block)).permute(0, 2, 1, 3)')
    code_lines_raw.append('    tensor_nd = torch.reshape(tensor_nd_tmp, (b, s, h))')
    code_lines_raw.append('    return tensor_nd')
    code_lines_raw.append('')
    code_lines_raw.append('')
    code_lines_raw.append('def nz_to_nd(tensor_nz, data_type, is_pa=False):')
    code_lines_raw.append('    byte_map = {')
    code_lines_raw.append('        torch.int8: 1,')
    code_lines_raw.append('        torch.int16: 2,')
    code_lines_raw.append('        torch.int32: 4,')
    code_lines_raw.append('        torch.int64: 8,')
    code_lines_raw.append('        torch.float16: 2,')
    code_lines_raw.append('        torch.float32: 4,')
    code_lines_raw.append('        torch.bfloat16: 2,')
    code_lines_raw.append('    }')
    code_lines_raw.append('    byte = byte_map.get(data_type, None)')
    code_lines_raw.append('    if byte is None:')
    code_lines_raw.append('        assert False, f"unknown data_type"')
    code_lines_raw.append('    block = 32 // byte')
    code_lines_raw.append('')
    code_lines_raw.append('    if len(tensor_nz.shape) == 2:')
    code_lines_raw.append('        return nz_to_nd_2dim(tensor_nz, block)')
    code_lines_raw.append('    elif len(tensor_nz.shape) == 3 and is_pa:')
    code_lines_raw.append('        return nz_to_nd_3dim_PA(tensor_nz, block)')
    code_lines_raw.append('    elif len(tensor_nz.shape) == 3 and not is_pa:')
    code_lines_raw.append('        return nz_to_nd_3dim_MLA(tensor_nz, block)')
    code_lines_raw.append('    else:')
    code_lines_raw.append('        assert False, f"no support shape"')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def nd_to_nz_2dim(tensor_nd, block):')
    code_lines_raw.append('    s = tensor_nd.shape[0]')
    code_lines_raw.append('    h = tensor_nd.shape[1]')
    code_lines_raw.append('    tensor_nz_tmp = torch.reshape(tensor_nd, (s, h // block, block)).permute(1, 0, 2)')
    code_lines_raw.append('    tensor_nz = torch.reshape(tensor_nz_tmp, (s, h))')
    code_lines_raw.append('    return tensor_nz')
    code_lines_raw.append('')
    code_lines_raw.append('')
    code_lines_raw.append('def nd_to_nz_3dim_MLA(tensor_nd, block):')
    code_lines_raw.append('    b = tensor_nd.shape[0]')
    code_lines_raw.append('    s = tensor_nd.shape[1]')
    code_lines_raw.append('    h = tensor_nd.shape[2]')
    code_lines_raw.append('    tensor_nz_tmp = torch.reshape(tensor_nd, (b * s, h // block, block)).permute(1, 0, 2)')
    code_lines_raw.append('    tensor_nz = torch.reshape(tensor_nz_tmp, (b, s, h))')
    code_lines_raw.append('    return tensor_nz')
    code_lines_raw.append('')
    code_lines_raw.append('')
    code_lines_raw.append('def nd_to_nz_3dim_PA(tensor_nd, block):')
    code_lines_raw.append('    b = tensor_nd.shape[0]')
    code_lines_raw.append('    s = tensor_nd.shape[1]')
    code_lines_raw.append('    h = tensor_nd.shape[2]')
    code_lines_raw.append('    tensor_nz_tmp = torch.reshape(tensor_nd, (b, s, h // block, block)).permute(0, 2, 1, 3)')
    code_lines_raw.append('    tensor_nz = torch.reshape(tensor_nz_tmp, (b, s, h))')
    code_lines_raw.append('    return tensor_nz')
    code_lines_raw.append('')
    code_lines_raw.append('')
    code_lines_raw.append('def nd_to_nz(tensor_nd, data_type, is_pa=False):')
    code_lines_raw.append('    byte_map = {')
    code_lines_raw.append('        torch.int8: 1,')
    code_lines_raw.append('        torch.int16: 2,')
    code_lines_raw.append('        torch.int32: 4,')
    code_lines_raw.append('        torch.int64: 8,')
    code_lines_raw.append('        torch.float16: 2,')
    code_lines_raw.append('        torch.float32: 4,')
    code_lines_raw.append('        torch.bfloat16: 2,')
    code_lines_raw.append('    }')
    code_lines_raw.append('    byte = byte_map.get(data_type, None)')
    code_lines_raw.append('    if byte is None:')
    code_lines_raw.append('        assert False, f"unknown data_type{data_type}"')
    code_lines_raw.append('    block = 32 // byte')
    code_lines_raw.append('')
    code_lines_raw.append('    if len(tensor_nd.shape) == 2:')
    code_lines_raw.append('        return nd_to_nz_2dim(tensor_nd, block)')
    code_lines_raw.append('    elif len(tensor_nd.shape) == 3 and is_pa:')
    code_lines_raw.append('        return nd_to_nz_3dim_PA(tensor_nd, block)')
    code_lines_raw.append('    elif len(tensor_nd.shape) == 3 and not is_pa:')
    code_lines_raw.append('        return nd_to_nz_3dim_MLA(tensor_nd, block)')
    code_lines_raw.append('    else:')
    code_lines_raw.append('        assert False, f"no support shape {tensor_nd.shape}"')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def round_away_from_zero(x):')
    code_lines_raw.append('    pos_part = torch.where(x > 0, torch.floor(x + 0.5), x)')
    code_lines_raw.append('    neg_part = torch.where(x < 0, torch.ceil(x - 0.5), x)')
    code_lines_raw.append('    result = torch.where(x >= 0, pos_part, neg_part)')
    code_lines_raw.append('    return result')
    code_lines_raw.append('')
    code_lines_raw.append('')
    code_lines_raw.append('def round_to_odd(tensor):')
    code_lines_raw.append('    rounded = torch.round(tensor)')
    code_lines_raw.append('    mask = torch.abs(tensor - rounded) == 0.5')
    code_lines_raw.append('    adjusted = torch.where(rounded > tensor, rounded - 1, rounded + 1)')
    code_lines_raw.append('    return torch.where(mask, adjusted, rounded)')
    code_lines_raw.append('')
    code_lines_raw.append('')
    code_lines_raw.append('def cast_tensor(tensor, date_type, mode):')
    code_lines_raw.append('    if mode == 0:  # default is the same as  CAST_RINT')
    code_lines_raw.append('        return torch.round(tensor).to(date_type)')
    code_lines_raw.append('    elif mode == 1:  # CAST_RINT round to nearest, tie to even')
    code_lines_raw.append('        return torch.round(tensor).to(date_type)')
    code_lines_raw.append('    elif mode == 2:  # CAST_ROUND round to nearest, tie away from zero')
    code_lines_raw.append('        return round_away_from_zero(tensor).to(date_type)')
    code_lines_raw.append('    elif mode == 3:  # CAST_FLOOR round to minus infinity')
    code_lines_raw.append('        return torch.floor(tensor).to(date_type)')
    code_lines_raw.append('    elif mode == 4:  # CAST_CEIL round to positive infinity')
    code_lines_raw.append('        return torch.ceil(tensor).to(date_type)')
    code_lines_raw.append('    elif mode == 5:  # CAST_TRUNC round to zero')
    code_lines_raw.append('        return torch.trunc(tensor).to(date_type)')
    code_lines_raw.append('    elif mode == 6:  # CAST_ODD round to odd (Von Neumann rounding)')
    code_lines_raw.append('        return round_to_odd(tensor).to(date_type)')
    code_lines_raw.append('    else:')
    code_lines_raw.append('        raise ValueError("Invalid CastMode")')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_pos(d):')
    code_lines_raw.append('    return d')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_neg(d):')
    code_lines_raw.append('    return -1 * d')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_not(d):')
    code_lines_raw.append('    return not d')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_add(a, b):')
    code_lines_raw.append('    return a + b')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_sub(a, b):')
    code_lines_raw.append('    return a - b')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_mul(a, b):')
    code_lines_raw.append('    return a * b')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_div(a, b):')
    code_lines_raw.append('    return a // b')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_mod(a, b):')
    code_lines_raw.append('    return a % b')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_eq(a, b):')
    code_lines_raw.append('    return a == b')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_ne(a, b):')
    code_lines_raw.append('    return a != b')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_lt(a, b):')
    code_lines_raw.append('    return a < b')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_le(a, b):')
    code_lines_raw.append('    return a <= b')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_gt(a, b):')
    code_lines_raw.append('    return a > b')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_ge(a, b):')
    code_lines_raw.append('    return a >= b')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_min(a, b):')
    code_lines_raw.append('    return min(a, b)')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines_raw.append('def ssop_max(a, b):')
    code_lines_raw.append('    return max(a, b)')
    code_lines_raw.append('')
    code_lines_raw.append('')

    code_lines = gen_build_code_lines(code_lines_raw)
    return code_lines


def gen_op_tensor(op_code, meta_op, meta_fn=None):
    code_lines_mod = []
    for d_idx, d in enumerate(meta_op['data_i_list']):
        v_name = d['t_name']
        t_shape_fmt = ','.join([f'{i}' for i in d['shape']])
        if len(get_gen_conf('golden_filapath')) > 0:
            filename_in = f"{get_gen_conf('golden_filapath')[d_idx]}"
            dtype = get_dtype(d['rawtensor']['datatype'])
            pt_file = f"{get_gen_conf('golden_data_file_path')}/funcHash{meta_op['meta_fn']['hash']}"
            pt_file += f"-tensorMagic{d['magic']}-input.pt"
            code_str = f"{v_name} = file_bin_to_pt('{filename_in}', {dtype}, '{pt_file}', {d['shape']})"
            code_lines_mod += gen_build_code_lines(code_str, 4)
        else:
            code_str = ''
            if d['rawtensor']['datatype'] == 4 and get_gen_conf('check_mode') == 5:
                code_str = f"{v_name} = torch.zeros({t_shape_fmt}).to({get_dtype(d['rawtensor']['datatype'])})"
            else:
                code_str = f"{v_name} = torch.randn({t_shape_fmt}).to({get_dtype(d['rawtensor']['datatype'])})"
            code_lines_mod += gen_build_code_lines(code_str, 4)
    return code_lines_mod


def gen_op_input(op_code, meta_op, meta_fn=None):
    gen_op_code_fn = get_op_func(op_code)
    op_info = None
    code_lines_raw = gen_build_code_lines(gen_op_code_fn(meta_op), 4)
    code_lines_mod = []
    key = 0
    if gen_op_code_fn(meta_op):
        if meta_op['opcode'] in ['TRANSPOSE_DATAMOVE']:
            key = 0 if len(meta_op['data_i_list'][0]['producers']) != 0 else 1
        for d_idx, d in enumerate(meta_op['data_i_list']):
            if (meta_op['opcode'] in ['TRANSPOSE_DATAMOVE']) and (d_idx != key):
                meta_op['meta_fn']['state_']['data_dump_done_set'].add(d['t_name'])
                continue
            if (meta_op['opcode'] in ['ROWSUM_SINGLE', 'TRANSPOSE_VNCHWCONV', 'ROWMAX_SINGLE', 
                                      'ROWMAX_COMBINE_AXIS_SINGLE', 'ROWSUM_COMBINE_AXIS_SINGLE']):
                continue
            if meta_op['opcode'] in ['ADD_BRC', 'SUB_BRC', 'MUL_BRC', 'DIV_BRC'] and d_idx == 2:
                continue
            meta_op['meta_fn']['state_']['data_dump_done_set'].add(d['t_name'])
            t_shape_fmt = '-'.join([f'{i}' for i in d['shape']])
            cond0 = get_gen_conf('check_mode') != 5
            cond1 = (meta_op['opcode'] == 'CALL')
            cond2 = (get_gen_conf('check_mode') == 0)
            copys_op_list = ['COPY_IN', 'COPY_OUT', 'VIEW', 'ASSEMBLE']
            cond3 = (get_gen_conf('check_mode') == -1) and (meta_op['opcode'] in copys_op_list)
            cond4 = cond1 or cond2 or cond3
            if cond0 and cond4:
                di_shape = d['shape']
                di_ori_shape = d['ori_shape']
                api_args = ''
                if di_shape != di_ori_shape:
                    view_slice_args = []
                    for shape in di_ori_shape:
                        view_slice_args.append(f'0:{shape}')
                    api_args = '[' + ','.join(view_slice_args) + ']'
                pt_file = f"{get_gen_conf('dump_data_file_path')}/timeId{meta_op['meta_fn']['state_']['timeId']}"
                pt_file += f"-funcHash{meta_op['meta_fn']['hash']}-tensorMagic{d['magic']}-input.pt"
                code_lines_mod += gen_build_code_lines(f"    torch.save({d['t_name']}{api_args}, '{pt_file}')")
                meta_op['meta_fn']['state_']['timeId'] += 1
            cond4 = (get_gen_conf('check_mode') == -2)
            code_str = ""
            if (get_gen_conf('check_mode') == 5) and meta_op['in_param_loc'] is not None:
                assert_shape = d['shape'].copy()
                if meta_op['opcode'] == 'COPY_IN':
                    dims = len(meta_op['data_o_list'][0]['ori_shape'])
                    for i in range(dims):
                        assert_shape[i] = gen_symbol_scalar_expr(meta_op['attr']['axis'][4 + dims * 2 + 2 * i])
                in_param_loc = f"_{meta_op['in_param_loc'][0]}"
                # scatter update,A_MUL_B只有第三个输出位于子图边界需要loc信息
                if meta_op['opcode'] in ['INDEX_OUTCAST', 'A_MUL_B'] and d_idx != 2:
                    in_param_loc = ''
                code_str = f"assert ({assert_shape} == list({d['t_name']}{in_param_loc}.shape)), "
                code_str += f"f'{d['t_name']}{in_param_loc} json shape: {assert_shape}, '"
                code_str += f"f'torch shape: {{{d['t_name']}{in_param_loc}.shape}}'"
            else:
                code_str = f"assert ({d['shape']} == list({d['t_name']}.shape)), "
                code_str += f"f'{d['t_name']} json shape: {d['shape']}, torch shape: {{{d['t_name']}.shape}}'"
            code_lines_mod += gen_build_code_lines(code_str, 4)
            if cond4:
                code_lines_mod += gen_build_code_lines(f"    print('{d['t_name']} torch shape: ', {d['t_name']}.shape)")
    for line in code_lines_raw:
        if (line == '    \n'):
            code_lines_mod.append('\n')
        elif (line == '    '):
            code_lines_mod.append('')
        else:
            code_lines_mod.append(f'{line}  {gen_node_comment(meta_op)}')
    return code_lines_mod


def gen_op_output(op_code, meta_op, meta_fn=None):
    code_lines_mod = []
    gen_op_code_fn = get_op_func(op_code)
    if gen_op_code_fn(meta_op) and (meta_op['opcode'] not in ['ASSEMBLE']):
        for idx, d in enumerate(meta_op['data_o_list']):
            do0_shape = d['shape']
            do0_ori_shape = d['ori_shape']
            do0_name = d['t_name']
            key_dims = []
            repeat_op_list = ['COPY_IN', 'ASSEMBLE', 'ROWSUMLINE', 'ROWSUM_SINGLE', 'ROWMAX_SINGLE',
                              'RESHAPE', 'A_MUL_B', 'A_MULACC_B']
            if (meta_op['opcode'] in repeat_op_list) and (len(do0_ori_shape) > 0) and (do0_shape != do0_ori_shape):
                view_slice_args1 = []
                for i, v in enumerate(do0_shape):
                    if v % do0_ori_shape[i] != 0:
                        key_dims.append((v // do0_ori_shape[i]) + 1)
                    else:
                        key_dims.append(v // do0_ori_shape[i])
                    view_slice_args1.append(f'0:{v}') 
                view_slice_args1 = []
                api_args1 = '[' + ','.join(view_slice_args1) + ']'
                t_shape_fmt = ','.join([f'{i}' for i in key_dims])
                code_lines_mod += gen_build_code_lines(f'    {do0_name} = {do0_name}.repeat({t_shape_fmt}){api_args1}')
            meta_op['meta_fn']['state_']['data_dump_done_set'].add(d['t_name'])
            if (meta_op['opcode'] in ['ROWSUM_SINGLE', 'TRANSPOSE_VNCHWCONV', 'ROWMAX_SINGLE']):
                continue
            if (meta_op['opcode'] in ['ROWSUM_SINGLE', 'ROWMAX_SINGLE', 
                    'ROWMAX_COMBINE_AXIS_SINGLE', 'ROWSUM_COMBINE_AXIS_SINGLE']) and idx == 1:
                continue
            if meta_op['opcode'] in ['ADD_BRC', 'SUB_BRC', 'MUL_BRC', 'DIV_BRC'] and idx == 1:
                continue
            cond0 = get_gen_conf('check_mode') != 5
            cond1 = (meta_op['opcode'] == 'CALL')
            cond2 = (get_gen_conf('check_mode') == 0)
            copys_op_list = ['COPY_IN', 'COPY_OUT', 'VIEW', 'ASSEMBLE']
            cond3 = (get_gen_conf('check_mode') == -1) and (meta_op['opcode'] in copys_op_list)
            cond4 = cond1 or cond2 or cond3
            if cond0 and cond4:
                api_args = ''
                if do0_shape != do0_ori_shape:
                    view_slice_args = []
                    for shape in do0_ori_shape:
                        view_slice_args.append(f'0:{shape}')
                    api_args = '[' + ','.join(view_slice_args) + ']'
                pt_file = f"{get_gen_conf('dump_data_file_path')}/timeId{meta_op['meta_fn']['state_']['timeId']}"
                pt_file += f"-funcHash{meta_op['meta_fn']['hash']}-tensorMagic{d['magic']}-output.pt"
                code_lines_mod += gen_build_code_lines(f"    torch.save({d['t_name']}{api_args}, '{pt_file}')")
                meta_op['meta_fn']['state_']['timeId'] += 1
            cond4 = (get_gen_conf('check_mode') == -2)
            code_str = ""
            if (get_gen_conf('check_mode') == 5) and meta_op['out_param_loc'] is not None:
                assert_shape = d['shape'].copy()
                # leaf图上的输出和invoke info的shape不一致，要以invoke info的信息为准
                if meta_op['opcode'] in ['COPY_OUT', 'TRANSPOSE_DATAMOVE', 'INDEX_OUTCAST']:
                    dims = len(meta_op['data_i_list'][0]['ori_shape'])
                    for i in range(dims):
                        assert_shape[i] = gen_symbol_scalar_expr(meta_op['attr']['axis'][4 + dims * 2 + 2 * i])
                out_param_loc = meta_op['out_param_loc'][0]
                code_str = f"assert ({assert_shape} == list({d['t_name']}_{out_param_loc}.shape)), "
                code_str += f"f'{d['t_name']}_{out_param_loc} json shape: {assert_shape}, "
                code_str += f"torch shape: {{{d['t_name']}_{out_param_loc}.shape}}'"
            else:
                code_str = f"assert ({d['shape']} == list({d['t_name']}.shape)), "
                code_str += f"f'{d['t_name']} json shape: {d['shape']}, torch shape: {{{d['t_name']}.shape}}'"
            code_lines_mod += gen_build_code_lines(code_str, 4)
            if cond4:
                code_lines_mod += gen_build_code_lines(f"    print('{d['t_name']} torch shape: ', {d['t_name']}.shape)")

    return code_lines_mod


def gen_entry_code(meta_app, is_topo_sort=True):
    meta_gen_code_dict = meta_app['meta_gen_code_dict'][0]
    for meta_fn in meta_app['meta_g']['meta_fn_dict'][0].values():
        meta_gen_code_dict['def_name'] = gen_def_head(meta_fn['rawname'])
        code_str = f"    {gen_def_func_head(meta_fn['rawname'])}"
        meta_gen_code_dict['main_name'] = gen_build_code_lines(code_str)
        meta_op_dict = meta_fn['meta_op_dict']
        for meta_op in meta_op_dict.values():
            meta_gen_code_dict['input'] = gen_op_tensor(meta_op['opcode'], meta_op, meta_fn)
            meta_gen_code_dict['input'] += gen_op_input(meta_op['opcode'], meta_op, meta_fn)
            meta_gen_code_dict['output'] = gen_op_output(meta_op['opcode'], meta_op, meta_fn)


def gen_code_op(op_code, meta_op, meta_fn=None):
    code_lines_mod = []
    # callop的incast为workspace时需要提前初始化
    if meta_op['opcode'] in ['CALL', 'COPY_IN']:
        for _, d in enumerate(meta_op['data_i_list']):
            if d['t_name'] not in meta_op['meta_fn']['state_']['data_dump_done_set']:
                meta_op['meta_fn']['state_']['data_dump_done_set'].add(d['t_name'])
                code_str = f"{d['t_name']} = torch.zeros({d['shape']})"
                code_str += f".to({get_dtype(d['rawtensor']['datatype'])}).to('{get_gen_conf('device_str')}')"
                code_lines_mod += gen_build_code_lines(code_str, 4)
                meta_op['meta_fn']['state_']['workspace_or_alloc'].add(d['t_name'])

    code_lines_mod += gen_op_input(op_code, meta_op)
    code_lines_mod += gen_op_output(op_code, meta_op)
    return code_lines_mod


def gen_func_code(meta_app, functype, is_topo_sort=False):
    code_lines = []
    for meta_fn in meta_app['meta_g']['meta_fn_dict'][functype].values():
        op_list, op_isolated_list = parser_topo_sort(meta_fn, is_topo_sort)
        code_lines += gen_def_head(meta_fn['rawname'], meta_fn, meta_app['meta_g'])
        #NZ适配 PA场景等主线方案确定后再适配
        if functype == 4 or functype == 3 or functype == 2:
            incasts = list(meta_app['meta_g']['meta_fn_dict'][functype].values())[0]['incasts']
            meta_t = list(meta_app['meta_g']['meta_fn_dict'][functype].values())[0]['meta_t']
            for _, incast_magic in enumerate(incasts):
                format_nz = meta_t[incast_magic]['format']
                t_name = f"t{incast_magic}"
                date_type = get_dtype(meta_t[incast_magic]['rawtensor']['datatype'])
                if format_nz == 1:
                    code_lines += gen_build_code_lines(f"{t_name} = nz_to_nd({t_name}_nz, {date_type}, False)", 4)
        #为scatter update规避的，具体方案待定，暂时先注释
        if (functype == 4):
            incasts = list(meta_app['meta_g']['meta_fn_dict'][functype].values())[0]['incasts']
            outcasts = list(meta_app['meta_g']['meta_fn_dict'][functype].values())[0]['outcasts']
            meta_t = list(meta_app['meta_g']['meta_fn_dict'][functype].values())[0]['meta_t']
            for _, incast_magic in enumerate(incasts):
                for _, outcast_magic in enumerate(outcasts):
                    incast_rawmagic = meta_t[incast_magic]['rawtensor']['rawmagic']
                    incast_shape = meta_t[incast_magic]['shape']
                    incast_offset = meta_t[incast_magic]['offset']
                    outcast_rawmagic = meta_t[outcast_magic]['rawtensor']['rawmagic']
                    outcast_shape = meta_t[outcast_magic]['shape']
                    outcast_offset = meta_t[outcast_magic]['offset']
                    if incast_rawmagic == outcast_rawmagic and incast_shape == \
                       outcast_shape and incast_offset == outcast_offset:
                        code_lines += gen_build_code_lines(f"t{outcast_magic} = t{incast_magic}", 4)
                        meta_fn['state_']['data_dump_done_set'].add(f"t{outcast_magic}")
        for op_k in op_list:
            meta_op = meta_fn['meta_op_dict'][op_k]
            code_lines += gen_code_op(meta_op['opcode'], meta_op, meta_fn)
        for op_k in list(op_isolated_list):
            meta_op = meta_fn['meta_op_dict'][op_k]
            code_lines += gen_code_op(meta_op['opcode'], meta_op, meta_fn)
        code_lines_out = "return "
        for id, v in enumerate(meta_fn['state_']['fn_i_o_magic_dict']['outcasts']):
            if (id == len(meta_fn['state_']['fn_i_o_magic_dict']['outcasts']) - 1):
                code_lines_out += f"{v}"
            else:
                code_lines_out += f"{v}, "
            meta_fn['state_']['data_dump_done_set'].add(f't{v}')
        code_lines += gen_build_code_lines(code_lines_out, 4)
        code_lines += gen_build_code_lines('\n')
    if (functype == 5):
        meta_app['meta_gen_code_dict'][functype][meta_fn['hash']] = code_lines
    else:
        meta_app['meta_gen_code_dict'][functype] = code_lines


def gen_def_code(meta_app, is_topo_sort=False):
    meta_gen_code_dict = {
        'gen_code_head': gen_code_head(),
        5: dict(),
        4: [],
        3: [],
        2: [],
        1: [],
        0: dict(),
    }
    meta_app['meta_gen_code_dict'] = meta_gen_code_dict
    for functype in meta_app['meta_g']['meta_fn_dict']:
        if (functype == 0):
            gen_entry_code(meta_app)
        elif (functype == 4):
            gen_func_code(meta_app, functype, False)
        elif (functype == 3) and (get_gen_conf('check_mode') == 5):
            continue
        else:
            gen_func_code(meta_app, functype, False)
    return


def gen_entry_for_tile_graph(meta_app):
    code_lines_mod = gen_build_code_lines(f"def PROGRAM_ENTRY():")
    functype = get_functype(meta_app, meta_app['meta_g']['entryhash'])
    entry_func = meta_app['meta_g']['meta_fn_dict'][functype][meta_app['meta_g']['entryhash']]
    incast = entry_func['incasts']
    outcasts = entry_func['outcasts']
    entry_t = entry_func['meta_t']
    # incast
    for idx, magic_t in enumerate(incast):
        v_name = f"t{magic_t}"
        t_shape_fmt = ','.join([f'{i}' for i in entry_t[magic_t]['shape']])
        if len(get_gen_conf('golden_filapath')) > 0:
            filename_in = f"{get_gen_conf('golden_filapath')[idx]}"
            dtype = get_dtype(entry_t[magic_t]['rawtensor']['datatype'])
            pt_file = f"{get_gen_conf('golden_data_file_path')}/funcHash{entry_func['hash']}"
            pt_file += f"-tensorMagic{entry_t[magic_t]['magic']}-input.pt"
            code_str = f"{v_name} = file_bin_to_pt('{filename_in}', {dtype}, '{pt_file}', {entry_t[magic_t]['shape']})"
            code_lines_mod += gen_build_code_lines(code_str, 4)
        else:
            dtype = get_dtype(entry_t[magic_t]['rawtensor']['datatype'])
            code_str = f"{v_name} = torch.randn({t_shape_fmt}).to({dtype})"
            code_lines_mod += gen_build_code_lines(code_str, 4)

    # def
    def_name = f"{entry_func['rawname']}("
    incasts_args = ''
    for idx, v in enumerate(incast):
        if (idx == len(incast) - 1):
            incasts_args += f"t{v})"
        else:
            incasts_args += f"t{v}, "
    outcasts_args = ''
    for idx, v in enumerate(outcasts):
        if (idx == len(outcasts) - 1):
            outcasts_args += f"t{v} "
        else:
            outcasts_args += f"t{v}, "
    code_lines_mod += gen_build_code_lines(f"{outcasts_args} = {def_name}{incasts_args}", 4)
    
    #outcasts
    for _, magic_t in enumerate(outcasts):
        pt_file = f"{get_gen_conf('dump_data_file_path')}/timeId{entry_func['state_']['timeId']}"
        pt_file += f"-entry-tensorMagic{magic_t}-output.pt"
        code_lines_mod += gen_build_code_lines(f"    torch.save(t{magic_t}, '{pt_file}')")
        entry_func['state_']['timeId'] += 1
    
    code_lines_mod += gen_build_code_lines('\n')
    return code_lines_mod


def gen_main_code(meta_app):
    code_lines = []
    meta_gen_code = meta_app['meta_gen_code_dict']
    code_lines += meta_gen_code['gen_code_head']
    if (get_gen_conf('check_mode') in [5]):
        for value in meta_gen_code[5].values():
            code_lines += value
        code_lines += meta_gen_code[4]
    else:
        code_lines += meta_gen_code[3]
        code_lines += meta_gen_code[2]
        code_lines += meta_gen_code[1]
    #只有program.json有program entry
    if 0 in meta_gen_code and get_gen_conf('check_mode') == 5:
        code_lines += meta_gen_code[0]['def_name']
        code_lines += meta_gen_code[0]['input']
        code_lines += meta_gen_code[0]['output']
        code_lines += gen_build_code_lines(f"if __name__ == '__main__':")
        code_lines += meta_gen_code[0]['main_name']
    else:
        code_lines += gen_entry_for_tile_graph(meta_app)
        code_lines += gen_build_code_lines(f"if __name__ == '__main__':")
        code_lines += gen_build_code_lines(f"PROGRAM_ENTRY()", 4)

    print(get_gen_conf('code_file_path'))
    with open(get_gen_conf('code_file_path'), 'w') as file:
        file.write('\n'.join(code_lines))


def parser_topo_sort(meta_fn, is_topo_sort=True):
    meta_op_dict = meta_fn['meta_op_dict']
    if not is_topo_sort:
        return list(meta_op_dict.keys()), []

    op_list_sorted = []
    op_done_set = set()
    op_todo_list = meta_fn['state_']['op_todo_list']
    # potential dead loop?
    depth = 0
    threshold = len(meta_fn['meta_op_dict']) * 10
    while op_todo_list:
        assert (depth < threshold), f'depth > threshold {threshold}'
        depth += 1
        op_k_curr = op_todo_list.pop(0)
        op_v_curr = meta_op_dict[op_k_curr]

        def check_dependency():
            for d in op_v_curr['data_i_list']:
                for op_k in d['producers']:
                    if op_k not in op_done_set:
                        return False
            return True

        if check_dependency() == False:
            op_todo_list.append(op_k_curr)
            continue

        for d in op_v_curr['data_o_list']:
            for op_k in d['consumers']:
                if op_k not in op_done_set:
                    if op_k not in op_todo_list:
                        op_todo_list.append(op_k)

        op_list_sorted.append(op_k_curr)
        op_done_set.add(op_k_curr)

    op_isolated_set = set(meta_op_dict.keys())
    assert (op_done_set.issubset(op_isolated_set))
    op_isolated_set.symmetric_difference_update(op_done_set)
    return op_list_sorted, list(op_isolated_set)
