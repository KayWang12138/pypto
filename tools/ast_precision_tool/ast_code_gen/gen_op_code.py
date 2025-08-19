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
import struct
from pprint import pprint
from ast_code_gen.common import get_gen_conf, get_dtype, error_checker
from ast_code_gen.common import gen_build_code_lines, gen_func_head, gen_symbol_scalar_expr


# None
def __gen_op_todo_(meta_op=None):
    code_lines = gen_build_code_lines(f'#TODO')
    return code_lines


def __gen_op_binary(api_name, meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 2, meta_op=meta_op)
    if meta_op['opcode'] in ['ADD_BRC', 'SUB_BRC', 'MUL_BRC', 'DIV_BRC'] and meta_op['meta_fn']['functype'] != 2:
        error_checker(len(do_list) == 2, meta_op=meta_op)
    else:
        error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di1_name = di_list[1]['t_name']
    do0_name = do_list[0]['t_name']
    code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}.{api_name}({di1_name})')
    return code_lines


# ADD
# S_ADD
def _gen_op_add(meta_op=None):
    api_name = f'add'
    return __gen_op_binary(api_name, meta_op)


def __gen_op_binary_scalar(api_name, meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 1, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    do0_name = do_list[0]['t_name']

    data_type = meta_op['attr']['scalarDataType']
    if data_type == 0:
        scalar = meta_op['attr']['scalarDataValue']
    else:
        packed = struct.pack('i', meta_op['attr']['scalarDataValue'])
        scalar = struct.unpack('f', packed)[0]
    is_reverse = meta_op['attr']['scalarReverse'] if meta_op['attr'].get('scalarReverse') else 0
    if is_reverse:
        code_lines += gen_build_code_lines(f'{do0_name} = torch.{api_name}(torch.tensor({scalar}), {di0_name})')
    else:
        code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}.{api_name}({scalar})')
    return code_lines


# ADDS
def _gen_op_adds(meta_op=None):
    api_name = f'add'
    return __gen_op_binary_scalar(api_name, meta_op)


# ASSEMBLE
def _gen_op_assemble(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 1, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di0_shape = di_list[0]['shape']
    do0_name = do_list[0]['t_name']
    do0_shape = do_list[0]['shape']
    if (do0_name == di0_shape):
        code_lines += gen_build_code_lines(f"{do0_name} = {di0_name}")
        return code_lines
    from_offset = [0 for _ in range(meta_op['attr']['axis'][1])]
    for i in range(meta_op['attr']['axis'][1]):
        from_offset[i] = meta_op['attr']['axis'][2 + i]
    view_off = from_offset
    view_slice_args = []
    for off, shape in zip(view_off, di0_shape):
        view_slice_args.append(f'{off}:{off + shape}')
    api_args = '[' + ','.join(view_slice_args) + ']'
    if do_list[0]['rawtensor']['rt_name'] not in meta_op['meta_fn']['state_']['data_dump_done_set']:
        meta_op['meta_fn']['state_']['data_dump_done_set'].add(do_list[0]['rawtensor']['rt_name'])
        t_shape_fmt = ','.join([f'{i}' for i in do_list[0]['rawtensor']['rawshape']])
        code_str = f"{do_list[0]['rawtensor']['rt_name']} = torch.zeros({t_shape_fmt})."
        code_str += f"to({get_dtype(di_list[0]['rawtensor']['datatype'])}).to('{get_gen_conf('device_str')}')"
        code_lines += gen_build_code_lines(code_str)
    code_lines += gen_build_code_lines(f"{do_list[0]['rawtensor']['rt_name']}{api_args} = {di0_name}")
    if do0_name not in meta_op['meta_fn']['state_']['data_dump_done_set']:
        meta_op['meta_fn']['state_']['data_dump_done_set'].add(do0_name)
        t_shape_fmt = ','.join([f'{i}' for i in do0_shape])
        code_str = f"{do0_name} = torch.zeros({t_shape_fmt}).to({get_dtype(di_list[0]['rawtensor']['datatype'])})"
        code_str += f".to('{get_gen_conf('device_str')}')"
        code_lines += gen_build_code_lines(code_str)
    view_slice_args = []
    for shape in do0_shape:
        view_slice_args.append(f'{0}:{0 + shape}')
    api_args = '[' + ','.join(view_slice_args) + ']'
    view_slice_args2 = []
    for off, shape in zip(do_list[0]['offset'], do0_shape):
        view_slice_args2.append(f'{off}:{off + shape}')
    api_args2 = '[' + ','.join(view_slice_args2) + ']'
    code_lines += gen_build_code_lines(f"{do0_name}{api_args} = {do_list[0]['rawtensor']['rt_name']}{api_args2}")
    code_lines += gen_build_code_lines(f"{do0_name}.to({get_dtype(do_list[0]['rawtensor']['datatype'])})")
    return code_lines


def __gen_op_matmul(api_name, is_at=False, is_bt=False, is_acc=False, meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    di0_name = di_list[0]['t_name']
    di1_name = di_list[1]['t_name']
    di0_shape = di_list[0]['shape']
    di1_shape = di_list[1]['shape']
    di0_orishape = di_list[0]['ori_shape']
    di1_orishape = di_list[1]['ori_shape']
    view_slice_args = []
    api_args0 = ''
    if di0_shape != di0_orishape:
        for shape in di0_orishape:
            view_slice_args.append(f'0:{shape}')
        api_args0 = '[' + ','.join(view_slice_args) + ']'
    view_slice_args = []
    api_args1 = ''
    if di1_shape != di1_orishape:
        for shape in di1_orishape:
            view_slice_args.append(f'0:{shape}')
        api_args1 = '[' + ','.join(view_slice_args) + ']'
    if is_acc:
        di2_name = di_list[2]['t_name']
        di2_shape = di_list[2]['shape']
        di2_orishape = di_list[2]['ori_shape']
        view_slice_args = []
        api_args2 = ''
        if di2_shape != di2_orishape:
            for shape in di2_orishape:
                view_slice_args.append(f'0:{shape}')
            api_args2 = '[' + ','.join(view_slice_args) + ']'
    do0_name = do_list[0]['t_name']
    di0_t = '.transpose(-1, -2)' if is_at else ''
    di1_t = '.transpose(-1, -2)' if is_bt else ''
    do0_add_op = f'.add({di2_name}{api_args2})' if is_acc else ''
    di0_to_fp32 = '.to(torch.float32)' if di_list[0]['is_to_fp32'] else ''
    di1_to_fp32 = '.to(torch.float32)' if di_list[1]['is_to_fp32'] else ''
    do0_dtype = f".to({get_dtype(do_list[0]['rawtensor']['datatype'])})"
    code_str = ''
    quant_dtype = [0, 1, 2, 3, 4]
    if di_list[0]['rawtensor']['datatype'] not in quant_dtype:
        to_fp64 = '.to(torch.float64)'
        do0_add_op = f'.add({di2_name}{api_args2}{to_fp64})' if is_acc else ''
        code_str = f'{do0_name} = '
        code_str += f'{di0_name}{api_args0}{di0_t}{to_fp64}.{api_name}'
        code_str += f'({di1_name}{api_args1}{di1_t}{to_fp64}){do0_add_op}{to_fp64}'
    elif di_list[0]['rawtensor']['datatype'] in quant_dtype:
        to_int64 = '.to(torch.int64)'
        do0_add_op = f'.add({di2_name}{api_args2}{to_int64})' if is_acc else ''
        code_str = f'{do0_name} = '
        code_str += f'{di0_name}{api_args0}{di0_t}{to_int64}.{api_name}'
        code_str += f'({di1_name}{api_args1}{di1_t}{to_int64}){do0_add_op}{to_int64}'
    else:
        code_str = f'{do0_name} = '
        code_str += f'{di0_name}{api_args0}{di0_t}{di0_to_fp32}.{api_name}'
        code_str += f'({di1_name}{api_args1}{di1_t}{di1_to_fp32}){do0_add_op}{do0_dtype}'
    code_lines += gen_build_code_lines(code_str)

    if get_gen_conf('check_mode') != 5:
        next_op_tuple = meta_op['meta_fn']['consumers_mapping'][do_list[0]['magic']]
        next_op = meta_op['meta_fn']['meta_op_dict'].get(next_op_tuple[0])
        next_opcode = next_op['opcode']
        if next_opcode not in ['A_MUL_Bt', 'A_MUL_Bt_FA2', 'A_MUL_B', 'A_MULACC_B', 'A_MULACC_Bt']:
            dtype = get_dtype(do_list[0]['rawtensor']['datatype'])
            code_lines += gen_build_code_lines(f"{do0_name} = {do0_name}.to({dtype})")
    return code_lines


# A_MUL_B
def _gen_op_matmul_ab(meta_op=None):
    api_name = f'matmul'
    return __gen_op_matmul(api_name, meta_op=meta_op)


# A_MUL_Bt
# A_MUL_Bt_FA2
def _gen_op_matmul_abt(meta_op=None):
    api_name = f'matmul'
    if meta_op['meta_fn']['functype'] == 2:
        return __gen_op_matmul(api_name, is_bt=True, meta_op=meta_op)
    return __gen_op_matmul(api_name, is_bt=False, meta_op=meta_op)


# A_MULACC_B
def _gen_op_matmul_acc_ab(meta_op=None):
    api_name = f'matmul'
    return __gen_op_matmul(api_name, is_acc=True, meta_op=meta_op)


# A_MULACC_Bt
def _gen_op_matmul_acc_abt(meta_op=None):
    api_name = f'matmul'
    if meta_op['meta_fn']['functype'] == 2:
        return __gen_op_matmul(api_name, is_bt=True, is_acc=True, meta_op=meta_op)
    return __gen_op_matmul(api_name, is_bt=False, is_acc=True, meta_op=meta_op)


def __gen_op_unary(api_name, api_args='', meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 1, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    do0_name = do_list[0]['t_name']
    quant_dtype = [0, 1, 2, 3, 4]
    if meta_op['opcode'] in ['CAST'] and do_list[0]['rawtensor']['datatype'] in quant_dtype:
        cast_mode = meta_op['op_attr']['op_attr_mode']
        code_lines += gen_build_code_lines(f'{do0_name} = cast_tensor({di0_name}, {api_args}, {cast_mode})')
    else:
        code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}.{api_name}({api_args})')
    di0_shape = di_list[0]['shape']
    do0_shape = do_list[0]['shape']
    if (meta_op['opcode'] in ['CAST']) and (di0_shape != do0_shape):
        repeat_flag = True
        key_dims = []
        for i, v in enumerate(do0_shape):
            key_dim = v // di0_shape[i]
            if key_dim < 1:
                repeat_flag = False
                continue
            key_dims.append(key_dim)
        if repeat_flag:
            t_shape_fmt = ','.join([f'{i}' for i in key_dims])
            code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}.repeat({t_shape_fmt})')
        else:
            t_shape_fmt = '[' + ', '.join([f':{n}' for n in do0_shape]) + ']'
            code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}{t_shape_fmt})')
    return code_lines


# ABS
def _gen_op_abs(meta_op=None):
    api_name = f'abs'
    return __gen_op_unary(api_name, meta_op=meta_op)


# BT_ALLOC
# BAR_V
# BAR_M
# CV_SYNC_SRC
# CV_SYNC_DST
# COMM_WAIT_FLAG
# FIX_ALLOC
# L0A_ALLOC
# L0B_ALLOC
# L0C_ALLOC
# L1_ALLOC
# L0A_ALLOC
# L0B_ALLOC
# PHASE1
# PHASE2
# SYNC_SRC
# SYNC_DST
# UB_ALLOC
def _gen_op_none(meta_op=None):
    return []


# BT_COPY_IN
# FIX_COPY_IN
# L0C_COPY_OUT
# L1_TO_L0A
# L1_TO_L0B
# L1_TO_L0Bt
# L1_COPY_IN_DMA
# REGISTER_COPY
# UB_COPY_OUT
# UB_TO_UB
# VIEW
def _gen_op_view(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 1, meta_op=meta_op)
    if (len(do_list) != 1):
        return code_lines
    di0_name = di_list[0]['t_name']
    do0_name = do_list[0]['t_name']
    from_offset = []
    if meta_op['opcode'] == 'VIEW':
        from_offset = [0 for _ in range(meta_op['attr']['axis'][1])]
        for i in range(meta_op['attr']['axis'][1]):
            from_offset[i] = meta_op['attr']['axis'][2 + i]
    view_off = [a - b for a, b in zip(from_offset, di_list[0]['offset'])]
    view_slice_args = []
    if (get_gen_conf('check_mode') == 5) and meta_op['in_param_loc'] is not None:
        in_param_loc = meta_op['in_param_loc'][0]
        out_param_loc = meta_op['out_param_loc'][0]
        di0_name = f'{di0_name}_{in_param_loc}'
        do0_name = f'{do0_name}_{out_param_loc}'
        api_args = '[' + ','.join(f'0:{x}' for x in do_list[0]['shape']) + ']'
        code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}{api_args}')
        return code_lines
    api_args = ""
    for off, shape in zip(view_off, do_list[0]['shape']):
        view_slice_args.append(f'{off}:{off + shape}')
    if len(view_slice_args) > 1:
        api_args = '[' + ','.join(view_slice_args) + ']'
    if meta_op['opcode'] == 'L1_TO_L0Bt':
        code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}{api_args}.transpose(0, 1)')
    else :
        code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}{api_args}')
    return code_lines


# BITSORT
def _gen_op_bitsort(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    di0_name = di_list[0]['t_name']
    do0_name = do_list[0]['t_name']
    axis = meta_op['op_attr']['op_attr_axis']
    is_largest = meta_op['op_attr']['op_attr_order']
    code_lines += gen_build_code_lines(f'{do0_name} = bitsort({di0_name}, {axis}, {is_largest} )')
    return code_lines


# CONVERT
def _gen_op_convert(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    di0_name = di_list[0]['t_name']
    do0_name = do_list[0]['t_name']
    error_checker(len(di_list) == 1, meta_op=meta_op)
    if (len(do_list) != 1):
        return code_lines
    api_args = '[' + ', '.join([f':{n}' for n in do_list[0]['shape']]) + ']'
    code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}{api_args}')
    return code_lines


# CALL
def __gen_op_call_(meta_op=None):
    if (get_gen_conf('check_mode') == 5) and (meta_op['meta_fn']['functype'] == 4):
        return __gen_op_call_in_root_(meta_op)
    if (get_gen_conf('check_mode') == 6) and (meta_op['meta_fn']['functype'] == 9):
        return __gen_op_call_dyn_entry(meta_op)
    code_lines = []
    data_in_gen_mode = get_gen_conf('data_in_gen_mode')
    consumers_mapping = dict()
    opmagic = meta_op['opmagic']
    subgraphid = meta_op['subgraphid']
    incasts = meta_op['data_i_list']
    outcasts = meta_op['data_o_list']
    tile_fn_outcasts = []
    if (get_gen_conf('check_mode') in [5]):
        tile_fn = meta_op['meta_fn_dict'][3]
        func_dict = tile_fn[next(iter(tile_fn))]
        tile_fn_outcasts = func_dict['outcasts']
        consumers_mapping = func_dict['consumers_mapping']
    for data_i in meta_op['data_i_list']:
        if (data_i['t_name']) in meta_op['meta_fn']['state_']['data_dump_done_set']:
            continue
        meta_op['meta_fn']['state_']['data_dump_done_set'].add(data_i['t_name'])
        view_slice_args = []
        for off, shape in zip(data_i['offset'], data_i['shape']):
            view_slice_args.append(f'{off}:{off + shape}')
        api_args = '[' + ','.join(view_slice_args) + ']'
        t_dtype_gen = get_dtype(data_i['rawtensor']['datatype'])
        t_name = data_i['t_name']
        t_shape = data_i['shape']
        t_shape_fmt = '-'.join([f'{i}' for i in t_shape])
        dev_str = get_gen_conf('device_str')
        if data_in_gen_mode in ['randn']:
            code_lines += gen_build_code_lines(
                f"{t_name} = torch.randn({t_shape}).to({t_dtype_gen}).to('{dev_str}')")
        elif data_in_gen_mode in ['file']:
            opmagic = consumers_mapping[data_i['magic']][0][1]
            infile_path = f"dump_tensor_pt/subgraphId{subgraphid}-rawMagic{data_i['rawtensor']['rawmagic']}"
            infile_path += f"-opMagic{opmagic}-dataType{data_i['rawtensor']['datatype']}-shape{t_shape_fmt}.pt"
            code_lines += gen_build_code_lines(
                f"{t_name} = torch.load('{infile_path}', map_location='cpu').to({t_dtype_gen}).to('{dev_str}')")
        else:
            assert ()

    for data_o in meta_op['data_o_list']:
        if (data_o['t_name'] in meta_op['meta_fn']['state_']['data_dump_done_set']):
            continue
        meta_op['meta_fn']['state_']['data_dump_done_set'].add(data_o['t_name'])
        t_dtype_gen = get_dtype(data_o['rawtensor']['datatype'])
        code_str = f"{data_o['t_name']} = torch.zeros({data_o['shape']})"
        code_str += f".to({get_dtype(data_o['rawtensor']['datatype'])}).to('{get_gen_conf('device_str')}')"
        code_lines += gen_build_code_lines(code_str)
    if (meta_op['meta_fn']['functype'] == 0):
        if (get_gen_conf('check_mode') in [5]):
            code_lines = gen_build_code_lines(
                f"{gen_func_head(next(iter(meta_op['meta_fn_dict'][4].values()))['rawname'], meta_op)}")
        else:
            if (meta_op['meta_fn_dict'].get(3)):
                code_lines = gen_build_code_lines(
                    gen_func_head(next(iter(meta_op['meta_fn_dict'][3].values()))['rawname'], meta_op))
            if (meta_op['meta_fn_dict'].get(2)):
                code_lines = gen_build_code_lines(
                    gen_func_head(next(iter(meta_op['meta_fn_dict'][2].values()))['rawname'], meta_op))
    else:
        leaf_func = meta_op['meta_fn_dict'][5][meta_op['calleehash']]
        code_lines += gen_build_code_lines(f"{gen_func_head(leaf_func['rawname'], meta_op)}")
    code_lines += gen_build_code_lines("\n")
    return code_lines


# CALL
def __gen_op_call_in_root_(meta_op=None):
    in_operand_info = []
    out_operand_info = []
    code_lines = []
    incast_params = meta_op['invoke_info']['incast_params']
    outcast_params = meta_op['invoke_info']['outcast_params']
    tensor_params = meta_op['invoke_info']['tensor_params']
    for incast_param in incast_params:
        in_operand_info.append(incast_param)
    for output_param in outcast_params:
        out_operand_info.append(output_param)
    for tensor_param in tensor_params:
        is_output = tensor_param.get('is_output')
        if is_output:
            out_operand_info.append(tensor_param)
        else:
            in_operand_info.append(tensor_param)

    # input
    inputs = meta_op['data_i_o_magic_dict']['ioperands']
    data_i_list = meta_op['data_i_list']
    for i in range(len(data_i_list)):
        data_i = data_i_list[i]
        iop_info = in_operand_info[i]
        rt = data_i['rawtensor']
        t_rt_shape_equal_flag = data_i['shape'] == rt['rawshape']
        timeid = meta_op['meta_fn']['state_']['timeId']
        if not t_rt_shape_equal_flag and inputs[i] in meta_op['scatter_update_loc']['input_loc']:
            for j, _ in enumerate(data_i_list):
                if rt['rt_name'] == data_i_list[j]['rawtensor']['rt_name'] and data_i_list[j]['rawtensor'][
                    'rawshape'] == data_i_list[j]['shape']:
                    view_slice_args = []
                    for off, shape in zip(data_i['offset'], data_i['shape']):
                        view_slice_args.append(f'{off}:{off + shape}')
                    api_args = '[' + ','.join(view_slice_args) + ']'
                    code_lines += gen_build_code_lines(f"{inputs[i]} = {data_i_list[j]['t_name']}{api_args}")
                    break
            subgraphid = meta_op['subgraphid']
            rawmagic = rt['rawmagic'] if (rt['actual_rawmagic'] == -1) else rt['actual_rawmagic']
            opmagic = meta_op['tensor_loc']['input_loc_info'][i]['op_magic']
            t_shape_fmt = '-'.join([f'{i}' for i in meta_op['tensor_loc']['input_loc_info'][i]['shape']])
            dtype = rt['datatype']
            pt_file = f"{get_gen_conf('dump_data_file_path')}/timeId{timeid}-subgraphId{subgraphid}"
            pt_file += f"-rawMagic{rawmagic}-opMagic{opmagic}-shape{t_shape_fmt}-dataType{dtype}-input.pt"
            code_lines += gen_build_code_lines(f"torch.save({inputs[i]}, '{pt_file}')")
            meta_op['meta_fn']['state_']['timeId'] += 1
            continue

        if (data_i['t_name'] not in meta_op['meta_fn']['state_']['data_dump_done_set']):
            meta_op['meta_fn']['state_']['data_dump_done_set'].add(data_i['t_name'])
            code_str = f"{data_i['t_name']} = torch.zeros({data_i['shape']})"
            code_str += f".to({get_dtype(rt['datatype'])}).to('{get_gen_conf('device_str')}')"
            code_lines += gen_build_code_lines(code_str)
        view_slice_args = []
        for off, shape in zip(iop_info['offset'], iop_info['shape']):
            view_slice_args.append(f'{off}:{off + shape}')
        api_args = '[' + ','.join(view_slice_args) + ']'
        if not t_rt_shape_equal_flag:
            if rt['rt_name'] not in meta_op['meta_fn']['state_']['data_dump_done_set']:
                meta_op['meta_fn']['state_']['data_dump_done_set'].add(rt['rt_name'])
                code_str = f"{rt['rt_name']} = torch.zeros({rt['rawshape']})"
                code_str += f".to({get_dtype(rt['datatype'])}).to('{get_gen_conf('device_str')}')"
                code_lines += gen_build_code_lines(code_str)
            view_slice_args1 = []
            for off, shape in zip(data_i['offset'], data_i['shape']):
                view_slice_args1.append(f'{off}:{off + shape}')
            api_args1 = '[' + ','.join(view_slice_args1) + ']'
            code_lines += gen_build_code_lines(f"{rt['rt_name']}{api_args1} = {data_i['t_name']}")
            code_lines += gen_build_code_lines(f"{inputs[i]} = {rt['rt_name']}{api_args}")
        else:
            code_lines += gen_build_code_lines(f"{inputs[i]} = {data_i['t_name']}{api_args}")
        #workspace和子图执行前需要alloc的未初始化的tensor不需要save对比，此时该tensor是脏数据，主要是reshape场景涉及
        if data_i['t_name'] not in meta_op['meta_fn']['state_']['workspace_or_alloc']:
            subgraphid = meta_op['subgraphid']
            rawmagic = rt['rawmagic'] if (rt['actual_rawmagic'] == -1) else rt['actual_rawmagic']
            opmagic = meta_op['tensor_loc']['input_loc_info'][i]['op_magic']
            t_shape_fmt = '-'.join([f'{i}' for i in meta_op['tensor_loc']['input_loc_info'][i]['shape']])
            dtype = rt['datatype']
            pt_file = f"{get_gen_conf('dump_data_file_path')}/timeId{timeid}-subgraphId{subgraphid}"
            pt_file += f"-rawMagic{rawmagic}-opMagic{opmagic}-shape{t_shape_fmt}-dataType{dtype}-input.pt"
            save_tname = f"{inputs[i]}"
            if f"{data_i['t_name']}_nz" in meta_op['meta_fn']['state_']['data_dump_done_set']:
                save_tname = f"{data_i['t_name']}_nz{api_args}"
            code_lines += gen_build_code_lines(f"torch.save({save_tname}, '{pt_file}')")
            meta_op['meta_fn']['state_']['timeId'] += 1

    # call leaf func
    leaf_func = meta_op['meta_fn_dict'][5][meta_op['calleehash']]
    code_lines += gen_build_code_lines(f"{gen_func_head(leaf_func['rawname'], meta_op)}")

    # output
    outputs = meta_op['data_i_o_magic_dict']['ooperands']
    data_o_list = meta_op['data_o_list']
    for i, _ in enumerate(data_o_list):
        data_o = data_o_list[i]
        dat_o_shape = data_o_list[i]['shape']
        oop_info = out_operand_info[i]
        rt = data_o['rawtensor']
        t_rt_shape_equal_flag = data_o['shape'] == rt['rawshape']
        timeid = meta_op['meta_fn']['state_']['timeId']
        if not t_rt_shape_equal_flag:
            is_scatterupdate_output = False
            for j, _ in enumerate(data_o_list):
                if (rt['rt_name'] == data_o_list[j]['rawtensor']['rt_name'] and 
                        outputs[j] in meta_op['scatter_update_loc']['output_loc']):
                    view_slice_args = []
                    for off, shape in zip(oop_info['offset'], oop_info['shape']):
                        view_slice_args.append(f'{off}:{off + shape}')
                    api_args = '[' + ','.join(view_slice_args) + ']'
                    code_lines += gen_build_code_lines(f"{outputs[i]} = {data_o_list[j]['t_name']}{api_args}")
                    code_lines += gen_build_code_lines(f"{data_o['t_name']} = {data_o_list[j]['t_name']}{api_args}")
                    is_scatterupdate_output = True
                    break
            if is_scatterupdate_output:
                subgraphid = meta_op['subgraphid']
                rawmagic = rt['rawmagic'] if (rt['actual_rawmagic'] == -1) else rt['actual_rawmagic']
                opmagic = meta_op['tensor_loc']['output_loc_info'][i]['op_magic']
                t_shape_fmt = '-'.join([f'{i}' for i in meta_op['tensor_loc']['output_loc_info'][i]['shape']])
                dtype = rt['datatype']
                pt_file = f"{get_gen_conf('dump_data_file_path')}/timeId{timeid}-subgraphId{subgraphid}"
                pt_file += f"-rawMagic{rawmagic}-opMagic{opmagic}-shape{t_shape_fmt}-dataType{dtype}-output.pt"
                code_lines += gen_build_code_lines(f"torch.save({outputs[i]}, '{pt_file}')")
                meta_op['meta_fn']['state_']['timeId'] += 1
                continue

            if rt['rt_name'] not in meta_op['meta_fn']['state_']['data_dump_done_set']:
                meta_op['meta_fn']['state_']['data_dump_done_set'].add(rt['rt_name'])
                code_str = f"{rt['rt_name']} = torch.zeros({rt['rawshape']})"
                code_str += f".to({get_dtype(rt['datatype'])}).to('{get_gen_conf('device_str')}')"
                code_lines += gen_build_code_lines(code_str)
            view_slice_args = []
            for off, shape in zip(oop_info['offset'], oop_info['shape']):
                view_slice_args.append(f'{off}:{off + shape}')
            api_args = '[' + ','.join(view_slice_args) + ']'
            if meta_op['calleehash'] in meta_op['meta_fn']['copy_out_acc_func']:
                code_lines += gen_build_code_lines(
                    f"{rt['rt_name']}{api_args} = {rt['rt_name']}{api_args}.add({outputs[i]})")
                code_lines += gen_build_code_lines(f"{outputs[i]} = {rt['rt_name']}{api_args}")
            else:
                code_lines += gen_build_code_lines(f"{rt['rt_name']}{api_args} = {outputs[i]}")
            view_slice_args = []
            for off, shape in zip(data_o['offset'], data_o['shape']):
                view_slice_args.append(f'{off}:{off + shape}')
            api_args = '[' + ','.join(view_slice_args) + ']'
            code_lines += gen_build_code_lines(f"{data_o['t_name']} = {rt['rt_name']}{api_args}")
        else:
            if data_o['t_name'] not in meta_op['meta_fn']['state_']['data_dump_done_set']:
                meta_op['meta_fn']['state_']['data_dump_done_set'].add(data_o['t_name'])
                code_str = f"{data_o['t_name']} = torch.zeros({data_o['shape']})"
                code_str += f".to({get_dtype(rt['datatype'])}).to('{get_gen_conf('device_str')}')"
                code_lines += gen_build_code_lines(code_str)
            view_slice_args = []
            output_shape = oop_info['shape'].copy()
            for off, shape in zip(oop_info['offset'], output_shape):
                view_slice_args.append(f'{off}:{off + shape}')
            api_args = '[' + ','.join(view_slice_args) + ']'
            
            if outputs[i] in meta_op['scatter_update_loc']['output_loc']:
                t_index = meta_op['scatter_update_loc']['index_loc'][outputs[i]]
                view_slice_args = []
                for j, v in enumerate(output_shape):
                    if j == len(output_shape) - 2:
                        view_slice_args.append(
                            f"{oop_info['offset'][j]}+{t_index}[0][0]:{oop_info['offset'][j]}+{t_index}[0][0]+{v}")
                    else:
                        view_slice_args.append(
                            f"{oop_info['offset'][j]}:{oop_info['offset'][j] + v}")
                api_args = '[' + ','.join(view_slice_args) + ']'
            if meta_op['calleehash'] in meta_op['meta_fn']['copy_out_acc_func']:
                code_lines += gen_build_code_lines(
                    f"{data_o['t_name']}{api_args} = {data_o['t_name']}{api_args}.add({outputs[i]})")
                code_lines += gen_build_code_lines(f"{outputs[i]} = {data_o['t_name']}{api_args}")
            else:
                code_lines += gen_build_code_lines(f"{data_o['t_name']}{api_args} = {outputs[i]}")
        
        subgraphid = meta_op['subgraphid']
        rawmagic = rt['rawmagic'] if (rt['actual_rawmagic'] == -1) else rt['actual_rawmagic']
        opmagic = meta_op['tensor_loc']['output_loc_info'][i]['op_magic']
        t_shape_fmt = '-'.join([f'{i}' for i in meta_op['tensor_loc']['output_loc_info'][i]['shape']])
        dtype = rt['datatype']
        pt_file = f"{get_gen_conf('dump_data_file_path')}/timeId{timeid}-subgraphId{subgraphid}"
        pt_file += f"-rawMagic{rawmagic}-opMagic{opmagic}-shape{t_shape_fmt}-dataType{dtype}-output.pt"
        save_tname = f"{outputs[i]}"
        if f"{data_o['t_name']}_nd_to_nz" in meta_op['meta_fn']['state_']['data_dump_done_set']:
            d0_dtype = get_dtype(dtype)
            save_tname = f"nd_to_nz({outputs[i]}, {d0_dtype})"
        code_lines += gen_build_code_lines(f"torch.save({save_tname}, '{pt_file}')")
        meta_op['meta_fn']['state_']['timeId'] += 1
    code_lines.append('')
    return code_lines


def __gen_op_call_dyn_entry(meta_op=None):
    code_lines = []
    for func in meta_op['meta_fn_dict'].values():
        if meta_op['calleehash'] in func:
            leaf_func = func[meta_op['calleehash']]
            break
    if len(meta_op['meta_fn']['dynamic']) == 0:
        code_lines += gen_build_code_lines(f"{gen_func_head(leaf_func['rawname'], meta_op)}")
        return code_lines
    begin = meta_op['meta_fn']['dynamic']['begin'][1]
    end = meta_op['meta_fn']['dynamic']['end'][1]
    step = meta_op['meta_fn']['dynamic']['step'][1]
    itername = meta_op['meta_fn']['dynamic']['itername']
    code_lines += gen_build_code_lines(f"for {itername} in range({begin}, {end}, {step}):")
    if meta_op['meta_fn']['dynamic'].get('paths') is None:
        code_lines += gen_build_code_lines(f"{gen_func_head(leaf_func['rawname'], meta_op)}")
        return code_lines
    for call_magic, conds in meta_op['meta_fn']['dynamic']['paths']:
        conditions = []
        for cond in conds:
            cond_left = gen_symbol_scalar_expr(cond[0])
            assert (cond[1] == True or cond[1] == False)
            conditions.append(f'{cond_left} == {cond[1]}')
        conditions_code = ' and '.join(conditions)
        code_lines += gen_build_code_lines(f"    if {conditions_code}:")
        for key, callee_meta_op in meta_op['meta_fn']['meta_op_dict'].items():
            if key[1] == call_magic:
                for func in callee_meta_op['meta_fn_dict'].values():
                    if callee_meta_op['calleehash'] in func:
                        leaf_func = func[callee_meta_op['calleehash']]
                        code_lines += (
                            gen_build_code_lines(f"        {gen_func_head(leaf_func['rawname'], callee_meta_op)}"))
                        break
                break
    return code_lines


# CAST
def _gen_op_cast(meta_op=None):
    api_name = f'to'
    api_args = get_dtype(meta_op['data_o_list'][0]['rawtensor']['datatype'])
    return __gen_op_unary(api_name, api_args, meta_op=meta_op)


# COPY_OUT
# L1_COPY_OUT_DMA
def _gen_op_assemble_out(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 1, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di0_shape = di_list[0]['shape']
    di0_orishape = di_list[0]['ori_shape']
    do0_name = do_list[0]['t_name']
    do0_shape = do_list[0]['shape']
    do0_offset = do_list[0]['offset']
    attr_size = meta_op['attr']['axis'][2]
    offset = []
    for i in range(attr_size):
        offset.append(gen_symbol_scalar_expr(meta_op['attr']['axis'][4 + 2 * i]))
    view_off = offset
    view_slice_args = []
    for off, shape in zip(view_off, di0_orishape):
        view_slice_args.append(f'{off}:{off + shape}')
    api_args = '[' + ','.join(view_slice_args) + ']'

    if (get_gen_conf('check_mode') == 5):
        assert (meta_op['out_param_loc'] is not None)
        rt = do_list[0]['rawtensor']
        rt_leaf_name = f"{rt['rt_name']}_subgraph{meta_op['subgraphid']}"
        if di_list[0]['mem_type']['asis'] == 4:
            code_lines += gen_build_code_lines(f"{di0_name} = {di0_name}.to({get_dtype(rt['datatype'])})")
        param_loc = meta_op['out_param_loc'][0]
        do0_name = f'{do0_name}_{param_loc}'
        api_ori = [f'0:{x}' for x in  di0_orishape]
        fmt_ori = ','.join(api_ori)
        code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}[{fmt_ori}]')
        if rt_leaf_name not in meta_op['meta_fn']['state_']['data_dump_done_set']:
            meta_op['meta_fn']['state_']['data_dump_done_set'].add(rt_leaf_name)
            code_str = f"{rt_leaf_name} = torch.zeros({rt['rawshape']})"
            code_str += f".to({get_dtype(rt['datatype'])}).to('{get_gen_conf('device_str')}')"
            code_lines += gen_build_code_lines(code_str)
        code_lines += gen_build_code_lines(f"{rt_leaf_name}{api_args} = {do0_name}")
        return code_lines
    orishape_slice_args = []
    for shape in di0_orishape:
        orishape_slice_args.append(f'0:{shape}')
    orishape_api_args = '[' + ','.join(orishape_slice_args) + ']'

    view_slice_args2 = []
    for off, shape in zip(do0_offset, do0_shape):
        view_slice_args2.append(f'{off}:{off + shape}')
    api_args2 = '[' + ','.join(view_slice_args2) + ']'

    view_slice_args3 = []
    for shape in do0_shape:
        view_slice_args3.append(f'0:{shape}')
    api_args3 = '[' + ','.join(view_slice_args3) + ']'

    if do_list[0]['rawtensor']['rt_name'] not in meta_op['meta_fn']['state_']['data_dump_done_set']:
        meta_op['meta_fn']['state_']['data_dump_done_set'].add(do_list[0]['rawtensor']['rt_name'])
        t_shape_fmt = ','.join([f'{i}' for i in do_list[0]['rawtensor']['rawshape']])
        code_str = f"{do_list[0]['rawtensor']['rt_name']} = torch.zeros({t_shape_fmt})"
        code_str += f".to({get_dtype(di_list[0]['rawtensor']['datatype'])}).to('{get_gen_conf('device_str')}')"
        code_lines += gen_build_code_lines(code_str)

    if 'op_attr' in meta_op and meta_op['op_attr'].get('op_attr_atomic_add') == 1:
        if do0_name not in meta_op['meta_fn']['state_']['data_dump_done_set']:
            meta_op['meta_fn']['state_']['data_dump_done_set'].add(do0_name)
            t_shape_fmt = ','.join([f'{i}' for i in do0_shape])
            code_str = \
                f"{do0_name} = torch.zeros({t_shape_fmt}).to({get_dtype(di_list[0]['rawtensor']['datatype'])})"
            code_str += f".to('{get_gen_conf('device_str')}')"
            code_lines += gen_build_code_lines(code_str)
        code_str = f"{do_list[0]['rawtensor']['rt_name']}{api_args2} = "
        code_str += f"{do_list[0]['rawtensor']['rt_name']}{api_args2}.add({di0_name}{orishape_api_args})"
        code_lines += gen_build_code_lines(code_str)
        code_lines += gen_build_code_lines(
            f"{do0_name}{api_args3} = {do_list[0]['rawtensor']['rt_name']}{api_args2}")
        return code_lines

    code_lines += gen_build_code_lines(
        f"{do_list[0]['rawtensor']['rt_name']}{api_args} = {di0_name}{orishape_api_args}")
    if do0_name not in meta_op['meta_fn']['state_']['data_dump_done_set']:
        meta_op['meta_fn']['state_']['data_dump_done_set'].add(do0_name)
        t_shape_fmt = ','.join([f'{i}' for i in do0_shape])
        code_str = f"{do0_name} = torch.zeros({t_shape_fmt}).to({get_dtype(di_list[0]['rawtensor']['datatype'])})"
        code_str += f".to('{get_gen_conf('device_str')}')"
        code_lines += gen_build_code_lines(code_str)
    if di0_name not in meta_op['meta_fn']['state_']['data_dump_done_set']:
        meta_op['meta_fn']['state_']['data_dump_done_set'].add(di0_name)
        t_shape_fmt = ','.join([f'{i}' for i in di0_shape])
        code_str = f"{di0_name} = torch.zeros({t_shape_fmt}).to({get_dtype(di_list[0]['rawtensor']['datatype'])})"
        code_str += f".to('{get_gen_conf('device_str')}')"
        code_lines += gen_build_code_lines(code_str)
    code_lines += gen_build_code_lines(f"{do0_name}{api_args3} = {do_list[0]['rawtensor']['rt_name']}{api_args2}")
    return code_lines


# COPY_IN
def _gen_op_copy_in(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 1, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di0_shape = di_list[0]['shape']
    do0_name = do_list[0]['t_name']
    do0_shape = do_list[0]['shape']
    do0_ori_shape = do_list[0]['ori_shape']
    if (get_gen_conf('check_mode') == 5):
        assert(meta_op['in_param_loc'] is not None)
        param_loc = meta_op['in_param_loc'][0]
        di0_name = f'{di0_name}_{param_loc}'
        api_args = '[' + ','.join(f'0:{x}' for x in do0_ori_shape) + ']'
        rt = di_list[0]['rawtensor']
        rt_leaf_name = f"{rt['rt_name']}_subgraph{meta_op['subgraphid']}"
        if rt_leaf_name in meta_op['meta_fn']['state_']['data_dump_done_set']:
            view_slice_args1 = []
            for off, shape in zip(di_list[0]['offset'], di_list[0]['shape']):
                view_slice_args1.append(f'{off}:{off + shape}')
            api_args1 = '[' + ','.join(view_slice_args1) + ']'
            code_lines += gen_build_code_lines(f"{di0_name} = {rt_leaf_name}{api_args1}")
        code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}{api_args}')
        return code_lines
    attr_size = meta_op['attr']['axis'][2]
    offset = []
    for i in range(attr_size):
        offset.append(gen_symbol_scalar_expr(meta_op['attr']['axis'][4 + 2 * i]))
    view_off = [a - b for a, b in zip(offset, di_list[0]['offset'])]
    view_slice_args = []
    for off, shape in zip(view_off, do0_ori_shape):
        view_slice_args.append(f'{off}:{off + shape}')
    if di0_name not in meta_op['meta_fn']['state_']['data_dump_done_set'] :
        meta_op['meta_fn']['state_']['data_dump_done_set'].add(di0_name)
        t_shape_fmt = ','.join([f'{i}' for i in di0_shape])
        code_str = f"{di0_name} = torch.randn({t_shape_fmt}).to({get_dtype(di_list[0]['rawtensor']['datatype'])})"
        code_str += f".to('{get_gen_conf('device_str')}')"
        code_lines += gen_build_code_lines(code_str)
    api_args = '[' + ','.join(view_slice_args) + ']'
    code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}{api_args}')
    return code_lines


# DIV
def _gen_op_div(meta_op=None):
    api_name = f'div'
    return __gen_op_binary(api_name, meta_op)


# DIVS
def _gen_op_divs(meta_op=None):
    api_name = f'div'
    return __gen_op_binary_scalar(api_name, meta_op)


# EXPAND
def _gen_op_expand(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 1, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di0_shape = di_list[0]['shape']
    do0_name = do_list[0]['t_name']
    do0_shape = do_list[0]['shape']
    key_dims = []
    error_checker(len(di0_shape) == len(do0_shape))
    for i, v in enumerate(di0_shape):
        key_dims.append(do0_shape[i] // v)
    t_shape_fmt = ','.join([f'{i}' for i in key_dims])
    code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}.repeat({t_shape_fmt})')
    return code_lines


# EXP
def _gen_op_exp(meta_op=None):
    api_name = f'exp'
    return __gen_op_unary(api_name, meta_op=meta_op)


# EXTRACT
def _gen_op_extract(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    di0_name = di_list[0]['t_name']
    do0_name = do_list[0]['t_name']
    is_largest = meta_op['op_attr']['op_attr_order']
    mod = meta_op['op_attr']['op_attr_makeMode']
    code_lines += gen_build_code_lines(f'{do0_name} = extract({di0_name}, {mod}, {is_largest})')
    return code_lines


# GATHER_ELEMENT
def _gen_op_gather_element(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 2, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di1_name = di_list[1]['t_name']
    do0_name = do_list[0]['t_name']
    axis = meta_op['attr']['axis'][0]
    code_lines += gen_build_code_lines(f'{do0_name} = torch.gather({di0_name},{axis},{di1_name})')
    return code_lines


# GATHER
def _gen_op_gather_v2(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 2, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di1_name = di_list[1]['t_name']
    do0_name = do_list[0]['t_name']
    axis = meta_op['attr']['axis'][0]
    code_str = f"{do0_name} = gather_v2({di0_name}, {di1_name}, {di_list[1]['ori_shape']}, {axis})"
    code_lines += gen_build_code_lines(code_str)
    return code_lines


# INDEX_PUT
def _gen_code_index_put(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 2, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di1_name = di_list[1]['t_name']
    do0_name = do_list[0]['t_name']
    code_lines += gen_build_code_lines(f'{do0_name}[{di0_name}] = {di1_name}')
    return code_lines


# INDEX_OUTCAST  规避方案
def _gen_op_scatter_update1(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 2, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di1_name = di_list[1]['t_name']
    do0_name = do_list[0]['t_name']
    do0_symbol = do_list[0]['rawtensor']['symbol']
    di1_ori_shape = di_list[1]['ori_shape']
    dst_symbol = do0_symbol
    dst_shape = do_list[0]['shape']
    scatter_dst_shape = do_list[0]['shape']
    dst_offset = [0 for _ in range(len(do_list[0]['shape']))]
    disable_assemble = True
    view_slice_args = []
    for shape in di1_ori_shape:
        view_slice_args.append(f'0:{shape}')
    api_args = '[' + ','.join(view_slice_args) + ']'
    if do_list[0]['magic'] in meta_op['meta_fn']['consumers_mapping']:
        next_op_tuple = meta_op['meta_fn']['consumers_mapping'][do_list[0]['magic']]
        next_op = meta_op['meta_fn']['meta_op_dict'].get(next_op_tuple[0])
        next_opcode = next_op['opcode']
        if next_opcode != 'ASSEMBLE' and len(meta_op['attr']['axis']) > 1:
            size = meta_op['attr']['axis'][2]
            scatter_dst_shape = meta_op['attr']['axis'][3 + 2 * size:]
            for i, v in enumerate(dst_offset):
                dst_offset[i] = (meta_op['attr']['axis'][4 + 2 * i])
        while next_opcode == 'ASSEMBLE':
            disable_assemble = False
            next_symbol = next_op['data_o_list'][0]['rawtensor']['symbol']
            dst_symbol = next_symbol
            for i, v in enumerate(dst_offset):
                dst_offset[i] += next_op['attr']['fromOffset'][i]
            dst_shape = next_op['data_o_list'][0]['shape']
            if next_op['data_o_list'][0]['magic'] in next_op['meta_fn']['consumers_mapping']:
                next_next_op_tuple = next_op['meta_fn']['consumers_mapping'][next_op['data_o_list'][0]['magic']]
                next_next_op = next_op['meta_fn']['meta_op_dict'].get(next_next_op_tuple[0])
                next_op = next_next_op
                next_opcode = next_op['opcode']
            else:
                next_opcode = ''
    else:
        size = meta_op['attr']['axis'][2]
        scatter_dst_shape = meta_op['attr']['axis'][3 + 2 * size:]
        for i, v in enumerate(dst_offset):
            dst_offset[i] = (meta_op['attr']['axis'][4 + 2 * i])
    dst_symbol_name = f't_{dst_symbol}'
    if dst_symbol_name not in meta_op['meta_fn']['state_']['data_dump_done_set']:
        t_shape_dst = ','.join([f'{i}' for i in dst_shape])
        code_str = f"hash_value_{dst_symbol_name} = hashlib.md5('{dst_symbol_name}'.encode()).hexdigest()"
        code_lines += gen_build_code_lines(code_str)
        code_lines += gen_build_code_lines(f"seed_{dst_symbol_name}= int(hash_value_{dst_symbol_name}[:8], 16)")
        code_lines += gen_build_code_lines(f"gen_scatter_update_{dst_symbol_name} = torch.Generator()")
        code_lines += gen_build_code_lines(f"gen_scatter_update_{dst_symbol_name}.manual_seed(seed_{dst_symbol_name})")
        d0_dtype = get_dtype(do_list[0]['rawtensor']['datatype'])
        code_str = f"{dst_symbol_name} = torch.randn({t_shape_dst}, dtype = {d0_dtype}, "
        code_str += f"generator = gen_scatter_update_{dst_symbol_name})"
        code_lines += gen_build_code_lines(code_str)
        meta_op['meta_fn']['state_']['data_dump_done_set'].add(dst_symbol_name)
    loc_list = []
    for i, v in enumerate(dst_offset):
        r_boundary = dst_offset[i] + scatter_dst_shape[i]
        loc_list.append(f'{dst_offset[i]}:{r_boundary}')
    loc = ','.join(loc_list)
    if disable_assemble:
        if do0_name not in meta_op['meta_fn']['state_']['data_dump_done_set']:
            code_lines += gen_build_code_lines(f'{do0_name} = {dst_symbol_name}')
            meta_op['meta_fn']['state_']['data_dump_done_set'].add(do0_name)
        code_lines += gen_build_code_lines(f'tmp = {do0_name}[{loc}]')
        code_lines += gen_build_code_lines(f'tmp = scatter_update(tmp, {di0_name}, {di1_name}{api_args}, {-2})')
    else:
        code_lines += gen_build_code_lines(f'{do0_name} = {dst_symbol_name}[{loc}]')
        code_str = f'{do0_name} = scatter_update({do0_name}, {di0_name}, {di1_name}{api_args}, {-2})'
        code_lines += gen_build_code_lines(code_str)
    return code_lines


# INDEX_OUTCAST 正式方案
def _gen_op_scatter_update(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 3, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di1_name = di_list[1]['t_name']
    di2_name = di_list[2]['t_name']
    do0_name = do_list[0]['t_name']
    di1_ori_shape = di_list[1]['ori_shape']
    do0_shape = do_list[0]['shape']
    do0_offset = do_list[0]['offset']
    dst_offset = [0 for _ in range(len(do_list[0]['shape']))]
    scatter_dst_shape = do_list[0]['shape'].copy()

    di2_size = 1
    do_size = 1
    for i in range(len(do0_shape)):
        di2_size *= di_list[2]['shape'][i]
        do_size *= do_list[0]['shape'][i]
    if len(meta_op['attr']['axis']) > 1:
        for i, _ in enumerate(dst_offset):
            dst_offset[i] = gen_symbol_scalar_expr(meta_op['attr']['axis'][4 + 2 * i]) - do0_offset[i]
        size = meta_op['attr']['axis'][2]
        for i in range(size):
            scatter_dst_shape[i] = gen_symbol_scalar_expr(meta_op['attr']['axis'][4 + size * 2 + 2 * i])
        scatter_dst_shape[-2] = do0_shape[-2] # -2轴不能切

    view_slice_args = []
    for shape in di1_ori_shape:
        view_slice_args.append(f'0:{shape}')
    api_args = '[' + ','.join(view_slice_args) + ']'

    view_slice_args1 = []
    for shape in do0_shape:
        view_slice_args1.append(f'{0}:{0 + shape}')
    api_args1 = '[' + ','.join(view_slice_args1) + ']'

    loc_list = []
    for i, _ in enumerate(dst_offset):
        r_boundary = dst_offset[i] + scatter_dst_shape[i]
        loc_list.append(f'{dst_offset[i]}:{r_boundary}')
    loc = ','.join(loc_list)
    if (get_gen_conf('check_mode') == 5):
        assert (meta_op['out_param_loc'] is not None)
        in_param_loc = meta_op['in_param_loc'][0]
        out_param_loc = meta_op['out_param_loc'][0]
        di2_name_str = f"{di2_name}_{in_param_loc}"
        do0_name_str = f"{do0_name}_{out_param_loc}"
        size = meta_op['attr']['axis'][2]
        output_shape = [0 for _ in range(size)]
        for i in range(size):
            output_shape[i] = gen_symbol_scalar_expr(meta_op['attr']['axis'][4 + size * 2 + 2 * i])
        view_slice_args2 = []
        for j, v in enumerate(output_shape):
            if j == len(output_shape) - 2:
                view_slice_args2.append(
                    f"0+{di1_name}[0][0]:0+{di1_name}[0][0]+{v}")
            else:
                view_slice_args2.append(f"0:{0 + v}")
        api_args2 = '[' + ','.join(view_slice_args2) + ']'
        code_lines += gen_build_code_lines(
            f"{do0_name_str}_tmp = scatter_update({di2_name_str}, {di0_name}, {di1_name}{api_args}, {-2})")
        code_lines += gen_build_code_lines(
            f"{do0_name_str} = {do0_name_str}_tmp{api_args2} ")
        return code_lines

    if do0_name not in meta_op['meta_fn']['state_']['data_dump_done_set']:
        meta_op['meta_fn']['state_']['data_dump_done_set'].add(do0_name)
        t_shape_fmt = ','.join([f'{i}' for i in do0_shape])
        code_str = f"{do0_name} = torch.zeros({t_shape_fmt}).to({get_dtype(di_list[2]['rawtensor']['datatype'])})"
        code_str += f".to('{get_gen_conf('device_str')}')"
        code_lines += gen_build_code_lines(code_str)
        code_lines += gen_build_code_lines(f"{do0_name}[{loc}] = {di2_name}")
 
    code_lines += gen_build_code_lines(
        f"{do0_name}[{loc}] = scatter_update({di2_name}, {di0_name}, {di1_name}{api_args}, {-2})")
    return code_lines


# L1_COPY_IN
# UB_COPY_IN
def _gen_op_l1_copy_in(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 1, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    do0_name = do_list[0]['t_name']
    if di_list[0]['rawtensor']['symbol'] not in meta_op['meta_fn']['symbol_dict']:
        return code_lines
    if (5 not in meta_op['meta_fn_dict']):
        rt_name = f"t{meta_op['meta_fn']['symbol_dict'][di_list[0]['rawtensor']['symbol']]}"
        if rt_name not in meta_op['meta_fn']['state_']['data_dump_done_set']:
            meta_op['meta_fn']['state_']['data_dump_done_set'].add(rt_name)
            code_lines += gen_build_code_lines(f"{di_list[0]['rawtensor']['rt_name']} = {rt_name}")
        view_slice_args = []
        for off, shape in zip(di_list[0]['offset'], di_list[0]['shape']):
            view_slice_args.append(f'{off}:{off + shape}')
        api_args = '[' + ','.join(view_slice_args) + ']'
        code_lines += gen_build_code_lines(f"{di0_name} = {di_list[0]['rawtensor']['rt_name']}{api_args}")

    api_args = '[' + ', '.join([f':{n}' for n in do_list[0]['shape']]) + ']'
    code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}{api_args}')
    return code_lines


# MUL
def _gen_op_mul(meta_op=None):
    api_name = f'mul'
    return __gen_op_binary(api_name, meta_op)


# MULS
def _gen_op_muls(meta_op=None):
    api_name = f'mul'
    return __gen_op_binary_scalar(api_name, meta_op)


# MAXIMUM
def _gen_op_maximum(meta_op=None):
    code_lines = []
    api_name = f'maximum'
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 2, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di1_name = di_list[1]['t_name']
    do0_name = do_list[0]['t_name']
    code_lines += gen_build_code_lines(f'{do0_name} = torch.{api_name}({di0_name}, {di1_name})')
    return code_lines


# MRGSORT
def _gen_op_mrgsort(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    di0_name = di_list[0]['t_name']
    do0_name = do_list[0]['t_name']
    axis = meta_op['op_attr']['op_attr_axis']
    is_largest = meta_op['op_attr']['op_attr_order']
    k = meta_op['op_attr']['op_attr_kvalue']
    code_lines += gen_build_code_lines(f'{do0_name} = mrgsort({di0_name}, {axis}, {is_largest}, {k} )')
    return code_lines


# PAIRMAX
def _gen_op_pairmax(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 2, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di1_name = di_list[1]['t_name']
    do0_name = do_list[0]['t_name']
    code_lines += gen_build_code_lines(f'{do0_name} = torch.max({di0_name},{di1_name})')
    return code_lines


# PAIRSUM
def _gen_op_pairsum(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 2, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di1_name = di_list[1]['t_name']
    do0_name = do_list[0]['t_name']
    code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}+{di1_name}')
    return code_lines


# RECIPROCAL
def _gen_op_reciprocal(meta_op=None):
    api_name = f'reciprocal'
    return __gen_op_unary(api_name, meta_op=meta_op)


def __gen_op_unary_workspace(api_name, api_args, meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 1, meta_op=meta_op)
    if meta_op['meta_fn']['functype'] == 2:
        error_checker(len(di_list) == 1, meta_op=meta_op)
    else:
        error_checker(len(do_list) == 2, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    do0_name = do_list[0]['t_name']
    args_for_max = '[0]' if api_name in ['max'] else ''
    code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}.{api_name}({api_args}){args_for_max}')
    return code_lines


# REDUCE_MAX_SINGLE
def _gen_op_reducemax_tail_dim(meta_op=None):
    api_name = f'max'
    api_args = f'dim=-1, keepdim=True'
    return __gen_op_unary_workspace(api_name, api_args, meta_op)


# REDUCE_SUM_SINGLE
def _gen_op_reducesum_tail_dim(meta_op=None):
    api_name = f'sum'
    api_args = f'dim=-1, keepdim=True'
    return __gen_op_unary_workspace(api_name, api_args, meta_op)


def __gen_op_rowexpand(api_name, api_args, dim, meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    di0_name = di_list[0]['t_name']
    do0_name = do_list[0]['t_name']
    di0_ori_shape = di_list[0]['ori_shape']
    do0_shape = do_list[0]['ori_shape']
    view_slice_args = []
    for shape in di0_ori_shape:
        view_slice_args.append(f'0:{shape}')
    api_args1 = '[' + ','.join(view_slice_args) + ']'
    api_args_mod = f'dim={dim}, keepdim=True'
    api_args_mod = f'{api_args_mod}, {api_args}' if api_args else api_args_mod
    args_for_max = '[0]' if api_name in ['max'] else ''
    code_lines += gen_build_code_lines(
        f'{do0_name} = {di0_name}{api_args1}.{api_name}({api_args_mod}){args_for_max}.expand({do0_shape})')
    return code_lines


# ROWEXPMAX
# ROWMAX_SINGLE
def _gen_op_rowexpandmax(meta_op=None):
    api_name = f'max'
    api_args = ''
    dim = -1
    if 'op_attr_AXIS' in meta_op['op_attr']:
        if meta_op['op_attr']['op_attr_AXIS'] is not None:
            dim = meta_op['op_attr']['op_attr_AXIS'] 
    return __gen_op_rowexpand(api_name, api_args, dim, meta_op)


# ROWEXPSUM
# ROWSUM_SINGLE
def _gen_op_rowexpandsum(meta_op=None):
    api_name = f'sum'
    api_args = ''
    dim = -1
    if 'op_attr_AXIS' in meta_op['op_attr']:
        if meta_op['op_attr']['op_attr_AXIS'] is not None:
            dim = meta_op['op_attr']['op_attr_AXIS'] 
    return __gen_op_rowexpand(api_name, api_args, dim, meta_op)


# RESAHPE
def _gen_op_reshape(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 1, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    do0_name = do_list[0]['t_name']
    v_shape_fmt = ','.join([f'{i}' for i in do_list[0]['ori_shape']])
    view_slice_args = []
    for shape in di_list[0]['ori_shape']:
        view_slice_args.append(f'0:{shape}')
    api_args = '[' + ','.join(view_slice_args) + ']'
    code_str_do0_rt = ''
    code_str_do0 = ''
    if (meta_op['out_param_loc'] or meta_op['in_param_loc']) is not None:
        if meta_op['out_param_loc']:
            param_loc = meta_op['out_param_loc'][0]
            do0_name = f'{do0_name}_{param_loc}'
            rt = do_list[0]['rawtensor']
            rt_leaf_name = f"{rt['rt_name']}_subgraph{meta_op['subgraphid']}"
            if rt_leaf_name not in meta_op['meta_fn']['state_']['data_dump_done_set']:
                meta_op['meta_fn']['state_']['data_dump_done_set'].add(rt_leaf_name)
                code_str_do0_rt = f"{rt_leaf_name} = torch.zeros({rt['rawshape']})"
                code_str_do0_rt += f".to({get_dtype(rt['datatype'])}).to('{get_gen_conf('device_str')}')"
            view_slice_args1 = []
            for off, shape in zip(do_list[0]['offset'], do_list[0]['shape']):
                view_slice_args1.append(f'{off}:{off + shape}')
            api_args1 = '[' + ','.join(view_slice_args1) + ']'
            code_str_do0 = f"{rt_leaf_name}{api_args1} = {do0_name}"
        if meta_op['in_param_loc']:
            param_loc = meta_op['in_param_loc'][0]
            di0_name = f'{di0_name}_{param_loc}'
            rt = di_list[0]['rawtensor']
            rt_leaf_name = f"{rt['rt_name']}_subgraph{meta_op['subgraphid']}"
            if rt_leaf_name in meta_op['meta_fn']['state_']['data_dump_done_set']:
                view_slice_args1 = []
                for off, shape in zip(di_list[0]['offset'], di_list[0]['shape']):
                    view_slice_args1.append(f'{off}:{off + shape}')
                api_args1 = '[' + ','.join(view_slice_args1) + ']'
                code_lines += gen_build_code_lines(f"{di0_name} = {rt_leaf_name}{api_args1}")
        code_lines += gen_build_code_lines(f"{do0_name} = torch.reshape({di0_name}, ({v_shape_fmt}))")
        code_lines += gen_build_code_lines(code_str_do0_rt)
        code_lines += gen_build_code_lines(code_str_do0)
        return code_lines
    if do_list[0]['magic'] in meta_op['meta_fn']['consumers_mapping']:
        code_lines += gen_build_code_lines(f"{do0_name} = torch.reshape({di0_name}{api_args}, ({v_shape_fmt}))")
    else:
        code_lines += gen_build_code_lines(f"{do0_name} = torch.reshape({di0_name}, ({v_shape_fmt}))")
    return code_lines


# ROWSUMLINE
def _gen_op_rowsumline(meta_op=None):
    api_name = f'sum'
    api_args = ''
    dim = meta_op['attr']['axis'][0]
    return __gen_op_rowexpand(api_name, api_args, dim, meta_op)


# REDUCE_ACC
def _gen_op_reduce_acc(meta_op=None):
    code_lines = []
    do_list = meta_op['data_o_list']
    di_list = meta_op['data_i_list']
    do0_name = do_list[0]['t_name']
    do0_shape = do_list[0]['shape']
    if do0_name not in meta_op['meta_fn']['state_']['data_dump_done_set']:
        meta_op['meta_fn']['state_']['data_dump_done_set'].add(do0_name)
        t_shape_fmt = ','.join([f'{i}' for i in do0_shape])
        code_str = f"{do0_name} = torch.zeros({t_shape_fmt}).to({get_dtype(di_list[0]['rawtensor']['datatype'])})"
        code_str += f".to('{get_gen_conf('device_str')}')"
        code_lines += gen_build_code_lines(code_str)
    api_name = f'add'
    for _, di in enumerate(di_list):
        di_name = di['t_name']
        code_lines += gen_build_code_lines(f'{do0_name} = {do0_name}.{api_name}({di_name})')
    code_lines += gen_build_code_lines(f"{do0_name} = {do0_name}.to({get_dtype(do_list[0]['rawtensor']['datatype'])})")
    return code_lines


# SUB
def _gen_op_sub(meta_op=None):
    api_name = f'sub'
    return __gen_op_binary(api_name, meta_op)


# SUBS
def _gen_op_subs(meta_op=None):
    api_name = f'sub'
    return __gen_op_binary_scalar(api_name, meta_op)


# SQRT
def _gen_op_sqrt(meta_op=None):
    api_name = f'sqrt'
    return __gen_op_unary(api_name, meta_op=meta_op)


# S_ADDS
def _gen_op_s_adds(meta_op=None):
    api_name = f'add'
    return __gen_op_binary_scalar(api_name, meta_op)


# S_DIVS
def _gen_op_s_divs(meta_op=None):
    api_name = f'div'
    return __gen_op_binary_scalar(api_name, meta_op)


# S_MULS
def _gen_op_s_muls(meta_op=None):
    api_name = f'mul'
    return __gen_op_binary_scalar(api_name, meta_op)


# S_SUBS
def _gen_op_s_subs(meta_op=None):
    api_name = f'sub'
    return __gen_op_binary_scalar(api_name, meta_op)


def __gen_op_s_maxmins(api_name, meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 1, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    do0_name = do_list[0]['t_name']
    data_type = meta_op['attr']['scalarDataType']
    if data_type == 0:
        di1_name = meta_op['attr']['scalarDataValue']
    else:
        di1_name = float_value = struct.unpack('f', struct.pack('I', meta_op['attr']['scalarDataValue']))[0]
    code_lines += gen_build_code_lines(f'{do0_name} = torch.{api_name}({di0_name},torch.tensor({di1_name}))')
    return code_lines


# S_MAXS
def _gen_op_s_maxs(meta_op=None):
    api_name = f'max'
    return __gen_op_s_maxmins(api_name, meta_op)


# S_MINS
def _gen_op_s_mins(meta_op=None):
    api_name = f'min'
    return __gen_op_s_maxmins(api_name, meta_op)


# SCATTER_ELEMENT
def _gen_op_scatter_element(meta_op=None):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    error_checker(len(di_list) == 2, meta_op=meta_op)
    error_checker(len(do_list) == 1, meta_op=meta_op)
    di0_name = di_list[0]['t_name']
    di1_name = di_list[1]['t_name']
    do0_name = do_list[0]['t_name']
    axis = meta_op['attr']['scalarAxis']
    data_type = meta_op['attr']['scalarDataType']
    if data_type == 0:
        di_value = meta_op['attr']['scalarDataValue']
    else:
        di_value = float_value = struct.unpack('f', struct.pack('I', meta_op['attr']['scalarDataValue']))[0]
    code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}.scatter_({axis}, {di1_name}, {di_value})')
    return code_lines


# TRANSPOSE_MOVEOUT
def _gen_op_transpose_datamove(meta_op=None):
    return __gen_op_transpose(meta_op, True)


# TRANSPOSE_VNCHWCONV
def _gen_op_transpose_vnchwconv(meta_op=None):
    return __gen_op_transpose(meta_op, False)


def __gen_op_transpose(meta_op=None, is_datamove=False):
    code_lines = []
    di_list = meta_op['data_i_list']
    do_list = meta_op['data_o_list']
    di_idx = 0 if len(di_list[0]['producers']) != 0 else 1
    di0_name = di_list[di_idx]['t_name']
    do0_name = do_list[0]['t_name']
    do0_shape = do_list[0]['shape']
    transpose_axis = meta_op['op_attr']['op_attr_shape']
    di0_t = f'.transpose({transpose_axis[0]}, {transpose_axis[1]})'
    dst_offset = [0 for _ in range(len(do_list[0]['shape']))]
    di_size = 1
    do_size = 1
    di_ori_size = 1
    do_ori_size = 1
    for i in range(len(do0_shape)):
        di_ori_size *= di_list[0]['ori_shape'][i]
        do_ori_size *= do_list[0]['ori_shape'][i]
        di_size *= di_list[0]['shape'][i]
        do_size *= do_list[0]['shape'][i]
    if (get_gen_conf('check_mode') == 5 and is_datamove is True):
        assert (meta_op['out_param_loc'] is not None)
        param_loc = meta_op['out_param_loc'][0]
        do0_name = f'{do0_name}_{param_loc}'
        code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}{di0_t}')
        rt = do_list[0]['rawtensor']
        rt_leaf_name = f"{rt['rt_name']}_subgraph{meta_op['subgraphid']}"
        if rt_leaf_name not in meta_op['meta_fn']['state_']['data_dump_done_set']:
            meta_op['meta_fn']['state_']['data_dump_done_set'].add(rt_leaf_name)
            code_str = f"{rt_leaf_name} = torch.zeros({rt['rawshape']})"
            code_str += f".to({get_dtype(rt['datatype'])}).to('{get_gen_conf('device_str')}')"
            code_lines += gen_build_code_lines(code_str)
        do_shape = do_list[0]['shape'].copy()
        do_offset = do_list[0]['offset'].copy()
        if di_ori_size != do_ori_size:
            offset_size = meta_op['attr']['axis'][2]
            for i in range(offset_size):
                do_shape[i] = gen_symbol_scalar_expr(meta_op['attr']['axis'][4 + offset_size * 2 + 2 * i])
            for i, _ in enumerate(do_offset):
                do_offset[i] = gen_symbol_scalar_expr(meta_op['attr']['axis'][4 + 2 * i])
        view_slice_args1 = []
        for off, shape in zip(do_offset, do_shape):
            view_slice_args1.append(f'{off}:{off + shape}')
        api_args1 = '[' + ','.join(view_slice_args1) + ']'
        code_lines += gen_build_code_lines(f"{rt_leaf_name}{api_args1} = {do0_name}")
        return code_lines

    do0_offset = do_list[0]['offset']
    if di_ori_size != do_ori_size:
        for i, v in enumerate(dst_offset):
            dst_offset[i] = gen_symbol_scalar_expr(meta_op['attr']['axis'][4 + 2 * i]) - do0_offset[i]

    if do0_name not in meta_op['meta_fn']['state_']['data_dump_done_set']:
        meta_op['meta_fn']['state_']['data_dump_done_set'].add(do0_name)
        t_shape_fmt = ','.join([f'{i}' for i in do0_shape])
        code_str = f"{do0_name} = torch.zeros({t_shape_fmt}).to({get_dtype(di_list[0]['rawtensor']['datatype'])})"
        code_str += f".to('{get_gen_conf('device_str')}')"
        code_lines += gen_build_code_lines(code_str)
    loc_list = []
    dst_shape = di_list[0]['shape']
    tmp = dst_shape[transpose_axis[0]]
    dst_shape[transpose_axis[0]] = dst_shape[transpose_axis[1]]
    dst_shape[transpose_axis[1]] = tmp
    for i, v in enumerate(dst_offset):
        r_boundary = dst_offset[i] + dst_shape[i]
        loc_list.append(f'{dst_offset[i]}:{r_boundary}')
    loc = ','.join(loc_list)
    if (do0_shape == di_list[0]['shape']):
        code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}{di0_t}')
    elif (di_size != di_ori_size and di_ori_size == do_ori_size):
        view_slice_args = []
        for shape in do_list[0]['ori_shape']:
            view_slice_args.append(f'0:{shape}')
        api_args = '[' + ','.join(view_slice_args) + ']'
        code_lines += gen_build_code_lines(f'{do0_name} = {di0_name}{di0_t}{api_args}')
    elif (di_size != di_ori_size and di_ori_size != do_ori_size):
        view_slice_args = []
        for shape in do_list[0]['ori_shape']:
            view_slice_args.append(f'0:{shape}')
        api_args = '[' + ','.join(view_slice_args) + ']'
        code_lines += gen_build_code_lines(f'{do0_name}[{loc}] = {di0_name}{di0_t}{api_args}')
    else:
        code_lines += gen_build_code_lines(f'{do0_name}[{loc}] = {di0_name}{di0_t}')
    return code_lines


# VECTORMAX
def _gen_op_vectormax(meta_op=None):
    api_name = f'maximum'
    return __gen_op_binary(api_name, meta_op)


# VEC_DUP
def _gen_code_vec_dup(meta_op=None):
    code_lines = []
    do_list = meta_op['data_o_list']
    error_checker(len(do_list) == 1, meta_op=meta_op)
    do0_name = do_list[0]['t_name']
    shape = []
    for v in do_list[0]['shape']:
        shape.append(f'{v}')
    do0_shape = '(' + ','.join(shape) + ')'
    data_type = meta_op['attr']['scalarDataType']
    if data_type == 0:
        di1_name = meta_op['attr']['scalarDataValue']
    else:
        di1_name = struct.unpack('f', struct.pack('I', meta_op['attr']['scalarDataValue']))[0]
    code_lines += gen_build_code_lines(f'{do0_name} = torch.full({do0_shape}, {di1_name})')
    return code_lines


_gen_op_code_dict = {
    None: __gen_op_todo_,
    'ADD': _gen_op_add,
    'ADD_BRC': _gen_op_add,
    'ADDS': _gen_op_adds,
    'ASSEMBLE': _gen_op_assemble,
    'A_MUL_Bt': _gen_op_matmul_abt,
    'A_MUL_Bt_FA2': _gen_op_matmul_abt,
    'A_MUL_B': _gen_op_matmul_ab,
    'A_MULACC_B': _gen_op_matmul_acc_ab,
    'A_MULACC_Bt': _gen_op_matmul_acc_abt,
    'ABS': _gen_op_abs,

    'BT_ALLOC': _gen_op_none,
    'BAR_V': _gen_op_none,
    'BAR_M': _gen_op_none,
    'BT_COPY_IN': _gen_op_view,
    'BITSORT': _gen_op_bitsort,

    'CALL': __gen_op_call_,
    'CAST': _gen_op_cast,
    'CONVERT': _gen_op_convert,
    'COPY_OUT': _gen_op_assemble_out,
    'COPY_IN': _gen_op_copy_in,
    'CV_SYNC_SRC': _gen_op_none,
    'CV_SYNC_DST': _gen_op_none,
    'COMM_WAIT_FLAG': _gen_op_none,

    'DIV': _gen_op_div,
    'DIV_BRC': _gen_op_div,
    'DIVS': _gen_op_divs,

    'EXPAND': _gen_op_expand,
    'EXP': _gen_op_exp,
    'EXTRACT': _gen_op_extract,

    'FIX_ALLOC': _gen_op_none,
    'FIX_COPY_IN': _gen_op_view,

    'GATHER_ELEMENT': _gen_op_gather_element,
    'GATHER': _gen_op_gather_v2,

    'INDEX_PUT': _gen_code_index_put,
    'INDEX_OUTCAST': _gen_op_scatter_update,

    'L0A_ALLOC': _gen_op_none,
    'L0B_ALLOC': _gen_op_none,
    'L0C_ALLOC': _gen_op_none,
    'L0C_COPY_OUT': _gen_op_view,
    'L1_ALLOC': _gen_op_none,
    'L1_COPY_IN': _gen_op_l1_copy_in,
    'L1_TO_L0A': _gen_op_view,
    'L1_TO_L0B': _gen_op_view,
    'L1_TO_L0Bt': _gen_op_view,
    'L1_COPY_IN_DMA': _gen_op_view,
    'L1_COPY_OUT_DMA': _gen_op_assemble_out,

    'MUL': _gen_op_mul,
    'MUL_BRC': _gen_op_mul,
    'MULS': _gen_op_muls,
    'MAXIMUM': _gen_op_maximum,
    'MRGSORT': _gen_op_mrgsort,

    'NOP': _gen_op_reshape,

    'PAIRMAX': _gen_op_pairmax,
    'PAIRSUM': _gen_op_pairsum,
    'PHASE1': _gen_op_none,
    'PHASE2': _gen_op_none,

    'RECIPROCAL': _gen_op_reciprocal,
    'REDUCE_MAX_SINGLE': _gen_op_reducemax_tail_dim,
    'ROWMAX_COMBINE_AXIS_SINGLE': _gen_op_reducemax_tail_dim,
    'REDUCE_SUM_SINGLE': _gen_op_reducesum_tail_dim,
    'ROWSUM_COMBINE_AXIS_SINGLE': _gen_op_reducesum_tail_dim,
    'REGISTER_COPY': _gen_op_view,
    'ROWEXPMAX': _gen_op_rowexpandmax,
    'ROWEXPSUM': _gen_op_rowexpandsum,
    'RESHAPE': _gen_op_reshape,
    'ROWSUM_SINGLE': _gen_op_rowexpandsum,
    'ROWMAX_SINGLE': _gen_op_rowexpandmax,
    'ROWSUMLINE': _gen_op_rowsumline,
    'REDUCE_ACC': _gen_op_reduce_acc,

    'SUB': _gen_op_sub,
    'SUB_BRC': _gen_op_sub,
    'SUBS': _gen_op_subs,
    'SQRT': _gen_op_sqrt,
    'S_ADD': _gen_op_add,
    'S_ADDS': _gen_op_s_adds,
    'S_DIV': _gen_op_div,
    'S_DIVS': _gen_op_s_divs,
    'S_MUL': _gen_op_mul,
    'S_MULS': _gen_op_s_muls,
    'S_SUB': _gen_op_sub,
    'S_SUBS': _gen_op_s_subs,
    'S_MAXS': _gen_op_s_maxs,
    'S_MINS': _gen_op_s_mins,
    'SYNC_SRC': _gen_op_none,
    'SYNC_DST': _gen_op_none,
    'SCATTER_ELEMENT': _gen_op_scatter_element,

    'TRANSPOSE_MOVEOUT': _gen_op_transpose_datamove,
    'TRANSPOSE_VNCHWCONV': _gen_op_transpose_vnchwconv,

    'UB_ALLOC': _gen_op_none,
    'UB_COPY_IN': _gen_op_l1_copy_in,
    'UB_COPY_OUT': _gen_op_view,
    'UB_TO_UB': _gen_op_view,

    'VECTORMAX': _gen_op_vectormax,
    'VIEW': _gen_op_view,
    'VEC_DUP': _gen_code_vec_dup,
}


def get_op_func(op_code):
    return _gen_op_code_dict.get(op_code, __gen_op_todo_)