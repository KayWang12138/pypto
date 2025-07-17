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
import os
import sys
import json
from pprint import pprint
from ast_code_gen.common import init_gen_conf, get_functype
from ast_code_gen.gen_main_code import gen_def_code, gen_main_code


_checkmode_dict = {
    "PrintShape": -2,
    "DumpCopysOp": -1,
    "DumpAllOp": 0,
    "RemoveRedundentReshape": 1,
    "ExpandFunction": 1,
    "DuplicateView": 1,
    "MergeViewAssemble": 1,
    "AssignMemoryType": 1,
    "InsertConvertOp": 1,
    "SplitLargeFanoutTensor": 1,
    "SplitReshapeOpPVC2": 1,
    "RemoveRedundentOp": 1,
    "GenerateMoveOp": 1,
    "GraphInitPass": 1,
    "PartitionVCPass": 1,
    "GraphPartitionPass": 1,
    "SplitLargeLocalRawPass": 1,
    "InsertCopyOpPass": 1,
    "CommonOperationEliminate": 1,
    "PreGraphPass": 1,
    "PadLocalBuffer": 1,
    "SubgraphToFunction": 2,
    "SrcDstBufferMergePVC2Pass": 2,
    "AddAllocNewPass": 2,
    "OoOSchedulePVC2Pass": 2,
    "RemoveAllocPass": 2,
    "InsertSyncNewPass": 2,
    "CodegenPreprocPass": 2,
    "program": 5,
}


def parse_rawtensors(rt):
    mete_rt = {
        'actual_rawmagic': rt.get('actual_rawmagic', -1),
        'datatype': rt['datatype'],
        'rawmagic': rt['rawmagic'],
        'rawshape': rt['rawshape'],
        'symbol': rt.get('symbol', 'None'),
    }
    return mete_rt


def parse_tensors(t, meta_rt):
    meta_t = {
        'format': t['format'],
        'magic': t['magic'],
        'offset': t['offset'],
        'shape': t['shape'],
        'ori_shape': t.get('validshape'),
        'nodetype': t['nodetype'],
        'rawtensor': meta_rt.get(t['rawtensor']),
    }
    if 'mem_type' in t:
        meta_t['mem_type'] = t['mem_type']
    return meta_t


def generate_mapping(fn, meta_fn):
    producers_mapping = {}
    consumers_mapping = {}
    for op in fn['operations']:
        for oop in op['ooperands']:
            tmp_list = producers_mapping.get(oop, [])
            tmp_list.append((fn['hash'], op['opmagic']))
            producers_mapping[oop] = [tuple(i) for i in tmp_list]
        for iop in op['ioperands']:
            tmp_list = consumers_mapping.get(iop, [])
            tmp_list.append((fn['hash'], op['opmagic']))
            consumers_mapping[iop] = [tuple(i) for i in tmp_list]
    meta_fn['producers_mapping'] = producers_mapping
    meta_fn['consumers_mapping'] = consumers_mapping


def parse_tensor_loc_info(meta_op, meta_g):
    if meta_op['invoke_info'] is not None:
        call_tensor_loc = {'input_loc': [], 'output_loc': [], 'input_loc_info': [], 'output_loc_info': []}
        for in_cast_it in meta_op['invoke_info']['incast_params']:
            call_tensor_loc['input_loc'].append(in_cast_it['param_loc'])
            call_tensor_loc['input_loc_info'].append(in_cast_it)
        for out_cast_it in meta_op['invoke_info']['outcast_params']:
            call_tensor_loc['output_loc'].append(out_cast_it['param_loc'])
            call_tensor_loc['output_loc_info'].append(out_cast_it)
        for tensor_param in meta_op['invoke_info']['tensor_params']:
            is_output = tensor_param.get('is_output')
            if is_output:
                call_tensor_loc['output_loc'].append(tensor_param['param_loc'])
                call_tensor_loc['output_loc_info'].append(tensor_param)
            else:
                call_tensor_loc['input_loc'].append(tensor_param['param_loc'])
                call_tensor_loc['input_loc_info'].append(tensor_param)
        meta_op['tensor_loc'] = {
            'input_loc_info': call_tensor_loc['input_loc_info'],
            'output_loc_info': call_tensor_loc['output_loc_info'],
        }
        if meta_op['calleehash'] not in meta_g['tensor_loc']:
            meta_g['tensor_loc'][meta_op['calleehash']] = call_tensor_loc


def parse_copy_out_acc_func_info(meta_op, meta_g):
    if meta_op['opcode'] == 'COPY_OUT' and 'op_attr' in meta_op and meta_op['op_attr'].get('op_attr_atomic_add') == 1:
        meta_g['copy_out_acc_func'].add(meta_op['meta_fn']['hash'])


def generate_data_info_for_call(op, meta_op, meta_g):
    tensor_loc_list = [[], []]
    tensor_loc = dict()
    if (meta_op['meta_fn']['functype'] == 4):
        tensor_loc = meta_g['tensor_loc'][meta_op['calleehash']]
        tensor_loc_list = [tensor_loc['input_loc'], tensor_loc['output_loc']]
    for k, k_list in zip(['ioperands', 'ooperands'], tensor_loc_list):
        for i in range(len(op[k])):
            t_magic = op[k][i]
            if (len(k_list) == 0):
                meta_op['data_i_o_magic_dict'][k].append(f't{t_magic}')
            else:
                meta_op['data_i_o_magic_dict'][k].append(f"t{t_magic}_{k_list[i]}")
    ooperands_info = meta_op['data_i_o_magic_dict']['ooperands']
    ioperands_info = meta_op['data_i_o_magic_dict']['ioperands']
    if tensor_loc:
        assert meta_g['meta_fn_dict'].get(3)
        tile_graph_meta_op_dict = list(meta_g['meta_fn_dict'][3].values())[0]['meta_op_dict']
        tile_graph_funchash = list(meta_g['meta_fn_dict'][3].keys())[0]
        meta_op['scatter_update_loc'] = {
            'input_loc': set(),
            'index_loc': {},
            'output_loc': set(),
        }
        meta_op['scatter_update_out_loc'] = set()
        for i in range(len(tensor_loc['output_loc_info'])):
            op_key = (tile_graph_funchash, tensor_loc['output_loc_info'][i]['op_magic'])
            if (tile_graph_meta_op_dict[op_key]['opcode'] == 'INDEX_OUTCAST'):
                meta_op['scatter_update_loc']['output_loc'].add(ooperands_info[i])

                for j in range(len(tensor_loc['input_loc_info'])):
                    op_j_key = (tile_graph_funchash, tensor_loc['input_loc_info'][j]['op_magic'])
                    index_magic = tile_graph_meta_op_dict[op_j_key]['data_o_list'][0]['magic']
                    if index_magic == tile_graph_meta_op_dict[op_key]['data_i_list'][1]['magic']:
                        meta_op['scatter_update_loc']['index_loc'][ooperands_info[i]] = ioperands_info[j]

        for i in range(len(tensor_loc['input_loc_info'])):
            op_key = (tile_graph_funchash, tensor_loc['input_loc_info'][i]['op_magic'])
            if (tile_graph_meta_op_dict[op_key]['opcode'] == 'INDEX_OUTCAST'):
                meta_op['scatter_update_loc']['input_loc'].add(ioperands_info[i])


def parse_operation(op, meta_fn, meta_g):
    data_i_list = []
    data_o_list = []
    for k, k_list in zip(['ioperands', 'ooperands'], [data_i_list, data_o_list]):
        for t_magic in op[k]:
            tensor = meta_fn['meta_t'][t_magic]
            data_dict = {
                'format': tensor['format'],
                't_name': f"t{tensor['magic']}",
                'magic': tensor['magic'],
                'nodetype': tensor['nodetype'],
                'offset': tensor['offset'],
                'ori_shape': tensor['ori_shape'],
                'shape': tensor['shape'],
                'producers': meta_fn['producers_mapping'].get(t_magic, []),
                'consumers': meta_fn['consumers_mapping'].get(t_magic, []),
                'rawtensor': {
                    'rt_name': f"rt{tensor['rawtensor']['rawmagic']}" if tensor['rawtensor'] else None,
                    'actual_rawmagic': tensor['rawtensor']['actual_rawmagic'] if tensor['rawtensor'] else None,
                    'datatype': tensor['rawtensor']['datatype'] if tensor['rawtensor'] else None,
                    'rawmagic': tensor['rawtensor']['rawmagic'] if tensor['rawtensor'] else None,
                    'rawshape': tensor['rawtensor']['rawshape'] if tensor['rawtensor'] else None,
                    'symbol': tensor['rawtensor']['symbol'] if tensor['rawtensor'] else None,
                    },
                'is_to_fp32': False
            }
            if 'mem_type' in tensor:
                data_dict['mem_type'] = tensor['mem_type']
            # Special Treatment: some ops not support fp16 in cpu
            if (meta_conf['device_str'] == 'cpu') and tensor['rawtensor'] and (tensor['rawtensor']['datatype'] == 6):
                data_dict['is_to_fp32'] = True
            k_list.append(data_dict)

    op_attr = op.get('attr', [])
    meta_op = {
        'key': (meta_fn['hash'], op['opmagic']),
        'opcode': op['opcode'],
        'opmagic': op['opmagic'],
        'data_i_list': data_i_list,
        'data_o_list': data_o_list,
        'attr': {
            'axis': op_attr,
            'scalarDataType': op_attr[0] if len(op_attr) >= 1 else None,
            'scalarDataValue': op_attr[1] if len(op_attr) >= 2 else None,
            'scalarReverse': op_attr[2] if len(op_attr) >= 3 else None,
            'scalarAxis': op_attr[3] if len(op_attr) >= 4 else None,
            },
        'op_attr': dict(),
        'program_funcmagic': op.get('program_funcmagic'),
        'calleehash': op.get('calleehash'),
        'subgraphid': op.get('subgraphid'),
        'meta_fn': meta_fn,
        'invoke_info': op.get('invoke_info'),
        'in_param_loc': op.get('in_param_loc'),
        'out_param_loc': op.get('out_param_loc'),
        'data_i_o_magic_dict': {
            'ioperands': [],
            'ooperands': [],
        },
    }

    # parse op attr
    if (op.get('op_attr')) and op['op_attr'] is not None:
        for key, value in op['op_attr'].items():
            meta_op['op_attr'][key] = value

    if (op['opcode'] in ['CALL', 'L1_COPY_IN', 'UB_COPY_IN', 'COPY_IN', 'L1_TO_L0B', 'L1_TO_L0A']):
        meta_op['meta_fn_dict'] = meta_g['meta_fn_dict']

    parse_tensor_loc_info(meta_op, meta_g)
    parse_copy_out_acc_func_info(meta_op, meta_g)

    if (op['opcode'] == 'CALL'):
        generate_data_info_for_call(op, meta_op, meta_g)
 
    return meta_op


def generate_data_info_for_def_func(fn, meta_fn, meta_g):
    leaf_tensor_loc_list = [[], []]
    if (meta_fn['functype'] == 5):
        leaf_tensor_loc = meta_g['tensor_loc'][meta_fn['hash']]
        leaf_tensor_loc_list = [leaf_tensor_loc['input_loc'], leaf_tensor_loc['output_loc']]
    for k, k_list in zip(['incasts', 'outcasts'], leaf_tensor_loc_list):
        for i in range(len(fn[k])):
            t_magic = fn[k][i][0]  # 静态图无需处理slot 先规避
            if (len(k_list) == 0):
                meta_fn['state_']['fn_i_o_magic_dict'][k].append(f't{t_magic}')
            else:
                meta_fn['state_']['fn_i_o_magic_dict'][k].append(f't{t_magic}_{k_list[i]}')


def parse_function(fn, meta_g):
    meta_fn = {
        'funcmagic': fn['funcmagic'],
        'hash': fn['hash'],
        'functype': fn['functype'],
        'rawname': fn['rawname'],
        #zzz 'incasts': fn['incasts'],
        #zzz 'outcasts': fn['outcasts'],
        'incasts': [],
        'outcasts': [],
        'meta_op_dict': {},
        'meta_rt': {},
        'meta_t': {},
        'state_': {
            'data_dump_done_set': set(),
            'workspace_or_alloc': set(),
            'op_todo_list': [],
            'fn_i_o_magic_dict': {
                'incasts': [],
                'outcasts': [],
            },
            'timeId': 0,
        },
        'semantic_label': fn.get('semantic_label'),
        'dynamic': fn.get('dynamic'),
    }
    # 静态图无需处理slot,先规避
    for incast in fn['incasts']:
        meta_fn['incasts'].append(incast[0])
    for outcast in fn['outcasts']:
        meta_fn['outcasts'].append(outcast[0])

    for rt in fn['rawtensors']:
        meta_fn['meta_rt'][rt['rawmagic']] = parse_rawtensors(rt)
    for t in fn['tensors']:
        meta_fn['meta_t'][t['magic']] = parse_tensors(t, meta_fn['meta_rt'])

    generate_mapping(fn, meta_fn)
    generate_data_info_for_def_func(fn, meta_fn, meta_g)

    meta_op_dict = meta_fn['meta_op_dict']
    op_todo_list = meta_fn['state_']['op_todo_list']
    for op in fn['operations']:
        if (fn['functype'] == 3) and (op['opcode'] == 'CALL'):
            continue
        meta_op = parse_operation(op, meta_fn, meta_g)
        op_k = meta_op['key']
        meta_op_dict[op_k] = meta_op
        if not meta_op['data_o_list']:
            continue
        if not meta_op['data_i_list']:
            if op_k not in op_todo_list:
                op_todo_list.append(op_k)
        for di in meta_op['data_i_list']:
            if len(di['producers']) == 0:
                if op_k not in op_todo_list:
                    op_todo_list.append(op_k)
    return meta_fn


def parse_graph(g):
    meta_g = {
        'version': g.get('version', '2.0'),
        'meta_fn_dict': {},
        'tensor_loc': dict(),
        'copy_out_acc_func': set(),
        'entryhash': g.get('entryhash'),
    }
    g['functions'].reverse()
    for fn in g['functions']:
        if (fn['functype'] in [4, 5]) and (meta_conf['check_mode'] not in [5]):
            continue
        meta_func = parse_function(fn, meta_g)
        meta_func['copy_out_acc_func'] = meta_g['copy_out_acc_func']
        if not meta_g['meta_fn_dict'].get(fn['functype']):
            meta_g['meta_fn_dict'][fn['functype']] = {}
        meta_g['meta_fn_dict'][fn['functype']][fn['hash']] = meta_func
    return meta_g


if __name__ == '__main__':
    meta_conf = {
        'version_gen_torch': '1.0.5',
        'device_str': 'cpu',  # cpu,
        'data_in_gen_mode': 'randn',  # file, randn
        'check_mode': 0,
        'golden_filapath': [f"{sys.argv[2]}/{sys.argv[i]}.bin"
                           for i in range(3, len(sys.argv))] if (len(sys.argv) > 3) else [],
    }

    json_file_path = sys.argv[1]
    json_file_dir = os.path.dirname(json_file_path)
    json_file_dir_mod = json_file_dir if json_file_dir else '.'
    json_file_name = os.path.basename(json_file_path).split('.')[0]
    meta_conf['code_file_path'] = f'{json_file_path}.py'
    meta_conf['dump_data_file_path'] = f'{json_file_dir_mod}/{json_file_name}_torch_data'
    meta_conf['golden_data_file_path'] = f'{json_file_dir_mod}/{json_file_name}_golden_data'

    # choose check_mode by pass name or user config
    pass_name = json_file_name.split('_')[-1]
    meta_conf['check_mode'] = _checkmode_dict.get(pass_name)
    assert meta_conf['check_mode'] != 2, f'Not support the pass check: {pass_name} please use subgraph check'
    if (len(sys.argv) == 3):
        meta_conf['check_mode'] = _checkmode_dict.get(sys.argv[2])
    init_gen_conf(meta_conf)
    meta_app = {
        'meta_gen_code_dict': {},
    }
    with open(json_file_path) as file_in:
        json_raw = json.load(file_in)
        meta_graph = parse_graph(json_raw)
        meta_app['meta_g'] = meta_graph
        gen_def_code(meta_app, is_topo_sort=False)
        gen_main_code(meta_app)

