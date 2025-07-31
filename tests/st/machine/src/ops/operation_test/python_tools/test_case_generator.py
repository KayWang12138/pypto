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

import argparse
import os
import json
from itertools import product
from typing import Dict, List, Any, Optional

import yaml


def resolve_variable_ref(value: str, config: Dict, context: Optional[Dict] = None) -> Any:
    """解析YAML中的变量引用（如$input0_dtype）"""
    if not isinstance(value, str) or not value.startswith('$'):
        return value
    
    ref = value[1:].split('_')
    if len(ref) < 2:
        return value  # 无效引用
    
    tensor_name, attr = ref[0], '_'.join(ref[1:])
    
    # 特殊处理复数形式的引用（如$input0_dtypes -> $input0_dtype）
    if attr.endswith('s') and attr[:-1] in ['dtype', 'shape']:
        attr = attr[:-1]  # 转为单数形式
    
    # 从配置中查找引用的张量
    for tensor in config.get('input_tensors', []):
        if tensor['name'] == tensor_name:
            # 对于shape/dtype，从上下文中获取实际值
            if attr in ['dtype', 'shape'] and context and tensor_name in context:
                return context[tensor_name][attr]
            
            resolved = tensor.get(attr, value)
            if attr == 'dtypes':
                return flatten_dtypes(resolved)
            return resolved
    
    if context and tensor_name in context:
        return context[tensor_name].get(attr, value)
    
    return value


def flatten_dtypes(dtypes: List) -> List:
    """处理嵌套的dtypes列表，确保正确解析"""
    if not dtypes:
        return []
    
    if isinstance(dtypes[0], list):
        return [item[0] for item in dtypes]
    
    return dtypes


def generate_test_suite(input_yaml: str, output_json: str):
    """根据YAML配置生成测试用例JSON"""
    with open(input_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    resolved_inputs = []
    ref_tensors = {}
    
    for tensor in config.get('input_tensors', []):
        resolved_tensor = tensor.copy()
        
        # 解析dtypes和shapes中的变量引用
        if isinstance(tensor['dtypes'], str):
            ref_tensor_name = tensor['dtypes'].split('_')[0][1:]
            resolved_tensor['dtypes'] = resolve_variable_ref(tensor['dtypes'], config)
            resolved_tensor['ref'] = ref_tensor_name  # 标记这个张量引用了另一个
            
            # 记录被引用的张量配置
            if ref_tensor_name not in ref_tensors:
                ref_config = next(t for t in config['input_tensors'] if t['name'] == ref_tensor_name)
                ref_tensors[ref_tensor_name] = ref_config
        
        if isinstance(tensor['shapes'], str):
            ref_tensor_name = tensor['shapes'].split('_')[0][1:]
            resolved_tensor['shapes'] = resolve_variable_ref(tensor['shapes'], config)
            resolved_tensor['ref'] = ref_tensor_name
            
            if ref_tensor_name not in ref_tensors:
                ref_config = next(t for t in config['input_tensors'] if t['name'] == ref_tensor_name)
                ref_tensors[ref_tensor_name] = ref_config
        
        resolved_tensor['dtypes'] = flatten_dtypes(resolved_tensor['dtypes'])
        resolved_inputs.append(resolved_tensor)
    
    # 生成输入张量的所有可能组合（处理引用关系）
    input_combinations = []
    
    # 首先处理没有引用其他张量的独立张量
    independent_tensors = [t for t in resolved_inputs if 'ref' not in t]
    dependent_tensors = [t for t in resolved_inputs if 'ref' in t]
    
    # 生成独立张量的组合
    independent_combos = []
    for tensor in independent_tensors:
        tensor_combos = []
        for dtype in tensor['dtypes']:
            for shape in tensor['shapes']:
                tensor_combos.append({
                    'name': tensor['name'],
                    'dtype': dtype,
                    'shape': shape
                })
        independent_combos.append(tensor_combos)
    
    # 计算独立张量的所有组合
    if independent_combos:
        base_combos = [list(combo) for combo in product(*independent_combos)]
    else:
        base_combos = [[]]
    
    # 为每个基础组合添加依赖张量的组合
    all_input_combos = []
    
    for base_combo in base_combos:
        # 创建上下文映射（名称到配置）
        context = {item['name']: item for item in base_combo}
        
        # 处理依赖张量
        valid_dependent_combos = [[]]  # 存储当前基础组合的有效依赖组合
        
        for tensor in dependent_tensors:
            ref_tensor_name = tensor['ref']
            ref_config = ref_tensors[ref_tensor_name]
            
            # 从上下文中获取被引用张量的实际值
            if ref_tensor_name in context:
                ref_dtype = context[ref_tensor_name]['dtype']
                ref_shape = context[ref_tensor_name]['shape']
                
                # 为当前依赖张量创建匹配的组合
                new_combos = []
                for combo in valid_dependent_combos:
                    new_combos.append(combo + [{
                        'name': tensor['name'],
                        'dtype': ref_dtype,
                        'shape': ref_shape
                    }])
                
                valid_dependent_combos = new_combos
        
        # 将基础组合与每个有效的依赖组合合并
        for dependent_combo in valid_dependent_combos:
            all_input_combos.append(base_combo + dependent_combo)
    
    tile_shapes = config.get('tile_shapes', [])
    view_shapes = config.get('view_shapes', [])
    
    # 生成测试用例
    test_cases = []
    case_counter = 0
    
    # 正交组合：输入组合 × tile_shape × view_shape × 参数
    for inputs, tile_shape, view_shape, params in product(
        all_input_combos, tile_shapes, view_shapes, config.get('params_list', [{}])
    ):
        context = {input['name']: input for input in inputs}
        
        # 解析输出张量配置
        resolved_outputs = []
        for output in config.get('output_tensors', []):
            resolved_output = {}
            for key, value in output.items():
                if isinstance(value, str):
                    if key in ['dtype', 'shape']:
                        resolved_output[key] = resolve_variable_ref(value, config, context)
                    else:
                        resolved_output[key] = value
                else:
                    resolved_output[key] = value
            resolved_outputs.append(resolved_output)
        
        # 生成测试用例
        test_case = {
            'name': f"{config['operation']}_test{case_counter}",
            'operation': config['operation'],
            'input_tensors': list(inputs),
            'output_tensors': resolved_outputs,
            'tile_shape': tile_shape,
            'view_shape': view_shape,
            'op_function': config.get('op_function', ''),
            'golden_function': config.get('golden_function', ''),
            'params': params
        }
        
        test_cases.append(test_case)
        case_counter += 1
    
    # 生成最终的JSON结构
    result = {
        'golden_dir': './golden',
        'test_cases': test_cases
    }
    
    # 保存为JSON文件
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"生成完成！共{len(test_cases)}个测试用例，保存至{output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成测试用例JSON")
    parser.add_argument("--input", required=True, help="YAML配置文件路径")
    parser.add_argument("--output", default="test_suite.json", help="输出JSON文件路径")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误：输入文件{args.input}不存在")
    else:
        generate_test_suite(args.input, args.output)