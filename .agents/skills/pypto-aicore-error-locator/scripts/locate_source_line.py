#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import json
import os
import re
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


def read_cce_file(cce_path: str) -> Tuple[str, Dict[int, str], Dict[str, List[int]]]:
    with open(cce_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    func_hash = None
    for line in lines:
        match = re.search(r'//\s*funcHash:\s*(\d+)', line)
        if match:
            func_hash = match.group(1)
            break
    
    if not func_hash:
        raise ValueError(f"无法找到 funcHash: {cce_path}")
    
    line_code_map = {idx: line.strip() for idx, line in enumerate(lines, start=1)}
    
    return func_hash, line_code_map

def get_cce_op(line_code_map):
    cce_op = {}
    target_op = ['set_flag', 'wait_flag', 'pipe_barrier', 'SUBKERNEL']
    for idx, line in line_code_map.items():
        is_t_operation = line.startswith('T') and '<' in line and '>' in line
        is_target_op = any(keyword in line for keyword in target_op)
        if is_t_operation or is_target_op:
            cce_op.update({idx:line})
    
    return cce_op

def extract_operation_type(cce_op_val):
    cce_op_type = []
    for val in cce_op_val:
        if val.startswith('T') and '<' in val and '>' in val:
            cce_op_type.append(val.split("<")[0])
        elif (val.startswith('w') or val.startswith('s')) and '(' in val and ')' in val:
            cce_op_type.append(val.split("(")[0])
        else:
            cce_op_type.append(val)

    return cce_op_type


def find_source_location(cce_path: str, json_path: str, cce_line_number: int) -> Dict[str, Any]:
    func_hash, line_code_map = read_cce_file(cce_path)

    cce_line = line_code_map[cce_line_number]
    cce_op = get_cce_op(line_code_map)
    cce_op_line = list(cce_op.keys())
    cce_op_type = extract_operation_type(list(cce_op.values()))

    if cce_line_number not in line_code_map:
        raise ValueError(f"CCE 文件中没有第 {cce_line_number} 行")

    if cce_line_number not in cce_op_line:
        return {
            'matched': False,
            'reason': '该代码没有前端代码与之映射',
            'cce_line_code': cce_line
        }
    
    cce_op_index = cce_op_line.index(cce_line_number)
    cce_op_name = cce_op_type[cce_op_index]
    
    with open(json_path, 'r', encoding='utf-8') as f:
        program_data = json.load(f)
    
    func_data = None
    for func in program_data.get('functions', []):
        if func.get('hash') == func_hash:
            func_data = func
            break
    
    if not func_data:
        raise ValueError(f"未找到 hash 为 {func_hash} 的函数")
    
    program_op = []
    program_opcode = []
    for op in func_data.get('operations', []):
        program_opcode.append(op.get('opcode'))
        program_op.append(op)

    # 检查操作数量是否匹配
    cce_count = len(cce_op_line)
    json_count = len(program_op)

    logger.info("")
    logger.info("[统计信息]")
    logger.info("  CCE 文件中操作数: %d 个", cce_count)
    logger.info("  program.json 中 操作数: %d 个", json_count)
    
    if cce_count != json_count:
        logger.error("CCE 文件中的操作为：")
        logger.error(cce_op_type)

        logger.error("program.json对应hash中的操作为： ")
        logger.error(program_opcode)
        return {
            'matched': False,
            'reason': 'CCE 文件中操作 与 program.json 中操作的个数不一样，请仔细检查',
            'cce_line_code': cce_line
        }
    
    matched_op = program_op[cce_op_index]
    
    return {
        'matched': True,
        'cce_line_code': cce_line,
        'operation_type': cce_op_name,
        'operation_index': cce_op_index + 1,
        'opcode': matched_op.get('opcode'),
        'source_file': matched_op.get('file'),
        'source_line': matched_op.get('line'),
    }


def print_source_code_line(file_path: str, line_number: int) -> None:
    if not file_path or not line_number:
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if line_number < 1 or line_number > len(lines):
            return

        start = max(0, line_number - 3)
        end = min(len(lines), line_number + 2)

        logger.info("")
        logger.info("[源代码] %s:%d", file_path, line_number)
        logger.info("-" * 80)
        for i in range(start, end):
            marker = ">>>" if i == line_number - 1 else "   "
            logger.info("%s %4d: %s", marker, i+1, lines[i].rstrip())
        logger.info("-" * 80)
    except Exception as e:
        logger.error("[ERROR] 无法读取源代码文件: %s", e)


def main():
    if len(sys.argv) < 4:
        logger.info("用法: python locate_source_line.py <cce_file> <program.json> <cce_line_number>")
        sys.exit(1)

    cce_path = sys.argv[1]
    json_path = sys.argv[2]
    cce_line_number = int(sys.argv[3])

    logger.info("CCE 文件: %s", cce_path)
    logger.info("问题行号: %d", cce_line_number)
    logger.info("代码: %s:%d", cce_path, cce_line_number)
    logger.info("-" * 80)

    result = find_source_location(cce_path, json_path, cce_line_number)

    logger.info("")
    logger.info("[CCE 问题代码]")
    logger.info("  %s", result['cce_line_code'])

    if result['matched']:
        logger.info("")
        logger.info("✓ 操作: %s (第 %d 个)", result['operation_type'], result['operation_index'])
        logger.info("✓ 匹配: %s", result['opcode'])

        if result['source_file'] and result['source_line']:
            logger.info("✓ 源代码: %s:%d", result['source_file'], result['source_line'])
            print_source_code_line(result['source_file'], result['source_line'])
        else:
            logger.info("✗ 该操作无源代码位置")
    else:
        logger.info("")
        logger.info("✗ 无法匹配")
        logger.info("  原因: %s", result['reason'])
        


if __name__ == '__main__':
    main()
