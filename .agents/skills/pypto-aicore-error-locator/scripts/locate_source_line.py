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


OPERATION_MAPPING = {
    'TCast': {'CAST'},
    'TLoad': {'COPY_IN'},
    'TStore': {'COPY_OUT'},
    'TMatmul': {'A_MUL_B', 'A_MULACC_B'},
    'TExtract': {'L1_TO_L0A', 'L1_TO_L0B'},
    'TVecDup': {'VEC_DUP'},
    'TMul': {'MUL'},
    'TMulS': {'MULS'},
    'TAdd': {'ADD'},
    'TAddS': {'ADDS'},
    'TSub': {'SUB'},
    'TDiv': {'DIV'},
    'TExp': {'EXP'}
}


def extract_operation_type(code_line: str) -> Optional[str]:
    match = re.match(r'^(T[A-Z]\w*)', code_line.strip())
    return match.group(1) if match else None


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
    
    operation_type_to_lines = {}
    for idx, line in enumerate(lines, start=1):
        op_type = extract_operation_type(line)
        if op_type:
            if op_type not in operation_type_to_lines:
                operation_type_to_lines[op_type] = []
            operation_type_to_lines[op_type].append(idx)
    
    return func_hash, line_code_map, operation_type_to_lines


def find_source_location(cce_path: str, json_path: str, cce_line_number: int) -> Dict[str, Any]:
    func_hash, line_code_map, operation_type_to_lines = read_cce_file(cce_path)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        program_data = json.load(f)
    
    func_data = None
    for func in program_data.get('functions', []):
        if func.get('hash') == func_hash:
            func_data = func
            break
    
    if not func_data:
        raise ValueError(f"未找到 hash 为 {func_hash} 的函数")
    
    if cce_line_number not in line_code_map:
        raise ValueError(f"CCE 文件中没有第 {cce_line_number} 行")

    cce_line = line_code_map[cce_line_number]
    cce_op_type = extract_operation_type(cce_line)
    
    if not cce_op_type:
        return {
            'matched': False,
            'reason': '该代码没有对应前端代码与之映射',
            'cce_line_code': cce_line
        }
    
    target_opcodes = OPERATION_MAPPING.get(cce_op_type, set())
    if not target_opcodes:
        return {
            'matched': False,
            'reason': f'未知操作类型: {cce_op_type}',
            'cce_line_code': cce_line,
            'operation_type': cce_op_type
        }
    
    op_type_lines = operation_type_to_lines.get(cce_op_type, [])
    operation_index = op_type_lines.index(cce_line_number)
    
    matched_operations = []
    for op in func_data.get('operations', []):
        if op.get('opcode') in target_opcodes:
            matched_operations.append(op)
    
    # 检查操作数量是否匹配
    cce_count = len(op_type_lines)
    json_count = len(matched_operations)

    logger.info("")
    logger.info("[统计信息]")
    logger.info("  CCE 文件中 %s: %d 个", cce_op_type, cce_count)
    logger.info("  program.json 中 %s: %d 个", target_opcodes, json_count)
    
    if cce_count != json_count:
        return {
            'matched': False,
            'reason': 'CCE 文件中 {} 与  program.json 中 {} 的个数不一样，请仔细检查'.format(cce_op_type, target_opcodes),
            'cce_line_code': cce_line
        }

    
    matched_op = matched_operations[operation_index]
    
    return {
        'matched': True,
        'cce_line_code': cce_line,
        'operation_type': cce_op_type,
        'operation_index': operation_index + 1,
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
