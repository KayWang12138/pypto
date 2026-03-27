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
import re
import os
import glob
import sys
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PassContext:
    pass_pattern: re.Pattern
    current_pass_lines: List[str] = field(default_factory=list)
    current_pass_name: Optional[str] = None
    current_function_name: Optional[str] = None
    pass_count: int = 0
    function_index_map: Dict[str, int] = field(default_factory=dict)


def process_log_line(line, context, output_dir):
    """
    处理单行日志
    
    Args:
        line: 日志行
        context: PassContext对象
        output_dir: 输出目录
    """
    match = context.pass_pattern.search(line)
    
    if match:
        if context.current_pass_name is not None:
            save_pass_log(context.current_pass_lines, context.current_pass_name,
                         context.current_function_name, output_dir,
                         context.function_index_map)
            context.pass_count += 1
        
        context.current_pass_name = match.group(1)
        context.current_function_name = match.group(2)
        context.current_pass_lines = [line]
    else:
        if context.current_pass_name is not None:
            context.current_pass_lines.append(line)


def extract_pass_logs(log_file_paths, output_dir):
    """
    提取日志文件中的每个pass部分并保存到单独的文件中
    支持多个输入文件，处理跨文件的pass日志
    
    Args:
        log_file_paths: 输入日志文件路径列表
        output_dir: 输出目录文件路径
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    pass_pattern = re.compile(r'Apply pass <([^>]+)> on function: ([^.]+)\.')
    context = PassContext(pass_pattern=pass_pattern)
    
    for log_file_path in log_file_paths:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            process_log_line(line, context, output_dir)
    
    if context.current_pass_name is not None:
        save_pass_log(context.current_pass_lines, context.current_pass_name,
                     context.current_function_name, output_dir,
                     context.function_index_map)
        context.pass_count += 1


def save_pass_log(pass_lines, pass_name, function_name, output_dir, function_index_map):
    """
    保存单个pass的日志到文件
    
    Args:
        pass_lines: pass日志行列表
        pass_name: pass名称
        function_name: function名称
        output_dir: 输出目录
        function_index_map: 用于跟踪每个function_name的序号字典
    """
    if function_name not in function_index_map:
        function_index_map[function_name] = 0
    
    index = function_index_map[function_name]
    filename = f"Pass_{index:02d}_{pass_name}_{function_name}.log"
    
    subdir_name = f"Pass_{index:02d}_{pass_name}"
    subdir_path = os.path.join(output_dir, subdir_name)
    os.makedirs(subdir_path, exist_ok=True)
    
    filepath = os.path.join(subdir_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(pass_lines)
    
    function_index_map[function_name] += 1


def main():
    """
    主函数，处理命令行参数并执行日志提取
    """
    parser = argparse.ArgumentParser(
        description='提取PyPTO日志中的pass部分并保存到单独的文件中',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python extract_pass_logs.py pypto-log-xxx.log
  python extract_pass_logs.py pypto-log-xxx.log -o ./output
  python extract_pass_logs.py log1.log log2.log log3.log -o ./output
  python extract_pass_logs.py 'log*.log' -o ./output
        """
    )
    parser.add_argument('log_files', nargs='+', help='日志文件路径（支持通配符）')
    parser.add_argument('-o', '--output', default='pass_logs',
                       help='输出目录（默认：pass_logs）')
    
    args = parser.parse_args()
    
    log_patterns = args.log_files
    output_directory = args.output
    
    log_files = []
    for pattern in log_patterns:
        matched_files = glob.glob(pattern)
        if matched_files:
            log_files.extend(matched_files)
    
    if not log_files:
        sys.exit("Error, no log has been found.")
    
    log_files.sort()
    extract_pass_logs(log_files, output_directory)


if __name__ == "__main__":
    main()
