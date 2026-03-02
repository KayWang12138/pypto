#!/usr/bin/env python3
"""
PyPTO 算子日志搜索工具

用于在 PyPTO 算子开发过程中搜索和分析日志文件。
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional


def search_in_file(
    file_path: Path,
    pattern: str,
    use_regex: bool = False,
    context_lines: int = 0,
    ignore_case: bool = False
) -> List[Tuple[int, str, List[str]]]:
    """
    在文件中搜索关键字
    
    Args:
        file_path: 文件路径
        pattern: 搜索模式
        use_regex: 是否使用正则表达式
        context_lines: 上下文行数
        ignore_case: 是否忽略大小写
    
    Returns:
        匹配结果列表，每个元素为 (行号, 匹配行, 上下文行列表)
    """
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Warning: 无法读取文件 {file_path}: {e}", file=sys.stderr)
        return results
    
    flags = re.IGNORECASE if ignore_case else 0
    
    for line_num, line in enumerate(lines, 1):
        line = line.rstrip('\n\r')
        
        if use_regex:
            try:
                if re.search(pattern, line, flags):
                    context = []
                    if context_lines > 0:
                        start = max(0, line_num - context_lines - 1)
                        end = min(len(lines), line_num + context_lines)
                        for i in range(start, end):
                            if i != line_num - 1:
                                context.append((i + 1, lines[i].rstrip('\n\r')))
                    results.append((line_num, line, context))
            except re.error as e:
                print(f"Error: 无效的正则表达式 '{pattern}': {e}", file=sys.stderr)
                return results
        else:
            search_pattern = pattern.lower() if ignore_case else pattern
            search_line = line.lower() if ignore_case else line
            
            if search_pattern in search_line:
                context = []
                if context_lines > 0:
                    start = max(0, line_num - context_lines - 1)
                    end = min(len(lines), line_num + context_lines)
                    for i in range(start, end):
                        if i != line_num - 1:
                            context.append((i + 1, lines[i].rstrip('\n\r')))
                results.append((line_num, line, context))
    
    return results


def count_matches(
    file_path: Path,
    pattern: str,
    use_regex: bool = False,
    ignore_case: bool = False
) -> int:
    """
    统计文件中匹配的数量
    
    Args:
        file_path: 文件路径
        pattern: 搜索模式
        use_regex: 是否使用正则表达式
        ignore_case: 是否忽略大小写
    
    Returns:
        匹配数量
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: 无法读取文件 {file_path}: {e}", file=sys.stderr)
        return 0
    
    flags = re.IGNORECASE if ignore_case else 0
    
    if use_regex:
        try:
            matches = re.findall(pattern, content, flags)
            return len(matches)
        except re.error as e:
            print(f"Error: 无效的正则表达式 '{pattern}': {e}", file=sys.stderr)
            return 0
    else:
        search_pattern = pattern.lower() if ignore_case else pattern
        search_content = content.lower() if ignore_case else content
        return search_content.count(search_pattern)


def search_in_directory(
    directory: Path,
    pattern: str,
    use_regex: bool = False,
    context_lines: int = 0,
    ignore_case: bool = False,
    count_only: bool = False
) -> dict:
    """
    在目录中搜索关键字
    
    Args:
        directory: 目录路径
        pattern: 搜索模式
        use_regex: 是否使用正则表达式
        context_lines: 上下文行数
        ignore_case: 是否忽略大小写
        count_only: 是否仅统计数量
    
    Returns:
        搜索结果字典
    """
    results = {}
    
    if not directory.exists():
        print(f"Error: 路径不存在: {directory}", file=sys.stderr)
        return results
    
    if directory.is_file():
        files = [directory]
    else:
        files = list(directory.rglob('*'))
        files = [f for f in files if f.is_file()]
    
    for file_path in files:
        if count_only:
            count = count_matches(file_path, pattern, use_regex, ignore_case)
            if count > 0:
                results[str(file_path)] = count
        else:
            matches = search_in_file(file_path, pattern, use_regex, context_lines, ignore_case)
            if matches:
                results[str(file_path)] = matches
    
    return results


def format_results(
    results: dict,
    count_only: bool = False,
    context_lines: int = 0
) -> str:
    """
    格式化搜索结果
    
    Args:
        results: 搜索结果
        count_only: 是否仅统计数量
        context_lines: 上下文行数
    
    Returns:
        格式化的结果字符串
    """
    output = []
    
    if count_only:
        total = sum(results.values())
        output.append(f"总计匹配: {total} 次")
        output.append("")
        output.append("按文件统计:")
        for file_path, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
            output.append(f"  {file_path}: {count} 次")
    else:
        total_matches = sum(len(matches) for matches in results.values())
        output.append(f"找到 {total_matches} 处匹配，在 {len(results)} 个文件中")
        output.append("")
        
        for file_path, matches in results.items():
            output.append(f"文件: {file_path}")
            output.append("-" * 80)
            
            for line_num, line, context in matches:
                if context_lines > 0:
                    output.append("")
                    for ctx_line_num, ctx_line in context:
                        marker = " > " if ctx_line_num < line_num else " < "
                        output.append(f"{marker} {ctx_line_num}: {ctx_line}")
                output.append(f" * {line_num}: {line}")
            
            output.append("")
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description='PyPTO 算子日志搜索工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础搜索
  %(prog)s -k "ERROR" -p output/
  
  # 带上下文
  %(prog)s -k "ERROR" -p output/ -C 2
  
  # 正则表达式
  %(prog)s -k "ERROR.*timeout" -p output/ -r
  
  # 统计次数
  %(prog)s -k "ERROR" -p output/ -c
  
  # 忽略大小写
  %(prog)s -k "error" -p output/ -i
        """
    )
    
    parser.add_argument(
        '-k', '--keyword',
        required=True,
        help='搜索关键字'
    )
    
    parser.add_argument(
        '-p', '--path',
        required=True,
        help='日志文件或目录路径'
    )
    
    parser.add_argument(
        '-r', '--regex',
        action='store_true',
        help='使用正则表达式匹配'
    )
    
    parser.add_argument(
        '-C', '--context',
        type=int,
        default=0,
        help='显示前后行数（默认: 0）'
    )
    
    parser.add_argument(
        '-c', '--count',
        action='store_true',
        help='仅统计出现次数'
    )
    
    parser.add_argument(
        '-i', '--ignore-case',
        action='store_true',
        help='忽略大小写'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出结果到文件'
    )
    
    args = parser.parse_args()
    
    # 执行搜索
    results = search_in_directory(
        Path(args.path),
        args.keyword,
        use_regex=args.regex,
        context_lines=args.context,
        ignore_case=args.ignore_case,
        count_only=args.count
    )
    
    # 格式化结果
    output = format_results(results, count_only=args.count, context_lines=args.context)
    
    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"结果已保存到: {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()
