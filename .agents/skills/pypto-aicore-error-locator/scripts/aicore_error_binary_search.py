#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil
import re
from pathlib import Path


def read_file(file_path):
    """读取文件内容"""
    with open(file_path, 'r') as f:
        return f.readlines()


def write_file(file_path, lines):
    """写入文件内容"""
    with open(file_path, 'w') as f:
        f.writelines(lines)


def get_commentable_lines(lines):
    """识别可注释的代码（只注释关键的代码行）"""
    commentable_lines = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # 跳过空行、注释、包含大括号的行
        # 跳过 pipe_barrier、set_flag 和 wait_flag 行
        if not stripped or stripped.startswith('//') or '{' in stripped or '}' in stripped or '#' in stripped or 'set_flag' in stripped or 'wait_flag' in stripped or 'pipe_barrier' in stripped:
            continue
        else:
            commentable_lines.append(i)

    return commentable_lines


def comment_lines(lines, line_indices):
    """注释指定的行（行号从1开始）"""
    lines_to_commentary = set(line_indices)
    
    # 执行注释
    for line_num in sorted(lines_to_commentary, reverse=True):
        line_idx = line_num - 1
        lines[line_idx] = '// ' + lines[line_idx]
    
    return lines


def uncomment_lines(lines, line_indices):
    """取消注释指定的行（行号从1开始）"""
    lines_to_uncomment = set(line_indices)
    
    # 执行取消注释（从前往后）
    for line_num in sorted(lines_to_uncomment):
        line_idx = line_num - 1
        if lines[line_idx].strip().startswith('//'):
            lines[line_idx] = lines[line_idx][3:]  # 移除 '// '
    
    return lines


def has_error(returncode, output):
    """检查输出中是否有 error 或返回码非0
    
    Args:
        returncode: 命令返回码
        output: 命令输出内容
    
    Returns:
        bool: True 表示有错误，False 表示无错误
    """
    # 检查返回码
    if returncode != 0:
        return True
    
    # 检查输出中的错误关键词（排除误报）
    output_lower = output.lower()
    
    # 真正的错误关键词
    true_error_keywords = [
        ' error',
        'error ',
        'exception',
        'segmentation fault',
        'core dump',
    ]
    
    for keyword in true_error_keywords:
        if keyword in output_lower:
            return True
    
    return False


def run_test(test_cmd, run_dir):
    """运行测试命令"""
    result = subprocess.run(
        test_cmd,
        shell=True,
        cwd=run_dir,
        capture_output=True,
        text=True,
        errors='ignore',
        timeout=1800
    )
    return result.returncode, result.stdout + result.stderr


def binary_search(cce_file, test_cmd, run_dir):
    """二分查找定位问题代码行"""
    print(f"开始二分查找定位问题代码...")
    print(f"CCE 文件: {cce_file}")
    print(f"测试命令: {test_cmd}")
    print(f"运行目录: {run_dir}")
    print()

    # 先检查原始文件是否有 error
    print("检查原始文件是否有 error...")
    returncode, output = run_test(test_cmd, run_dir)
    error_exists = has_error(returncode, output)
    
    if not error_exists:
        print("错误：原始文件运行无 error，无需进行二分查找")
        print("可能原因：")
        print("  1. 测试命令或运行目录不正确")
        print("  2. CCE 文件本身没有问题")
        raise Exception("原始文件运行无 error，无需进行二分查找")
    # 打印 error 相关信息
    error_lines = [line for line in output.split('\n') if 'error' in line.lower()]
    if error_lines:
        print("Error 信息:")
        for line in error_lines[:10]:  # 只打印前 10 行
            print(f"  {line}")
    print("原始文件运行有 error，开始二分查找")
    print()

    # 备份原始文件
    backup_file = cce_file + ".bak"
    shutil.copy(cce_file, backup_file)
    cce_lines = read_file(cce_file)
    original_lines = cce_lines.copy()

    # 注释所有 pipe_barrier、set_flag 和 wait_flag 行
    for i, line in enumerate(cce_lines):
        if 'set_flag' in line or 'wait_flag' in line or 'pipe_barrier' in line:
            if not line.strip().startswith('//'):
                cce_lines[i] = '// ' + line

    # 识别可注释的行
    commentable_lines = get_commentable_lines(cce_lines)
    print(f"可注释的行数: {len(commentable_lines)}")
    print()

    if len(commentable_lines) == 0:
        print("错误：没有可注释的行")
        # 恢复原始文件并删除备份
        write_file(cce_file, original_lines)
        os.remove(backup_file)
        return None

    # 注释所有可注释的行
    print("注释所有可注释的行")
    current_lines = cce_lines.copy()
    current_lines = comment_lines(current_lines, commentable_lines)
    already_commented = set(commentable_lines)

    # 运行测试
    write_file(cce_file, current_lines)
    print("运行测试...")
    returncode, output = run_test(test_cmd, run_dir)
    error_exists = has_error(returncode, output)

    if error_exists:
        print("错误：注释所有行后仍有 error，无法定位问题")
        # 打印 error 相关信息
        error_lines = [line for line in output.split('\n') if 'error' in line.lower()]
        if error_lines:
            print("Error 信息:")
            for line in error_lines[:10]:  # 只打印前 10 行
                print(f"  {line}")
        # 恢复原始文件并删除备份
        write_file(cce_file, original_lines)
        os.remove(backup_file)
        return None

    print("注释所有行后运行成功（无 error），开始逐步取消注释")
    print()

    # 二分查找：从前往后逐步取消注释
    left = 0
    right = len(commentable_lines) - 1
    iteration = 0

    while left < right:
        iteration += 1
        mid = (left + right) // 2

        print(f"迭代 {iteration}: left={left}, right={right}, mid={mid}")

        # 从注释所有行的状态开始，取消注释 [0, mid] 范围的行
        print(f"取消注释范围 [0, mid] 的行: {commentable_lines[0:mid+1]}")
        
        current_lines = cce_lines.copy()
        current_lines = comment_lines(current_lines, commentable_lines)
        
        # 取消注释从 0 到 mid 的行
        lines_to_uncomment = commentable_lines[0:mid+1]
        current_lines = uncomment_lines(current_lines, lines_to_uncomment)

        # 运行测试
        write_file(cce_file, current_lines)
        print("运行测试...")
        returncode, output = run_test(test_cmd, run_dir)
        error_exists = has_error(returncode, output)

        if error_exists:
            print("结果: 运行失败（有 error），问题在 [0, mid] 中")
            # 打印 error 相关信息
            error_lines = [line for line in output.split('\n') if 'error' in line.lower()]
            if error_lines:
                print("Error 信息:")
                for line in error_lines[:10]:  # 只打印前 10 行
                    print(f"  {line}")
            right = mid
        else:
            print("结果: 运行成功（无 error），问题在 [mid+1. end] 中")
            left = mid + 1

        print()

    # 精确定位
    print(f"精确定位: left={commentable_lines[left]}, right={commentable_lines[right]}")
    problem_line = commentable_lines[left]
    print(f"问题代码行: {problem_line}")

    # 恢复原始文件并删除备份
    write_file(cce_file, original_lines)
    os.remove(backup_file)

    return problem_line


def main():
    """主函数"""
    if len(sys.argv) < 4:
        print("用法: python3 aicore_error_binary_search.py <cce_file> <test_cmd> <run_dir>")
        print()
        print("参数说明:")
        print("  cce_file: CCE 文件路径（绝对路径）")
        print("  test_cmd: 触发 aicore error 的测试命令")
        print("  run_dir: 运行测试命令的目录路径（绝对路径）")
        print()
        print("示例:")
        print("  python3 aicore_error_binary_search.py \\")
        print("    /path/to/kernel_aicore/xxx.cpp \\")
        print("    'python test_2.py' \\")
        print("    /path/to/run_dir")
        sys.exit(1)

    cce_file = sys.argv[1]
    test_cmd = sys.argv[2]
    run_dir = sys.argv[3]

    # 将所有路径转换为绝对路径
    cce_file = os.path.abspath(cce_file)
    run_dir = os.path.abspath(run_dir)

    if not os.path.exists(cce_file):
        print(f"错误: CCE文件不存在: {cce_file}")
        sys.exit(1)

    if not os.path.exists(run_dir):
        print(f"错误: 运行目录不存在: {run_dir}")
        sys.exit(1)

    # 执行二分查找
    problem_line = binary_search(cce_file, test_cmd, run_dir)

    if problem_line:
        print(f"\n找到问题代码行: {problem_line}")

        # 显示问题代码
        lines = read_file(cce_file)
        if problem_line <= len(lines):
            print(f"问题代码: {lines[problem_line - 1].strip()}")

        return problem_line
    else:
        print("\n未找到问题代码行")
        return None


if __name__ == '__main__':
    main()
