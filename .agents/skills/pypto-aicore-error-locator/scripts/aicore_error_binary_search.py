#!/usr/bin/env python3

import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    read_file,
    write_file,
    get_commentable_lines,
    comment_lines,
    uncomment_lines,
    has_error,
    run_test,
    comment_special_lines
)


def print_error_info(output):
    error_lines = [line for line in output.split('\n') if 'error' in line.lower()]
    if error_lines:
        print("Error 信息:")
        for line in error_lines[:10]:
            print(f"  {line}")


def binary_search(cce_file, test_cmd, run_dir):
    print(f"开始二分查找定位问题代码...")
    print(f"CCE 文件: {cce_file}")
    print(f"测试命令: {test_cmd}")
    print(f"运行目录: {run_dir}")
    print()

    print("检查原始文件是否有 error...")
    returncode, output = run_test(test_cmd, run_dir)
    error_exists = has_error(returncode, output)
    
    if not error_exists:
        print("错误：原始文件运行无 error，无需进行二分查找")
        print("可能原因：")
        print("  1. 测试命令或运行目录不正确")
        print("  2. CCE 文件本身没有问题")
        raise Exception("原始文件运行无 error，无需进行二分查找")
    
    print_error_info(output)
    print("原始文件运行有 error，开始二分查找")
    print()

    backup_file = cce_file + ".bak"
    shutil.copy(cce_file, backup_file)
    cce_lines = read_file(cce_file)
    original_lines = cce_lines.copy()

    cce_lines = comment_special_lines(cce_lines)

    commentable_lines = get_commentable_lines(cce_lines)
    print(f"可注释的行数: {len(commentable_lines)}")
    print()

    if len(commentable_lines) == 0:
        print("错误：没有可注释的行")
        write_file(cce_file, original_lines)
        os.remove(backup_file)
        return None

    print("注释所有可注释的行")
    current_lines = cce_lines.copy()
    current_lines = comment_lines(current_lines, commentable_lines)

    write_file(cce_file, current_lines)
    print("运行测试...")
    returncode, output = run_test(test_cmd, run_dir)
    error_exists = has_error(returncode, output)

    if error_exists:
        print("错误：注释所有行后仍有 error，无法定位问题")
        print_error_info(output)
        write_file(cce_file, original_lines)
        os.remove(backup_file)
        return None

    print("注释所有行后运行成功（无 error），开始逐步取消注释")
    print()

    left = 0
    right = len(commentable_lines) - 1
    iteration = 0

    while left < right:
        iteration += 1
        mid = (left + right) // 2

        print(f"迭代 {iteration}: left={left}, right={right}, mid={mid}")

        print(f"取消注释范围 [0, mid] 的行: {commentable_lines[0:mid+1]}")
        
        current_lines = cce_lines.copy()
        current_lines = comment_lines(current_lines, commentable_lines)
        
        lines_to_uncomment = commentable_lines[0:mid+1]
        current_lines = uncomment_lines(current_lines, lines_to_uncomment)

        write_file(cce_file, current_lines)
        print("运行测试...")
        returncode, output = run_test(test_cmd, run_dir)
        error_exists = has_error(returncode, output)

        if error_exists:
            print("结果: 运行失败（有 error），问题在 [0, mid] 中")
            print_error_info(output)
            right = mid
        else:
            print("结果: 运行成功（无 error），问题在 [mid+1. end] 中")
            left = mid + 1

        print()

    print(f"精确定位: left={commentable_lines[left]}, right={commentable_lines[right]}")
    problem_line = commentable_lines[left]
    print(f"问题代码行: {problem_line}")

    write_file(cce_file, original_lines)
    os.remove(backup_file)

    return problem_line


def print_usage():
    print("用法:参数说明:")
    print("  cce_file: CCE 文件路径（绝对路径）")
    print("  test_cmd: 触发 aicore error 的测试命令")
    print("  run_dir: 运行测试命令的目录路径（绝对路径）")
    print()
    print("示例:")
    print("  python3 aicore_error_binary_search.py \\")
    print("    /path/to/kernel_aicore/xxx.cpp \\")
    print("    'python test_2.py' \\")
    print("    /path/to/run_dir")


def main():
    if len(sys.argv) < 4:
        print_usage()
        sys.exit(1)

    cce_file = sys.argv[1]
    test_cmd = sys.argv[2]
    run_dir = sys.argv[3]

    cce_file = os.path.abspath(cce_file)
    run_dir = os.path.abspath(run_dir)

    if not os.path.exists(cce_file):
        print(f"错误: CCE文件不存在: {cce_file}")
        sys.exit(1)

    if not os.path.exists(run_dir):
        print(f"错误: 运行目录不存在: {run_dir}")
        sys.exit(1)

    problem_line = binary_search(cce_file, test_cmd, run_dir)

    if problem_line:
        print(f"\n找到问题代码行: {problem_line}")

        lines = read_file(cce_file)
        if problem_line <= len(lines):
            print(f"问题代码: {lines[problem_line - 1].strip()}")

        return problem_line
    else:
        print("\n未找到问题代码行")
        return None


if __name__ == '__main__':
    main()
