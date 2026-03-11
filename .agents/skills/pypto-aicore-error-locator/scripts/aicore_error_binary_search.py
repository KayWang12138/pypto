#!/usr/bin/env python3
"""
AICore Error 二分查找定位脚本
用于定位 CCE 文件中导致 aicore error 的问题代码行
"""

import os
import subprocess
import sys
import shutil
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
    """识别可注释的代码行（只注释关键的代码行）"""
    commentable_lines = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # 跳过空行、注释行和大括号行
        if not stripped or stripped.startswith('//') or stripped in ['{', '}', '};']:
            continue
        # 只注释TLoad、TExtract、TMatmul、TStore、wait_flag、set_flag等关键行
        if 'TLoad' in stripped or 'TExtract' in stripped or 'TMatmul' in stripped or 'TStore' in stripped:
            commentable_lines.append(i)
        elif 'wait_flag' in stripped or 'set_flag' in stripped:
            commentable_lines.append(i)
    return commentable_lines


def find_wait_set_pairs(lines):
    """识别 wait_flag 和 set_flag 的成对行号"""
    wait_set_pairs = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if 'wait_flag' in stripped and not stripped.startswith('//'):
            # 找到 wait_flag，寻找对应的 set_flag
            for j in range(i + 1, len(lines) + 1):
                if 'set_flag' in lines[j - 1] and not lines[j - 1].strip().startswith('//'):
                    wait_set_pairs.append((i, j))
                    break
    return wait_set_pairs


def comment_lines(lines, line_indices, wait_set_pairs=None):
    """注释指定的行（行号从1开始），自动处理 wait_flag 和 set_flag 成对注释"""
    lines_to_commentary = set(line_indices)
    
    # 如果提供了 wait_set_pairs，自动添加成对的行
    if wait_set_pairs:
        for wait_line, set_line in wait_set_pairs:
            # 如果注释了 wait_flag，也要注释对应的 set_flag
            if wait_line in lines_to_commentary:
                lines_to_commentary.add(set_line)
            # 如果注释了 set_flag，也要注释对应的 wait_flag
            if set_line in lines_to_commentary:
                lines_to_commentary.add(wait_line)
    
    # 执行注释
    for line_num in sorted(lines_to_commentary, reverse=True):
        line_idx = line_num - 1
        lines[line_idx] = '// ' + lines[line_idx]
    
    return lines


def uncomment_lines(lines, line_indices, wait_set_pairs=None):
    """取消注释指定的行（行号从1开始），自动处理 wait_flag 和 set_flag 成对取消注释"""
    lines_to_uncomment = set(line_indices)
    
    # 如果提供了 wait_set_pairs，自动添加成对的行
    if wait_set_pairs:
        for wait_line, set_line in wait_set_pairs:
            # 如果取消注释 wait_flag，也要取消注释对应的 set_flag
            if wait_line in lines_to_uncomment:
                lines_to_uncomment.add(set_line)
            # 如果取消注释 set_flag，也要取消注释对应的 wait_flag
            if set_line in lines_to_uncomment:
                lines_to_uncomment.add(wait_line)
    
    # 执行取消注释（从前往后）
    for line_num in sorted(lines_to_uncomment):
        line_idx = line_num - 1
        if lines[line_idx].strip().startswith('//'):
            lines[line_idx] = lines[line_idx][3:]  # 移除 '// '
    
    return lines


def has_aicore_error(output):
    """检查输出中是否有 aicore error. """
    return 'aicore error' in output or 'ErrorTracking' in output


def run_test(test_cmd, run_dir):
    """运行测试命令"""
    result = subprocess.run(
        test_cmd,
        shell=True,
        cwd=run_dir,
        capture_output=True,
        text=True,
        errors='ignore',
        timeout=120
    )
    return result.returncode, result.stdout + result.stderr


def binary_search(cce_file, test_cmd, run_dir):
    """二分查找定位问题代码行"""
    print(f"开始二分查找定位问题代码...")
    print(f"CCE 文件: {cce_file}")
    print(f"测试命令: {test_cmd}")
    print(f"运行目录: {run_dir}")
    print()

    # 备份原始文件
    backup_file = cce_file + ".bak"
    shutil.copy(cce_file, backup_file)
    original_lines = read_file(cce_file)

    # 识别可注释的行
    commentable_lines = get_commentable_lines(original_lines)
    print(f"可注释的行数: {len(commentable_lines)}")
    print()

    if len(commentable_lines) == 0:
        print("错误：没有可注释的行")
        return None

    # 识别 wait_flag 和 set_flag 成对
    wait_set_pairs = find_wait_set_pairs(original_lines)
    if wait_set_pairs:
        print(f"识别到 {len(wait_set_pairs)} 对 wait_flag/set_flag:")
        for wait_line, set_line in wait_set_pairs:
            print(f"  wait_flag: 行 {wait_line}, set_flag: 行 {set_line}")
        print()

    # 第一轮：先不注释代码，直接运行原始文件
    print("第一轮：运行原始文件（不注释任何代码）")
    write_file(cce_file, original_lines)
    returncode, output = run_test(test_cmd, run_dir)
    has_error = has_aicore_error(output)

    if not has_error:
        print("错误：第一轮运行原始文件无 aicore error，但应该有 error")
        print("请检查测试命令或 CCE 文件")
        return None

    print("第一轮结果：运行失败（有 aicore error），开始二分查找")
    print()

    # 注释所有可注释的行
    print("注释所有可注释的行")
    current_lines = original_lines.copy()
    current_lines = comment_lines(current_lines, commentable_lines, wait_set_pairs)
    already_commented = set(commentable_lines)

    # 运行测试
    write_file(cce_file, current_lines)
    print("运行测试...")
    returncode, output = run_test(test_cmd, run_dir)
    has_error = has_aicore_error(output)

    if has_error:
        print("错误：注释所有行后仍有 aicore error，无法定位问题")
        return None

    print("注释所有行后运行成功（无 aicore error），开始逐步取消注释")
    print()

    # 二分查找：逐步取消注释
    left = 0
    right = len(commentable_lines) - 1
    iteration = 0

    while (right - left) > 1:
        iteration += 1
        mid = (left + right) // 2

        print(f"迭代 {iteration}: left={left}, right={right}, mid={mid}")

        # 从当前文件开始，取消注释从 left 到 mid 的行（从前往后取消注释）
        print(f"取消注释范围 [left, mid] 的行: {commentable_lines[left:mid+1]}")
        
        # 重新从注释所有行的状态开始
        current_lines = original_lines.copy()
        current_lines = comment_lines(current_lines, commentable_lines, wait_set_pairs)
        
        # 取消注释从 left 到 mid 的行
        lines_to_uncomment = commentable_lines[left:mid+1]
        current_lines = uncomment_lines(current_lines, lines_to_uncomment, wait_set_pairs)

        print(f"实际取消注释的行: {lines_to_uncomment}")

        # 运行测试
        write_file(cce_file, current_lines)
        print("运行测试...")
        returncode, output = run_test(test_cmd, run_dir)
        has_error = has_aicore_error(output)

        if has_error:
            print("结果: 运行失败（有 aicore error），问题在取消注释的行中")
            right = mid
        else:
            print("结果: 运行成功（无 aicore error），问题在仍被注释的行中")
            left = mid

        print()

    # 精确定位
    print(f"精确定位: left={left}, right={right}")
    problem_line = commentable_lines[right]
    print(f"问题代码行: {problem_line}")

    # 恢复原始文件
    write_file(cce_file, original_lines)

    return problem_line


def main():
    """主函数"""
    if len(sys.argv) < 4:
        print("用法: python3 aicore_error_binary_search.py <cce_file> <test_cmd> <run_dir>")
        print("示例: python3 aicore_error_binary_search.py kernel_aicore/xxx.cpp 'python test_2.py' .")
        sys.exit(1)

    cce_file = sys.argv[1]
    test_cmd = sys.argv[2]
    run_dir = sys.argv[3]

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
