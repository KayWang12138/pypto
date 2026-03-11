#!/usr/bin/env python3

import os
import re
import sys
import shutil
import subprocess
from pathlib import Path


def read_file(file_path):
    """读取文件内容"""
    with open(file_path, 'r') as f:
        return f.readlines()


def write_file(file_path, lines):
    """写入文件内容"""
    with open(file_path, 'w') as f:
        f.writelines(lines)


def parse_luid(luid_str):
    """解析 LUid 字符串，提取各个字段"""
    # 格式: LUid{deviceTaskId, funcId, rootIndex, opIdx, leafIndex}
    match = re.search(r'LUid\{(\d+),(\d+),(\d+),(\d+),(\d+)\}', luid_str)
    if match:
        return {
            'deviceTaskId': int(match.group(1)),
            'funcId': int(match.group(2)),
            'rootIndex': int(match.group(3)),
            'opIdx': int(match.group(4)),
            'leafIndex': int(match.group(5))
        }
    return None


def parse_lactstart(lactstart_str):
    """解析 LActStart 字符串，提取 coreIdx"""
    # 格式: LActStart{coreIdx}
    match = re.search(r'LActStart\{(\d+)\}', lactstart_str)
    if match:
        return int(match.group(1))
    return None


def parse_lactfinish(lactfinish_str):
    """解析 LActFinish 字符串，提取 coreIdx"""
    # 格式: LActFinish{coreIdx}
    match = re.search(r'LActFinish\{(\d+)\}', lactfinish_str)
    if match:
        return int(match.group(1))
    return None


def analyze_trace( log_file):
    """分析追踪日志，找出缺失的 coreIdx 和对应的 leafIndex"""
    lactstart_events = []
    lactfinish_events = []

    print("=" * 80)
    print("分析追踪日志")
    print("=" * 80)
    print(f"日志文件: {log_file}")
    print()

    with open(log_file, 'r') as f:
        for line in f:
            if 'trace' in line and 'LActStart' in line:
                # 提取 LUid 和 LActStart
                luid_match = re.search(r'LUid\{[^}]+\}', line)
                lactstart_match = re.search(r'LActStart\{[^}]+\}', line)
                
                if luid_match and lactstart_match:
                    luid = parse_luid(luid_match.group(0))
                    core_idx = parse_lactstart(lactstart_match.group(0))
                    
                    if luid and core_idx is not None:
                        lactstart_events.append({
                            'luid': luid,
                            'coreIdx': core_idx
                        })
            
            elif 'trace' in line and 'LActFinish' in line:
                # 提取 LUid 和 LActFinish
                luid_match = re.search(r'LUid\{[^}]+\}', line)
                lactfinish_match = re.search(r'LActFinish\{[^}]+\}', line)
                
                if luid_match and lactfinish_match:
                    luid = parse_luid(luid_match.group(0))
                    core_idx = parse_lactfinish(lactfinish_match.group(0))
                    
                    if luid and core_idx is not None:
                        lactfinish_events.append({
                            'luid': luid,
                            'coreIdx': core_idx
                        })

    # 提取所有 coreIdx
    lactstart_core_idxs = sorted(set(event['coreIdx'] for event in lactstart_events))
    lactfinish_core_idxs = sorted(set(event['coreIdx'] for event in lactfinish_events))

    print(f"LActStart 事件数量: {len(lactstart_events)}")
    print(f"LActStart coreIdxs: {lactstart_core_idxs}")
    print(f"LActFinish 事件数量: {len(lactfinish_events)}")
    print(f"LActFinish coreIdxs: {lactfinish_core_idxs}")

    # 找出缺失的 coreIdx
    missing_core_idxs = [idx for idx in lactstart_core_idxs if idx not in lactfinish_core_idxs]
    
    print(f"\n缺失的 coreIdxs: {missing_core_idxs}")

    # 提取缺失 coreIdx 对应的 leafIndex
    missing_leaf_indices = []
    for event in lactstart_events:
        if event['coreIdx'] in missing_core_idxs:
            leaf_idx = event['luid']['leafIndex']
            if leaf_idx not in missing_leaf_indices:
                missing_leaf_indices.append(leaf_idx)

    print(f"对应的 leafIndices: {missing_leaf_indices}")
    print()

    return missing_leaf_indices, lactstart_events, lactfinish_events


def find_cce_file(kernel_aicore_dir, leaf_index):
    """定位问题 CCE 文件"""
    print(f"定位问题 CCE 文件 (leafIndex: {leaf_index})")
    print(f"kernel_aicore 目录: {kernel_aicore_dir}")
    print()
    
    # 查找 Record 文件
    record_files = list(Path(kernel_aicore_dir).glob("**/sub_func_*_call_*.h"))
    
    print(f"找到 {len(record_files)} 个 Record 文件")
    
    target_record_file = None
    core_type = None
    func_name = None
    
    for record_file in record_files:
        with open(record_file, 'r') as f:
            content = f.read()
            # 搜索 case <leafIndex>
            if f'case {leaf_index}:' in content:
                print(f"\n在文件中找到 leafIndex {leaf_index}: {record_file}")
                target_record_file = record_file
                
                # 提取 core_type
                # Record 文件名称格式: sub_func_<core_type>_call_<ID>.h
                match = re.search(r'sub_func_(\w+)_call_\d+\.h', record_file.name)
                if match:
                    core_type = match.group(1)
                    print(f"Core type: {core_type}")
                
                # 提取函数名
                # 查找 case <leaf_index>: 后面的函数调用
                pattern = rf'case {leaf_index}:\s*\{{\s*\n\s*(\w+)\('
                match = re.search(pattern, content)
                if match:
                    func_name = match.group(1)
                    print(f"Function name: {func_name}")
                break
    
    if not target_record_file:
        print(f"\n错误：无法在任何 Record 文件中找到 leafIndex {leaf_index}")
        return None
    
    # 从函数名提取 CCE_pre_name 和 CCE_ID
    # 函数名格式: <CCE_pre_name>_<CCE_ID>_<ID>_<func_hash>
    # 例如: TENSOR_s0_Unroll1_PATH0_hiddenfunc0_8_0_4503599627370496
    # 其中: CCE_pre_name = TENSOR_s0_Unroll1_PATH0_hiddenfunc0, CCE_ID = 8, ID = 0, func_hash = 4503599627370496
    
    # CCE 文件名格式: <CCE_pre_name>_<CCE_ID>_<hash>_<ID>_<core_type>.cpp
    # 例如: TENSOR_s0_Unroll1_PATH0_hiddenfunc0_8_3768358353088068894_0_aic.cpp
    # 其中: hash 是编译时生成的哈希值，与函数名中的 func_hash 不同
    
    if func_name:
        # 查找最后一个数字作为 CCE_ID
        # 从后往前找，找到第一个数字，然后继续往前找完整的数字
        match = re.search(r'_(\d+)_(\d+)_(\d+)$', func_name)
        if match:
            cce_id = match.group(1)
            id_val = match.group(2)
            func_hash = match.group(3)
            cce_pre_name = func_name[:func_name.rfind(f'_{cce_id}_{id_val}_{func_hash}')]
            
            print(f"\n从函数名提取信息:")
            print(f"  函数名: {func_name}")
            print(f"  CCE_pre_name: {cce_pre_name}")
            print(f"  CCE_ID: {cce_id}")
            print(f"  ID: {id_val}")
            print(f"  func_hash: {func_hash}")
            
            # 查找 CCE 文件
            # CCE 文件名称格式: <CCE_pre_name>_<CCE_ID>_<hash>_<ID>_<core_type>.cpp
            # hash 是未知的，使用通配符匹配
            cce_pattern = f"{cce_pre_name}_{cce_id}_*_{id_val}_{core_type}.cpp"
            cce_files = list(Path(kernel_aicore_dir).glob(f"**/{cce_pattern}"))
            
            print(f"\n搜索 CCE 文件模式: {cce_pattern}")
            print(f"找到 {len(cce_files)} 个匹配文件")
            
            if cce_files:
                print(f"\n找到 CCE 文件: {cce_files[0]}")
                return str(cce_files[0])
            else:
                print(f"\n错误：无法查找到匹配模式的 CCE 文件: {cce_pattern}")
                return None
        else:
            print(f"\n错误：无法解析函数名格式: {func_name}")
            return None
    
    return None


def get_commentable_lines(lines):
    """识别可注释的代码（只注释关键的代码行）"""
    commentable_lines = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # 跳过空行、注释行和包含大括号的行
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


def test_cce_file(cce_file, test_cmd, run_dir):
    """测试 CCE 文件：注释所有可注释的行，运行测试"""
    print(f"测试 CCE 文件: {cce_file}")
    
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
    
    if len(commentable_lines) == 0:
        print("错误：没有可注释的行")
        # 恢复原始文件并删除备份
        write_file(cce_file, original_lines)
        os.remove(backup_file)
        return False, None
    
    # 注释所有可注释的行
    print("注释所有可注释的行...")
    current_lines = cce_lines.copy()
    current_lines = comment_lines(current_lines, commentable_lines)
    
    # 运行测试
    write_file(cce_file, current_lines)
    print("运行测试...")
    returncode, output = run_test(test_cmd, run_dir)
    error_exists = has_error(returncode, output)
    
    # 恢复原始文件并删除备份
    write_file(cce_file, original_lines)
    os.remove(backup_file)
    
    if error_exists:
        print("结果: 注释所有行后仍有 error，此文件可能不是问题文件")
        return False, None
    else:
        print("结果: 注释所有行后运行成功（无 error），此文件可能是问题文件")
        return True, commentable_lines


def find_trace_log_file(device_log_path):
    """在 device_log_path 下搜索包含 trace 的日志文件"""
    print(f"在 {device_log_path} 下搜索包含 trace 的日志文件...")
    
    # 递归搜索所有 .log 文件
    log_files = list(Path(device_log_path).rglob("*.log"))
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                if 'trace' in content and ('LActStart' in content or 'LActFinish' in content):
                    print(f"找到 trace 日志文件: {log_file}")
                    return str(log_file)
        except Exception as e:
            continue
    
    print(f"错误：在 {device_log_path} 下未找到包含 trace 的日志文件")
    return None

def main():
    if len(sys.argv) < 3:
        print("用法: python3 analyze_trace_log.py <device_log_path> <kernel_aicore_dir> [test_cmd] [run_dir]")
        print()
        print("参数说明:")
        print("  device_log_path: device log 落盘路径")
        print("  kernel_aicore_dir: kernel_aicore 目录路径")
        print("  test_cmd: （可选）触发 aicore error 的测试命令")
        print("  run_dir: （可选）运行测试命令的目录路径")
        sys.exit(1)
    
    device_log_path = sys.argv[1]
    kernel_aicore_dir = sys.argv[2]
    test_cmd = sys.argv[3] if len(sys.argv) > 3 else None
    run_dir = sys.argv[4] if len(sys.argv) > 4 else None
    
    # 将所有路径转换为绝对路径
    device_log_path = os.path.abspath(device_log_path)
    kernel_aicore_dir = os.path.abspath(kernel_aicore_dir)
    
    if test_cmd and run_dir:
        run_dir = os.path.abspath(run_dir)
    
    if not os.path.exists(device_log_path):
        print(f"错误：device log 路径不存在: {device_log_path}")
        sys.exit(1)
    
    if not os.path.exists(kernel_aicore_dir):
        print(f"错误：kernel_aicore 目录不存在: {kernel_aicore_dir}")
        sys.exit(1)
    
    if run_dir and not os.path.exists(run_dir):
        print(f"错误：运行目录不存在: {run_dir}")
        sys.exit(1)
    
    # 查找 trace 日志文件
    log_file = find_trace_log_file(device_log_path)
    if not log_file:
        sys.exit(1)
    
    # 分析追踪日志
    missing_leaf_indices, lactstart_events, lactfinish_events = analyze_trace(log_file)
    
    if not missing_leaf_indices:
        print("=" * 80)
        print("结果：没有发现缺失的 leaf index")
        print("所有任务已成功完成，无需定位问题 CCE 文件")
        print("=" * 80)
        return
    
    # 查找 CCE 文件
    problem_cce_files = []
    for leaf_index in missing_leaf_indices:
        print("=" * 80)
        cce_file = find_cce_file(kernel_aicore_dir, leaf_index)
        if cce_file:
            problem_cce_files.append(cce_file)
    
    # 输出结果
    print("=" * 80)
    print("定位结果")
    print("=" * 80)
    
    if not problem_cce_files:
        print("\n未找到问题 CCE 文件")
        print("=" * 80)
        return
    
    print(f"\n找到 {len(problem_cce_files)} 个问题 CCE 文件:")
    for i, cce_file in enumerate(problem_cce_files, 1):
        print(f"  {i}. {cce_file}")
    
    # 如果存在多个问题 CCE 文件且提供了测试命令，依次排查
    if len(problem_cce_files) > 1 and test_cmd and run_dir:
        print("\n存在多个问题 CCE 文件，依次排查每个文件...")
        print()
        
        for i, cce_file in enumerate(problem_cce_files, 1):
            print("=" * 80)
            print(f"排查文件 {i}/{len(problem_cce_files)}: {cce_file}")
            print("=" * 80)
            
            is_problem_file, commentable_lines = test_cce_file(cce_file, test_cmd, run_dir)
            
            if is_problem_file:
                print(f"\n此文件可能是问题文件")
                print("\n请使用二分查找脚本定位具体的问题代码行:")
                print(f"  python3 .agents/skills/pypto-aicore-error-locator/scripts/aicore_error_binary_search.py \\")
                print(f"    {cce_file} \\")
                print(f"    '{test_cmd}' \\")
                print(f"    {run_dir}")
                print("\n已找到问题文件，停止排查其他文件")
                print("=" * 80)
                return
            else:
                print(f"\n此文件不是问题文件，继续排查下一个文件")
                print()
        
        print("=" * 80)
        print("所有文件排查完毕，未找到问题文件")
        print("=" * 80)
    elif len(problem_cce_files) == 1 and test_cmd and run_dir:
        cce_file = problem_cce_files[0]
        print(f"\n只有一个问题文件")
        print("\n请使用二分查找脚本定位具体的问题代码行:")
        print(f"  python3 .agents/skills/pypto-aicore-error-locator/scripts/aicore_error_binary_search.py \\")
        print(f"    {cce_file} \\")
        print(f"    '{test_cmd}' \\")
        print(f"    {run_dir}")
        print("=" * 80)
    else:
        print("\n下一步：")
        print("  1. 如果只有一个问题文件，直接使用二分查找脚本定位问题代码行")
        print("  2. 如果存在多个问题文件，需要先测试每个文件确定具体的问题文件")
        print("\n使用方法：")
        print("  # 提供测试命令，自动测试多个问题文件")
        print("  python3 analyze_trace_log.py <device_log_path> <kernel_aicore_dir> <test_cmd> <run_dir>")
        print()
        print("  # 或手动测试每个文件，然后使用二分查找脚本")
        print("  python3 .agents/skills/pypto-aicore-error-locator/scripts/aicore_error_binary_search.py \\")
        print("    <cce_file> <test_cmd> <run_dir>")
        print("=" * 80)


if __name__ == "__main__":
    main()
