#!/usr/bin/env python3

import os
import re
import sys
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    read_file,
    write_file,
    get_commentable_lines,
    comment_lines,
    has_error,
    run_test,
    comment_special_lines,
    parse_luid,
    parse_core_idx
)


def analyze_trace(log_file):
    lactstart_events = []
    lactfinish_events = []

    logging.info("=" * 80)
    logging.info("分析追踪日志")
    logging.info("=" * 80)
    logging.info(f"日志文件: {log_file}")
    logging.info("")

    with open(log_file, 'r') as f:
        for line in f:
            if 'trace' in line and 'LActStart' in line:
                luid_match = re.search(r'LUid\{[^}]+\}', line)
                lactstart_match = re.search(r'LActStart\{[^}]+\}', line)
                
                if luid_match and lactstart_match:
                    luid = parse_luid(luid_match.group(0))
                    core_idx = parse_core_idx(lactstart_match.group(0), r'LActStart\{(\d+)\}')
                    
                    if luid and core_idx is not None:
                        lactstart_events.append({
                            'luid': luid,
                            'coreIdx': core_idx
                        })
            
            elif 'trace' in line and 'LActFinish' in line:
                luid_match = re.search(r'LUid\{[^}]+\}', line)
                lactfinish_match = re.search(r'LActFinish\{[^}]+\}', line)
                
                if luid_match and lactfinish_match:
                    luid = parse_luid(luid_match.group(0))
                    core_idx = parse_core_idx(lactfinish_match.group(0), r'LActFinish\{(\d+)\}')
                    
                    if luid and core_idx is not None:
                        lactfinish_events.append({
                            'luid': luid,
                            'coreIdx': core_idx
                        })

    lactstart_core_idxs = sorted(set(event['coreIdx'] for event in lactstart_events))
    lactfinish_core_idxs = sorted(set(event['coreIdx'] for event in lactfinish_events))

    logging.info(f"LActStart 事件数量: {len(lactstart_events)}")
    logging.info(f"LActStart coreIdxs: {lactstart_core_idxs}")
    logging.info(f"LActFinish 事件数量: {len(lactfinish_events)}")
    logging.info(f"LActFinish coreIdxs: {lactfinish_core_idxs}")

    missing_core_idxs = [idx for idx in lactstart_core_idxs if idx not in lactfinish_core_idxs]
    
    logging.info(f"\n缺失的 coreIdxs: {missing_core_idxs}")

    missing_leaf_indices = []
    for event in lactstart_events:
        if event['coreIdx'] in missing_core_idxs:
            leaf_idx = event['luid']['leafIndex']
            if leaf_idx not in missing_leaf_indices:
                missing_leaf_indices.append(leaf_idx)

    logging.info(f"对应的 leafIndices: {missing_leaf_indices}")
    logging.info("")

    return missing_leaf_indices, lactstart_events, lactfinish_events


def find_cce_file(kernel_aicore_dir, leaf_index):
    logging.info(f"定位问题 CCE 文件 (leafIndex: {leaf_index})")
    logging.info(f"kernel_aicore 目录: {kernel_aicore_dir}")
    logging.info("")
    
    record_files = list(Path(kernel_aicore_dir).glob("**/sub_func_*_call_*.h"))
    
    logging.info(f"找到 {len(record_files)} 个 Record 文件")
    
    target_record_file = None
    core_type = None
    func_name = None
    
    for record_file in record_files:
        with open(record_file, 'r') as f:
            content = f.read()
            if f'case {leaf_index}:' in content:
                logging.info(f"\n在文件中找到 leafIndex {leaf_index}: {record_file}")
                target_record_file = record_file
                
                match = re.search(r'sub_func_(\w+)_call_\d+\.h', record_file.name)
                if match:
                    core_type = match.group(1)
                    logging.info(f"Core type: {core_type}")
                
                pattern = rf'case {leaf_index}:\s*\{{\s*\n\s*(\w+)\('
                match = re.search(pattern, content)
                if match:
                    func_name = match.group(1)
                    logging.info(f"Function name: {func_name}")
                break
    
    if not target_record_file:
        logging.info(f"\n错误：无法在任何 Record 文件中找到 leafIndex {leaf_index}")
        return None
    
    if func_name:
        match = re.search(r'_(\d+)_(\d+)_(\d+)$', func_name)
        if match:
            cce_id = match.group(1)
            id_val = match.group(2)
            func_hash = match.group(3)
            cce_pre_name = func_name[:func_name.rfind(f'_{cce_id}_{id_val}_{func_hash}')]
            
            logging.info(f"\n从函数名提取信息:")
            logging.info(f"  函数名: {func_name}")
            logging.info(f"  CCE_pre_name: {cce_pre_name}")
            logging.info(f"  CCE_ID: {cce_id}")
            logging.info(f"  ID: {id_val}")
            logging.info(f"  func_hash: {func_hash}")
            
            cce_pattern = f"{cce_pre_name}_{cce_id}_*_{id_val}_{core_type}.cpp"
            cce_files = list(Path(kernel_aicore_dir).glob(f"**/{cce_pattern}"))
            
            logging.info(f"\n搜索 CCE 文件模式: {cce_pattern}")
            logging.info(f"找到 {len(cce_files)} 个匹配文件")
            
            if cce_files:
                logging.info(f"\n找到 CCE 文件: {cce_files[0]}")
                return str(cce_files[0])
            else:
                logging.info(f"\n错误：无法查找到匹配模式的 CCE 文件: {cce_pattern}")
                return None
        else:
            logging.info(f"\n错误：无法解析函数名格式: {func_name}")
            return None
    
    return None


def test_cce_file(cce_file, test_cmd, run_dir):
    logging.info(f"测试 CCE 文件: {cce_file}")
    
    backup_file = cce_file + ".bak"
    shutil.copy(cce_file, backup_file)
    cce_lines = read_file(cce_file)
    original_lines = cce_lines.copy()
    
    cce_lines = comment_special_lines(cce_lines)
    
    commentable_lines = get_commentable_lines(cce_lines)
    logging.info(f"可注释的行数: {len(commentable_lines)}")
    
    if not commentable_lines:
        logging.info("错误：没有可注释的行")
        write_file(cce_file, original_lines)
        os.remove(backup_file)
        return False, None
    
    logging.info("注释所有可注释的行...")
    current_lines = cce_lines.copy()
    current_lines = comment_lines(current_lines, commentable_lines)
    
    write_file(cce_file, current_lines)
    logging.info("运行测试...")
    returncode, output = run_test(test_cmd, run_dir)
    error_exists = has_error(returncode, output)
    
    write_file(cce_file, original_lines)
    os.remove(backup_file)
    
    if error_exists:
        logging.info("结果: 注释所有行后仍有 error，此文件可能不是问题文件")
        return False, None
    else:
        logging.info("结果: 注释所有行后运行成功（无 error），此文件可能是问题文件")
        return True, commentable_lines


def find_trace_log_file(device_log_path):
    logging.info(f"在 {device_log_path} 下搜索包含 trace 的日志文件...")
    
    log_files = list(Path(device_log_path).rglob("*.log"))
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                if 'trace' in content and ('LActStart' in content or 'LActFinish' in content):
                    logging.info(f"找到 trace 日志文件: {log_file}")
                    return str(log_file)
        except Exception:
            continue
    
    logging.info(f"错误：在 {device_log_path} 下未找到包含 trace 的日志文件")
    return None


def print_usage():
    logging.info("用法: python3 analyze_trace_log.py <device_log_path> <kernel_aicore_dir> [test_cmd] [run_dir]")
    logging.info("")
    logging.info("参数说明:")
    logging.info("  device_log_path: device log 落盘路径")
    logging.info("  kernel_aicore_dir: kernel_aicore 目录路径")
    logging.info("  test_cmd: （可选）触发 aicore error 的测试命令")
    logging.info("  run_dir: （可选）运行测试命令的目录路径")


def validate_paths(device_log_path, kernel_aicore_dir, run_dir):
    if not os.path.exists(device_log_path):
        logging.info(f"错误：device log 路径不存在: {device_log_path}")
        return False
    
    if not os.path.exists(kernel_aicore_dir):
        logging.info(f"错误：kernel_aicore 目录不存在: {kernel_aicore_dir}")
        return False
    
    if run_dir and not os.path.exists(run_dir):
        logging.info(f"错误：运行目录不存在: {run_dir}")
        return False
    
    return True


def main():
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)
    
    device_log_path = sys.argv[1]
    kernel_aicore_dir = sys.argv[2]
    test_cmd = sys.argv[3] if len(sys.argv) > 3 else None
    run_dir = sys.argv[4] if len(sys.argv) > 4 else None
    
    device_log_path = os.path.abspath(device_log_path)
    kernel_aicore_dir = os.path.abspath(kernel_aicore_dir)
    
    if test_cmd and run_dir:
        run_dir = os.path.abspath(run_dir)
    
    if not validate_paths(device_log_path, kernel_aicore_dir, run_dir):
        sys.exit(1)
    
    log_file = find_trace_log_file(device_log_path)
    if not log_file:
        sys.exit(1)
    
    missing_leaf_indices, lactstart_events, lactfinish_events = analyze_trace(log_file)
    
    if not missing_leaf_indices:
        logging.info("=" * 80)
        logging.info("结果：没有发现缺失的 leaf index")
        logging.info("所有任务已成功完成，无需定位问题 CCE 文件")
        logging.info("=" * 80)
        return
    
    problem_cce_files = []
    for leaf_index in missing_leaf_indices:
        logging.info("=" * 80)
        cce_file = find_cce_file(kernel_aicore_dir, leaf_index)
        if cce_file:
            problem_cce_files.append(cce_file)
    
    logging.info("=" * 80)
    logging.info("定位结果")
    logging.info("=" * 80)
    
    if not problem_cce_files:
        logging.info("\n未找到问题 CCE 文件")
        logging.info("=" * 80)
        return
    
    logging.info(f"\n找到 {len(problem_cce_files)} 个问题 CCE 文件:")
    for i, cce_file in enumerate(problem_cce_files, 1):
        logging.info(f"  {i}. {cce_file}")
    
    if len(problem_cce_files) > 1 and test_cmd and run_dir:
        logging.info("\n存在多个问题 CCE 文件，依次排查每个文件...")
        logging.info("")
        
        for i, cce_file in enumerate(problem_cce_files, 1):
            logging.info("=" * 80)
            logging.info(f"排查文件 {i}/{len(problem_cce_files)}: {cce_file}")
            logging.info("=" * 80)
            
            is_problem_file, commentable_lines = test_cce_file(cce_file, test_cmd, run_dir)
            
            if is_problem_file:
                logging.info(f"\n此文件可能是问题文件")
                logging.info("\n请使用二分查找脚本定位具体的问题代码行:")
                logging.info(f"  python3 .agents/skills/pypto-aicore-error-locator/scripts/aicore_error_binary_search.py \\")
                logging.info(f"    {cce_file} \\")
                logging.info(f"    '{test_cmd}' \\")
                logging.info(f"    {run_dir}")
                logging.info("\n已找到问题文件，停止排查其他文件")
                logging.info("=" * 80)
                return
            else:
                logging.info(f"\n此文件不是问题文件，继续排查下一个文件")
                logging.info("")
        
        logging.info("=" * 80)
        logging.info("所有文件排查完毕，未找到问题文件")
        logging.info("=" * 80)
    elif len(problem_cce_files) == 1 and test_cmd and run_dir:
        cce_file = problem_cce_files[0]
        logging.info(f"\n只有一个问题文件")
        logging.info("\n请使用二分查找脚本定位具体的问题代码行:")
        logging.info(f"  python3 .agents/skills/pypto-aicore-error-locator/scripts/aicore_error_binary_search.py \\")
        logging.info(f"    {cce_file} \\")
        logging.info(f"    '{test_cmd}' \\")
        logging.info(f"    {run_dir}")
        logging.info("=" * 80)
    else:
        logging.info("\n下一步：")
        logging.info("  1. 如果只有一个问题文件，直接使用二分查找脚本定位问题代码行")
        logging.info("  2. 如果存在多个问题文件，需要先测试每个文件确定具体的问题文件")
        logging.info("\n使用方法：")
        logging.info("  # 提供测试命令，自动测试多个问题文件")
        logging.info("  python3 analyze_trace_log.py <device_log_path> <kernel_aicore_dir> <test_cmd> <run_dir>")
        logging.info("")
        logging.info("  # 或手动测试每个文件，然后使用二分查找脚本")
        logging.info("  python3 .agents/skills/pypto-aicore-error-locator/scripts/aicore_error_binary_search.py \\")
        logging.info("    <cce_file> <test_cmd> <run_dir>")
        logging.info("=" * 80)


if __name__ == "__main__":
    main()
