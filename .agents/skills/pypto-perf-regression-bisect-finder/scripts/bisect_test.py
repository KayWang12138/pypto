#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""
PyPTO 性能退化二分查找 - 自动化测试脚本
用于 git bisect run，自动化执行测试、分析和评估流程
"""

import os
import sys
import subprocess
import json
import shutil
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, check=True, capture_output=False, cwd=None):
    """运行命令并返回结果"""
    logger.info(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=check,
        capture_output=capture_output,
        text=True,
        cwd=cwd
    )
    return result


def get_commit_info():
    """获取当前 commit 信息"""
    commit_hash = run_command(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True
    ).stdout.strip()
    
    commit_msg = run_command(
        ["git", "log", "--format=%s", "-1", "HEAD"],
        capture_output=True
    ).stdout.strip()
    
    commit_date = run_command(
        ["git", "log", "--format=%ai", "-1", "HEAD"],
        capture_output=True
    ).stdout.strip()
    
    commit_author = run_command(
        ["git", "log", "--format=%an <%ae>", "-1", "HEAD"],
        capture_output=True
    ).stdout.strip()
    
    return {
        "hash": commit_hash,
        "message": commit_msg,
        "date": commit_date,
        "author": commit_author
    }


def add_debug_options(test_case_path):
    """添加泳道图配置到测试案例"""
    add_debug_script = os.path.join(
        os.path.dirname(__file__),
        "add_debug_options.py"
    )
    
    if not os.path.exists(add_debug_script):
        logger.error(f"错误: 找不到 add_debug_options.py 脚本: {add_debug_script}")
        return False
    
    result = run_command(
        ["python3", add_debug_script, "add", test_case_path],
        check=False
    )
    
    if result.returncode == 0:
        logger.info("✓ 泳道图配置添加成功")
        return True
    else:
        logger.error("✗ 泳道图配置添加失败")
        return False


def restore_test_case(test_case_path):
    """恢复测试案例"""
    add_debug_script = os.path.join(
        os.path.dirname(__file__),
        "add_debug_options.py"
    )
    
    if not os.path.exists(add_debug_script):
        return False
    
    result = run_command(
        ["python3", add_debug_script, "restore", test_case_path],
        check=False
    )
    
    return result.returncode == 0


def compile_pypto(work_dir):
    """编译 PyPTO"""
    logger.info("编译 PyPTO...")
    
    build_dir = os.path.join(work_dir, "build_out")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    
    # 编译
    result = run_command(
        ["python3", "-m", "pip", "install", ".", "--force-reinstall", "--no-deps", "--quiet"],
        check=False,
        cwd=work_dir
    )
    
    if result.returncode == 0:
        logger.info("✓ PyPTO 编译安装成功")
        return True
    else:
        logger.error("✗ PyPTO 编译安装失败")
        return False


def run_test_case(test_case_path, env_vars=None):
    """运行测试案例"""
    logger.info(f"运行测试案例: {test_case_path}")
    
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    
    result = subprocess.run(
        ["python3", test_case_path],
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        logger.info("✓ 测试案例运行成功")
        return True
    else:
        logger.error(f"✗ 测试案例运行失败")
        logger.error(f"错误输出: {result.stderr}")
        return False


def analyze_batch_gaps(output_dir, script_dir):
    """分析批次间隙"""
    logger.info("分析批次间隙...")
    
    swimlane_file = None
    topo_file = None
    
    # 查找泳道图和拓扑文件
    for root, dirs, files in os.walk(output_dir):
        if "merged_swimlane.json" in files:
            swimlane_file = os.path.join(root, "merged_swimlane.json")
        if "dyn_topo.txt" in files:
            topo_file = os.path.join(root, "dyn_topo.txt")
        if swimlane_file and topo_file:
            break
    
    if not swimlane_file or not topo_file:
        logger.error(f"✗ 找不到泳道图或拓扑文件")
        logger.error(f"  swimlane: {swimlane_file}")
        logger.error(f"  topo: {topo_file}")
        return False
    
    output_file = os.path.join(output_dir, "batch_gaps.json")
    
    result = run_command([
        "python3",
        os.path.join(script_dir, "analyze_batch_gaps.py"),
        "--swimlane", swimlane_file,
        "--topo", topo_file,
        "--output", output_file
    ], check=False)
    
    if result.returncode == 0:
        logger.info("✓ 批次间隙分析成功")
        return True
    else:
        logger.error("✗ 批次间隙分析失败")
        return False


def evaluate_performance(output_dir, bisect_condition_file, script_dir):
    """评估当前版本性能"""
    logger.info("评估当前版本性能...")
    
    batch_gaps_file = os.path.join(output_dir, "batch_gaps.json")
    
    if not os.path.exists(batch_gaps_file):
        logger.error(f"✗ 找不到批次间隙文件: {batch_gaps_file}")
        return None
    
    result = run_command([
        "python3",
        os.path.join(script_dir, "auto_diff.py"),
        "--evaluate",
        "--current", batch_gaps_file,
        "--config", bisect_condition_file
    ], check=False)
    
    # 返回退出码 (0=good, 1=bad, 125=skip)
    logger.info(f"评估结果: {result.returncode} (0=good, 1=bad, 125=skip)")
    return result.returncode


def save_commit_info(output_dir, commit_info):
    """保存 commit 信息到输出目录"""
    info_file = os.path.join(output_dir, "commit_info.txt")
    
    with open(info_file, "w", encoding="utf-8") as f:
        f.write(f"Commit: {commit_info['hash']}\n")
        f.write(f"Message: {commit_info['message']}\n")
        f.write(f"Date: {commit_info['date']}\n")
        f.write(f"Author: {commit_info['author']}\n")
    
    logger.info(f"✓ Commit 信息已保存到 {info_file}")


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO 性能退化二分查找 - 自动化测试脚本"
    )
    parser.add_argument(
        "--test-case",
        required=True,
        help="测试案例路径"
    )
    parser.add_argument(
        "--work-dir",
        default=os.getcwd(),
        help="工作目录（默认为当前目录）"
    )
    parser.add_argument(
        "--bisect-condition",
        required=True,
        help="bisect_condition.json 文件路径"
    )
    parser.add_argument(
        "--env-device-id",
        default="1",
        help="NPU 设备 ID（默认为 1）"
    )
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    logger.info("=" * 60)
    logger.info("PyPTO 性能退化二分查找 - 自动化测试")
    logger.info("=" * 60)
    
    # 获取当前 commit 信息
    logger.info("\n[1/6] 获取当前 commit 信息...")
    commit_info = get_commit_info()
    logger.info(f"  Commit: {commit_info['hash']}")
    logger.info(f"  Message: {commit_info['message']}")
    logger.info(f"  Author: {commit_info['author']}")
    
    # 编译 PyPTO
    logger.info("\n[2/6] 编译 PyPTO...")
    if not compile_pypto(args.work_dir):
        sys.exit(125)  # Skip
    
    # 添加泳道图配置
    logger.info("\n[3/6] 添加泳道图配置...")
    if not add_debug_options(args.test_case):
        sys.exit(125)  # Skip
    
    # 设置环境变量
    env_vars = {
        "TILE_FWK_DEVICE_ID": args.env_device_id,
    }
    
    # 加载 PTO_TILE_LIB_CODE_PATH（如果已设置）
    if "PTO_TILE_LIB_CODE_PATH" in os.environ:
        env_vars["PTO_TILE_LIB_CODE_PATH"] = os.environ["PTO_TILE_LIB_CODE_PATH"]
    
    logger.info(f"  环境变量: TILE_FWK_DEVICE_ID={env_vars['TILE_FWK_DEVICE_ID']}")
    
    # 清理旧的 output 目录
    logger.info("\n[4/6] 清理旧的 output 目录...")
    output_dir = os.path.join(args.work_dir, "output")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        logger.info(f"  已删除旧的 output 目录")
    
    # 运行测试案例
    logger.info("\n[5/6] 运行测试案例...")
    if not run_test_case(args.test_case, env_vars):
        restore_test_case(args.test_case)
        sys.exit(125)  # Skip
    
    # 重命名 output 目录
    logger.info("\n[6/6] 处理输出目录...")
    output_dir_new = os.path.join(args.work_dir, f"output_{commit_info['hash']}")
    if os.path.exists(output_dir):
        shutil.move(output_dir, output_dir_new)
        logger.info(f"  已重命名: output -> output_{commit_info['hash']}")
    else:
        logger.error(f"  ✗ 找不到 output 目录")
        restore_test_case(args.test_case)
        sys.exit(125)  # Skip
    
    # 保存 commit 信息
    save_commit_info(output_dir_new, commit_info)
    
    # 恢复测试案例
    restore_test_case(args.test_case)
    
    # 分析批次间隙
    logger.info("\n[7/7] 分析批次间隙...")
    if not analyze_batch_gaps(output_dir_new, script_dir):
        sys.exit(125)  # Skip
    
    # 评估性能
    logger.info("\n[8/8] 评估性能...")
    exit_code = evaluate_performance(
        output_dir_new,
        args.bisect_condition,
        script_dir
    )
    
    if exit_code is None:
        sys.exit(125)  # Skip
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"最终结果: {exit_code} (0=good, 1=bad)")
    logger.info(f"{'=' * 60}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()