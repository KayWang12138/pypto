#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import os
import glob
import shutil
import subprocess
import pytest


def test_msprof_profiling_pypto_op_summary():
    """
    看护用例：验证 msprof 性能采集功能
    1. 执行 msprof python examples/01_beginner/basic/add_direct.py
    2. 验证 PROF*/mindstudio_profiler_output/op_summary_*.csv 文件生成
    3. 验证 CSV 文件中包含 PyPTO 字样
    """
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 从 python/tests/st 回到根目录
    root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    
    # 清理旧的 PROF* 文件夹
    old_prof_dirs = glob.glob(os.path.join(root_dir, "PROF*"))
    for old_dir in old_prof_dirs:
        shutil.rmtree(old_dir, ignore_errors=True)
    
    # 构建要执行的脚本路径
    add_direct_script = os.path.join(root_dir, "examples", "01_beginner", "basic", "add_direct.py")
    assert os.path.exists(add_direct_script), f"脚本不存在: {add_direct_script}"
    
    # 执行 msprof 命令
    cmd = ["msprof", "python", add_direct_script]
    try:
        result = subprocess.run(
            cmd,
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        # 打印输出以便调试
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        print(f"returncode: {result.returncode}")
    except subprocess.TimeoutExpired:
        pytest.fail("msprof 命令执行超时")
    except FileNotFoundError:
        pytest.fail("msprof 命令未找到，请确保 CANN 环境已正确配置")
    
    # 验证 PROF* 文件夹生成
    prof_dirs = glob.glob(os.path.join(root_dir, "PROF*"))
    assert len(prof_dirs) > 0, f"未在 {root_dir} 下找到 PROF* 文件夹"
    
    # 验证 op_summary_*.csv 文件生成并包含 PyPTO
    pypto_found = False
    op_summary_files_found = []
    
    for prof_dir in prof_dirs:
        # 搜索 mindstudio_profiler_output/op_summary_*.csv 文件
        op_summary_pattern = os.path.join(
            prof_dir, "mindstudio_profiler_output", "op_summary_*.csv"
        )
        op_summary_files = glob.glob(op_summary_pattern)
        op_summary_files_found.extend(op_summary_files)
        
        for csv_file in op_summary_files:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "PyPTO" in content:
                        pypto_found = True
                        print(f"在 {csv_file} 中找到 PyPTO 字样")
                        break
            except Exception as e:
                print(f"读取 {csv_file} 时出错: {e}")
        
        if pypto_found:
            break
    
    # 断言检查
    assert len(op_summary_files_found) > 0, (
        f"未在 PROF* 文件夹中找到 mindstudio_profiler_output/op_summary_*.csv 文件。\n"
        f"已检查的 PROF 目录: {prof_dirs}"
    )
    
    assert pypto_found, (
        f"在 op_summary CSV 文件中未找到 PyPTO 字样。\n"
        f"已检查的 CSV 文件: {op_summary_files_found}"
    )
    
    print("✓ msprof 性能采集测试通过，成功找到 PyPTO 字样")
