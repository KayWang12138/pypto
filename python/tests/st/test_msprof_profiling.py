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


def _get_root_dir() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", "..", ".."))


def _clean_prof_dirs(root_dir: str) -> None:
    for old_dir in glob.glob(os.path.join(root_dir, "PROF*")):
        shutil.rmtree(old_dir, ignore_errors=True)


def _run_msprof(root_dir: str, script_path: str) -> subprocess.CompletedProcess:
    cmd = ["msprof", "python", script_path]
    try:
        result = subprocess.run(
            cmd,
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result
    except subprocess.TimeoutExpired as exc:
        raise pytest.fail("msprof 命令执行超时") from exc
    except FileNotFoundError as exc:
        raise pytest.fail("msprof 命令未找到，请确保 CANN 环境已正确配置") from exc


def _collect_op_summary_files(prof_dirs):
    op_summary_files_found = []
    for prof_dir in prof_dirs:
        op_summary_pattern = os.path.join(
            prof_dir, "mindstudio_profiler_output", "op_summary_*.csv"
        )
        op_summary_files_found.extend(glob.glob(op_summary_pattern))
    return op_summary_files_found


def _find_pypto_in_csv(op_summary_files):
    for csv_file in op_summary_files:
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                if "PyPTO" in f.read():
                    return True
        except Exception as exc:
            _ = exc
    return False


def test_msprof_profiling_pypto_op_summary():
    """
    看护用例：验证 msprof 性能采集功能
    1. 执行 msprof python examples/01_beginner/basic/add_direct.py
    2. 验证 PROF*/mindstudio_profiler_output/op_summary_*.csv 文件生成
    3. 验证 CSV 文件中包含 PyPTO 字样
    """
    root_dir = _get_root_dir()
    _clean_prof_dirs(root_dir)
    add_direct_script = os.path.join(
        root_dir, "examples", "01_beginner", "basic", "add_direct.py"
    )
    assert os.path.exists(add_direct_script), f"脚本不存在: {add_direct_script}"

    _run_msprof(root_dir, add_direct_script)
    prof_dirs = glob.glob(os.path.join(root_dir, "PROF*"))
    assert len(prof_dirs) > 0, f"未在 {root_dir} 下找到 PROF* 文件夹"

    op_summary_files_found = _collect_op_summary_files(prof_dirs)
    pypto_found = _find_pypto_in_csv(op_summary_files_found)
    assert len(op_summary_files_found) > 0, (
        f"未在 PROF* 文件夹中找到 mindstudio_profiler_output/op_summary_*.csv 文件。\n"
        f"已检查的 PROF 目录: {prof_dirs}"
    )

    assert pypto_found, (
        f"在 op_summary CSV 文件中未找到 PyPTO 字样。\n"
        f"已检查的 CSV 文件: {op_summary_files_found}"
    )

    for prof_dir in prof_dirs:
        shutil.rmtree(prof_dir, ignore_errors=True)
