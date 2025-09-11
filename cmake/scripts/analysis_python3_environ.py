#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
""" Python3环境分析.

Python3环境分析.
"""
import argparse
import logging
import importlib
from pathlib import Path
from typing import Optional


class Analysis:

    def __init__(self, args):
        self.executable: Path = Path(args.executable[0]).resolve()
        self.print_pybind11_dir: bool = args.print_pybind11_dir
        self.print_torch_version: bool = args.print_torch_version
        self.judge_pytest_installed: bool = args.judge_pytest_installed
        self.judge_pytest_forked_installed: bool = args.judge_pytest_forked_installed

    @staticmethod
    def analysis_pybind11_dir() -> str:
        """获取 pybind11_DIR, 以便外层 CMake 处理

        :return: pybind11_DIR
        """
        pybind11_dir: Optional[Path] = None
        try:
            import pybind11
            pybind11_dir = Path(pybind11.get_cmake_dir()).resolve()
        except ModuleNotFoundError:
            pass
        return str(pybind11_dir) if pybind11_dir else ""

    @staticmethod
    def analysis_torch_version() -> str:
        torch_version: Optional[str] = None
        try:
            import torch
            torch_version = str(torch.__version__)
        except ModuleNotFoundError:
            pass
        return str(torch_version) if torch_version else ""

    @staticmethod
    def analysis_pytest() -> str:
        installed: bool = False
        try:
            importlib.import_module("pytest")
            installed = True
        except (ModuleNotFoundError or ImportError):
            pass
        return "True" if installed else ""

    @staticmethod
    def analysis_pytest_forked() -> str:
        installed: bool = False
        try:
            importlib.import_module("pytest_forked")
            installed = True
        except (ModuleNotFoundError or ImportError):
            pass
        return "True" if installed else ""

    @staticmethod
    def main() -> str:
        """主处理流程
        """
        # 参数注册
        parser = argparse.ArgumentParser(description=f"Python3-Environ Analysis.", epilog="Best Regards!")
        parser.add_argument("-e", "--executable", nargs=1, type=str, required=True,
                            help="Specific python3 executable path.")
        parser.add_argument("--print_pybind11_dir", action="store_true", default=False,
                            help="Print pip3::pybind11 dir.")
        parser.add_argument("--print_torch_version", action="store_true", default=False,
                            help="Print pip3::torch version.")
        parser.add_argument("--judge_pytest_installed", action="store_true", default=False,
                            help="Judge pip3::pytest installed.")
        parser.add_argument("--judge_pytest_forked_installed", action="store_true", default=False,
                            help="Judge pip3::pytest-forked installed.")
        # 流程处理
        ctrl = Analysis(args=parser.parse_args())
        return ctrl.analysis()

    def analysis(self) -> str:
        if self.print_pybind11_dir:
            return self.analysis_pybind11_dir()
        elif self.print_torch_version:
            return self.analysis_torch_version()
        elif self.judge_pytest_installed:
            return self.analysis_pytest()
        elif self.judge_pytest_forked_installed:
            return self.analysis_pytest_forked()
        else:
            return ""


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    print(Analysis.main(), end='')
