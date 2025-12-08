#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 setuptools 及 CMake 集成配置.
"""
import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Any, List

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self):
        super().__init__(name="", sources=[])  # 源文件列表为空，因为实际构建由 CMake 处理


class CMakeUserOption:
    # 额外的命令行配置, 格式: 长选项, 短选项, 描述, 默认值
    USER_OPTION: List[Any] = [
        ('clean-first', None, 'Clean before build', None),
        ('cmake-generator=', None, 'CMake Generator', None),
        ('cmake-args=', None, 'Additional CMake parameters', None),
        ('backend=', None, 'Backend type', None),
        ('disable-install-strip', None, 'Disable strip when install', None),
    ]

    def __init__(self):
        self.clean_first: Optional[bool] = None
        self.cmake_generator: Optional[str] = None
        self.cmake_args: Optional[str] = None
        self.backend: Optional[str] = None
        self.disable_install_strip: bool = False
        self.initialize_options_default()

    def __str__(self):
        ver = sys.version_info
        desc: str = ""
        desc += f"\nEnviron"
        desc += f"\n    Python3               : {sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"
        desc += f"\n{self.__class__.__name__}"
        desc += f"\n    clean-first           : {self.clean_first}"
        desc += f"\n    cmake_generator       : {self.cmake_generator}"
        desc += f"\n    cmake-args            : {self.cmake_args}"
        desc += f"\n    backend               : {self.backend}"
        desc += f"\n    disable-install-strip : {self.disable_install_strip}"
        desc += f"\n"
        return desc

    @classmethod
    def _has_ninja(cls) -> bool:
        """检查 Ninja 是否可用
        """
        try:
            subprocess.check_output(['ninja', '--version'], stderr=subprocess.DEVNULL)
            return True
        except (OSError, subprocess.CalledProcessError):
            return False

    @classmethod
    def _get_cmake_generator(cls, generator: Optional[str]) -> Optional[str]:
        if generator:
            generator = generator.replace(" ", "\ ")
        else:
            if cls._has_ninja():
                generator = "Ninja"
        return generator

    def initialize_options_default(self):
        self.clean_first = None
        self.cmake_generator = None
        self.cmake_args = None
        self.backend = None
        self.disable_install_strip = False

    def initialize_options_from_env(self):
        env_build_ext_args = os.environ.get("PYPTO_BUILD_EXT_ARGS", "")
        if env_build_ext_args:
            parser = argparse.ArgumentParser(description=f"Setuptools CMakeBuild Ext.", add_help=False)
            parser.add_argument("--clean-first", action="store_true", default=False, dest="clean")
            parser.add_argument("--cmake-generator", nargs="?", type=str, default="", dest="cmake_generator")
            parser.add_argument("--cmake-args", nargs="?", type=str, default="", dest="cmake_args")
            parser.add_argument("--backend", nargs="?", type=str, default="", dest="backend")
            parser.add_argument("--disable-install-strip", action="store_true", default=False, dest="strip")
            args, _ = parser.parse_known_args(env_build_ext_args.split())
            self.clean_first = args.clean
            self.cmake_generator = self._get_cmake_generator(generator=args.cmake_generator)
            self.cmake_args = str(args.cmake_args).replace("'", "")
            self.backend = str(args.backend).lower()
            self.disable_install_strip = args.strip

    def finalize_options_normal(self):
        self.clean_first = True if self.clean_first else False
        self.cmake_generator = self._get_cmake_generator(generator=self.cmake_generator)
        self.cmake_args = None if not self.cmake_args else self.cmake_args
        self.backend = str(self.backend).lower() if self.backend else ""
        self.disable_install_strip = True if self.disable_install_strip else False


class CMakeBuild(build_ext, CMakeUserOption):
    """自定义构建命令，调用 CMake 构建系统
    """
    user_options = build_ext.user_options + CMakeUserOption.USER_OPTION

    def initialize_options(self):
        """通过控制命令行选项初始化顺序, 实现实际命令行选项优先生效.
        """
        super().initialize_options()
        # 从环境变量中解析并初始化
        self.initialize_options_default()
        self.initialize_options_from_env()

    def finalize_options(self):
        super().finalize_options()
        self.finalize_options_normal()

    def run(self):
        """执行构建流程
        """
        logging.info("%s", self)
        # 准备构建目录, 使用扩展名创建唯一的构建目录
        build_dir: Path = Path(self.build_temp).resolve()
        if build_dir.exists() and self.clean_first:
            logging.info("Clean Build-Tree(%s)", build_dir)
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        # CMake Configure
        src: Path = Path(__file__).parent.resolve()
        cmd: str = f"cmake -S {src} -B {build_dir}"
        cmd += f" -G {self.cmake_generator}" if self.cmake_generator else ""
        cmd += f" -DPython3_EXECUTABLE={sys.executable} -DCMAKE_INSTALL_PREFIX={self.build_lib}"
        cmd += f" -DENABLE_FEATURE_PYTHON_FRONT_END=pypto"
        cmd += f" -DBUILD_WITH_CANN=OFF" if self.backend in ["cost_model", ] else f" -DBUILD_WITH_CANN=ON"
        cmd += f" {self.cmake_args}" if self.cmake_args else ""
        logging.info("CMake Configure, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8')
        ret.check_returncode()

        # CMake Build
        cmd: str = f"cmake --build {build_dir}" + (f" -- -j {self.parallel}" if self.parallel else "")
        logging.info("CMake Build, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8')
        ret.check_returncode()

        # CMake Install
        cmd: str = f"cmake --install {build_dir} --prefix {self.build_lib}"
        cmd += f"" if self.disable_install_strip else f" --strip"
        logging.info("CMake Install, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8')
        ret.check_returncode()


class SetupCtrl:
    """SetupTools 流程控制
    """

    @classmethod
    def main(cls):
        """主处理流程
        """
        # Setuptools 配置
        setup(
            # 扩展模块配置
            ext_modules=[
                CMakeExtension(),
            ],
            cmdclass={
                'build_ext': CMakeBuild,
            },
        )


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    SetupCtrl.main()
