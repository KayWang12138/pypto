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
import sys
import subprocess
import shutil

from typing import Optional, Any, List
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel


class MetaHelper:

    _SRC_ROOT: Path
    _CONFIG: Any

    @staticmethod
    def has_ninja():
        """检查 Ninja 是否可用
        """
        try:
            subprocess.check_output(['ninja', '--version'], stderr=subprocess.DEVNULL)
            return True
        except (OSError, subprocess.CalledProcessError):
            return False

    @classmethod
    def init(cls):
        cls._SRC_ROOT = Path(__file__).parent.resolve()
        toml: Path = Path(cls._SRC_ROOT, "pyproject.toml")
        ver = sys.version_info
        if ver >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
        with open(toml, 'rb') as fh:
            cls._CONFIG = tomllib.load(fh)

    @classmethod
    def get_metadata(cls, k: str, t: str = "project", d: Optional[str] = "") -> Any:
        """从 pyproject.toml 读取项目元数据
        """
        return cls._CONFIG.get(t, {}).get(k, d)

    @classmethod
    def name(cls) -> str:
        return cls.get_metadata(k="name")

    @classmethod
    def readme(cls) -> str:
        return cls._read_file(sub_path=cls.get_metadata(k="readme"))

    @classmethod
    def license(cls) -> str:
        return cls._read_file(sub_path=cls.get_metadata(k="license").get("file", ""))

    @classmethod
    def src_root(cls) -> Path:
        return cls._SRC_ROOT

    @classmethod
    def _read_file(cls, sub_path: str) -> str:
        s: str = ""
        f: Path = Path(cls.src_root(), sub_path)
        if f.exists() and f.is_file():
            with open(f, 'r') as fh:
                s = fh.read()
        return s


class CMakeExtension(Extension):
    def __init__(self):
        super().__init__(name=MetaHelper.name(), sources=[])  # 源文件列表为空，因为实际构建由 CMake 处理


class CMakeUserOption:

    # 额外的命令行配置, 格式: 长选项, 短选项, 描述, 默认值
    USER_OPTION: List[Any] = [
        ('clean-first', None, 'Clean before build', None),
        ('cmake-args=', None, 'Additional CMake parameters', None),
        ('disable-install-strip', None, 'Disable strip when install', None),
    ]

    def __init__(self):
        self.clean_first: Optional[bool] = None
        self.cmake_args: Optional[str] = None
        self.disable_install_strip: bool = False
        self.initialize_options_default()

    def __str__(self):
        ver = sys.version_info
        desc: str = ""
        desc += f"\nEnviron"
        desc += f"\n    Python3               : {sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"
        desc += f"\n{self.__class__.__name__}"
        desc += f"\n    clean-first           : {self.clean_first}"
        desc += f"\n    cmake-args            : {self.cmake_args}"
        desc += f"\n    disable-install-strip : {self.disable_install_strip}"
        desc += f"\n"
        return desc

    def initialize_options_default(self):
        self.clean_first = None
        self.cmake_args = None
        self.disable_install_strip = False

    def initialize_options_from_env(self):
        env_build_ext_args = os.environ.get("PYPTO_BUILD_EXT_ARGS", "")
        if env_build_ext_args:
            parser = argparse.ArgumentParser(description=f"Setuptools CMakeBuild Ext.", add_help=False)
            parser.add_argument("--clean-first", action="store_true", default=False, dest="clean")
            parser.add_argument("--cmake-args", nargs="?", type=str, default="", dest="cmake_args")
            parser.add_argument("--disable-install-strip", action="store_true", default=False, dest="strip")
            args, _ = parser.parse_known_args(env_build_ext_args.split())
            self.clean_first = args.clean
            self.cmake_args = str(args.cmake_args).replace("'", "")
            self.disable_install_strip = args.strip

    def initialize_options_from_distribution(self, distribution):
        if hasattr(distribution, 'clean_first'):
            self.clean_first = distribution.clean_first
        if hasattr(distribution, 'cmake_args'):
            if distribution.cmake_args is not None:
                self.cmake_args = distribution.cmake_args
        if hasattr(distribution, 'disable_install_strip'):
            self.disable_install_strip = distribution.disable_install_strip

    def finalize_options_normal(self):
        self.clean_first = True if self.clean_first else False
        self.cmake_args = None if not self.cmake_args else self.cmake_args
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
        self.initialize_options_from_distribution(distribution=self.distribution)

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
        generator: str = "Ninja" if MetaHelper.has_ninja() else "'Unix Makefiles'"
        cmd: str = f"cmake -S {MetaHelper.src_root()} -B {build_dir} -G {generator}"
        cmd += f" -DPython3_EXECUTABLE={sys.executable} -DCMAKE_INSTALL_PREFIX={self.build_lib}"
        cmd += f" -DENABLE_FEATURE_PYTHON_FRONT_END={MetaHelper.name()} -DBUILD_WITH_CANN=ON"
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


class CustomBdistWheel(bdist_wheel, CMakeUserOption):
    user_options = bdist_wheel.user_options + CMakeUserOption.USER_OPTION

    def initialize_options(self):
        """通过控制命令行选项初始化顺序, 实现实际命令行选项优先生效.
        """
        super().initialize_options()
        self.initialize_options_default()

    def finalize_options(self):
        super().finalize_options()
        self.finalize_options_normal()
        # 将参数存储到 distribution 中
        self.distribution.clean_first = self.clean_first
        self.distribution.cmake_args = self.cmake_args
        self.distribution.disable_install_strip = self.disable_install_strip

    def run(self):
        # 在运行前确保参数已传递
        build_ext_cmd = self.distribution.get_command_obj('build_ext')
        if build_ext_cmd:
            if self.cmake_args:
                build_ext_cmd.cmake_args = self.cmake_args
            if self.cmake_args:
                build_ext_cmd.cmake_args = self.cmake_args
            if self.disable_install_strip:
                build_ext_cmd.disable_install_strip = self.disable_install_strip
        super().run()


class SetupCtrl:
    """SetupTools 流程控制
    """

    @classmethod
    def main(cls):
        """主处理流程
        """
        # Setuptools 配置
        setup(
            # 基本元数据
            name=MetaHelper.name(),
            version=MetaHelper.get_metadata(k="version"),
            description=MetaHelper.get_metadata(k="description"),
            long_description=MetaHelper.readme(),
            long_description_content_type="text/markdown",
            license=MetaHelper.license(),

            # 包结构配置
            packages=find_packages(where="python"),
            package_dir={'': "python"},
            include_package_data=True,

            # 扩展模块配置
            ext_modules=[
                CMakeExtension(),
            ],
            cmdclass={
                'build_ext': CMakeBuild,
                'bdist_wheel': CustomBdistWheel,
            },

            # 依赖和兼容性配置
            install_requires=MetaHelper.get_metadata(k="dependencies"),
            python_requires=MetaHelper.get_metadata(k="requires-python"),
            zip_safe=False,
        )


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    MetaHelper.init()
    SetupCtrl.main()
