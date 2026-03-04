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
"""在线编译二进制.
"""
import os
import shlex
import shutil
import logging
import subprocess
import tempfile
import ctypes
from pathlib import Path
from typing import Optional


class BuildOnline:

    def __init__(self):
        self.cmake: Optional[Path] = self._which_cmake()
        self.py_mod_torch_version: str = ""
        self.py_mod_torch_root_dir: str = ""
        self.py_mod_torch_cmake_dir: str = ""
        self.py_mod_torch_c_use_cxx11_abi: int = 1
        self._init_torch_param()

    @classmethod
    def _which_cmake(cls) -> Optional[Path]:
        """查找系统级 CMake 可执行文件路径

        排除 cmake pip 包的干扰, 通过遍历 PATH 环境变量查找 ELF 格式的 CMake 可执行文件.

        :return: 系统 CMake 可执行文件路径, 找不到则返回 None
        :rtype: Optional[Path]
        """
        # 拆分 PATH 环境变量为单个目录列表(排除空目录)
        path_dir_lst = [d.strip() for d in os.environ.get("PATH", "").split(os.pathsep) if d.strip()]

        # 遍历每个 PATH 目录, 逐个调用 shutil.which 检查, 限定 shutil.which 只在当前单个目录下查找 cmake
        valid_path_lst = []
        for path_dir in path_dir_lst:
            # 避免 PATH 环境变量中有重复的单元
            if path_dir in valid_path_lst:
                continue
            valid_path_lst.append(path_dir)
            # 检查当前目录
            cmake_str = shutil.which("cmake", path=path_dir)
            if not cmake_str:
                continue
            cmake_file = Path(cmake_str).resolve()
            if not cmake_file.exists() or not cmake_file.is_file():
                continue
            if cmake_file.stat().st_size <= 4:  # 下文读取前 4 字节判断文件是否是 ELF 文件
                continue
            with open(cmake_file, 'rb') as fh:
                header = fh.read(4)  # 前 4 字节是 ELF 文件标识
            if header != b'\x7fELF':
                continue
            return cmake_file
        return None

    @classmethod
    def _build_calculator(cls, tmp_dir: Path, src_dir: Path, build_job_num: int = 4,
                          capture_output: bool = True) -> Path:
        builder = BuildOnline()

        # 路径准备
        build_dir = Path(tmp_dir, "build")
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        install_dir = Path(tmp_dir, "install")

        # CMake Configure
        cmd = f"{builder.cmake} -S {src_dir} -B {build_dir} -DCMAKE_INSTALL_PREFIX={install_dir} "
        cmd += f" -DCMAKE_BUILD_TYPE=Release"
        cmd += f" -DPY3_MOD_TORCH_ROOT_PATH={builder.py_mod_torch_root_dir}"
        cmd += f" -DPY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI={builder.py_mod_torch_c_use_cxx11_abi}"
        logging.debug("CMake Configure, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=capture_output, check=True, text=True, encoding='utf-8')
        ret.check_returncode()

        # CMake Build
        cmd = f"{builder.cmake} --build {build_dir}" + (f" -j {build_job_num}" if build_job_num else "")
        logging.debug("CMake Build, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=capture_output, check=True, text=True, encoding='utf-8')
        ret.check_returncode()

        # CMake Install
        cmd = f"{builder.cmake} --install {build_dir} --prefix={install_dir}"
        logging.debug("CMake Install, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=capture_output, check=True, text=True, encoding='utf-8')
        ret.check_returncode()

        calc_shared = Path(install_dir, "lib/libtile_fwk_calculator.so")
        if not calc_shared.exists():
            raise RuntimeError(f"{calc_shared} not exists.")
        return calc_shared

    @classmethod
    def build_and_load_calculator(cls):
        src_dir = Path(Path(__file__).parent.resolve(), "calculator")
        with tempfile.TemporaryDirectory() as _tmp_dir:
            tmp_dir = Path(_tmp_dir)
            calc_shared = cls._build_calculator(tmp_dir=tmp_dir, src_dir=src_dir)
            ctypes.CDLL(str(calc_shared), mode=ctypes.RTLD_GLOBAL)
            logging.debug("Load %s success.", calc_shared)

    def _init_torch_param(self):
        os_env = os.environ.copy()
        try:
            os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
            import torch
            self.py_mod_torch_version = str(torch.__version__)
            self.py_mod_torch_root_dir = str(Path(torch.__file__).parent)
            self.py_mod_torch_cmake_dir = str(Path(torch.utils.cmake_prefix_path).resolve())
            self.py_mod_torch_c_use_cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
        except (ModuleNotFoundError or ImportError) as e:
            raise RuntimeError(f"Can not import torch, please check your python environment. Error: {e}") from e
        finally:
            os.environ = os_env
