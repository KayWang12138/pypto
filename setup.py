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
"""集成 CMake 处理的 setuptools 配置.
"""
import argparse
import importlib
import logging
import hashlib
import math
import multiprocessing
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import site
import sysconfig
import warnings
from pathlib import Path
from typing import Optional, Any, List, Tuple, Union
from importlib import metadata

from setuptools import setup, Extension
from setuptools.command.editable_wheel import editable_wheel
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self):
        super().__init__(name="", sources=[])  # 源文件列表为空, 因为实际构建由 CMake 处理


class EditModeHelper:
    @staticmethod
    def get_pip_edit_mode_install_path() -> Path:
        """获取 pip edit 模式实际安装包的路径

        以 editable 模式执行时, editable_wheel 结束后, whl 包的安装由 pip 接管.
        无法在通过自定义 setuptools 子命令('install') 的方式获取 whl 安装路径.
        """
        # 优先取用户级路径, 再取系统级路径
        site_paths = site.getsitepackages()
        # 遍历找到包含 "dist-packages" 的路径(Debian/Ubuntu), 无则取第一个 site-packages
        for path in site_paths:
            if "dist-packages" in path:
                return Path(path)
        # 非 Debian 系统(如 CentOS/Windows), 返回默认 platlib
        rst: Path = Path(site_paths[0]) if site_paths else Path(sysconfig.get_path("platlib"))
        return rst


class CustomEditableWheel(editable_wheel, EditModeHelper):
    """自定义 editable_wheel 命令

    1. 感知 -e 模式, 传递给 build_ext 以便其处理 CMake install 路径;
    2. 接收 build_ext 传递的 CMake install files 并回写入 whl 包的 RECORD 文件, 以便 -e 模式下对应文件可以随 uninstall 流程删除;

    回写 RECORD 文件功能暂未使能, 因插入 RECORD 记录内对应的文件本质未受 whl 包管理.
    在多次执行 pip install -e 的场景下, 会执行先编译(Install 或 Rewrite对应 CMake Install 文件), 再卸载, 再安装的流程.
    卸载阶段因 RECORD 文件内有相关文件记录, 会导致对应文件被删除, 进而导致重复 pip install -e 结束后, 对应文件被删除的问题.
    """

    def run(self):
        # 传递 -e 模式标记给 build_ext 命令
        build_ext_cmd = self.distribution.get_command_obj("build_ext")
        build_ext_cmd.pypto_editable_mode = True  # 设置标记
        # 继续执行标准的命令流程(这会触发 build_ext, egg_info)
        super().run()

    def _insert_cmake_install_files_to_whl_record_file(self):
        """接收 build_ext 传递的 CMake install files 并回写入 whl 包的 RECORD 文件
        """
        # 获取 RECORD 新增条目字符串
        record_str, record_num = self._get_cmake_install_files_record_info()
        if not record_str:
            return

        # 获取 setuptools 内置的 wheel 包, 避免直接使用 wheel 包, 减少因与 setuptools 内置的 wheel 包版本不一致导致兼容性问题的
        try:
            vendor_wheel = importlib.import_module('setuptools._vendor.wheel.wheelfile')
            setuptools_wheel = getattr(vendor_wheel, 'WheelFile')
        except ImportError as e:
            raise ImportError("Could not import setuptools wheel module") from e

        # 找到 whl 包的 RECORD 文件路径(如: pypto-0.0.1.dist-info/RECORD)
        whl_file: Path = self._get_editable_whl_file()
        with setuptools_wheel(whl_file, "r") as wf:
            record_file_lst: List[Path] = [Path(p) for p in wf.namelist() if p.endswith(".dist-info/RECORD")]
        if not record_file_lst:
            raise RuntimeError(f"Can't find RECORD file in {whl_file}")
        record_file: Path = record_file_lst[0]

        # 将新增条目插入 whl 包的 RECORD 文件
        logging.info("Overwrite RECORD(%s), will insert %s entries.", record_file, record_num)
        with setuptools_wheel(whl_file, "a") as wf:
            wf.writestr(str(record_file), record_str.encode("utf-8"))

    def _get_cmake_install_files_record_info(self) -> Tuple[str, int]:
        # 获取 cmake install 文件列表
        install_files: List[Path] = self._get_cmake_install_files()
        if not install_files:
            return "", 0
        # 生成 cmake install 文件的 RECORD 条目(符合 setuptools 标准格式)
        record_entries: List[str] = []
        site_pkg: Path = self.get_pip_edit_mode_install_path()  # site-packages 绝对路径
        for abs_file in install_files:
            # 跳过不存在的文件
            if not abs_file.exists():
                continue
            # 计算 RECORD 条目
            # 格式1: 相对于 site-packages 的路径, RECORD 中必须用此路径, 否则 pip 无法识别
            # 格式2: SHA256 哈希(格式: sha256=xxx)
            # 格式3: 文件大小(字节数)
            rel_path: str = os.path.relpath(str(abs_file), str(site_pkg))
            with open(abs_file, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            file_hash = f"sha256={file_hash}"  # 与 setuptools 原生格式一致
            file_size = os.path.getsize(abs_file)
            # 拼接 RECORD 条目(格式: 路径,哈希,大小)
            entry = f"{rel_path},{file_hash},{file_size}"
            record_entries.append(entry)
        return "\n".join(record_entries), len(record_entries)

    def _get_cmake_install_files(self) -> List[Path]:
        install_files: List[Path] = []
        if not (hasattr(self, 'pypto_install_manifest_lst') and self.pypto_install_manifest_lst):
            logging.warning("Can't get any CMake install manifest.")
        else:
            install_files = [Path(p) for p in getattr(self, "pypto_install_manifest_lst", [])]
        return install_files

    def _get_editable_whl_file(self) -> Path:
        dist_info = self.get_finalized_command("dist_info")
        dist_name: str = getattr(dist_info, "name", "pypto")
        whl_pattern = f"{dist_name}-*.editable-*.whl"
        whl_file_lst = list(Path(self.dist_dir).glob(whl_pattern))
        if not whl_file_lst:
            raise RuntimeError(f"Can't get whl file, Dir: {self.dist_dir}, pattern: {whl_pattern}")
        return Path(whl_file_lst[0])


class CMakeUserOption:
    # 额外的命令行配置, 格式: 长选项, 短选项, 描述, 默认值
    USER_OPTION: List[Any] = [
        ('cmake-generator=', None, 'CMake Generator', None),
        ('cmake-build-type=', None, 'CMake Build Type', None),
        ('cmake-options=', None, 'CMake Options', None),
        ('cmake-verbose', None, 'Enable CMake Verbose Output', None),
    ]

    def __init__(self):
        self.cmake_generator: Optional[str] = None
        self.cmake_build_type: Optional[str] = None
        self.cmake_options: Optional[str] = None
        self.cmake_verbose: bool = False
        # 获取 CMake 路径
        self.cmake: Optional[Path] = None

    def __str__(self):
        ver = sys.version_info
        ver1 = metadata.version("setuptools")
        ver2 = metadata.version("pybind11")

        desc: str = ""
        desc += f"\nEnviron"
        desc += f"\n    Python3               : {sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"
        desc += f"\n        pip"
        desc += f"\n               setuptools : {ver1}"
        desc += f"\n                 pybind11 : {ver2}"
        desc += f"\n    CMake                 : {self.cmake}"
        desc += f"\n{self.__class__.__name__}"
        desc += f"\n    cmake_generator       : {self.cmake_generator}"
        desc += f"\n    cmake_build_type      : {self.cmake_build_type}"
        desc += f"\n    cmake-options         : {self.cmake_options}"
        desc += f"\n    cmake-verbose         : {self.cmake_verbose}"
        desc += f"\n"
        return desc

    @staticmethod
    def is_macos() -> bool:
        """检测是否为 macOS 平台"""
        import platform
        return platform.system() == "Darwin"

    @staticmethod
    def which_cmake() -> Optional[Path]:
        """查找系统级 CMake 可执行文件路径

        支持 Linux (ELF) 和 macOS (Mach-O 或 Python 脚本) 平台
        """
        import platform
        is_macos = platform.system() == "Darwin"

        # 拆分 PATH 环境变量为单个目录列表（排除空目录）
        path_dir_lst = [d.strip() for d in os.environ.get("PATH", "").split(os.pathsep) if d.strip()]

        # macOS/Linux: 添加常见的 cmake 安装路径（pip 用户安装、Homebrew 等）
        extra_paths = []
        if is_macos:
            # pip 用户安装路径
            user_base = subprocess.run([sys.executable, '-m', 'site', '--user-base'],
                                      capture_output=True, text=True, timeout=5)
            if user_base.returncode == 0:
                user_bin = os.path.join(user_base.stdout.strip(), 'bin')
                if os.path.exists(user_bin):
                    extra_paths.append(user_bin)
            # Homebrew 路径
            extra_paths.extend(['/opt/homebrew/bin', '/usr/local/bin'])
        else:
            # Linux: 常见路径
            extra_paths.extend(['/usr/local/bin', '/usr/bin', os.path.expanduser('~/.local/bin')])

        # 合并路径列表，extra_paths 优先
        all_paths = extra_paths + path_dir_lst

        # 遍历每个目录查找 cmake
        valid_path_lst: List[str] = []
        for path_dir in all_paths:
            if not path_dir or path_dir in valid_path_lst:
                continue
            valid_path_lst.append(path_dir)
            if not os.path.exists(path_dir):
                continue

            cmake_str: Optional[Union[Path, str]] = shutil.which("cmake", path=path_dir)
            if not cmake_str:
                continue
            cmake_file: Path = Path(cmake_str).resolve()
            if not cmake_file.exists() or not cmake_file.is_file():
                continue
            if cmake_file.stat().st_size <= 4:
                continue
            with open(cmake_file, 'rb') as fh:
                header = fh.read(4)

            # Linux: 检查 ELF 文件标识
            if header == b'\x7fELF':
                return cmake_file

            # macOS: 检查 Mach-O 文件标识 (fat binary 或 64-bit)
            if is_macos:
                # Mach-O magic numbers: cafebabe (fat), feedface (32-bit), feedfacf (64-bit)
                if header in [b'\xca\xfe\xba\xbe', b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf',
                              b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe']:
                    return cmake_file
                # Python 脚本 (pip 安装的 cmake)
                if header.startswith(b'#!/'):
                    try:
                        result = subprocess.run([str(cmake_file), '--version'],
                                              capture_output=True, timeout=5, text=True)
                        if result.returncode == 0 and 'cmake version' in result.stdout:
                            return cmake_file
                    except Exception:
                        continue
            continue
        return None

    @staticmethod
    def find_homebrew_gcc() -> Tuple[Optional[str], Optional[str]]:
        """在 macOS 上查找 Homebrew 安装的 GCC

        Returns:
            Tuple[Optional[str], Optional[str]]: (gcc_path, g++_path) 或 (None, None)
        """
        homebrew_paths = ["/opt/homebrew/bin", "/usr/local/bin"]
        for brew_path in homebrew_paths:
            if not os.path.exists(brew_path):
                continue
            # 查找 gcc-XX 版本，优先使用最新版本
            gcc_versions = []
            for f in os.listdir(brew_path):
                import re
                match = re.match(r'gcc-(\d+)$', f)
                if match:
                    gcc_versions.append((int(match.group(1)), f))
            gcc_versions.sort(reverse=True)  # 降序排列，最新版本优先

            for ver, gcc_name in gcc_versions:
                gcc_path = os.path.join(brew_path, gcc_name)
                gxx_path = os.path.join(brew_path, f"g++-{ver}")
                if os.path.exists(gcc_path) and os.path.exists(gxx_path):
                    return gcc_path, gxx_path
        return None, None

    @classmethod
    def setup_macos_gcc_env(cls) -> None:
        """在 macOS 上设置 GCC 环境变量（如果未设置）"""
        import platform
        if platform.system() != "Darwin":
            return

        # 如果已经设置了 CC/CXX 环境变量，检查是否为 GCC
        cc = os.environ.get("CC", "")
        cxx = os.environ.get("CXX", "")
        if cc and cxx:
            # 检查是否为 GCC
            try:
                result = subprocess.run([cc, "--version"], capture_output=True, text=True, timeout=5)
                if "gcc" in result.stdout.lower() or "Free Software Foundation" in result.stdout:
                    logging.info("Using pre-configured GCC: CC=%s, CXX=%s", cc, cxx)
                    return
            except Exception:
                pass

        # 查找 Homebrew GCC
        gcc_path, gxx_path = cls.find_homebrew_gcc()
        if gcc_path and gxx_path:
            os.environ["CC"] = gcc_path
            os.environ["CXX"] = gxx_path
            logging.info("macOS: Auto-configured Homebrew GCC: CC=%s, CXX=%s", gcc_path, gxx_path)
        else:
            # 提供安装指导
            logging.warning(
                "macOS: GNU GCC not found. Python frontend requires GCC.\n"
                "Please install GCC via Homebrew:\n"
                "  1. Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"\n"
                "  2. Install GCC: brew install gcc\n"
                "  3. Rebuild the project\n"
                "Or specify GCC compiler manually:\n"
                "  export CC=/opt/homebrew/bin/gcc-14\n"
                "  export CXX=/opt/homebrew/bin/g++-14"
            )

    def initialize_options_cmake(self):
        # 赋初值, 此处需赋初值, 否则 setuptools 会丢失对应参数
        self.cmake_generator = None
        self.cmake_build_type = None
        self.cmake_options = None
        self.cmake_verbose = False

        # macOS 平台：自动配置 Homebrew GCC
        self.setup_macos_gcc_env()

        self.cmake: Optional[Path] = self.which_cmake()
        if not self.cmake:
            raise RuntimeError(f"Can't find cmake")

        # 从环境变量(如有)中获取配置值作为默认值, 后续命令行中如果也设置了对应配置则会覆盖对应值, 达到命令行配置优先生效的效果.
        env_build_ext_args = os.environ.get("PYPTO_BUILD_EXT_ARGS", "")
        if not env_build_ext_args:
            return
        pattern = r'(?:[^\s\"\']|\"[^\"]*\"|\'[^\']*\')+'
        env_build_ext_args_split: List[str] = re.findall(pattern, env_build_ext_args)
        parser = argparse.ArgumentParser(description=f"Setuptools CMakeBuild Ext.", add_help=False)
        parser.add_argument("--cmake-generator", nargs="?", type=str, default=None, dest="cmake_generator")
        parser.add_argument("--cmake-build-type", nargs="?", type=str, default=None, dest="cmake_build_type")
        parser.add_argument("--cmake-options", nargs="?", type=str, default="", dest="cmake_options")
        parser.add_argument("--cmake-verbose", action="store_true", default=False, dest="cmake_verbose")
        args, _ = parser.parse_known_args(env_build_ext_args_split)
        self.cmake_generator = args.cmake_generator
        self.cmake_build_type = args.cmake_build_type
        self.cmake_options = args.cmake_options
        self.cmake_verbose = args.cmake_verbose

    def finalize_options_cmake(self):
        # 赋传参值
        self.cmake_generator = None if not self.cmake_generator else self.cmake_generator
        if self.cmake_generator:
            self.cmake_generator = self.cmake_generator.replace(r'"', "")
            self.cmake_generator = f"\"{self.cmake_generator}\""
        self.cmake_build_type = None if not self.cmake_build_type else self.cmake_build_type
        self.cmake_options = self.cmake_options.replace("'", "").replace('"', "") if self.cmake_options else None
        self.cmake_verbose = True if self.cmake_verbose else False
        # CMake Options 修正
        cmake_option_lst: List[str] = [o.replace(" ", "")
                                       for o in (self.cmake_options.split(" ") if self.cmake_options else [])]
        if self.cmake_generator:
            for option in cmake_option_lst:
                if option.startswith("-DCMAKE_GENERATOR="):
                    cmake_option_lst.remove(option)
                    logging.warning("Configuration via --cmake-generator has higher priority than --cmake-options; "
                                    "in case of conflict, the former prevails.")
        if self.cmake_build_type:
            for option in cmake_option_lst:
                if option.startswith("-DCMAKE_BUILD_TYPE="):
                    cmake_option_lst.remove(option)
                    logging.warning("Configuration via --cmake-build-type has higher priority than --cmake-options; "
                                    "in case of conflict, the former prevails.")
        self.cmake_options = " ".join(cmake_option_lst) if cmake_option_lst else self.cmake_options


class CMakeBuild(build_ext, CMakeUserOption, EditModeHelper):
    """自定义构建命令, 调用 CMake 构建系统
    """
    user_options = build_ext.user_options + CMakeUserOption.USER_OPTION

    @staticmethod
    def _get_job_num(job_num: Optional[int], generator: Optional[str]) -> Optional[int]:
        def_job_num: Optional[int] = min(int(math.ceil(float(multiprocessing.cpu_count()) * 0.9)), 48)  # 48 为缺省最大核数
        def_job_num = None if generator and generator.lower() in ["ninja", ] else def_job_num  # ninja 由其自身决定缺省核数
        job_num = job_num if job_num and job_num > 0 else def_job_num
        return job_num

    @staticmethod
    def _get_cmake_install_manifest(build_dir: Path, file_name: str = "install_manifest.txt") -> List[str]:
        installed_files: List[str] = []
        install_manifest_file: Path = Path(build_dir, file_name)
        if install_manifest_file.exists():
            with open(install_manifest_file, 'r', encoding="utf-8") as fh:
                installed_files = [line.strip() for line in fh if line.strip()]
        return installed_files

    def initialize_options(self):
        """通过控制命令行选项初始化顺序, 实现实际命令行选项优先生效.
        """
        super().initialize_options()
        self.initialize_options_cmake()

    def finalize_options(self):
        super().finalize_options()
        self.finalize_options_cmake()

    def run(self):
        """执行构建流程
        """
        logging.info("%s", self)
        # 源码根目录
        src: Path = Path(__file__).parent.resolve()
        # 准备构建目录, 使用扩展名创建唯一的构建目录
        build_dir: Path = Path(self.build_temp).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        # 获取 cmake install prefix
        cmake_install_prefix: Path = self._get_cmake_install_prefix()

        # CMake Configure
        cmd: str = f"{self.cmake} -S {src} -B {build_dir}"
        cmd += f" -G {self.cmake_generator}" if self.cmake_generator else ""
        cmd += f" -DCMAKE_BUILD_TYPE={self.cmake_build_type}" if self.cmake_build_type else ""
        cmd += f" -DPython3_EXECUTABLE={sys.executable} -DCMAKE_INSTALL_PREFIX={cmake_install_prefix}"
        cmd += f" {self.cmake_options}" if self.cmake_options else ""
        logging.info("CMake Configure, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8')
        ret.check_returncode()

        # CMake Build
        job_num: Optional[int] = self._get_job_num(job_num=self.parallel, generator=self.cmake_generator)
        cmd: str = f"{self.cmake} --build {build_dir}" + (f" -j {job_num}" if job_num else "")
        cmd += f" --verbose" if self.cmake_verbose else ""
        logging.info("CMake Build, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8')
        ret.check_returncode()

        # CMake Install
        cmake_install_prefix: Path = self._get_cmake_install_prefix()  # 重复获取触发提示
        cmd: str = f"{self.cmake} --install {build_dir} --prefix {cmake_install_prefix}"
        logging.info("CMake Install, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8')
        ret.check_returncode()

        # macOS with GCC: Fix missing LC_ID_DYLIB in pypto_impl.so
        # GCC on macOS doesn't properly set install_name for shared libraries
        if platform.system() == "Darwin":
            self._fix_macos_dylib(cmake_install_prefix)
            # Also fix in build_lib which is used for wheel packaging
            self._fix_macos_dylib(Path(self.build_lib))

        if self._edit_mode():
            installed_files: List[str] = self._get_cmake_install_manifest(build_dir=build_dir)
            if installed_files:
                # 向 editable_wheel 命令传递
                editable_wheel_cmd = self.distribution.get_command_obj("editable_wheel")
                editable_wheel_cmd.pypto_install_manifest_lst = installed_files
                logging.info("Command build_ext passes %s CMake install files to editable_wheel command",
                             len(editable_wheel_cmd.pypto_install_manifest_lst))

    def _fix_macos_dylib(self, install_prefix: Path):
        """Fix missing LC_ID_DYLIB in shared libraries built with GCC on macOS.

        GCC on macOS doesn't properly set install_name for shared libraries,
        which causes dlopen to fail with "MH_DYLIB is missing LC_ID_DYLIB".
        We fix this by relinking the library using clang++ from object files.
        """
        pypto_impl_path = install_prefix / "pypto" / "pypto_impl.so"
        if not pypto_impl_path.exists():
            logging.warning("pypto_impl.so not found at %s, skipping LC_ID_DYLIB fix", pypto_impl_path)
            return

        # Find object files directory
        build_dir = Path(self.build_temp)
        obj_dir = build_dir / "python" / "src" / "CMakeFiles" / "pypto_impl.dir"
        if not obj_dir.exists():
            logging.warning("Object files directory not found at %s, skipping LC_ID_DYLIB fix", obj_dir)
            return

        # Collect all object files
        obj_files = list(obj_dir.glob("*.o")) + list(obj_dir.glob("bindings/*.o"))
        if not obj_files:
            logging.warning("No object files found, skipping LC_ID_DYLIB fix")
            return

        logging.info("Fixing LC_ID_DYLIB for %s using clang++", pypto_impl_path)

        # Find libc_sec.dylib
        src_root = Path(__file__).parent.resolve()
        libc_sec_path = src_root / "third_party_path" / "Release" / "lib" / "libc_sec.dylib"
        if not libc_sec_path.exists():
            logging.warning("libc_sec.dylib not found at %s, skipping LC_ID_DYLIB fix", libc_sec_path)
            return

        # Create temporary output file
        fixed_path = str(pypto_impl_path) + ".fixed"

        # Relink using clang++ with proper install_name
        clang_cmd = [
            "clang++", "-dynamiclib", "-arch", "arm64",
            "-install_name", "@rpath/pypto_impl.so",
            "-undefined", "dynamic_lookup",
            "-o", fixed_path
        ]
        clang_cmd.extend([str(f) for f in obj_files])
        clang_cmd.extend([
            "-L/opt/homebrew/opt/gcc/lib/gcc/current", "-lstdc++",
            str(libc_sec_path)
        ])

        try:
            logging.info("Running: %s", " ".join(clang_cmd))
            ret = subprocess.run(clang_cmd, capture_output=True, text=True)
            if ret.returncode != 0:
                logging.warning("Failed to fix LC_ID_DYLIB: %s", ret.stderr)
                if os.path.exists(fixed_path):
                    os.remove(fixed_path)
                return

            # Replace original file with fixed one
            shutil.move(fixed_path, str(pypto_impl_path))
            logging.info("Successfully fixed LC_ID_DYLIB for pypto_impl.so")
        except Exception as e:
            logging.warning("Failed to fix LC_ID_DYLIB: %s", str(e))
            if os.path.exists(fixed_path):
                os.remove(fixed_path)

    def _edit_mode(self) -> bool:
        if hasattr(self, 'pypto_editable_mode') and self.pypto_editable_mode:
            return True
        return False

    def _get_cmake_install_prefix(self) -> Path:
        cmake_install_prefix: Path = Path(self.build_lib)
        if self._edit_mode():
            # 可编辑安装模式下, 设置 CMake Install Prefix 为源码相关路径
            src_root: Path = Path(__file__).parent.resolve()  # -e 模式下不会 copy 源码到 tmp 目录
            cmake_install_prefix = Path(src_root, "python")
            logging.warning("Run in editable mode, use %s as cmake install prefix.", cmake_install_prefix)
        return cmake_install_prefix.resolve()


class SetupCtrl:
    """SetupTools 流程控制
    """

    @classmethod
    def main(cls):
        """主处理流程
        """
        warnings.filterwarnings("ignore", category=UserWarning, module="setuptools.command.build_py")
        # Setuptools 配置
        setup(
            # 扩展模块配置
            ext_modules=[
                CMakeExtension(),
            ],
            cmdclass={
                'editable_wheel': CustomEditableWheel,  # setuptools>=58.0.0, pip install -e 会触发 editable_wheel
                'build_ext': CMakeBuild,
            },
        )


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    SetupCtrl.main()
