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
""" 构建总入口.

构建总入口.
"""
import abc
import os
import sys
import argparse
import logging
import multiprocessing
import shlex
import shutil
import signal
import subprocess
import json
import math
import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

if str(Path(Path(__file__).parent, "tools")) not in sys.path:
    sys.path.append(str(Path(Path(__file__).parent, "tools")))

import work_flow as wf


class CMakeParam(abc.ABC):
    """ 需要向 CMake 传入 Option 的参数
    """

    @staticmethod
    @abc.abstractmethod
    def reg_args(parser, ext: Optional[Any] = None):
        pass

    @classmethod
    def _cfg_require(cls, opt: str, ctr: bool = True, tv: str = "ON", fv: str = "OFF") -> str:
        """
        获取 CMake Config 阶段的必选 Option 配置

        :param opt: CMake 选项, 会最终体现到 CMake -D传入的参数中
        :param ctr: 控制变量
        :param tv: 控制变量为 True 时, 设置的值
        :param fv: 控制变量为 False 时, 设置的值
        :return: 设置的值
        """
        cmd: str = f" -D{opt}=" + (tv if ctr else fv)
        return cmd

    @classmethod
    def _cfg_optional(cls, opt: str, ctr: bool, v: str):
        """
        获取 CMake Config 阶段的可选 Option 配置

        :param opt: CMake 选项, 会最终体现到 CMake -D传入的参数中
        :param ctr: 控制变量
        :param v: 控制变量为 True 时, 设置的值
        """
        cmd: str = (f" -D{opt}=" + v) if ctr else ""
        return cmd

    @abc.abstractmethod
    def get_cfg_cmd(self):
        pass


@dataclasses.dataclass
class BuildParam(CMakeParam):
    """ 构建相关参数
    """
    targets: Optional[List[str]] = None  # 编译目标
    job_num: int = min(int(math.ceil(float(multiprocessing.cpu_count()) * 0.8)), 16) # 编译阶段使用核数
    clean: bool = False  # 强制清理 Build-Tree 及 Install-Tree 标记
    timeout: Optional[int] = None  # 构建超时时长
    type_: Optional[str] = None  # 构建类型
    asan: bool = False  # 使能 AddressSanitizer
    ubsan: bool = False  # 使能 UndefinedBehaviorSanitizer
    gcov: bool = False  # 使能 GNU Coverage
    clang_install_path: Optional[Path] = None  # Clang 安装位置

    def __init__(self, args):
        self.targets = args.targets
        self.job_num = args.job_num if args.job_num > 0 else self.job_num
        self.clean = args.clean
        self.timeout = None if args.timeout == 0 else args.timeout
        self.type_ = args.build_type
        self.asan = args.asan
        self.ubsan = args.ubsan
        self.gcov = args.gcov
        self.clang_install_path = self._get_clang_install_path(opt=args.clang)

    def __str__(self):
        desc: str = ""
        desc += f"\nBuild"
        desc += f"\n    Targets                 : {self.targets}"
        desc += f"\n    Job Num                 : {self.job_num}"
        desc += f"\n    Clean                   : {self.clean}"
        desc += f"\n    Timeout                 : {self.timeout}"
        desc += f"\n    BuildType               : {self.type_}"
        desc += f"\n    ASan                    : {self.asan}"
        desc += f"\n    UbSan                   : {self.ubsan}"
        desc += f"\n    GCov                    : {self.gcov}"
        desc += f"\n    ClangInstallPath        : {self.clang_install_path}"
        return desc

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument("-t", "--targets", nargs="?", type=str, action="append",
                            help="targets, specific build targets, "
                                 "If you specify more than one, all targets within the specified range are built.")
        parser.add_argument("-j", "--job_num", nargs="?", type=int, default=-1,
                            help="job num, specific job num of build.")
        parser.add_argument("-c", "--clean", action="store_true", default=False,
                            help="clean, clean Build-Tree and Install-Tree before build.")
        parser.add_argument("--timeout", nargs="?", type=int, default=0,
                            help="build task timeout.")
        parser.add_argument("--build_type", nargs="?", type=str, default=None,
                            choices=["Debug", "Release", "MinSizeRel", "RelWithDebInfo"],
                            help="build type.")
        parser.add_argument("--asan", action="store_true", default=False,
                            help="Enable AddressSanitizer.")
        parser.add_argument("--ubsan", action="store_true", default=False,
                            help="Enable UndefinedBehaviorSanitizer.")
        parser.add_argument("--gcov", action="store_true", default=False,
                            help="Enable GNU Coverage Instrumentation Tool.")
        parser.add_argument("--clang", nargs="?", type=str, default="",
                            help="Specify clang install path, such as /usr/bin/clang")

    @staticmethod
    def _get_clang_install_path(opt: Optional[str]) -> Optional[Path]:
        # 获取 Clang 安装目录
        if opt is None:  # 指定 clang 参数, 但未指定具体路径, 此时需尝试寻找
            cmd = "which clang"
            ret = subprocess.run(shlex.split(cmd), capture_output=True, check=True, text=True, encoding='utf-8')
            ret.check_returncode()
            clang_install_path = Path(ret.stdout).resolve()
        elif opt == "":  # 未指定 clang 参数
            clang_install_path = None
        else:  # 指定 clang 参数, 并指定具体路径
            clang_install_path = Path(opt)
        if clang_install_path is not None:
            clang_install_path = Path(clang_install_path).resolve().parent
            if not clang_install_path.exists():
                raise ValueError(f"Clang install path not exist, path={clang_install_path}")
        return clang_install_path

    def get_cfg_cmd(self) -> str:
        cmd: str = ""
        cmd += self._cfg_optional(opt="CMAKE_BUILD_TYPE", ctr=bool(self.type_), v=self.type_)
        cmd += self._cfg_require(opt="ENABLE_ASAN", ctr=self.asan)
        cmd += self._cfg_require(opt="ENABLE_UBSAN", ctr=self.ubsan)
        cmd += self._cfg_require(opt="ENABLE_GCOV", ctr=self.gcov)

        def _check_clang_toolchain(_opt: str, _b: str) -> Tuple[bool, str]:
            _p: Path = Path(self.clang_install_path, _b)
            if _p.exists():
                return True, self._cfg_require(opt=_opt, tv=str(_p))
            logging.error("Clang Toolchain %s not exist.", _p)
            return False, ""

        def _gen_clang_cmd() -> Tuple[bool, str]:
            _bin_opt_lst: List[List[str]] = [["clang", "CMAKE_C_COMPILER"],
                                             ["clang++", "CMAKE_CXX_COMPILER"]]
            _rst: bool = True
            _cmd: str = ""
            for _bin_opt in _bin_opt_lst:
                _sub_bin, _sub_opt = _bin_opt
                _sub_rst, _sub_cmd = _check_clang_toolchain(_opt=_sub_opt, _b=_sub_bin)
                _rst = _rst and _sub_rst
                _cmd = _cmd + _sub_cmd
            return _rst, _cmd if _rst else ""

        # Clang
        if self.clang_install_path is not None:
            ret, clang_cmd = _gen_clang_cmd()
            if not ret:
                raise RuntimeError(f"Clang({self.clang_install_path}) not complete.")
            cmd += clang_cmd
        return cmd


@dataclasses.dataclass
class FeatureParam(CMakeParam):
    """ 特性控制相关参数
    """
    frontend_type: Optional[str] = None # 前端类型, 支持 python3, cpp
    backend_type: Optional[str] = None # 后端类型, 支持 npu, cost_model

    def __init__(self, args):
        self.frontend_type = "python3" if args.frontend is None else args.frontend
        self.backend_type = "npu" if args.backend is None else args.backend

    def __str__(self):
        desc: str = ""
        desc += f"\nFeature"
        desc += f"\n    Frontend                : {self.frontend_type}"
        desc += f"\n    Backend                 : {self.backend_type}"
        return desc

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument("-f", "--frontend", nargs="?", type=str, default="cpp",
                            choices=["python3", "cpp"],
                            help="backend, such as npu/cost_model etc.")
        parser.add_argument("-b", "--backend", nargs="?", type=str, default="npu",
                            choices=["npu", "cost_model"],
                            help="backend, such as npu/cost_model etc.")

    def get_cfg_cmd(self) -> str:
        cmd: str = self._cfg_require(opt="BUILD_WITH_CANN", ctr=self.backend_type in ["npu"])
        return cmd


@dataclasses.dataclass
class TestsExecuteParam(CMakeParam):
    """ Tests 执行相关参数
    """
    changed_file: Optional[Path] = None  # 修改文件路径
    auto_execute: bool = False  # 用例自动执行
    auto_execute_parallel: bool = False  # 用例并行执行

    def __init__(self, args):
        self.changed_file = None if not args.changed_files else Path(args.changed_files).resolve()
        self.auto_execute = args.disable_auto_execute
        self.auto_execute_parallel = self.auto_execute and self.ci_model

    @property
    def ci_model(self) -> bool:
        return True if self.changed_file else False

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument("--changed_files", nargs="?", type=Path, default=None,
                            help="Specify the file of files changed, "
                                 "so that the corresponding test cases can be triggered incrementally.")
        parser.add_argument("--disable_auto_execute", action="store_false", default=True,
                            help="Disable auto execute STest/Utest with build.")

    def get_cfg_cmd(self) -> str:
        cmd: str = ""
        cmd += self._cfg_require(opt="ENABLE_TESTS_EXECUTE", ctr=self.auto_execute)
        cmd += self._cfg_require(opt="ENABLE_TESTS_EXECUTE_PARALLEL", ctr=self.auto_execute_parallel)
        return cmd


@dataclasses.dataclass
class TestsGoldenParam(CMakeParam):
    clean: bool = False  # 清理 Golden 标记
    path: Optional[Path] = None  # 指定 Golden 路径

    def __init__(self, args):
        self.clean = args.golden_clean
        if args.golden_path:
            # 传参且指定具体路径时, 使用指定路径, 否则具体缺省路径由 CMake 侧决定
            self.path = Path(args.golden_path).resolve()

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument("--golden_path", "--stest_golden_path", nargs="?", type=str, default="",
                            help="Specific Tests golden path.", dest="golden_path")
        parser.add_argument("--golden_clean", "--golden_path_clean", "--stest_golden_path_clean",
                            action="store_true", default=False,
                            help="Clean Tests golden.", dest="golden_clean")

    def get_cfg_cmd(self) -> str:
        cmd: str = ""
        cmd += self._cfg_require(opt="ENABLE_STEST_GOLDEN_PATH_CLEAN", ctr=self.clean)
        cmd += self._cfg_require(opt="ENABLE_STEST_GOLDEN_PATH", ctr=bool(self.path), tv=str(self.path))
        return cmd


@dataclasses.dataclass
class TestsFilterParam(CMakeParam):
    cmake_option: str = ""
    enable: bool = False
    filter_str: Optional[str] = None

    def __init__(self, argv: Optional[str], opt: str):
        self.cmake_option = opt
        if argv is None:
            self.enable, self.filter_str = True, "ON"      # 指定 对应参数, 但未指定内容
        elif argv == "":
            self.enable, self.filter_str = False, "OFF"   # 未指定 对应参数
        else:
            self.enable, self.filter_str = True, argv    # 指定 对应参数 且指定内容

    @property
    def filter_str_pytest(self) -> str:
        s: str = self.filter_str
        s = s.replace('::', '#').replace(':', ' ')
        s = s.replace('#', '::')
        return s

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        mark: str = str(ext).lower()
        mark_lst: List[str] = mark.split("_")
        have_char: bool = len(mark_lst) <= 1
        mark_word: str = mark.replace("_", " ")
        help_str: str = (f"Enable {mark_word} scene, specific {mark_word} filter, "
                         f"multiple cases are separated by ':'/',' .")
        if have_char:
            mark_char: Optional[str] = mark_lst[0][0] if have_char else None
            parser.add_argument(f"-{mark_char}", f"--{mark}", nargs="?", type=str, default="", help=help_str)
        else:
            parser.add_argument(f"--{mark}", nargs="?", type=str, default="", help=help_str)

    def get_cfg_cmd(self) -> str:
        cmd: str = self._cfg_require(opt=f"{self.cmake_option}", ctr=self.enable, tv=f"{self.filter_str}")
        return cmd


@dataclasses.dataclass
class STestExecuteParam(CMakeParam):
    auto_execute_device_id: str = ""
    interpreter_config: bool = False
    enable_binary_cache: bool = False
    dump_json: bool = False

    def __init__(self, args, enable_binary_cache: bool):
        devs = ["0"]
        if args.device is not None:
            devs = [str(d) for d in list(set(args.device)) if d is not None and str(d) != ""]
        self.auto_execute_device_id = ":".join(devs)
        self.dump_json = args.stest_dump_json
        self.interpreter_config = args.enable_interpreter_config
        self.enable_binary_cache = enable_binary_cache

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument("-d", "--device", nargs="?", type=int, action="append",
                            help="Device ID, default 0.")
        parser.add_argument("--stest_dump_json", action="store_true", default=False,
                            help="Dump json files.")
        parser.add_argument("--enable_interpreter_config", action="store_true", default=False,
                            help="enable STest Interpreter Config")

    def get_cfg_cmd(self) -> str:
        cmd: str = ""
        cmd += self._cfg_require(opt="ENABLE_STEST_EXECUTE_DEVICE_ID", tv=self.auto_execute_device_id)
        cmd += self._cfg_require(opt="ENABLE_STEST_DUMP_JSON", ctr=self.dump_json)
        cmd += self._cfg_require(opt="ENABLE_STEST_INTERPRETER_CONFIG", ctr=self.interpreter_config)
        cmd += self._cfg_require(opt="ENABLE_STEST_BINARY_CACHE", ctr=self.enable_binary_cache)
        return cmd


@dataclasses.dataclass
class STestToolsParam(CMakeParam):
    cases_csv_file: Optional[Path] = None
    intercept_flag: bool = False
    output_clean: bool = False

    prof_enable: bool = False
    prof_level: str = "l1"
    prof_warn_up_cnt: Optional[int] = None
    prof_try_cnt: Optional[int] = None
    prof_max_cnt: Optional[int] = None

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        # Tools
        parser_tools = parser.add_parser('tools', help="Tools")
        parser_tools.add_argument("--cases_csv_file", nargs=1, type=Path, default=None,
                                  help="Specify cases.csv")
        parser_tools.add_argument("--intercept", action="store_true", default=False,
                                  help="intercept, Intercept if have failed case result")
        parser_tools.add_argument("--tools_output_clean", action="store_true", default=False,
                                  help="clean, Specify clean flag, clean tools output dir")
        # Tools.Profiling
        sub_parser_prof = parser_tools.add_subparsers(dest="Tolls SubCommand")
        parser_prof = sub_parser_prof.add_parser('profiling', help="Profiling", aliases=['prof'])
        parser_prof.add_argument("-l", "--level", "--prof_level", dest="prof_level",
                                 nargs="?", type=str, default="l1", choices=["l1", "l2"],
                                 help="Specify profiling level")
        parser_prof.add_argument("-w", "--warn_up_cnt", "--prof_warn_up_cnt", dest="prof_warn_up_cnt",
                                 nargs=1, type=int, default=None,
                                 help="Specify profiling warn up cnt")
        parser_prof.add_argument("-t", "--try_cnt", "--prof_try_cnt", dest="prof_try_cnt",
                                 nargs=1, type=int, default=None,
                                 help="Specify profiling try cnt")
        parser_prof.add_argument("-m", "--max_cnt", "--prof_max_cnt", dest="prof_max_cnt",
                                 nargs=1, type=int, default=None,
                                 help="Specify profiling max cnt")
        parser_prof.set_defaults(func=SubCommandMgr.init_param_tools_profiling)

    def init_param(self, args):
        self.cases_csv_file = Path(args.cases_csv_file[0]).resolve() if args.cases_csv_file else None
        self.intercept_flag = args.intercept
        self.output_clean = args.tools_output_clean

    def init_param_profiling(self, args):
        self.prof_enable = True
        self.prof_level = args.prof_level
        self.prof_warn_up_cnt = args.prof_warn_up_cnt[0] if args.prof_warn_up_cnt else None
        self.prof_try_cnt = args.prof_try_cnt[0] if args.prof_try_cnt else None
        self.prof_max_cnt = args.prof_max_cnt[0] if args.prof_max_cnt else None

    def get_cfg_cmd(self) -> str:
        cmd: str = ""
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_PROF", ctr=self.prof_enable)

        # 当前 tools 下仅支持 prof 工具, 当其未使能时, 不需设置其他 option
        if not self.prof_enable:
            return cmd

        # 公共参数
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_CASE_FILE", ctr=bool(self.cases_csv_file),
                                 tv=str(self.cases_csv_file))
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_INTERCEPT", ctr=self.intercept_flag)
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_OUTPUT_CLEAN", ctr=self.output_clean)

        # Profiling 工具参数
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_PROF_LEVEL", tv=self.prof_level)
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_PROF_WARN_UP_CNT",
                                 ctr=self.prof_warn_up_cnt is not None,
                                 tv=f"{self.prof_warn_up_cnt}")
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_PROF_TRY_CNT",
                                 ctr=self.prof_try_cnt is not None,
                                 tv=f"{self.prof_try_cnt}")
        cmd += self._cfg_require(opt="ENABLE_STEST_TOOLS_PROF_MAX_CNT",
                                 ctr=self.prof_max_cnt is not None,
                                 tv=f"{self.prof_max_cnt}")
        return cmd


class TestsParam(CMakeParam):

    def __init__(self, args):
        self.exec: TestsExecuteParam = TestsExecuteParam(args=args)
        self.golden: TestsGoldenParam = TestsGoldenParam(args=args)
        self.utest: TestsFilterParam = TestsFilterParam(argv=args.utest, opt="ENABLE_UTEST")
        self.stest_exec: STestExecuteParam = STestExecuteParam(args=args, enable_binary_cache=self.exec.ci_model)
        self.stest_tools: STestToolsParam = STestToolsParam()
        self.stest: TestsFilterParam = TestsFilterParam(argv=args.stest, opt="ENABLE_STEST")
        self.stest_distributed: TestsFilterParam = TestsFilterParam(argv=args.stest_distributed,
                                                                    opt="ENABLE_STEST_DISTRIBUTED")

    def __str__(self):
        desc: str = ""
        if self.utest.enable or self.stest.enable or self.stest_distributed.enable:
            desc += f"\nTests"
            desc += f"\n    Execute"
            desc += f"\n               Changed File : {self.exec.changed_file}"
            desc += f"\n                       Auto : {self.exec.auto_execute}"
            desc += f"\n                   Parallel : {self.exec.auto_execute_parallel}"
            if self.utest.enable:
                desc += f"\n    Utest"
                desc += f"\n                     Enable : {self.utest.enable}"
                desc += f"\n                     Filter : {self.utest.filter_str}"
            if self.stest.enable or self.stest_distributed.enable:
                desc += f"\n    Golden"
                desc += f"\n                      Clean : {self.golden.clean}"
                desc += f"\n                       Path : {self.golden.path}"
                desc += f"\n    Stest Execute"
                desc += f"\n                     Device : {self.stest_exec.auto_execute_device_id}"
                desc += f"\n                   DumpJson : {self.stest_exec.dump_json}"
                desc += f"\n         Interpreter Config : {self.stest_exec.interpreter_config}"
                desc += f"\n        Enable Binary Cache : {self.stest_exec.enable_binary_cache}"
            if self.stest.enable:
                desc += f"\n    Stest"
                desc += f"\n                     Enable : {self.stest.enable}"
                desc += f"\n                     Filter : {self.stest.filter_str}"
                if self.stest_tools.prof_enable:
                    desc += f"\n        Tools"
                    desc += f"\n              Case Csv File : {self.stest_tools.cases_csv_file}"
                    desc += f"\n             Intercept Flag : {self.stest_tools.intercept_flag}"
                    desc += f"\n               Output Clean : {self.stest_tools.output_clean}"
                    desc += f"\n        Tools Profiling"
                    desc += f"\n                     Enable : {self.stest_tools.prof_enable}"
                    desc += f"\n                      Level : {self.stest_tools.prof_level}"
                    desc += f"\n                Warn Up Cnt : {self.stest_tools.prof_warn_up_cnt}"
                    desc += f"\n                    Try Cnt : {self.stest_tools.prof_try_cnt}"
                    desc += f"\n                    Max Cnt : {self.stest_tools.prof_max_cnt}"
            if self.stest_distributed.enable:
                desc += f"\n    Stest Distributed"
                desc += f"\n                     Enable : {self.stest_distributed.enable}"
                desc += f"\n                     Filter : {self.stest_distributed.filter_str}"
        return desc

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        TestsExecuteParam.reg_args(parser=parser)
        TestsGoldenParam.reg_args(parser=parser)
        TestsFilterParam.reg_args(parser=parser, ext="utest")
        STestExecuteParam.reg_args(parser=parser)
        STestToolsParam.reg_args(parser=ext)
        TestsFilterParam.reg_args(parser=parser, ext="stest")
        TestsFilterParam.reg_args(parser=parser, ext="stest_distributed")

    def get_cfg_cmd(self) -> str:
        cmd: str = ""
        cmd += self.utest.get_cfg_cmd()
        cmd += self.stest.get_cfg_cmd()
        cmd += self.stest_distributed.get_cfg_cmd()
        if self.utest.enable or self.stest.enable or self.stest_distributed.enable:
            cmd += self.exec.get_cfg_cmd()
            if self.stest.enable or self.stest_distributed.enable:
                cmd += self.golden.get_cfg_cmd()
                cmd += self.stest_exec.get_cfg_cmd()
            if self.stest.enable:
                cmd += self.stest_tools.get_cfg_cmd()
        return cmd


@dataclasses.dataclass
class ModelParam(CMakeParam):
    prof: int = 0
    pe: int = 2
    sim: bool = False
    sim_with_onboard_aicpu: bool = False
    back_annotation_aicpu: bool = False
    back_annotation_aicore: bool = False
    replay_file_path: Optional[str] = None
    calendar: bool = False
    pvmodel: bool = False

    def __init__(self, args):
        self.prof = args.prof
        self.pe = args.pe
        self.sim = args.sim
        self.sim_with_onboard_aicpu = args.sim_with_onboard_aicpu
        self.back_annotation_aicpu = args.back_annotation_aicpu
        self.back_annotation_aicore = args.back_annotation_aicore
        self.replay_file_path = args.replay_file_path
        self.calendar = args.calendar
        self.pvmodel = args.pvmodel

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument("--prof", nargs="?", type=int, default=0, choices=[1, 2],
                            help="Enable workflow.")
        parser.add_argument("--pe", nargs="?", type=int, default=2, choices=[1, 2, 4, 5, 6, 7, 8],
                            help="Enable pmuEvent.")
        parser.add_argument("-s1", "--sim", action="store_true", default=False,
                            help="enable simulation")
        parser.add_argument("-s2", "--sim_with_onboard_aicpu", action="store_true", default=False,
                            help="enable simulation with onboard aicpu code")
        parser.add_argument("-b1", "--back_annotation_aicpu", action="store_true", default=False,
                            help="enable back-annotation in simulation with aicpu onboard data.")
        parser.add_argument("-b2", "--back_annotation_aicore", action="store_true", default=False,
                            help="(WIP)enable back-annotation in simulation with aicore onboard data.")
        parser.add_argument("-rf", "--replay_file_path", type=str, default=None,
                            help="Specify replay file path for back annotation.")
        parser.add_argument("-cal", "--calendar", action="store_true", default=False,
                            help="Enable calendar mode.")
        parser.add_argument("-pv", "--pvmodel", action="store_true", default=False,
                            help="Enable PVModel mode.")

    @staticmethod
    def _save_simulation_json(simulation_json, src_root: Path):
        temp_json_path = os.path.join(str(src_root), "src/cost_model/simulation/scripts/tmp_simulation.json")
        os.makedirs(os.path.dirname(temp_json_path), exist_ok=True)
        with open(temp_json_path, 'w') as f:
            json.dump(simulation_json, f, indent=4)

    def get_cfg_cmd(self) -> str:
        return ""

    def gen_simulation_json(self, src_root: Path) -> None:
        simulation_json = {
            "global_configs": {
                "platform_configs": {},
                "simulation_configs": {}
            }
        }
        self._gen_simulation_json_sim(cfg=simulation_json)
        self._gen_simulation_json_sim_with_onboard_aicpu(cfg=simulation_json)
        self._gen_simulation_json_back_annotation_aicpu(cfg=simulation_json)
        self._gen_simulation_json_back_annotation_aicore(cfg=simulation_json)
        self._gen_simulation_json_calendar(cfg=simulation_json)
        self._gen_simulation_json_pvmodel(cfg=simulation_json)
        self._save_simulation_json(simulation_json, src_root=src_root)

    def _gen_simulation_json_sim(self, cfg: Dict[Any, Any]):
        if self.sim:
            cfg["global_configs"]["platform_configs"]["ENABLE_COST_MODEL"] = True

    def _gen_simulation_json_sim_with_onboard_aicpu(self, cfg: Dict[Any, Any]):
        if self.sim_with_onboard_aicpu:
            cfg["global_configs"]["platform_configs"]["ENABLE_COST_MODEL"] = True
            cfg["global_configs"]["simulation_configs"]["USE_ON_BOARD_INFO"] = True
            cfg["global_configs"]["simulation_configs"]["args"] = [
                "Model.statisticReportToFile=true",
                "Model.deviceArch=910B",
                "Model.useOOOPassSeq=true",
                "Core.logLabelMode=0"
            ]

    def _gen_simulation_json_back_annotation_aicpu(self, cfg: Dict[Any, Any]):
        if self.back_annotation_aicpu:
            if self.replay_file_path is None:
                logging.error("Error: replay_file_path is required when back_annotation_aicpu is enabled")
                raise ValueError("Missing required argument: -rf, --replay_file_path")
            cfg["global_configs"]["platform_configs"]["ENABLE_COST_MODEL"] = True
            cfg["global_configs"]["simulation_configs"]["args"] = [
                "Model.statisticReportToFile=true",
                "Model.deviceArch=910B",
                "Model.useOOOPassSeq=true",
                "Core.logLabelMode=0",
                "Model.replayAllMode=1",
                f"Model.replayFile={self.replay_file_path}"
            ]

    def _gen_simulation_json_back_annotation_aicore(self, cfg: Dict[Any, Any]):
        if self.back_annotation_aicore:
            if self.replay_file_path is None:
                logging.error("Error: replay_file_path is required when back_annotation_aicore is enabled")
                raise ValueError("Missing required argument: -rf, --replay_file_path")
            cfg["global_configs"]["platform_configs"]["ENABLE_COST_MODEL"] = True
            cfg["global_configs"]["simulation_configs"]["USE_ON_BOARD_INFO"] = True
            cfg["global_configs"]["simulation_configs"]["JSON_PATH"] = self.replay_file_path
            cfg["global_configs"]["simulation_configs"]["args"] = [
                "Model.statisticReportToFile=true",
                "Model.deviceArch=910B",
                "Model.useOOOPassSeq=true",
                "Core.logLabelMode=0"
            ]

    def _gen_simulation_json_calendar(self, cfg: Dict[Any, Any]):
        if self.calendar:
            if self.replay_file_path is None:
                logging.error("Error: replay_file_path is required when calendar is enabled")
                raise ValueError("Missing required argument: -rf, --replay_file_path")
            cfg["global_configs"]["platform_configs"]["ENABLE_COST_MODEL"] = True
            cfg["global_configs"]["simulation_configs"]["args"] = [
                "Model.statisticReportToFile=true",
                "Model.deviceArch=910B",
                "Model.useOOOPassSeq=true",
                "Core.logLabelMode=0",
                "Model.genCalendarScheduleCpp=true",
                "Model.simulationFixedLatencyTask=true",
                f"Model.fixedLatencyTaskInfoPath={self.replay_file_path}",
                "Model.fixedLatencyTimeConvert=1",
                "Model.aicpuMachineNumber=1",
                "Model.coreMachineNumberPerAICPU=54",
                "Model.cubeMachineNumberPerAICPU=27",
                "Model.vecMachineNumberPerAICPU=27",
            ]

    def _gen_simulation_json_pvmodel(self, cfg: Dict[Any, Any]):
        if self.pvmodel:
            cfg["global_configs"]["platform_configs"]["ENABLE_COST_MODEL"] = True
            cfg["global_configs"]["platform_configs"]["ENABLE_SOFT_MEMORY"] = True
            cfg["global_configs"]["platform_configs"]["ENABLE_PV_DATA"] = True
            cfg["global_configs"]["simulation_configs"]["PV_LEVEL"] = 2
            cfg["global_configs"]["simulation_configs"]["args"] = [
                "Model.statisticReportToFile=true",
                "Model.deviceArch=910B",
                "Model.useOOOPassSeq=true",
                "Core.logLabelMode=0",
            ]


class BuildCtrl:
    """ 构建过程控制.

    本类包含由命令行指定或解析出的控制标记/参数, 以控制构建过程执行.
    """

    def __init__(self, args):
        self.whl_prefix: str = "pto"
        self.src_root: Path = Path(__file__).parent.resolve()
        self.build_root: Path = Path(Path.cwd(), "build")
        self.install_root: Path = Path(self.build_root.parent, "output")
        self.build: BuildParam = BuildParam(args=args)
        self.feature: FeatureParam = FeatureParam(args=args)
        self.tests: TestsParam = TestsParam(args=args)
        self.model: ModelParam = ModelParam(args=args)

    def __str__(self):
        ver = sys.version_info
        desc = ""
        desc += f"\nEnviron"
        desc += f"\n    Python3                 : {sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"
        desc += f"\nPath"
        desc += f"\n    Source  Dir             : {self.src_root}"
        desc += f"\n    Build   Dir             : {self.build_root}"
        desc += f"\n    Install Dir             : {self.install_root}"
        desc += f"{self.build}"
        desc += f"{self.feature}"
        desc += f"{self.tests}"
        desc += f"\n"
        return desc

    @staticmethod
    def run_build_cmd(cmd: str, update_env: Optional[Dict[str, str]] = None, check: bool = False,
                      timeout: Optional[int] = None) -> Optional[subprocess.CompletedProcess]:
        """执行具体 build 命令行

        因以下原因, 设置本函数, 而非调用原生 subprocess.run
        1. 支持多 target 构建, 各 target 构建时长共享公共 timeout 配置;
        2. UTest/STest 并行执行场景下, 执行时进程调用关系为:
               build.py(主进程) -> 进程1(CMake) -> 进程2(CMake Generator, make/ninja) -> 进程3(Python)-> 进程4(executable)
           此时若 进程1 超时, 需要触发其子/孙进程感知, 进而结束

        :param cmd: Build 命令行
        :param update_env: 环境变量(额外更新内容)
        :param check: 检查返回值
        :param timeout: 执行超时时长
        """

        def _stop_pg(_p: subprocess.Popen):
            """通过 SIGINT 信号通知所有子/孙进程结束, python 并行脚本内会捕获该信号进行结算处理
            """
            _pgid = os.getpgid(_p.pid)
            logging.info("Send terminate event to CMake[%s]", _pgid)
            os.killpg(_pgid, signal.SIGINT)

        stdout: Optional[str] = None
        stderr: Optional[str] = None
        env = {**os.environ}
        env.update(update_env if update_env else {})
        with subprocess.Popen(shlex.split(cmd), env=env, text=True, encoding='utf-8',
                              start_new_session=True) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _stop_pg(_p=process)
                raise
            except KeyboardInterrupt:
                # 一般为用户主动触发, 不需再上报错误
                _stop_pg(_p=process)
            except Exception:
                process.kill()
                raise
            finally:
                stdout = stdout or ""
                stderr = stderr or ""
            ret_code = process.poll()
            if check and ret_code:
                raise subprocess.CalledProcessError(ret_code, process.args, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(process.args, ret_code, stdout, stderr)

    @staticmethod
    def pip_uninstall(name: str, path: Optional[Path] = None):
        """
        卸载对应 whl 包

        :param name: 包名
        :param path: 指定安装路径(可选)
        """
        cmd: str = f"{sys.executable} -m pip uninstall -y {name}"
        act_env = {**os.environ}
        if path:
            ori_env_python_path: str = act_env.get("PYTHONPATH", "")
            act_env_python_path: str = f"{path}:{ori_env_python_path}" if ori_env_python_path else f"{path}"
            act_env.update({"PYTHONPATH": act_env_python_path})
        ret = subprocess.run(shlex.split(cmd),
                            capture_output=False, check=True, text=True, encoding='utf-8', env=act_env)
        ret.check_returncode()
        logging.info("Success uninstall %s package%s", name, f" from {path}" if path else "")

    @staticmethod
    def find_match_whl(name: str, path: Path) -> Optional[Path]:
        """
        在指定路径下, 查找对应匹配的 whl 包文件

        :param name: 包名
        :param path: 指定路径
        :return: whl 包路径, None 表示未找到
        """
        cpp_desc: str = f"cp{sys.version_info.major}{sys.version_info.minor}"
        pattern: str = f"{name}-*-{cpp_desc}-{cpp_desc}-*.whl"
        whl_glob = path.glob(pattern=pattern)
        whl_files = [Path(f) for f in whl_glob]
        whl_file: Optional[Path] = whl_files[0] if whl_files else None
        if whl_file:
            logging.info("Success find match %s from %s", whl_file, path)
        else:
            logging.error("Failed to find match %s whl from %s, pattern=%s", name, path, pattern)
        return whl_file

    @staticmethod
    def pip_install(whl: Path, path: Optional[Path] = None):
        """
        安装指定 whl 包

        :param whl: 包文件
        :param path: 安装路径(可选), 未指定时会安装在默认路径
        :return: 安装路径
        """
        cmd: str = f"{sys.executable} -m pip install --no-compile {whl}"
        cmd += f" --target={path}" if path else ""
        ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8')
        ret.check_returncode()
        logging.info("Success install %s%s", whl, f" to {path}" if path else "")

    @classmethod
    def main(cls):
        """ 主处理流程
        """
        parser = argparse.ArgumentParser(description=f"Tile Framework C++ Build Ctrl.", epilog="Best Regards!")
        sub_parser = parser.add_subparsers()  # 子命令
        # 参数注册
        BuildParam.reg_args(parser=parser)
        FeatureParam.reg_args(parser=parser)
        TestsParam.reg_args(parser=parser, ext=sub_parser)
        ModelParam.reg_args(parser=parser)
        # 参数处理
        args = parser.parse_args()
        ctrl = BuildCtrl(args=args)
        # 流程处理
        # 区分 python3 前端和 cpp 前端
        logging.info("%s", ctrl)
        if ctrl.feature.frontend_type in ["python", "python3"]:
            logging.info("Front-end(python3), start process with scikit-build-core.")
            ctrl.py_clean()
            ctrl.py_build()
            ctrl.py_tests()
        else:
            logging.info("Front-end(cpp), start process with CMake.")
            if 'func' in args:
                args.func(args=args, ctrl=ctrl)
            ctrl.cmake_clean()
            ctrl.cmake_configure()
            ctrl.model.gen_simulation_json(src_root=ctrl.src_root)
            ctrl.cmake_build()

    def cmake_clean(self):
        """ 清理中间结果, 清理内容包括构建树, 安装树全部内容. """
        if self.build.clean:
            if self.build_root.exists():
                logging.info("Clean Build-Tree(%s)", self.build_root)
                shutil.rmtree(self.build_root)
            if self.install_root.exists():
                logging.info("Clean Install-Tree(%s)", self.install_root)
                shutil.rmtree(self.install_root)
        if self.tests.stest_exec.enable_binary_cache:
            binary_cache_path = Path(Path.home(), "ast_data")
            if binary_cache_path.exists():
                shutil.rmtree(binary_cache_path)
                logging.info("Clean Binary Cache Path(%s)", binary_cache_path)

    def py_clean(self):
        if self.build.clean:
            dist: Path = Path(self.src_root, "dist")
            if dist.exists():
                logging.info("Clean Install-Tree(%s)", dist)
                shutil.rmtree(dist)

    def cmake_configure(self):
        """ CMake Configure 阶段流程. """
        # 基本配置, 当前 CMake 中有调用 python3 的情况, 传入 python3 解释器, 保证所使用的 python3 版本一致
        cmd = f"cmake -S {self.src_root} -B {self.build_root} -DPython3_EXECUTABLE={sys.executable}"
        cmd += self.build.get_cfg_cmd()
        cmd += self.feature.get_cfg_cmd()
        cmd += self.tests.get_cfg_cmd()
        # 执行
        logging.info("CMake Configure, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8')
        ret.check_returncode()

    def cmake_build(self):
        """ CMake Build 阶段流程. """
        # prof使能初始化
        update_env = {}
        if self.model.prof == 1 or self.model.prof == 2:
            update_env = wf.ini(self.build_root, self.model.prof, self.model.pe)
        cmd_list: List[str] = []
        if self.build.targets:
            for t in self.build.targets:
                cmd = f"cmake --build {self.build_root} --target {t} -- -j {self.build.job_num}"
                cmd_list.append(cmd)
        else:
            cmd = f"cmake --build {self.build_root} -- -j {self.build.job_num}"
            cmd_list.append(cmd)
        for i, c in enumerate(cmd_list):
            ts = datetime.now(tz=timezone.utc)
            logging.info("CMake Build(%s/%s), Cmd: %s", i + 1, len(cmd_list), c)
            try:
                ret = self.run_build_cmd(cmd=c, update_env=update_env, check=True, timeout=self.build.timeout)
            except subprocess.CalledProcessError as e:
                logging.info(f"Run cmd {c} failed, ERROR CODE: {e.returncode}")
                # 一键绘图
                if self.model.prof == 1 or self.model.prof == 2:
                    wf.work_flow_plot(self.build_root, self.model.prof, self.model.pe)
                raise
            ret.check_returncode()
            duration: int = int((datetime.now(tz=timezone.utc) - ts).seconds)
            duration_str: str = f"{duration}/{self.build.timeout}" if self.build.timeout else f"{duration}"
            logging.info("CMake Build(%s/%s), Cmd: %s, Duration %s sec",
                         i + 1, len(cmd_list), c, duration_str)
            # 超时时长更新, 当指定多 target 时, 各 target 共享总超时时长
            self.build.timeout = self.build.timeout - duration if self.build.timeout else self.build.timeout
        # 一键绘图
        if self.model.prof == 1 or self.model.prof == 2:
            wf.work_flow_plot(self.build_root, self.model.prof, self.model.pe)

    def py_build(self):
        cmd: str = f"{sys.executable} -I -m build --no-isolation -v"
        ts = datetime.now(tz=timezone.utc)
        logging.info("Python3 Build, Cmd: %s", cmd)
        ret = self.run_build_cmd(cmd=cmd, check=True, timeout=self.build.timeout)
        ret.check_returncode()
        duration: int = int((datetime.now(tz=timezone.utc) - ts).seconds)
        duration_str: str = f"{duration}/{self.build.timeout}" if self.build.timeout else f"{duration}"
        logging.info("Python3 Build, Cmd: %s, Duration %s sec", cmd, duration_str)

    def py_tests(self):
        if not self.tests.utest.enable and not self.tests.stest.enable:
            return
        # 重装 whl
        dist: Path = Path(self.src_root, "dist")
        self.py_tests_install_whl(dist=dist)
        # 执行用例, UTest
        utest_ini: Path = Path(self.src_root, "python/tests/pytest_ut.ini")
        self.py_tests_run_pytest(dist=dist, tests=self.tests.utest, ini=utest_ini, ext="-n auto")
        # 执行用例, STest
        stest_ini: Path = Path(self.src_root, "python/tests/pytest_st.ini")
        self.py_tests_run_pytest(dist=dist, tests=self.tests.stest, ini=stest_ini, ext="--forked")

    def py_tests_install_whl(self, dist: Path):
        # 卸载 whl 包
        self.pip_uninstall(name=self.whl_prefix, path=dist)
        # 查找 whl 包
        whl: Optional[Path] = self.find_match_whl(name=self.whl_prefix, path=dist)
        if not whl:
            raise RuntimeError(f"Can't find {self.whl_prefix} whl file from {dist}")
        # 安装 whl 包
        self.pip_install(whl=whl, path=dist)

    def py_tests_run_pytest(self, dist: Optional[Path], tests: TestsFilterParam, ini: Path, ext: str = ""):
        if not tests.enable:
            return
        # cmd 拼接
        cmd: str = f"{sys.executable} -m pytest -vv -s --rootdir={self.src_root}"
        if tests.filter_str in ["ON"]:
            cmd += f" -c {ini} {ext}"
        else:
            cmd += f" {tests.filter_str_pytest} --forked"
        # cmd 执行
        origin_env = {**os.environ}
        update_env = {}
        if dist:
            ori_env_python_path: str = origin_env.get("PYTHONPATH", "")
            act_env_python_path: str = f"{dist}:{ori_env_python_path}" if ori_env_python_path else f"{dist}"
            update_env.update({"PYTHONPATH": act_env_python_path})
            #
            add_env_ld: str = str(Path(dist, f"{self.whl_prefix}", "lib"))
            ori_env_ld: str = origin_env.get("LD_LIBRARY_PATH", "")
            act_env_ld: str = f"{ori_env_ld}:{add_env_ld}" if ori_env_ld else f"{add_env_ld}"
            update_env.update({"LD_LIBRARY_PATH": act_env_ld})
        ts = datetime.now(tz=timezone.utc)
        logging.info("pytest run, Cmd: %s", cmd)
        ret = self.run_build_cmd(cmd=cmd, check=True, update_env=update_env)
        ret.check_returncode()
        duration: int = int((datetime.now(tz=timezone.utc) - ts).seconds)
        logging.info("pytest run, Cmd: %s, Duration %s sec", cmd, duration)


class SubCommandMgr:
    @classmethod
    def init_param_tools_profiling(cls, args, ctrl: BuildCtrl):
        ctrl.tests.stest_tools.init_param(args=args)
        ctrl.tests.stest_tools.init_param_profiling(args=args)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    BuildCtrl.main()
