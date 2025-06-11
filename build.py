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
import argparse
import logging
import multiprocessing
import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional, List


class BuildCtrl:
    """ 构建过程控制.

    本类包含由命令行指定或解析出的控制标记/参数, 以控制构建过程执行.
    """

    def __init__(self, args):
        # 路径
        self.src_root: Path = Path(__file__).parent.resolve()
        self.build_root: Path = Path(Path.cwd(), "build")
        self.install_root: Path = Path(self.build_root.parent, "output")
        # 控制标记/参数预处理(common)
        self.backend_type = "npu" if args.backend is None else args.backend
        self.build_targets: Optional[List[str]] = args.targets  # 编译阶段的编译目标
        self.build_job_num: int = args.job_num if args.job_num > 0 else int(multiprocessing.cpu_count())
        self.forced_clean: bool = args.clean  # 强制清理 Build-Tree 及 Install-Tree 标记
        self.timeout = None if args.timeout == 0 else args.timeout  # 构建超时时长
        self.init_param_common()
        # 控制标记/参数预处理(tests)
        self.utest_enable: bool = False  # UTest 使能标记
        self.utest_cases_filter: Optional[str] = None  # 指定 UTest 所需执行用例
        self.stest_enable: bool = False  # STest 使能标记
        self.stest_cases_filter: Optional[str] = None  # 指定 STest 所需执行用例
        self.stest_golden_path: Optional[Path] = None  # STest 指定 Golden 路径
        self.stest_golden_path_clean: bool = args.stest_golden_path_clean  # STest 清理 Golden 标记
        self.stest_device_id: str = ""
        self.tests_auto_execute: bool = args.disable_auto_execute
        self.tests_auto_execute_parallel: bool = False
        self.tests_enable_binary_cache: bool = False
        self.tests_changed_file: Optional[Path] = args.changed_files
        self.stest_dump_json: bool = args.stest_dump_json
        self.init_param_tests(args=args)
        # 控制标记/参数预处理(build_tools)
        self.asan: bool = args.asan
        self.ubsan: bool = args.ubsan
        self.gcov: bool = args.gcov
        self.experiment_copy_aicpu_binary: bool = args.experiment_copy_aicpu_binary
        self.prof = args.prof
        self.pe = args.pe
        # 控制标记/参数预处理(tools)
        self.tools_cases_csv_file: Optional[Path] = None
        self.tools_intercept_flag: bool = False
        self.tools_prof_enable: bool = False
        self.tools_prof_level: str = "l1"
        self.tools_output_clean: bool = False
        self.tools_prof_warn_up_cnt: Optional[int] = None
        self.tools_prof_try_cnt: Optional[int] = None
        self.tools_prof_max_cnt: Optional[int] = None

        self.sim = args.sim
        self.sim_with_onboard_aicpu = args.sim_with_onboard_aicpu
        self.back_annotation_aicpu = args.back_annotation_aicpu
        self.back_annotation_aicore = args.back_annotation_aicore
        self.calendar = args.calendar
        self.replay_file_path = args.replay_file_path

    def __str__(self):
        desc = ""
        desc += f"\nArgs Param"
        desc += f"\n\tBackend Type  : {self.backend_type}"
        desc += f"\n\tForced Clean  : {self.forced_clean}"
        desc += f"\n\tBuild Job Num : {self.build_job_num}"
        desc += f"\n\tBuild Targets : {self.build_targets}"
        desc += f"\n\tBuild UTest   : Flag({self.utest_enable}), Filter({self.utest_cases_filter})"
        desc += (f"\n\tBuild STest   : Flag({self.stest_enable}), Filter({self.stest_cases_filter}),"
                 f" DeviceID({self.stest_device_id})")
        desc += (f"\n\tTests Execute : Flag({self.tests_auto_execute}),"
                 f" Parallel({self.tests_auto_execute_parallel}), BinaryCache({self.tests_enable_binary_cache})"
                 f" PrintJson({self.stest_dump_json})")
        desc += f"\n\tTests Changed : File({self.tests_changed_file})"
        desc += f"\nOthers"
        desc += f"\n\tSource  Root Dir : {self.src_root}"
        desc += f"\n\tBuild   Root Dir : {self.build_root}"
        desc += f"\n\tInstall Root Dir : {self.install_root}"
        return desc

    def init_param_common(self):
        if self.timeout is not None:
            ret = subprocess.run(shlex.split("uname -m"), capture_output=True, check=True, text=True, encoding='utf-8')
            ret.check_returncode()
            hardware_processor_type = re.sub('[\r\n\t]', '', ret.stdout)
            self.timeout = self.timeout if hardware_processor_type == "x86_64" else self.timeout * 2

    def init_param_tests(self, args):
        self.tests_changed_file = None if not self.tests_changed_file else Path(self.tests_changed_file).resolve()
        self.tests_auto_execute_parallel = True if self.tests_changed_file is not None else False
        self.tests_enable_binary_cache = True if self.tests_changed_file is not None else False
        self.init_param_tests_utest(args=args)
        self.init_param_tests_stest(args=args)

    def init_param_tests_utest(self, args):
        # 识别具体需要触发的 Tests 范围, 暂不支持
        if args.utest is None:
            self.utest_enable = True
            self.utest_cases_filter = "ON"  # 指定 -u 但未指定 filter
        elif args.utest == "":
            self.utest_enable = False
            self.utest_cases_filter = None  # 未指定 -u
        else:
            self.utest_enable = True
            self.utest_cases_filter = args.utest  # 使用脚本传入的 filter

    def init_param_tests_stest(self, args):
        # 识别具体需要触发的 Tests 范围, 暂不支持
        if args.stest is None:
            self.stest_enable = True
            self.stest_cases_filter = "ON"  # 指定 -s 但未指定 filter
        elif args.stest == "":
            self.stest_enable = False
            self.stest_cases_filter = None  # 未指定 -u
        else:
            self.stest_enable = True
            self.stest_cases_filter = args.stest  # 使用脚本传入的 filter
        if args.stest_golden_path is None:  # 未传参
            self.stest_golden_path = Path(self.build_root, "tests/st/golden")
        elif args.stest_golden_path == "":  # 未指定
            self.stest_golden_path = Path(self.build_root, "tests/st/golden")
        else:
            self.stest_golden_path = Path(args.stest_golden_path).resolve()
        self.stest_golden_path.mkdir(parents=True, exist_ok=True)
        # STest 并行加速
        devs = ["0"]
        if args.device is not None:
            devs = [str(d) for d in list(set(args.device)) if d is not None and str(d) != ""]
        self.stest_device_id = ":".join(devs)

    def init_param_tools(self, args):
        self.tools_output_clean = args.tools_output_clean
        self.tools_intercept_flag = args.intercept
        self.tools_cases_csv_file = Path(args.cases_csv_file[0]).resolve() if args.cases_csv_file else None

    def init_param_tools_profiling(self, args):
        self.tools_prof_enable = True
        self.tools_prof_level = args.prof_level
        self.tools_prof_warn_up_cnt = args.prof_warn_up_cnt[0] if args.prof_warn_up_cnt else None
        self.tools_prof_try_cnt = args.prof_try_cnt[0] if args.prof_try_cnt else None
        self.tools_prof_max_cnt = args.prof_max_cnt[0] if args.prof_max_cnt else None

    @classmethod
    def main(cls):
        """ 主处理流程 """
        parser = argparse.ArgumentParser(description=f"Ascend C++ Build Ctrl.", epilog="Best Regards!")
        sub_parser = parser.add_subparsers()  # 子命令
        # 参数注册
        parser.add_argument("-b", "--backend", nargs="?", type=str, default="npu",
                            choices=["npu", "cost_model"],
                            help="backend, such as npu/cost_model etc.")
        parser.add_argument("-t", "--targets", nargs="?", type=str, action="append",
                            help="targets, specific build targets, "
                                 "If you specify more than one, all targets within the specified range are built.")
        parser.add_argument("-j", "--job_num", nargs="?", type=int, default=-1,
                            help="job num, specific job num of build.")
        parser.add_argument("-c", "--clean", action="store_true", default=False,
                            help="clean, clean Build-Tree and Install-Tree before build.")
        parser.add_argument("--timeout", nargs="?", type=int, default=0,
                            help="build task timeout.")
        cls._add_argument_tests(parser=parser)
        cls._add_argument_build_tools(parser=parser)
        cls._add_argument_tools(sub_parser=sub_parser)

        # 流程处理
        args = parser.parse_args()
        ctrl = BuildCtrl(args=args)
        if 'func' in args:
            args.func(args=args, ctrl=ctrl)
        logging.info("%s", ctrl)
        ctrl.clean()
        ctrl.configure()
        ctrl.build()

    @classmethod
    def _add_argument_tests(cls, parser):
        parser.add_argument("-u", "--utest", nargs="?", type=str, default="",
                            help="utest, enable UTest scene, specific UTest case filter, "
                                 "multiple test cases are separated by ':'/',' .")
        parser.add_argument("-s", "--stest", nargs="?", type=str, default="",
                            help="stest, enable STest scene, specific STest case filter, "
                                 "multiple test cases are separated by ':'/',' .")
        parser.add_argument("--stest_golden_path", nargs="?", type=str, default="",
                            help="Specific STest golden path.")
        parser.add_argument("--stest_golden_path_clean", action="store_true", default=False,
                            help="Clean STest golden.")
        parser.add_argument("--disable_auto_execute", action="store_false", default=True,
                            help="Disable auto execute STest/Utest with build.")
        parser.add_argument("-d", "--device", nargs="?", type=int, action="append",
                            help="Device ID, default 0.")
        parser.add_argument("--changed_files", nargs="?", type=Path, default=None,
                            help="Specify the file of files changed, "
                                 "so that the corresponding test cases can be triggered incrementally.")
        parser.add_argument("--pe", nargs="?", type=int, default=2, choices=[1, 2, 4, 5, 6, 7, 8],
                            help="Enable pmuEvent.")
        parser.add_argument("--stest_dump_json", action="store_true", default=False,
                            help="Dump json files.")
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

    @classmethod
    def _add_argument_build_tools(cls, parser):
        parser.add_argument("--asan", action="store_true", default=False,
                            help="Enable AddressSanitizer.")
        parser.add_argument("--ubsan", action="store_true", default=False,
                            help="Enable UndefinedBehaviorSanitizer.")
        parser.add_argument("--gcov", action="store_true", default=False,
                            help="Enable GNU Coverage Instrumentation Tool.")
        parser.add_argument("--experiment_copy_aicpu_binary", action="store_true", default=False,
                            help="Experiment, copy aicpu binary auto.")
        parser.add_argument("--prof", nargs="?", type=int, default=0, choices=[1, 2],
                            help="Enable workflow.")

    @classmethod
    def _add_argument_tools(cls, sub_parser):
        # Tools
        parser_tools = sub_parser.add_parser('tools', help="Tools")
        parser_tools.add_argument("--tools_output_clean", action="store_true", default=False,
                                  help="clean, Specify clean flag, clean tools output dir")
        parser_tools.add_argument("--intercept", action="store_true", default=False,
                                  help="intercept, Intercept if have failed case result")
        parser_tools.add_argument("--cases_csv_file", nargs=1, type=Path, default=None,
                                  help="Specify cases.csv")
        # Tools.Profiling
        sub_parser_prof = parser_tools.add_subparsers(dest="Tolls SubCommand")
        parser_prof = sub_parser_prof.add_parser('profiling', help="Profiling", aliases=['prof'])
        parser_prof.add_argument("--prof_level", nargs="?", type=str, default="l1", choices=["l1", "l2"],
                                 help="Specify profiling level")
        parser_prof.add_argument("--prof_warn_up_cnt", nargs=1, type=int, default=None,
                                 help="Specify profiling warn up cnt")
        parser_prof.add_argument("--prof_try_cnt", nargs=1, type=int, default=None,
                                 help="Specify profiling try cnt")
        parser_prof.add_argument("--prof_max_cnt", nargs=1, type=int, default=None,
                                 help="Specify profiling max cnt")
        parser_prof.set_defaults(func=SubCommandMgr.init_param_tools_profiling)

    def clean(self):
        """ 清理中间结果, 清理内容包括构建树, 安装树全部内容. """
        pass

    def configure(self):
        """ CMake Configure 阶段流程. """
        pass

    def build(self):
        """ CMake Build 阶段流程. """
        pass


class SubCommandMgr:
    @classmethod
    def init_param_tools_profiling(cls, args, ctrl: BuildCtrl):
        ctrl.init_param_tools(args=args)
        ctrl.init_param_tools_profiling(args=args)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    BuildCtrl.main()
