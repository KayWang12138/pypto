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


# Python3.7.5 等较低版本, 需要添加
g_src_root: Path = Path(__file__).parent.resolve()
g_src_tools: Path = Path(g_src_root, "tools")
if str(g_src_tools) not in sys.path:
    sys.path.append(str(g_src_tools))

import work_flow as wf


class ArgParser:

    @staticmethod
    def get_opt(opt: Optional[str]) -> Tuple[bool, Optional[str]]:
        """ 初始化可指定 str 的 args """
        if opt is None:
            return True, "ON"   # 指定 对应参数 但未指定内容
        elif opt == "":
            return False, None  # 未指定 对应参数
        else:
            return True, opt    # 指定 对应参数 且指定内容


class BuildCtrl:
    """ 构建过程控制.

    本类包含由命令行指定或解析出的控制标记/参数, 以控制构建过程执行.
    """


    @dataclasses.dataclass
    class BuildParam:
        """ 构建相关参数
        """
        targets: Optional[List[str]] = None  # 编译目标
        job_num: int = min(int(math.ceil(float(multiprocessing.cpu_count()) * 0.8)), 16) # 使用核数
        clean: bool = False  # 强制清理 Build-Tree 及 Install-Tree 标记
        timeout: Optional[int] = None  # 构建超时时长
        type_: Optional[str] = None  # 构建类型
        asan: bool = False
        ubsan: bool = False
        gcov: bool = False
        clang_path: Optional[Path] = None

        def __init__(self, args):
            self.targets = args.targets
            self.job_num = args.job_num if args.job_num > 0 else self.job_num
            self.clean = args.clean
            self.timeout = None if args.timeout == 0 else args.timeout
            self.type_ = args.build_type
            self.asan = args.asan
            self.ubsan = args.ubsan
            self.gcov = args.gcov
            # clang
            if args.clang is None:  # 指定 clang 参数, 但未指定具体路径, 此时需尝试寻找
                cmd = "which clang"
                ret = subprocess.run(shlex.split(cmd), capture_output=True, check=True, text=True, encoding='utf-8')
                ret.check_returncode()
                self.clang_path = Path(ret.stdout).resolve()
            elif args.clang == "":  # 未指定 clang 参数
                self.clang_path = None
            else:  # 指定 clang 参数, 并指定具体路径
                self.clang_path = Path(args.clang)
            if self.clang_path is not None:
                self.clang_path = Path(self.clang_path).resolve().parent
                if not self.clang_path.exists():
                    raise ValueError(f"Clang install path not exist, path={self.clang_path}")


    @dataclasses.dataclass
    class TestsParam:
        """ Tests 相关参数
        """
        changed_file: Optional[Path] = None  # 修改文件路径
        auto_execute: bool = False  # 用例自动执行
        auto_execute_parallel: bool = False  # 用例并行执行
        interpreter_config: bool = False

        def __init__(self, args):
            self.changed_file = None if not args.changed_files else Path(args.changed_files).resolve()
            self.auto_execute = args.disable_auto_execute
            self.auto_execute_parallel = self.ci_model
            self.interpreter_config = args.enable_interpreter_config

        @property
        def ci_model(self) -> bool:
            return True if self.changed_file else False


    @dataclasses.dataclass
    class UTestParam:
        """ UTest 相关参数
        """
        enable: bool = False  # UTest 使能标记
        cases_filter: Optional[str] = None  # 指定 UTest 所需执行用例
        python_enable: bool = False  # Python UTest 使能标记
        python_cases_filter: Optional[str] = None  # 指定 Python UTest 所需执行用例

        def __init__(self, args):
            self.enable, self.cases_filter = ArgParser.get_opt(opt=args.utest)
            self.python_enable, self.python_cases_filter = ArgParser.get_opt(opt=args.utest_python)


    @dataclasses.dataclass
    class STestParam:
        """ STest 相关参数
        """
        enable: bool = False  # STest 使能标记
        cases_filter: Optional[str] = None  # 指定 STest 所需执行用例
        distributed_enable: bool = False  # distributed STest 使能标记
        distributed_cases_filter: Optional[str] = None  # 指定 distributed STest 所需执行用例
        python_enable: bool = False  # Python STest 使能标记
        python_cases_filter: Optional[str] = None  # 指定 Python STest 所需执行用例
        golden_path_clean: bool = False  # STest 清理 Golden 标记
        golden_path: Optional[Path] = None  # STest 指定 Golden 路径
        device_id: str = ""
        enable_binary_cache: bool = False
        dump_json: bool = False

        def __init__(self, args, tests, build_root: Path):
            self.enable, self.cases_filter = ArgParser.get_opt(opt=args.stest)
            self.distributed_enable, self.distributed_cases_filter = ArgParser.get_opt(opt=args.stest_distributed)
            self.python_enable, self.python_cases_filter = ArgParser.get_opt(opt=args.stest_python)
            self.python_enable = True if self.enable else self.python_enable  # 依赖 CI 任务拆分
            # Golden Path
            self.golden_path_clean = args.stest_golden_path_clean
            if args.stest_golden_path is None:  # 未传参
                self.golden_path = Path(build_root, "tests/st/golden")
            elif args.stest_golden_path == "":  # 未指定
                self.golden_path = Path(build_root, "tests/st/golden")
            else:
                self.golden_path = Path(args.stest_golden_path).resolve()
            self.golden_path.mkdir(parents=True, exist_ok=True)
            # DeviceId
            devs = ["0"]
            if args.device is not None:
                devs = [str(d) for d in list(set(args.device)) if d is not None and str(d) != ""]
            self.device_id = ":".join(devs)
            #
            self.enable_binary_cache = tests.ci_model
            self.dump_json = args.stest_dump_json


    @dataclasses.dataclass
    class ToolsParam:
        """ Tools 相关参数
        """
        cases_csv_file: Optional[Path] = None
        intercept_flag: bool = False
        prof_enable: bool = False
        prof_level: str = "l1"
        output_clean: bool = False
        prof_warn_up_cnt: Optional[int] = None
        prof_try_cnt: Optional[int] = None
        prof_max_cnt: Optional[int] = None

        def init_param(self, args):
            self.output_clean = args.tools_output_clean
            self.intercept_flag = args.intercept
            self.cases_csv_file = Path(args.cases_csv_file[0]).resolve() if args.cases_csv_file else None

        def init_param_profiling(self, args):
            self.prof_enable = True
            self.prof_level = args.prof_level
            self.prof_warn_up_cnt = args.prof_warn_up_cnt[0] if args.prof_warn_up_cnt else None
            self.prof_try_cnt = args.prof_try_cnt[0] if args.prof_try_cnt else None
            self.prof_max_cnt = args.prof_max_cnt[0] if args.prof_max_cnt else None


    def __init__(self, args):
        # 路径
        self.src_root: Path = Path(__file__).parent.resolve()
        self.build_root: Path = Path(Path.cwd(), "build")
        self.install_root: Path = Path(self.build_root.parent, "output")
        # 控制标记/参数预处理(common)
        self.backend_type = "npu" if args.backend is None else args.backend
        self.build: BuildCtrl.BuildParam = BuildCtrl.BuildParam(args=args)
        # 控制标记/参数预处理(tests)
        self.tests: BuildCtrl.TestsParam = BuildCtrl.TestsParam(args=args)
        self.utest: BuildCtrl.UTestParam = BuildCtrl.UTestParam(args=args)
        self.stest: BuildCtrl.STestParam = BuildCtrl.STestParam(args=args, tests=self.tests, build_root=self.build_root)
        # 控制标记/参数预处理(build_tools)
        self.prof = args.prof
        self.pe = args.pe
        # 控制标记/参数预处理(tools)
        self.tools: BuildCtrl.ToolsParam = BuildCtrl.ToolsParam()
        # Model
        self.sim = args.sim
        self.sim_with_onboard_aicpu = args.sim_with_onboard_aicpu
        self.back_annotation_aicpu = args.back_annotation_aicpu
        self.back_annotation_aicore = args.back_annotation_aicore
        self.calendar = args.calendar
        self.pvmodel = args.pvmodel
        self.replay_file_path = args.replay_file_path

    def __str__(self):

        def get_filter_str(_filter: Optional[str]) -> str:
            if _filter is None:
                _filter_str = "0"
            elif _filter == "ON":
                _filter_str = "ON"
            else:
                _filter_list = _filter.split(':')
                _filter_str = f"{len(_filter_list)}"
            return _filter_str

        ver = sys.version_info
        desc = ""
        desc += f"\nEnviron"
        desc += f"\n\tPython3                  : {sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"
        desc += f"\nArgs Param"
        desc += f"\n\tBackend Type             : {self.backend_type}"
        desc += f"\n\tForced Clean             : {self.build.clean}"
        desc += f"\n\tBuild Job Num            : {self.build.job_num}"
        desc += f"\n\tBuild Type               : {self.build.type_}"
        desc += f"\n\tBuild Targets            : {self.build.targets}"
        desc += (f"\n\tBuild UTest              : Flag({self.utest.enable}), "
                 f"Filter({get_filter_str(self.utest.cases_filter)})")
        desc += (f"\n\tBuild UTest(Python)      : Flag({self.utest.python_enable}), "
                 f"Filter({get_filter_str(self.utest.python_cases_filter)})")
        desc += (f"\n\tBuild STest              : Flag({self.stest.enable}), "
                 f"Filter({get_filter_str(self.stest.cases_filter)}) DeviceID({self.stest.device_id})")
        desc += (f"\n\tBuild STest(Distributed) : Flag({self.stest.distributed_enable}), "
                 f"Filter({get_filter_str(self.stest.distributed_cases_filter)})")
        desc += (f"\n\tBuild STest(Python)      : Flag({self.stest.python_enable}), "
                 f"Filter({get_filter_str(self.stest.python_cases_filter)})")
        desc += (f"\n\tTests Execute            : Flag({self.tests.auto_execute}),"
                 f" Parallel({self.tests.auto_execute_parallel}),"
                 f" PrintJson({self.stest.dump_json}),"
                 f" BinaryCache({self.stest.enable_binary_cache})")
        desc += f"\n\tTests Changed            : File({self.tests.changed_file})"
        desc += f"\n\tTests Interpreter        : {self.tests.interpreter_config}"
        desc += f"\nOthers"
        desc += f"\n\tSource  Root Dir         : {self.src_root}"
        desc += f"\n\tBuild   Root Dir         : {self.build_root}"
        desc += f"\n\tInstall Root Dir         : {self.install_root}"
        return desc

    @property
    def tests_enable(self) -> bool:
        ut_enable: bool = self.utest.enable or self.utest.python_enable
        st_enable: bool = self.stest.enable or self.stest.distributed_enable or self.stest.python_enable
        return ut_enable or st_enable

    @classmethod
    def main(cls):
        """ 主处理流程 """
        parser = argparse.ArgumentParser(description=f"Tile Framework C++ Build Ctrl.", epilog="Best Regards!")
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
        parser.add_argument("--build_type", nargs="?", type=str, default=None,
                            choices=["Debug", "Release", "MinSizeRel", "RelWithDebInfo"],
                            help="build type.")
        cls._add_argument_tests(parser=parser)
        cls._add_argument_build_tools(parser=parser)
        cls._add_argument_tools(sub_parser=sub_parser)

        # 流程处理
        args = parser.parse_args()
        ctrl = BuildCtrl(args=args)
        if 'func' in args:
            args.func(args=args, ctrl=ctrl)
        logging.info("%s", ctrl)
        ctrl.clean_process()
        ctrl.configure_process()
        ctrl.build_process()

    @classmethod
    def _add_argument_tests(cls, parser):
        parser.add_argument("-u", "--utest", nargs="?", type=str, default="",
                            help="utest, enable UTest scene, specific UTest case filter, "
                                 "multiple test cases are separated by ':'/',' .")
        parser.add_argument("--utest_python", nargs="?", type=str, default="",
                            help="utest, enable Python UTest scene, specific Python UTest case filter, "
                                 "multiple Python test cases are separated by ':'/',' .")
        parser.add_argument("-s", "--stest", nargs="?", type=str, default="",
                            help="stest, enable STest scene, specific STest case filter, "
                                 "multiple test cases are separated by ':'/',' .")
        parser.add_argument("--stest_distributed", nargs="?", type=str, default="",
                            help="stest, enable Distributed STest scene, Distributed STest case filter, "
                                 "multiple distributed test cases are separated by ':'/',' .")
        parser.add_argument("--stest_python", nargs="?", type=str, default="",
                            help="stest, enable Python STest scene, Python STest case filter, "
                                 "multiple Python test cases are separated by ':'/',' .")
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
        parser.add_argument("-pv", "--pvmodel", action="store_true", default=False,
                            help="Enable PVModel mode.")
        parser.add_argument("--enable_interpreter_config", action="store_true", default=False,
                            help="enable STest Interpreter Config")

    @classmethod
    def _add_argument_build_tools(cls, parser):
        parser.add_argument("--clang", nargs="?", type=str, default="",
                            help="Specify clang install path, such as /usr/bin/clang")
        parser.add_argument("--asan", action="store_true", default=False,
                            help="Enable AddressSanitizer.")
        parser.add_argument("--ubsan", action="store_true", default=False,
                            help="Enable UndefinedBehaviorSanitizer.")
        parser.add_argument("--gcov", action="store_true", default=False,
                            help="Enable GNU Coverage Instrumentation Tool.")
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

    @classmethod
    def _gen_cmd(cls, opt: str, ctr: bool, tv: str = "ON", fv: str = "OFF") -> str:
        cmd: str = f" -D{opt}=" + (tv if ctr else fv)
        return cmd

    @classmethod
    def _gen_cmd_str(cls, opt: str, v: str) -> str:
        return cls._gen_cmd(opt=opt, ctr=True, tv=v)

    @classmethod
    def _gen_cmd_path(cls, opt: str, v: Path) -> str:
        return cls._gen_cmd_str(opt=opt, v=str(v))

    def clean_process(self):
        """ 清理中间结果, 清理内容包括构建树, 安装树全部内容. """
        if self.build.clean:
            if self.build_root.exists():
                logging.info("Clean Build-Tree(%s)", self.build_root)
                shutil.rmtree(self.build_root)
            if self.install_root.exists():
                logging.info("Clean Install-Tree(%s)", self.install_root)
                shutil.rmtree(self.install_root)
        if self.stest.enable_binary_cache:
            binary_cache_path = Path(Path.home(), "ast_data")
            if binary_cache_path.exists():
                shutil.rmtree(binary_cache_path)
                logging.info("Clean Binary Cache Path(%s)", binary_cache_path)

    def configure_process(self):
        """ CMake Configure 阶段流程. """
        # 基本配置, 当前 CMake 中有调用 python3 的情况, 传入 python3 解释器, 保证所使用的 python3 版本一致
        ver = sys.version_info
        cmd = f"cmake -S {self.src_root} -B {self.build_root} -DPython3_EXECUTABLE={sys.executable}"
        cmd += f" -DPython3_FIND_STRATEGY=LOCATION -DPython3_FIND_VERSION={ver.major}.{ver.minor}"
        cmd += f" -DCMAKE_BUILD_TYPE={self.build.type_}" if self.build.type_ else ""
        # common 相关配置
        #    SocVersion, Backend 相关配置, SocVersion相关配置暂不支持
        if self.backend_type == "npu":
            cmd += f" -DENABLE_BUILD_WITH_CANN=ON"
        if self.backend_type == "cost_model":
            cmd += f" -DENABLE_BUILD_WITH_CANN=OFF"
        # tests 相关配置
        cmd += self._configure_tests()
        # tools_build 相关配置
        cmd += self._configure_tools_build()
        # tools 相关配置
        cmd += self._configure_tools()
        # 执行
        logging.info("CMake Configure, Cmd: %s", cmd)
        ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8')
        ret.check_returncode()
        self._gen_simulation_json()

    def build_process(self):
        """ CMake Build 阶段流程. """
        # prof使能初始化
        update_env = {}
        if self.prof == 1 or self.prof == 2:
            update_env = wf.ini(self.build_root, self.prof, self.pe)
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
                ret = self.run_build_cmd(cmd=c, update_env=update_env, check=True)
            except subprocess.CalledProcessError as e:
                logging.info(f"Run cmd {c} failed, ERROR CODE: {e.returncode}")
                # 一键绘图
                if self.prof == 1 or self.prof == 2:
                    wf.work_flow_plot(self.build_root, self.prof, self.pe)
                raise
            ret.check_returncode()
            logging.info("CMake Build(%s/%s), Duration %s sec, Cmd: %s",
                         i + 1, len(cmd_list),
                         (datetime.now(tz=timezone.utc) - ts).seconds, c)
        # 一键绘图
        if self.prof == 1 or self.prof == 2:
            wf.work_flow_plot(self.build_root, self.prof, self.pe)

    def run_build_cmd(self, cmd: str, update_env: Optional[Dict[str, str]] = None,
                      check: bool = False) -> Optional[subprocess.CompletedProcess]:
        """执行具体 build 命令行

        因以下原因, 设置本函数, 而非调用原生 subprocess.run
        1. 支持多 target 构建, 各 target 构建时长共享公共 timeout 配置;
        2. UTest/STest 并行执行场景下, 执行时进程调用关系为:
               build.py(主进程) -> 进程1(CMake) -> 进程2(CMake Generator, make/ninja) -> 进程3(Python)-> 进程4(executable)
           此时若 进程1 超时, 需要触发其子/孙进程感知, 进而结束

        :param cmd: Build 命令行
        :param update_env: 环境变量(额外更新内容)
        :param check: 检查返回值
        """

        def _stop_pg(_p: subprocess.Popen):
            """通过 SIGINT 信号通知所有子/孙进程结束, python 并行脚本内会捕获该信号进行结算处理
            """
            _pgid = os.getpgid(_p.pid)
            logging.info("Send terminate event to CMake[%s]", _pgid)
            os.killpg(_pgid, signal.SIGINT)

        ts = datetime.now(tz=timezone.utc)
        stdout: Optional[str] = None
        stderr: Optional[str] = None
        env = {**os.environ}
        env.update(update_env if update_env else {})
        with subprocess.Popen(shlex.split(cmd), env=env, text=True, encoding='utf-8',
                              start_new_session=True) as process:
            try:
                stdout, stderr = process.communicate(timeout=self.build.timeout)
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
        if self.build.timeout:
            self.build.timeout = self.build.timeout - (datetime.now(tz=timezone.utc) - ts).seconds
        return subprocess.CompletedProcess(process.args, ret_code, stdout, stderr)

    def _configure_tests(self) -> str:
        cmd = ""
        # 公共
        if self.tests_enable:
            cmd += self._gen_cmd(opt="ENABLE_TESTS_EXECUTE", ctr=self.tests.auto_execute)
            cmd += self._gen_cmd(opt="ENABLE_TESTS_EXECUTE_PARALLEL",
                                 ctr=self.tests.auto_execute and self.tests.auto_execute_parallel)
        # UTest
        cmd += self._gen_cmd(opt="ENABLE_TESTS_UTEST", ctr=self.utest.enable, tv=f"{self.utest.cases_filter}")
        # UTest Python 场景
        cmd += self._gen_cmd(opt="ENABLE_TESTS_UTEST_PYTHON", ctr=self.utest.python_enable,
                             tv=f"{self.utest.python_cases_filter}")
        # STest 公共
        if self.stest.enable or self.stest.distributed_enable or self.stest.python_enable:
            # Golden
            cmd += self._gen_cmd(opt="ENABLE_TESTS_STEST_GOLDEN_PATH_CLEAN", ctr=self.stest.golden_path_clean)
            cmd += f" -DENABLE_TESTS_STEST_GOLDEN_PATH={self.stest.golden_path}"
            # BinaryCache
            cmd += self._gen_cmd(opt="ENABLE_TESTS_STEST_BINARY_CACHE", ctr=self.stest.enable_binary_cache)
            # DumJson
            cmd += self._gen_cmd(opt="ENABLE_TESTS_STEST_DUMP_JSON", ctr=self.stest.dump_json)
        # STest
        if self.stest.enable:
            # DeviceId
            cmd += f" -DENABLE_TESTS_EXECUTE_DEVICE_ID={self.stest.device_id}"
            cmd += f" -DENABLE_TESTS_STEST={self.stest.cases_filter}"
        else:
            cmd += f" -DENABLE_TESTS_STEST=OFF"
        # STest, Distributed
        cmd += self._gen_cmd(opt="ENABLE_TESTS_STEST_DISTRIBUTED", ctr=self.stest.distributed_enable,
                             tv=f"{self.stest.distributed_cases_filter}")
        # STest Interpreter Config
        cmd += self._gen_cmd(opt="ENABLE_TESTS_STEST_INTERPRETER_CONFIG", ctr=self.tests.interpreter_config)
        # STest Python
        cmd += self._gen_cmd(opt="ENABLE_TESTS_STEST_PYTHON", ctr=self.stest.python_enable)
        return cmd

    def _configure_tools_build(self) -> str:
        cmd = ""

        def _check_clang_toolchain(_opt: str, _b: str) -> Tuple[bool, str]:
            _p: Path = Path(self.build.clang_path, _b)
            if _p.exists():
                return True, self._gen_cmd_path(opt=_opt, v=_p)
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
        if self.build.clang_path is not None:
            ret, clang_cmd = _gen_clang_cmd()
            if not ret:
                raise RuntimeError(f"Clang({self.build.clang_path}) not complete.")
            cmd += clang_cmd

        cmd += self._gen_cmd(opt="ENABLE_ASAN", ctr=self.build.asan)
        cmd += self._gen_cmd(opt="ENABLE_UBSAN", ctr=self.build.ubsan)
        cmd += self._gen_cmd(opt="ENABLE_GCOV", ctr=self.build.gcov)
        return cmd

    def _configure_tools(self) -> str:
        cmd = ""
        # tools 公共参数
        if self.tools.prof_enable:
            cmd += self._gen_cmd(opt="ENABLE_TESTS_STEST_TOOLS_OUTPUT_CLEAN", ctr=self.tools.output_clean)
            cmd += self._gen_cmd(opt="ENABLE_TESTS_STEST_TOOLS_INTERCEPT", ctr=self.tools.intercept_flag)
            if self.tools.cases_csv_file:
                cmd += f" -DENABLE_TESTS_STEST_TOOLS_CASE_FILE={self.tools.cases_csv_file}"
        # Profiling 工具参数
        cmd += self._configure_tools_profiling()
        return cmd

    def _configure_tools_profiling(self) -> str:
        cmd = ""
        if self.tools.prof_enable:
            cmd += f" -DENABLE_TESTS_STEST_TOOLS_PROF=ON"
            cmd += f" -DENABLE_TESTS_STEST_TOOLS_PROF_LEVEL={self.tools.prof_level}"
            cmd += self._gen_cmd(opt="ENABLE_TESTS_STEST_TOOLS_PROF_WARN_UP_CNT",
                                 ctr=self.tools.prof_warn_up_cnt is not None, tv=f"{self.tools.prof_warn_up_cnt}")
            cmd += self._gen_cmd(opt="ENABLE_TESTS_STEST_TOOLS_PROF_TRY_CNT",
                                 ctr=self.tools.prof_try_cnt is not None, tv=f"{self.tools.prof_try_cnt}")
            cmd += self._gen_cmd(opt="ENABLE_TESTS_STEST_TOOLS_PROF_MAX_CNT",
                                 ctr=self.tools.prof_max_cnt is not None, tv=f"{self.tools.prof_max_cnt}")
        else:
            cmd += f" -DENABLE_TESTS_STEST_TOOLS_PROF=OFF"
        return cmd

    def _save_simulation_json(self, simulation_json):
        temp_json_path = os.path.join(str(self.src_root), "src/cost_model/simulation/scripts/tmp_simulation.json")
        os.makedirs(os.path.dirname(temp_json_path), exist_ok=True)
        with open(temp_json_path, 'w') as f:
            json.dump(simulation_json, f, indent=4)

    def _gen_simulation_json(self) -> None:
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
        self._save_simulation_json(simulation_json)

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


class SubCommandMgr:
    @classmethod
    def init_param_tools_profiling(cls, args, ctrl: BuildCtrl):
        ctrl.tools.init_param(args=args)
        ctrl.tools.init_param_profiling(args=args)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    BuildCtrl.main()
