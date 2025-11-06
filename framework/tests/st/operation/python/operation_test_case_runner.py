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
""" """
import argparse
import json
import os
import signal
import subprocess
import sys

import pytest
import torch

from convert_test_case_data_to_json import convert_data_to_json
from analyze_test_case_log import TestCaseLogAnalyzer


class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.logger = open(log_file, "w", encoding="utf-8")

    def __del__(self):
        self.terminal = None
        if self.logger is not None and not self.logger.closed:
            self.logger.close()
        self.logger = None

    def write(self, msg):
        self.terminal.write(msg)
        self.logger.write(msg)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.logger.flush()

    def isatty(self):
        return False


class OperationTestCaseRunner:
    def __init__(self, config):
        self.work_path = os.getcwd()
        self.input_file = os.path.abspath(config.input_file)
        self.op = config.op
        self.index = [config.start_index, config.end_index]
        self.report_file = os.path.abspath(config.report)
        self.device = config.device
        self.pto = config.pto
        self.json_only = config.json_only
        self.clean = config.clean
        self.save_data = config.save_data
        self.log_path = os.path.dirname(self.report_file) + "/test_case_log"
        self.plog_cache_path = f"{self.work_path}/plog"

    @classmethod
    def run_cmd(cls, cmd):
        print(f"Start exec : {cmd}")
        with subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            text=True,
            encoding="utf-8",
            start_new_session=True,
            shell=True,
        ) as process:
            try:
                stdout, stderr = process.communicate()
            except KeyboardInterrupt:
                _pgid = os.getpgid(process.pid)
                os.killpg(_pgid, signal.SIGINT)
            except Exception:
                process.kill()
                raise
            finally:
                stdout = stdout or ""
                stderr = stderr or ""
            ret_code = process.poll()
            if ret_code:
                raise subprocess.CalledProcessError(
                    ret_code, process.args, output=stdout, stderr=stderr
                )
        return subprocess.CompletedProcess(process.args, ret_code, stdout, stderr)

    def clear_cache_files(self):
        if os.path.exists(self.log_path):
            os.system(f"rm -rf {self.log_path}")
        os.mkdir(self.log_path)
        if os.path.exists(self.plog_cache_path):
            os.system(f"rm -rf {self.plog_cache_path}")
        os.mkdir(self.plog_cache_path)
        if os.path.exists(self.report_file):
            os.remove(self.report_file)

    def run_test_case(self, test_case_info):
        index = test_case_info["index"]
        case_name = test_case_info["case_name"]
        case_op = test_case_info["operation"]
        # run test
        test_case = f"Test{case_op}/{case_op}OperationTest.Test{case_op}/{index}"
        clean_str = "-c" if self.clean else ""
        log_file = f"{self.log_path}/{case_name}.log"
        cmd = f"{sys.executable} build.py -s='{test_case}' -d={self.device} {clean_str} 2>&1 | tee {log_file}"
        return self.run_cmd(cmd)

    def gen_test_report(self, test_case_info: dict):
        case_index = test_case_info["case_index"]
        case_name = test_case_info["case_name"]
        case_op = test_case_info["operation"]
        # test report(excel file)
        log_file = f"{self.log_path}/{case_name}.log"
        log_path = f"{self.log_path}/{case_op}/{case_index}"
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)
        analyzer = TestCaseLogAnalyzer(
            case_index, case_name, case_op, log_file, self.report_file
        )
        is_pass = analyzer.analyze()
        if not is_pass or self.save_data:
            os.system(f"cp -rf {self.plog_cache_path} {log_path}")
        os.system(f"mv {log_file} {log_path}")
        os.system(f"rm -rf {self.plog_cache_path}")

    def run_pto_test_case(self, test_case_info):
        pto_install_path = f"{self.work_path}/build/pypto_install"
        if pto_install_path not in sys.path:
            sys.path.insert(0, pto_install_path)
        torch_install_path = os.path.dirname(torch.__file__)
        os.environ["LD_LIBRARY_PATH"] = (
            f"{torch_install_path}:{os.getenv('LD_LIBRARY_PATH')}"
        )
        os.environ["TILEFWK_CONFIG_PATH"] = (
            f"{pto_install_path}/pto/tile_fwk_config.json"
        )
        os.environ["PLATFORM_CONFIG_PATH"] = (
            f"{pto_install_path}/pto/tile_fwk_platform_info.json"
        )
        os.environ["TILE_FWK_DEVICE_ID"] = f"{self.device}"

        case_name = test_case_info["case_name"]
        logger = Logger(f"{self.log_path}/{case_name}.log")
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = logger
        sys.stderr = logger
        os.chdir(f"{self.work_path}/framework/tests/st/python")
        os.environ["CUR_TEST_CASE_INFO"] = json.dumps(test_case_info)
        pytest.main(["-vs", "st/operation/vector_operation_test_case_launcher.py"])
        os.chdir(f"{self.work_path}")
        sys.stdout = stdout
        sys.stderr = stderr

    def run(self):
        os.environ["ASCEND_PROCESS_LOG_PATH"] = self.plog_cache_path
        json_path = f"{self.work_path}/framework/tests/st/operation/test_case/"
        test_case_info_list = convert_data_to_json(
            self.input_file, self.op, self.index, json_path
        )
        if self.json_only:
            return

        if self.pto:
            clean_str = "-c" if self.clean else ""
            self.run_cmd(
                f"{sys.executable} build.py -s -t=tile_fwk_stest_python {clean_str}"
            )

        for test_case_info in test_case_info_list:
            # run test
            (
                self.run_pto_test_case(test_case_info)
                if self.pto
                else self.run_test_case(test_case_info)
            )
            # generate test report
            self.gen_test_report(test_case_info)
            # clear golden data
            index = test_case_info["index"]
            case_op = test_case_info["operation"]
            test_case = f"Test{case_op}/{case_op}OperationTest.Test{case_op}/{index}"
            golden_path = f"{self.work_path}/build/framework/tests/st/golden/{test_case}"
            if os.path.exists(golden_path + "/golden_desc.json"):
                os.remove(golden_path + "/golden_desc.json")
            if not self.save_data:
                os.system(f"rm -rf {golden_path}")

        del os.environ["ASCEND_PROCESS_LOG_PATH"]


def add_test_case_args(parser):
    # 参数注册
    parser.add_argument("op", type=str, help="The operation will be test.")
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        default=None,
        help="The input test case data file.",
    )
    parser.add_argument(
        "-s",
        "--start_index",
        nargs="?",
        type=int,
        default=0,
        help="The start index of test case, it will be row id sub 2 for csv or excel.",
    )
    parser.add_argument(
        "-e",
        "--end_index",
        nargs="?",
        type=int,
        default=-1,
        help="The end index of test case,"
        " it will be reset to the max index when it is less than 0 or greater than the max.",
    )
    parser.add_argument(
        "--report",
        nargs="?",
        type=str,
        default="test_result_report.xlsx",
        help="The report of test case result.",
    )


def add_build_args(parser):
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="clean the compile result.",
    )


def add_run_args(parser):
    parser.add_argument(
        "-d",
        "--device",
        nargs="?",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        help="Select npu id.",
    )

    parser.add_argument("--pto", action="store_true", help="Test pto test case.")
    parser.add_argument(
        "--json_only", action="store_true", help="Convert csv to json only, not run."
    )
    parser.add_argument(
        "--save_data", action="store_true", help="Save golden and plog etc."
    )


def parse_args():
    """主处理流程"""
    parser = argparse.ArgumentParser(
        description=f"Run operation st test case.", epilog="Best Regards!"
    )
    add_test_case_args(parser)
    add_build_args(parser)
    add_run_args(parser)

    return parser.parse_args()


def dump_test_case_args(args):
    print("Run operation test case args :")
    print("Op : ", args.op)
    print("Input data : ", args.input_file)
    print("Start index : ", args.start_index)
    print("End index : ", args.end_index)
    print("Device : ", args.device)
    print("Clean : ", args.clean)
    print("Is pto case : ", args.pto)
    print("Save data : ", args.save_data)
    print("Is json only : ", args.json_only)
    print("Report : ", args.report)


def run_test_case(args):
    if args.input_file is None:
        args.input_file = (
            f"{os.getcwd()}/framework/tests/st/operation/test_case/{args.op}_st_test_cases.csv"
        )
    dump_test_case_args(args)
    if not os.path.exists(args.input_file):
        raise ValueError(args.input_file + " is not exists.")
    runner = OperationTestCaseRunner(args)
    runner.clear_cache_files()
    runner.run()
