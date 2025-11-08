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
import json
import os
from pathlib import Path
import sys

import pytest
import torch

from test_case_loader import TestCaseLoader
from test_case_log_analyzer import TestCaseLogAnalyzer
from test_case_logger import TestCaseLogger
from test_case_shell_actuator import TestCaseShellActuator


class TestCaseLauncher:
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
        self.pto_install_path = f"{self.work_path}/dist"
        self.golden_script = Path(config.golden_script).resolve()

    def tear_up(self):
        if os.path.exists(self.log_path):
            os.system(f"rm -rf {self.log_path}")
        os.mkdir(self.log_path)
        if os.path.exists(self.plog_cache_path):
            os.system(f"rm -rf {self.plog_cache_path}")
        os.mkdir(self.plog_cache_path)
        if os.path.exists(self.report_file):
            os.remove(self.report_file)

        os.environ["TILE_FWK_DEVICE_ID"] = f"{self.device}"
        os.environ["ASCEND_PROCESS_LOG_PATH"] = self.plog_cache_path

        if not self.pto:
            golden_dir = str(self.golden_script.parent)
            if golden_dir not in sys.path:
                sys.path.append(golden_dir)
            module_name = f"{self.golden_script.stem}"
            import importlib

            importlib.import_module(module_name)
            sys.path.remove(golden_dir)
        else:
            if self.pto_install_path not in sys.path:
                sys.path.insert(0, self.pto_install_path)
                sys.path.insert(0, self.pto_install_path + "/pto/lib")
            torch_install_path = os.path.dirname(torch.__file__)
            os.environ["LD_LIBRARY_PATH"] = (
                f"{torch_install_path}:{os.getenv('LD_LIBRARY_PATH')}"
            )
            os.environ["TILEFWK_CONFIG_PATH"] = (
                f"{self.pto_install_path}/pto/configs/tile_fwk_config.json"
            )
            os.environ["PLATFORM_CONFIG_PATH"] = (
                f"{self.pto_install_path}/pto/configs/tile_fwk_platform_info.json"
            )

    def tear_down(self):
        if self.pto_install_path in sys.path:
            sys.path.remove(self.pto_install_path)
            sys.path.remove(self.pto_install_path + "/pto/lib")
        del os.environ["TILE_FWK_DEVICE_ID"]
        del os.environ["ASCEND_PROCESS_LOG_PATH"]

    def compile_if_need(self):
        clean_str = "-c" if self.clean else ""
        cmd = f"{sys.executable} build.py {clean_str} -s="
        if self.pto:
            cmd += "python/tests/st/test_record_if_branch.py -f=python3"
        else:
            cmd += "'TestAdd/AddOperationTest.TestAdd/*' --disable_auto_execute"
        TestCaseShellActuator.run(cmd)

    def run_test_case(self, test_case_info):
        log_file = f"{self.log_path}/{test_case_info['case_name']}.log"
        test_case_info["log_file"] = log_file
        launcher_patch: Path = Path(
            Path(__file__).parent.parent.parent.parent, "st/operation/python"
        ).resolve()
        if str(launcher_patch) not in sys.path:
            sys.path.append(str(launcher_patch))
        from vector_operation_test_case_launcher import test_case_launcher

        test_case_launcher(test_case_info)

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
        is_pass = analyzer.run()
        if not is_pass or self.save_data:
            os.system(f"cp -rf {self.plog_cache_path} {log_path}")
        os.system(f"mv {log_file} {log_path}")
        os.system(f"rm -rf {self.plog_cache_path}")

    def run_pto_test_case(self, test_case_info):
        case_name = test_case_info["case_name"]
        logger = TestCaseLogger(f"{self.log_path}/{case_name}.log")
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = logger
        sys.stderr = logger
        pytest.main(
            [
                "-vs",
                "--test_case_info",
                json.dumps(test_case_info),
                "python/tests/st/operation/vector_operation_test_case_launcher.py",
            ]
        )
        sys.stdout = stdout
        sys.stderr = stderr

    def run(self):
        self.tear_up()
        self.compile_if_need()
        json_path = f"{self.work_path}/framework/tests/st/operation/test_case/"
        test_case_info_list = TestCaseLoader(
            self.input_file, self.op, self.index, json_path
        ).run()
        if self.json_only:
            return

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
            golden_path = f"{self.work_path}/build/tests/st/golden/{test_case}"
            if os.path.exists(golden_path + "/golden_desc.json"):
                os.remove(golden_path + "/golden_desc.json")
            if not self.save_data:
                os.system(f"rm -rf {golden_path}")
        self.tear_down()
