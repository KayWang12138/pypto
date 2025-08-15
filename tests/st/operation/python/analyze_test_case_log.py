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

import logging
import os
import re
import sys
import pandas as pd


class TestCaseResult:
    def __init__(
        self, index: int, name: str, is_pass: bool, duration: str, op, detail: str = ""
    ):
        self._index = index
        self._name = name
        self._is_pass = is_pass
        self._duration = duration
        self._op = op
        self._detail = detail

    @property
    def index(self) -> str:
        return self._index

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_pass(self) -> str:
        return self._is_pass

    @property
    def status(self) -> str:
        return "PASS" if self._is_pass else "FAILED"

    @property
    def duration(self) -> str:
        return self._duration

    @property
    def op(self) -> str:
        return self._op

    @property
    def detail(self) -> str:
        return self._detail

    def dump_to_excel(self):
        return {
            "index": [self.index],
            "case_name": [self.name],
            "status": [self.status],
            "duration": [self.duration],
            "operation": [self.op],
            "detail": [self.detail],
        }


class TestCaseLogAnalyzer:
    def __init__(self, log_file: str, report_file: str):
        self._log_file = log_file
        self._report_file = report_file

    def get_value_by_pattern(self, pattern, line: str):
        match = re.search(pattern, line)
        if match:
            return match["value"].strip()
        else:
            logging.error(f"{pattern} is no match in {line}.")
            return ""

    def get_duration(self, line: str):
        pattern = rf"\((?P<value>\d+?) ms total\)"
        return self.get_value_by_pattern(pattern, line)

    def get_case_name(self, line: str):
        pattern = rf'"case_name":"(?P<value>.*?)"'
        return self.get_value_by_pattern(pattern, line)

    def get_case_op(self, line: str):
        pattern = rf'"operation":"(?P<value>.*?)"'
        return self.get_value_by_pattern(pattern, line)

    def get_case_index(self, line: str):
        pattern = rf'"case_index":(?P<value>\d+)'
        return self.get_value_by_pattern(pattern, line)

    def parse_log_file(self):
        case_index = 0
        case_name = ""
        is_pass = False
        duration = ""
        case_op = None
        detail = ""
        with open(self._log_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if "case_index" in line:
                    case_index = self.get_case_index(line)
                if "case_name" in line:
                    case_name = self.get_case_name(line)
                if 'operation"' in line:
                    case_op = self.get_case_op(line)
                elif "[  PASSED  ] 1 test" in line:
                    is_pass = True
                elif "[  FAILED  ]" in line:
                    is_pass = False
                elif "ms total" in line:
                    duration = self.get_duration(line)

        return TestCaseResult(case_index, case_name, is_pass, duration, case_op, detail)

    def generate_excel_report(self, result: TestCaseResult) -> bool:
        report_data = result.dump_to_excel()
        data_frame = pd.DataFrame(report_data)
        if not os.path.exists(self._report_file):
            df = pd.DataFrame(list(report_data.keys()))
            df.to_excel(self._report_file, sheet_name=result.op)
        else:
            with pd.ExcelFile(self._report_file) as excel_file:
                if result.op in excel_file.sheet_names:
                    data_frame = pd.concat(
                        [pd.read_excel(excel_file, sheet_name=result.op), data_frame],
                        ignore_index=True,
                    )

        with pd.ExcelWriter(
            self._report_file, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            data_frame.to_excel(
                writer,
                sheet_name=result.op,
                index=False,
            )
        return result.is_pass

    def analyze(self) -> bool:
        test_case_result = self.parse_log_file()
        return self.generate_excel_report(test_case_result)


if __name__ == "__main__":
    analyzer = TestCaseLogAnalyzer(sys.argv[1], sys.argv[2])
    exit(0 if analyzer.analyze() else 1)
