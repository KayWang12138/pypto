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
"""获取 Tests 用例对应用例.
"""
import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import List, Any, Optional, Dict

import yaml


class CaseCtrl:
    def __init__(self, args):
        self.file: Path = Path(args.file[0]).resolve()
        self.type: str = str(args.type[0]).lower()
        self.type = "stest" if self.type in ["stest", "st"] else self.type
        self.type = "utest" if self.type in ["utest", "ut"] else self.type
        self.group: Optional[str] = args.group[0] if args.group and args.group[0] else None
        # yaml 字典
        with open(self.file, 'r', encoding='utf-8') as f:
            self.type_dict: Optional[Dict[str, Any]] = yaml.safe_load(f)

    @property
    def brief(self) -> List[Any]:
        ver = sys.version_info
        lst = [
            ["Python3", f"{sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"],
            ["Classify", self.file],
            ["Type", self.type],
            ["Group", self.group],
        ]
        return lst

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description=f"Testcase Parser", epilog="Best Regards!")
        parser.add_argument("-f", "--file", required=True, nargs=1, type=Path,
                            help="Specific classify_rule.yaml path.")
        parser.add_argument("-t", "--type", nargs=1, type=str, required=True,
                            help="Specific test type.")
        parser.add_argument("-g", "--group", nargs=1, type=str, required=False,
                            help="Specific test group name.")
        args = parser.parse_args()
        ctrl: CaseCtrl = CaseCtrl(args=args)
        ctrl.process()

    def process(self):
        rst_cases: List[str] = self._get_cases_list()
        rst_str: str = ":".join(rst_cases)
        print(rst_str, end='')

    def _get_cases_list(self) -> List[str]:
        rst_cases: List[str] = []
        group_dict = self.type_dict.get(self.type, {})

        if self.group and self.group != "OFF":
            sub_class = group_dict.get(self.group)
            if sub_class and isinstance(sub_class, dict):
                cases_list = sub_class.get("cases", [])
                if isinstance(cases_list, list):
                    rst_cases = copy.deepcopy(cases_list)
        else:
            for _, sub_class_data in group_dict.items():
                if not isinstance(sub_class_data, dict):
                    continue
                cases_list = sub_class_data.get("cases", [])
                if isinstance(cases_list, list):
                    rst_cases.extend(copy.deepcopy(cases_list))
        return rst_cases


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - PID[%(process)d] - %(levelname)s: %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ]
    )
    exit(0 if CaseCtrl.main() else 1)