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
"""分析修改文件清单.

分析修改文件清单, 判断当前测试场景是否需要执行及获取用例执行范围.
"""
import argparse
import dataclasses
import fnmatch
import logging
import sys
from pathlib import Path
from typing import List, Any, Optional, Dict, Tuple

import yaml


@dataclasses.dataclass
class Module:
    name: str
    cases: List[str]
    write: List[Path]

    @staticmethod
    def _relative_to(s: Path, d: Path) -> bool:
        try:
            if s.relative_to(d):
                return True
        except ValueError:
            pass
        return False

    def is_trigger(self, changed: List[Path]) -> Tuple[bool, List[str]]:
        # 若无 changed, 默认触发所有用例
        if not changed:
            return True, self.cases
        # 当所有 changed 均命中白名单, 无需触发
        for c in changed:
            c_skip = False
            for w in self.write:
                if self._relative_to(c, w):
                    c_skip = True
                    logging.debug("Changed(%s) hit writeList(%s), skip Module(%s)", c, w, self.name)
                    break
            if not c_skip:
                logging.debug("Changed(%s) not hit writeList, trigger Module(%s)", c, self.name)
                return True, self.cases
        return False, []


class Analysis:
    _KEY_WRITE_LIST: str = "write_list"
    _KEY_CASES: str = "cases"
    _ALL_FRONTEND_TYPE: List[str] = ["python", "cpp"]
    _ALL_TESTS_TYPE: List[str] = ["utest", "stest", "stest_distributed", "example", "models"]

    def __init__(self, rule: Path):
        self.modules: Dict[str, Module] = self._init_get_models(rule=rule)

    def __str__(self) -> str:
        ver = sys.version_info
        desc = f"\nPython3  : {sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"
        for frontend, frontend_dict in self.modules.items():
            desc += f"\n{frontend}"
            for tests, tests_dict in frontend_dict.items():
                desc += f"\n\t{tests}"
                for module in tests_dict.values():
                    desc += f"\n\t\tModule({module.name}) : {len(module.write)}"
        desc += f"\n"
        return desc

    @classmethod
    def main(cls) -> str:
        parser = argparse.ArgumentParser(description=f"Analysis Changed Files", epilog="Best Regards!")
        parser.add_argument("-r", "--rule", required=True, nargs=1, type=Path,
                            help="Specific classify_rule.yaml")
        parser.add_argument("-f", "--frontend", nargs=1, type=str, required=True, choices=["cpp", "python"],
                            help="Specific tests fronted")
        parser.add_argument("-t", "--type", nargs=1, type=str, required=True, choices=["utest", "stest"],
                            default="utest",
                            help="Specific tests type")
        parser.add_argument("-g", "--group", nargs='?', type=str, required=False, default="",
                            help="Specific tests group, multiple group are separated by ','")
        parser.add_argument("-c", "--changed_files", nargs=1, type=Path, required=False, dest="file",
                            help="Specific changed_files.txt")
        args = parser.parse_args()
        # 参数解析
        ctrl = Analysis(rule=Path(args.rule[0]).resolve())
        logging.info(ctrl)
        frontend = str(args.frontend[0]).lower()
        tests = str(args.type[0]).lower()
        group = args.group.split(",") if args.group else None
        changed = Path(args.file[0]).resolve() if args.file and args.file[0] else None
        # 流程处理
        logging.info(ctrl.analysis(frontend=frontend, tests=tests, group=group, changed_txt=changed))

    def analysis(self, frontend: str, tests: str,
                 group: Optional[List[str]] = None, changed_txt: Optional[Path] = None) -> List[str]:
        models_dict = self.modules.get(frontend, {}).get(tests, {})
        group = group if group else []
        changed: List[Path] = self._analysis_get_changed(file=changed_txt)
        cases = []
        for module in models_dict.values():
            match_group = False if group else True
            for cur_grp in group:
                # 支持 group 名称模糊匹配
                if fnmatch.fnmatch(module.name, cur_grp):
                    match_group = True
                    break
            if not match_group:
                logging.debug("Module(%s) not match group %s", module.name, group)
                continue
            logging.debug("Module(%s) match group %s", module.name, group)
            trigger, module_cases = module.is_trigger(changed=changed)
            if not trigger:
                continue
            if module_cases is not None:
                cases.extend(module_cases)
        return cases

    def _init_get_write_list(self, desc: Dict[str, Any]) -> List[Path]:
        lst = desc.get(self._KEY_WRITE_LIST, [])
        lst = lst if lst else []
        rst = [Path(_rel) for _rel in lst]
        desc.pop(self._KEY_WRITE_LIST, None)
        return rst

    def _init_get_models(self, rule: Path) -> Dict[str, Module]:
        modules = {}
        with open(rule, 'r', encoding='utf-8') as f:
            rule_dict = yaml.safe_load(f)
        rule_dict = rule_dict.get("pypto", {})
        for frontend, frontend_dict in rule_dict.items():
            if frontend not in self._ALL_FRONTEND_TYPE:
                continue
            frontend_model_dict = {}
            for test_type, test_dict in frontend_dict.items():
                if test_type not in self._ALL_TESTS_TYPE:
                    continue
                test_model_dict = {}
                test_type_write_list = self._init_get_write_list(desc=test_dict)
                for group_name, desc in test_dict.items():
                    # 处理 group 下白名单
                    write_list = self._init_get_write_list(desc=desc)
                    write_list.extend(test_type_write_list)
                    write_list = list[Path](set[Path](write_list))
                    # 获取 group 下用例列表
                    cases_list = desc.get(self._KEY_CASES, [])
                    cases_list = list[str](set[str](cases_list))
                    mod = Module(name=group_name, cases=cases_list, write=write_list)
                    test_model_dict[group_name] = mod
                frontend_model_dict[test_type] = test_model_dict
            modules[frontend] = frontend_model_dict
        return modules

    def _analysis_get_changed(self, file: Optional[Path]) -> List[Path]:
        changed = []
        if file:
            with open(file, 'r', encoding='utf-8') as f:
                changed = [Path(l.rstrip('\n')) for l in f]
        return changed


if __name__ == "__main__":
    # print(Analysis.main(), end='')
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    Analysis.main()
