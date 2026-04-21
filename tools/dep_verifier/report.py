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
"""违规聚合与报告输出。"""
import csv
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Dict, Iterable, List

from .models import Severity, Violation

logger = logging.getLogger(__name__)


#: 严重等级到"是否应作为失败退出码"的映射。
FAILURE_SEVERITIES = {Severity.ERROR, Severity.SUSPECT_HIGH}


class ViolationReport:
    """规则产出的违规的聚合、统计、打印与持久化。"""

    def __init__(self):
        self._violations: List[Violation] = []
        self._per_rule: Dict[str, List[Violation]] = defaultdict(list)

    def extend(self, rule_id: str, violations: Iterable[Violation]):
        for v in violations:
            self._violations.append(v)
            self._per_rule[rule_id].append(v)

    @property
    def all(self) -> List[Violation]:
        return self._violations

    def has_failure(self) -> bool:
        return any(v.severity in FAILURE_SEVERITIES for v in self._violations)

    # -------- 控制台输出 --------
    def print_console(self, max_per_rule: int = 50):
        print("=" * 80)
        print("  运行时依赖边建立正确性校验报告")
        print("=" * 80)

        severity_count = Counter(v.severity for v in self._violations)
        total = len(self._violations)
        print(f"  违规总数: {total}")
        for s in Severity:
            if severity_count.get(s):
                print(f"    {s.value:<16} {severity_count[s]}")
        print("-" * 80)

        if total == 0:
            print("\n  [PASS] 所有启用的规则均未发现异常。\n")
            return

        for rule_id, vlist in sorted(self._per_rule.items()):
            if not vlist:
                continue
            print(f"\n## 规则 {rule_id} —— {len(vlist)} 条违规")
            per_sev = Counter(v.severity for v in vlist)
            for s in Severity:
                if per_sev.get(s):
                    print(f"     {s.value:<16} {per_sev[s]}")
            shown = 0
            for v in vlist:
                if shown >= max_per_rule:
                    print(f"     ... 已省略 {len(vlist) - shown} 条，详见 JSON/CSV 报告")
                    break
                print(f"     - {v.short()}")
                if v.details:
                    compact = ", ".join(f"{k}={_compact(val)}" for k, val in v.details.items())
                    print(f"         {compact}")
                shown += 1

        print()
        if self.has_failure():
            print("  [FAIL] 存在 ERROR 或 SUSPECT-HIGH 级别违规，请重点关注。\n")
        else:
            print("  [WARN] 仅存在低级别提示，依赖建立大概率正确。\n")

    # -------- 文件持久化 --------
    def save_json(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = [
            {
                "rule_id": v.rule_id,
                "severity": v.severity.value,
                "message": v.message,
                "details": v.details,
            }
            for v in self._violations
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("违规 JSON 报告已保存: %s", path)

    def save_csv(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rule_id", "severity", "message", "details"])
            for v in self._violations:
                w.writerow([
                    v.rule_id,
                    v.severity.value,
                    v.message,
                    json.dumps(v.details, ensure_ascii=False),
                ])
        logger.info("违规 CSV 报告已保存: %s", path)


def _compact(v) -> str:
    """把 dict/list 等结构打平为单行以便控制台浏览。"""
    if isinstance(v, (dict, list, tuple, set)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return str(v)
