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
"""违规聚合与报告输出（仅 CSV + 极简控制台）。"""
import csv
import logging
import os
from typing import Iterable, List

from .models import Violation

logger = logging.getLogger(__name__)


class ViolationReport:
    """聚合 Violation，输出单个 CSV 报告与一行控制台结论。"""

    CSV_HEADER = ["rule", "slot", "cell", "message"]

    def __init__(self):
        self._violations: List[Violation] = []

    def extend(self, violations: Iterable[Violation]):
        for v in violations:
            self._violations.append(v)

    @property
    def violations(self) -> List[Violation]:
        return self._violations

    def has_failure(self) -> bool:
        return bool(self._violations)

    def print_console(self) -> None:
        if not self._violations:
            print("PASS: 未发现问题。")
            return

        # 按 (rule, slot) 聚合，每组只打印一行摘要。
        groups = {}
        for v in self._violations:
            key = (v.rule_id, v.slot_idx if v.slot_idx is not None else -1)
            groups.setdefault(key, []).append(v)

        print(f"FAIL: 发现 {len(self._violations)} 条问题，按 (rule, slot) 汇总如下：")
        for (rule, slot), vs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
            slot_s = "-" if slot < 0 else str(slot)
            cells = sorted({v.cell_idx for v in vs if v.cell_idx is not None})
            cells_s = ",".join(str(c) for c in cells) if cells else "-"
            sample = vs[0].message
            print(f"  {rule} | slot={slot_s} | cell={cells_s} | count={len(vs)} | {sample}")

    def save_csv(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.CSV_HEADER)
            for v in self._violations:
                w.writerow([
                    v.rule_id,
                    "" if v.slot_idx is None else v.slot_idx,
                    "" if v.cell_idx is None else v.cell_idx,
                    v.message,
                ])
        logger.info("报告已写入: %s", path)
