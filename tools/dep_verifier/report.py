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
"""Violation aggregation and compact CSV/console reporting."""
import csv
import logging
import os
from typing import Dict, Iterable, List, Optional

from .models import Violation

logger = logging.getLogger(__name__)


class ViolationReport:
    """Aggregate violations and output one CSV report plus concise console summary."""

    CSV_HEADER = ["rule", "slot", "tensor", "func", "cell", "message"]

    def __init__(
        self,
        slot_tensor_names: Optional[Dict[int, str]] = None,
        slot_func_names: Optional[Dict[int, str]] = None,
    ):
        self._violations: List[Violation] = []
        self._slot_tensor_names: Dict[int, str] = slot_tensor_names or {}
        self._slot_func_names: Dict[int, str] = slot_func_names or {}

    def extend(self, violations: Iterable[Violation]):
        for v in violations:
            self._violations.append(v)

    @property
    def violations(self) -> List[Violation]:
        return self._violations

    def has_failure(self) -> bool:
        return bool(self._violations)

    def _tensor_of(self, slot_idx: Optional[int]) -> str:
        if slot_idx is None:
            return ""
        return self._slot_tensor_names.get(slot_idx, "") or ""

    def _func_of(self, slot_idx: Optional[int]) -> str:
        if slot_idx is None:
            return ""
        return self._slot_func_names.get(slot_idx, "") or ""

    def print_console(self) -> None:
        if not self._violations:
            print("PASS: no issues found.")
            return

        groups = {}
        for v in self._violations:
            key = (v.rule_id, v.slot_idx if v.slot_idx is not None else -1)
            groups.setdefault(key, []).append(v)

        print(f"FAIL: found {len(self._violations)} issues (grouped by rule and slot):")
        for (rule, slot), vs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
            slot_s = "-" if slot < 0 else str(slot)
            tensor = self._tensor_of(slot if slot >= 0 else None)
            func = self._func_of(slot if slot >= 0 else None)
            slot_label = f"{slot_s} ({tensor})" if tensor else slot_s
            if func:
                slot_label = f"{slot_label} @ {func}"
            cells = sorted({v.cell_idx for v in vs if v.cell_idx is not None})
            cells_s = ",".join(str(c) for c in cells) if cells else "-"
            sample = vs[0].message
            print(f"  {rule} | slot={slot_label} | cell={cells_s} | count={len(vs)} | {sample}")

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
                    self._tensor_of(v.slot_idx),
                    self._func_of(v.slot_idx),
                    "" if v.cell_idx is None else v.cell_idx,
                    v.message,
                ])
        logger.info("report written to: %s", path)
