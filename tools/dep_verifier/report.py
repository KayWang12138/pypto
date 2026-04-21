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
"""Violation aggregation and operator-oriented CSV/console reporting."""
import csv
import logging
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from .models import CATEGORY_ORDER, CATEGORY_TITLES, Violation

logger = logging.getLogger(__name__)


class ViolationReport:
    """Aggregate violations and output one CSV report plus concise console summary.

    Output is grouped by problem category (ConcurrentWriteOverlap /
    MissingDependency / IllegalReadWriteLinkage) so that an operator developer
    can quickly tell what kind of dependency problem they are looking at,
    without needing to know about stitch / DeviceTask internals.
    """

    CSV_HEADER = ["category", "rule", "slot", "tensor", "func", "cell", "message"]

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

    def _format_subject(self, v: Violation) -> str:
        if v.slot_idx is None:
            return "(no tensor)"
        tensor = self._tensor_of(v.slot_idx)
        slot_label = f"slot={v.slot_idx}"
        if v.cell_idx is not None:
            slot_label = f"{slot_label}, cell={v.cell_idx}"
        if tensor:
            return f"tensor '{tensor}' ({slot_label})"
        return f"unnamed tensor ({slot_label})"

    def print_console(self) -> None:
        if not self._violations:
            print("PASS")
            return

        grouped: Dict[str, List[Violation]] = defaultdict(list)
        for v in self._violations:
            grouped[v.category or "Other"].append(v)

        print(f"FAIL: {len(self._violations)} issue(s) detected.")
        ordered_categories = [c for c in CATEGORY_ORDER if c in grouped]
        ordered_categories += [c for c in grouped if c not in CATEGORY_ORDER]

        for cat in ordered_categories:
            title = CATEGORY_TITLES.get(cat, cat)
            print(f"\n[{title}]")
            for v in grouped[cat]:
                print(f"  - {self._format_subject(v)}: {v.message}")

    def save_csv(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.CSV_HEADER)
            for v in self._violations:
                w.writerow([
                    v.category,
                    v.rule_id,
                    "" if v.slot_idx is None else v.slot_idx,
                    self._tensor_of(v.slot_idx),
                    self._func_of(v.slot_idx),
                    "" if v.cell_idx is None else v.cell_idx,
                    v.message,
                ])
        logger.info("report written to: %s", path)
