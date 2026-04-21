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
"""Consumer read must come from a legal source (rule 4-gamma).

For every kernel instance that reads a tensor region, there must either be
a producer that wrote that region (and is reachable in the dependency
graph), or the tensor must be a declared program input. Otherwise the
consumer's input data flow is missing.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

from ..models import Category, SlotAccessEvent, Violation
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


@register_rule
class CellReadSourceRule(Rule):
    RULE_ID = "rule_cell_read_source"
    DESCRIPTION = "Each consumer read must come from a legal producer or program input"
    CATEGORY = Category.MISSING_DEPENDENCY

    def check(self, ctx: RuleContext) -> List[Violation]:
        if not ctx.slot_accesses:
            return []
        writers_by_cell = ctx.writers_of_cell
        writer_seqs_by_slot_cell = ctx.writer_seqs_by_slot_cell

        no_writer_count: Dict[Tuple[int, int], int] = defaultdict(int)
        unreachable_count: Dict[Tuple[int, int], int] = defaultdict(int)

        for e in ctx.slot_accesses:
            if not e.is_reader:
                continue
            if not ctx.is_partial_slot(e.slot_idx):
                continue
            for cell in e.cell_idx_list:
                self._check_one(
                    ctx, e, cell, writers_by_cell, writer_seqs_by_slot_cell,
                    no_writer_count, unreachable_count)

        violations: List[Violation] = []
        for (slot, cell), n in sorted(no_writer_count.items()):
            violations.append(Violation(
                rule_id=self.RULE_ID,
                slot_idx=slot,
                cell_idx=cell,
                message=(
                    f"{n} consumer read(s) have no producer (the tensor is "
                    f"not a program input; producer dependency is missing)"
                ),
            ))
        for (slot, cell), n in sorted(unreachable_count.items()):
            violations.append(Violation(
                rule_id=self.RULE_ID,
                slot_idx=slot,
                cell_idx=cell,
                message=(
                    f"{n} consumer read(s) cannot reach any producer in the "
                    f"dependency graph; producer-to-consumer ordering is "
                    f"missing"
                ),
            ))
        return violations

    def _check_one(
            self, ctx: RuleContext, r: SlotAccessEvent, cell: int,
            writers_by_cell, writer_seqs_by_slot_cell,
            no_writer_count: Dict[Tuple[int, int], int],
            unreachable_count: Dict[Tuple[int, int], int]) -> None:
        slot = r.slot_idx
        seq_no = r.seq_no
        same_seq_writers = writers_by_cell.get((seq_no, slot, cell), [])
        prior_seqs = [s for s in writer_seqs_by_slot_cell.get((slot, cell), [])
                      if s < seq_no]
        has_prior_writer = bool(prior_seqs)

        if same_seq_writers and any(
                ctx.reaches(seq_no, w.task_id, r.task_id) for w in same_seq_writers):
            return

        if same_seq_writers:
            if r.all_concrete and not has_prior_writer:
                unreachable_count[(slot, cell)] += 1
            return

        if has_prior_writer:
            return

        if ctx.is_program_input_slot(slot):
            return
        if not r.all_concrete:
            return
        no_writer_count[(slot, cell)] += 1
