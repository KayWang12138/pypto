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
"""规则 4-γ：Cell 读有来源。"""
from typing import List

from ..models import SlotAccessEvent, Violation
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


@register_rule
class CellReadSourceRule(Rule):
    RULE_ID = "rule_cell_read_source"
    DESCRIPTION = "规则 4-γ：reader 必须能追溯到合法 writer 或外部输入"

    def check(self, ctx: RuleContext) -> List[Violation]:
        if not ctx.slot_accesses:
            return []
        writers_by_cell = ctx.writers_of_cell
        writer_seqs_by_slot_cell = ctx.writer_seqs_by_slot_cell

        violations: List[Violation] = []
        for e in ctx.slot_accesses:
            if not e.is_reader:
                continue
            if not ctx.is_partial_slot(e.slot_idx):
                continue
            for cell in e.cell_idx_list:
                self._check_one(ctx, e, cell, writers_by_cell,
                                writer_seqs_by_slot_cell, violations)
        return violations

    def _check_one(self, ctx: RuleContext, r: SlotAccessEvent, cell: int,
                   writers_by_cell, writer_seqs_by_slot_cell,
                   out: List[Violation]) -> None:
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
                out.append(Violation(
                    rule_id=self.RULE_ID,
                    slot_idx=slot,
                    cell_idx=cell,
                    message=f"同 seqNo 存在 writer 但 reader 不可达 "
                            f"(readerTaskId={r.task_id})",
                ))
            return

        if has_prior_writer:
            return

        if ctx.is_program_input_slot(slot):
            return
        if not r.all_concrete:
            return
        out.append(Violation(
            rule_id=self.RULE_ID,
            slot_idx=slot,
            cell_idx=cell,
            message=f"读取无任何 writer 且 slot 非 INPUT/INOUT "
                    f"(readerTaskId={r.task_id})",
        ))