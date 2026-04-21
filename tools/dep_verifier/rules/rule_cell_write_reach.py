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
"""规则 4-β：Cell 写有去向。"""
from typing import Dict, List, Tuple

from ..models import Violation
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


@register_rule
class CellWriteReachRule(Rule):
    RULE_ID = "rule_cell_write_reach"
    DESCRIPTION = "规则 4-β：writer 的数据必须能被消费或合法终结"

    def check(self, ctx: RuleContext) -> List[Violation]:
        if not ctx.slot_accesses:
            return []

        writers_by_cell = ctx.writers_of_cell
        readers_by_cell = ctx.readers_of_cell
        reader_seqs_by_slot_cell = ctx.reader_seqs_by_slot_cell

        last_writer_funcidx: Dict[Tuple[int, int, int], int] = {
            key: max(w.func_idx for w in ws) for key, ws in writers_by_cell.items()
        }

        violations: List[Violation] = []
        for (seq_no, slot, cell), writers in writers_by_cell.items():
            if not ctx.is_partial_slot(slot):
                continue
            same_seq_readers = readers_by_cell.get((seq_no, slot, cell), [])
            has_future_reader = any(
                s > seq_no for s in reader_seqs_by_slot_cell.get((slot, cell), []))

            for w in writers:
                if any(ctx.reaches(seq_no, w.task_id, r.task_id)
                       for r in same_seq_readers):
                    continue

                if same_seq_readers:
                    if w.all_concrete:
                        violations.append(Violation(
                            rule_id=self.RULE_ID,
                            slot_idx=slot,
                            cell_idx=cell,
                            message=f"同 seqNo 存在 reader 但 writer 不可达 "
                                    f"(writerTaskId={w.task_id})",
                        ))
                    continue

                if has_future_reader:
                    continue

                is_last_writer = (
                    last_writer_funcidx[(seq_no, slot, cell)] == w.func_idx)
                is_output_slot = ctx.is_program_output_slot(slot)

                if is_last_writer and is_output_slot and w.all_concrete:
                    continue
                if is_last_writer:
                    continue
                if not w.all_concrete:
                    continue

                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    slot_idx=slot,
                    cell_idx=cell,
                    message=f"写入被同 cell 后续 writer 覆盖且中间无 reader "
                            f"(writerTaskId={w.task_id})",
                ))
        return violations
