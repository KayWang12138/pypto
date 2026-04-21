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
"""
规则 4-γ：Cell 读有来源。

判定（见《Cell 粒度数据流正确性校验方案设计》§5.4）：
  对每个 reader r 读入的每个 cellIdx c：
    candidate_writers = writers_of_cell[(r.seqNo, slot, c)]
                      ∪ writers_of_cell[(s < r.seqNo, slot, c)]

    若 candidate_writers 为空：
      - slot ∈ program_input_slots ∪ zero_init_slots → PASS
      - 否则 → ERROR（野读）

    若 candidate_writers 非空：
      case A. 同 seqNo 存在可达 writer → PASS
      case B. 同 seqNo 存在 writer 但均不可达 → ERROR（漏 stitch 边）
      case C. 仅跨 seqNo writer：
              验证"最后一次写 seqNo"是否为 max(writer seqNo)
                是 → PASS
                否 → SUSPECT-HIGH（跨 task 读到非最终值）
                （这里简化：实际 runtime 上跨 seq 的 "最后一次写" 就是
                 max writer seqNo < reader seqNo；故默认 PASS，若存在跨 seq
                 writer 但被后续同 cell 覆盖过则提示 SUSPECT-MEDIUM）

本规则承担"漏依赖"检测的主责任。
"""
from typing import List, Set

from ..models import Severity, SlotAccessEvent, Violation
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


@register_rule
class CellReadSourceRule(Rule):
    RULE_ID = "rule_cell_read_source"
    DESCRIPTION = "规则 4-γ：Cell 读有来源（reader 必须能追溯到合法 writer，否则视为野读/漏边）"

    def check(self, ctx: RuleContext) -> List[Violation]:
        if not ctx.slot_accesses:
            return []

        writers_by_cell = ctx.writers_of_cell
        writer_seqs_by_slot_cell = ctx.writer_seqs_by_slot_cell

        violations: List[Violation] = []

        # 遍历所有 reader 事件
        for e in ctx.slot_accesses:
            if not e.is_reader:
                continue
            for cell in e.cell_idx_list:
                self._check_one_read(
                    ctx, e, cell,
                    writers_by_cell, writer_seqs_by_slot_cell,
                    violations)
        return violations

    # ------------------------------------------------------------------
    def _check_one_read(self, ctx: RuleContext, r: SlotAccessEvent, cell: int,
                        writers_by_cell, writer_seqs_by_slot_cell,
                        out: List[Violation]) -> None:
        slot = r.slot_idx
        seq_no = r.seq_no

        same_seq_writers = writers_by_cell.get((seq_no, slot, cell), [])
        prior_writer_seqs = [s for s in writer_seqs_by_slot_cell.get((slot, cell), [])
                             if s < seq_no]
        has_prior_writer = bool(prior_writer_seqs)

        # 1) 无任何 writer
        if not same_seq_writers and not has_prior_writer:
            if slot in ctx.program_input_slots or slot in ctx.zero_init_slots:
                return  # PASS：program input / zero-init
            out.append(self._violation(
                r, cell,
                severity=Severity.ERROR if r.all_concrete else Severity.SUSPECT_HIGH,
                reason="reader 无任何 writer 且 slot 未声明为 program input / zero-init：野读",
            ))
            return

        # 2) 同 seqNo 有 writer：优先校验是否可达
        if same_seq_writers:
            if any(ctx.reaches(seq_no, w.task_id, r.task_id) for w in same_seq_writers):
                return  # PASS

            # 同 seqNo 有 writer 但没有可达路径
            # 若还有跨 seq 的 writer，可降级为 SUSPECT-MEDIUM；否则 ERROR 漏边
            if has_prior_writer:
                out.append(self._violation(
                    r, cell,
                    severity=Severity.SUSPECT_MEDIUM,
                    reason=("同 seqNo 有 writer 但 reader 不可达；"
                            "同时存在跨 seq writer 可能作为来源（疑似漏 stitch 边）"),
                    extras={
                        "sameSeqWriterTaskIds": [w.task_id for w in same_seq_writers[:10]],
                        "priorWriterSeqs": prior_writer_seqs,
                    },
                ))
            else:
                out.append(self._violation(
                    r, cell,
                    severity=Severity.ERROR if r.all_concrete else Severity.SUSPECT_HIGH,
                    reason="同 seqNo 有 writer 但 reader 不可达，且无跨 seq writer：漏 stitch 边",
                    extras={
                        "sameSeqWriterTaskIds": [w.task_id for w in same_seq_writers[:10]],
                    },
                ))
            return

        # 3) 仅跨 seq writer（默认 PASS；若 reader 之前有 writer 被覆盖过，仍是合法行为）
        #    这里不再产生 violation —— 跨 task 数据传递交由物理内存 + seq 顺序保证。

    # ------------------------------------------------------------------
    def _violation(self, r: SlotAccessEvent, cell: int,
                   severity: Severity, reason: str, extras: dict = None) -> Violation:
        details = {
            "seqNo": r.seq_no,
            "slotIdx": r.slot_idx,
            "cellIdx": cell,
            "readerTaskId": r.task_id,
            "readerFuncIdx": r.func_idx,
            "readerOpIdx": r.op_idx,
            "allConcrete": r.all_concrete,
            "reason": reason,
        }
        if extras:
            details.update(extras)
        return Violation(
            rule_id=self.RULE_ID,
            severity=severity,
            message=f"Cell 读无来源：(seqNo={r.seq_no}, slot={r.slot_idx}, cell={cell})",
            details=details,
        )
