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
规则 4-β：Cell 写有去向。

判定（见《Cell 粒度数据流正确性校验方案设计》§5.3）：
  对每个 writer w 写入的每个 cellIdx c：
    consumers_same = readers_of_cell[(w.seqNo, slot, c)]
    consumers_future = readers_of_cell[(>w.seqNo, slot, c)]

    (1) 同 seqNo 内存在 reader，且 w→reader 可达 → PASS
    (2) 同 seqNo 内存在 reader，但无任何可达路径 → ERROR（漏 stitch 边）
    (3) 无同 seqNo reader，但跨 seqNo 存在 reader → PASS
        （靠 slot 物理内存 + sequential 调度传递）
    (4) 无 reader：
        - 是该 cell 在 (seqNo, slot) 中的"最后一次写"（funcIdx 最大）且 slot
          标记为 output / 无后续 seq 的覆盖 → INFO（允许：末尾写、program output）
        - 否则 → SUSPECT-HIGH（写了又被后续覆盖且中间无人读，engram-tmp 的
          127 个被覆盖实例命中这里）
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from ..models import Severity, SlotAccessEvent, Violation
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


@register_rule
class CellWriteReachRule(Rule):
    RULE_ID = "rule_cell_write_reach"
    DESCRIPTION = "规则 4-β：Cell 写有去向（writer 的 cell 有可达 reader 或合法末尾写）"

    def check(self, ctx: RuleContext) -> List[Violation]:
        if not ctx.slot_accesses:
            return []

        writers_by_cell = ctx.writers_of_cell
        readers_by_cell = ctx.readers_of_cell
        reader_seqs_by_slot_cell = ctx.reader_seqs_by_slot_cell

        # 同 (seqNo, slot, cell) 最大 funcIdx 的 writer，用于判定"末尾写"
        last_writer_funcidx: Dict[Tuple[int, int, int], int] = {}
        for key, writers in writers_by_cell.items():
            last_writer_funcidx[key] = max(w.func_idx for w in writers)

        violations: List[Violation] = []
        for (seq_no, slot, cell), writers in writers_by_cell.items():
            same_seq_readers = readers_by_cell.get((seq_no, slot, cell), [])
            future_seqs = [s for s in reader_seqs_by_slot_cell.get((slot, cell), [])
                           if s > seq_no]
            has_future_reader = bool(future_seqs)

            for w in writers:
                # (1) 同 seqNo reader 中有可达 → PASS
                if any(ctx.reaches(seq_no, w.task_id, r.task_id)
                       for r in same_seq_readers):
                    continue

                # (2) 同 seqNo reader 存在但无可达路径 → ERROR 漏边
                if same_seq_readers:
                    violations.append(self._violation(
                        w, slot, cell,
                        severity=Severity.ERROR if w.all_concrete else Severity.SUSPECT_HIGH,
                        reason="同 seqNo 内有 reader 但 writer 无可达路径：疑似漏 stitch 边",
                        extras={
                            "sameSeqReaderTaskIds": [r.task_id for r in same_seq_readers[:10]],
                        },
                    ))
                    continue

                # (3) 跨 seqNo reader 存在 → PASS（物理内存 + seq 顺序调度）
                if has_future_reader:
                    continue

                # (4) 无 reader
                is_last_writer = (last_writer_funcidx[(seq_no, slot, cell)] == w.func_idx)
                is_output_slot = slot in ctx.program_output_slots

                if is_last_writer and is_output_slot:
                    violations.append(self._violation(
                        w, slot, cell,
                        severity=Severity.INFO,
                        reason="末尾写 + program output slot（允许）",
                    ))
                elif is_last_writer:
                    # 未声明输出 slot，但是末尾写：降级为 SUSPECT-LOW 提示
                    violations.append(self._violation(
                        w, slot, cell,
                        severity=Severity.SUSPECT_LOW,
                        reason="末尾写但未声明为 program output slot（需人工确认）",
                    ))
                else:
                    # 核心 engram-tmp 信号：写了后被同 cell 下一个 writer 覆盖，中间无人读
                    sev = Severity.SUSPECT_HIGH if w.all_concrete else Severity.SUSPECT_MEDIUM
                    violations.append(self._violation(
                        w, slot, cell,
                        severity=sev,
                        reason="写入被同 cell 后续 writer 覆盖，期间无 reader（engram-tmp 型覆盖）",
                    ))
        return violations

    def _violation(self, w: SlotAccessEvent, slot: int, cell: int,
                   severity: Severity, reason: str, extras: dict = None) -> Violation:
        details = {
            "seqNo": w.seq_no,
            "slotIdx": slot,
            "cellIdx": cell,
            "writerTaskId": w.task_id,
            "writerFuncIdx": w.func_idx,
            "writerOpIdx": w.op_idx,
            "allConcrete": w.all_concrete,
            "reason": reason,
        }
        if extras:
            details.update(extras)
        return Violation(
            rule_id=self.RULE_ID,
            severity=severity,
            message=f"Cell 写无去向：(seqNo={w.seq_no}, slot={slot}, cell={cell})",
            details=details,
        )
