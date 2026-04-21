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
规则 4-α：Cell 级写竞争。

判定（见《Cell 粒度数据流正确性校验方案设计》§5.2）：
  对每个 (seqNo, slotIdx, cellIdx):
    W = writers_of_cell[(seqNo, slotIdx, cellIdx)]
    若 |W| <= 1：PASS
    否则：若 W 在依赖图 G（同 seqNo）上不构成全序链 → ERROR（同 cell 写竞争）

engram-tmp 病态场景命中方式：
  128 个 writer 全写 cellIdx=0，在同一 DeviceTask 内彼此无 stitch 边，
  → 任取两个 writer 都互不可达 → 触发 ERROR。

误判面：
  - reuse stitch alias 写：放行（检查 dyn_stitch_edges.csv 中 stitchKind=reuse）
  - allConcrete=0 的保守估计：降级为 SUSPECT-HIGH，并在报告中标注
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from ..models import Severity, SlotAccessEvent, Violation
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


@register_rule
class CellWriteConflictRule(Rule):
    RULE_ID = "rule_cell_write_conflict"
    DESCRIPTION = "规则 4-α：Cell 级写竞争（同一 cell 多 writer 且非全序链）"

    #: 单 cell 的 writer 数量超过该阈值时，跳过全序链穷举并直接标记为 suspect，
    #: 避免 O(n^2) BFS 成本。后续如需精确判定可通过 --cell-writers-hard-limit 调整。
    HARD_LIMIT = 512

    def check(self, ctx: RuleContext) -> List[Violation]:
        if not ctx.slot_accesses:
            return []

        # 构建：(seqNo, slotIdx) -> reuse 标记（若同 seqNo 某 slot 被任一 stitch 边标为 reuse 则放行）
        reuse_slots: Set[Tuple[int, int]] = set()
        for e in ctx.stitch_edges:
            kind = (e.stitch_kind or "").lower()
            if "reuse" in kind and e.inferred_seq_no is not None:
                reuse_slots.add((e.inferred_seq_no, e.slot_idx))

        violations: List[Violation] = []
        for (seq_no, slot, cell), writers in ctx.writers_of_cell.items():
            if len(writers) <= 1:
                continue
            if (seq_no, slot) in reuse_slots:
                continue  # reuse stitch：框架显式允许覆盖

            # 去重（同 taskId 可能出现多次，理论上一次 op 只 dump 一行，保险起见）
            uniq: Dict[int, SlotAccessEvent] = {}
            for w in writers:
                uniq.setdefault(w.task_id, w)
            writer_list = list(uniq.values())
            if len(writer_list) <= 1:
                continue

            all_concrete = all(w.all_concrete for w in writer_list)

            # 大规模 writer 集合：不做 O(n^2) 判定，直接按 suspect 处理
            if len(writer_list) > self.HARD_LIMIT:
                violations.append(self._make_violation(
                    seq_no, slot, cell, writer_list,
                    severity=Severity.SUSPECT_HIGH,
                    reason=(f"writer 数量 {len(writer_list)} 超过硬上限 {self.HARD_LIMIT}，"
                            "跳过全序链精确判定，按可疑标记"),
                    all_concrete=all_concrete,
                ))
                continue

            if self._is_total_order_chain(ctx, seq_no, writer_list):
                continue

            if all_concrete:
                severity = Severity.ERROR
                reason = "同 cell 多 writer 且无全序链，存在覆盖竞争"
            else:
                severity = Severity.SUSPECT_HIGH
                reason = "同 cell 多 writer 且无全序链（allConcrete=0，可能为保守估计）"

            violations.append(self._make_violation(
                seq_no, slot, cell, writer_list, severity, reason, all_concrete))
        return violations

    # ------------------------------------------------------------------
    # 工具：全序链判定
    # ------------------------------------------------------------------
    @staticmethod
    def _is_total_order_chain(
            ctx: RuleContext, seq_no: int, writers: List[SlotAccessEvent]) -> bool:
        """
        判定 writer 集合在同 seqNo DAG 上是否构成全序链。

        策略：
          1) 先按某种固定序（funcIdx, opIdx）排序 writers；
          2) 计算每个 writer 的后代集合；
          3) 若排序后相邻对都"前→后可达"，则判定为链；
             否则只要存在一对彼此不可达就 fail。
        """
        if len(writers) <= 1:
            return True

        ordered = sorted(writers, key=lambda w: (w.func_idx, w.op_idx, w.task_id))

        # 判定（先尝试启发式：按 funcIdx 升序相邻可达）
        for a, b in zip(ordered, ordered[1:]):
            if not ctx.reaches(seq_no, a.task_id, b.task_id):
                # 启发失败：走一般判定——检查是否存在任意两个 writer 互不可达
                return CellWriteConflictRule._all_pairs_ordered(ctx, seq_no, ordered)
        return True

    @staticmethod
    def _all_pairs_ordered(
            ctx: RuleContext, seq_no: int, writers: List[SlotAccessEvent]) -> bool:
        """O(n^2)：判定任意两个 writer 存在可达关系（方向任一）。"""
        n = len(writers)
        # 为减少 descendants 重复查询，预先求每个 writer 的后代集合
        desc_list: List[Set[int]] = [
            ctx.descendants(seq_no, w.task_id) for w in writers
        ]
        for i in range(n):
            wi_id = writers[i].task_id
            for j in range(i + 1, n):
                wj_id = writers[j].task_id
                if wj_id in desc_list[i]:
                    continue
                if wi_id in desc_list[j]:
                    continue
                return False
        return True

    # ------------------------------------------------------------------
    # 工具：构造 Violation
    # ------------------------------------------------------------------
    def _make_violation(
            self, seq_no: int, slot: int, cell: int,
            writers: List[SlotAccessEvent], severity: Severity, reason: str,
            all_concrete: bool) -> Violation:
        summarized_tids = [w.task_id for w in writers[:10]]
        return Violation(
            rule_id=self.RULE_ID,
            severity=severity,
            message=f"Cell 写竞争：(seqNo={seq_no}, slot={slot}, cell={cell})",
            details={
                "seqNo": seq_no,
                "slotIdx": slot,
                "cellIdx": cell,
                "writerCount": len(writers),
                "sampleWriterTaskIds": summarized_tids,
                "allConcrete": all_concrete,
                "reason": reason,
            },
        )
