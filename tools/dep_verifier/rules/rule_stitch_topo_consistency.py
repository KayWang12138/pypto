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
规则 3：Stitch 边与 Topo 双向一致性。

方向 A（edges → topo）：dyn_stitch_edges.csv 每条记录，producerTaskId 在其归属 seqNo
  的 dyn_topo 行中，consumerTaskId 必须出现在 successors[staticSuccCount:] 中。

方向 B（topo → edges）：dyn_topo.txt 每行 stitch_successors（非 reuse 类型）必须在
  dyn_stitch_edges.csv 中能找到对应 (producerTaskId, consumerTaskId) 记录。
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from ..models import Severity, Violation
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


REUSE_KINDS = {"reuse", "Reuse", "REUSE"}


@register_rule
class StitchTopoConsistencyRule(Rule):
    RULE_ID = "rule_stitch_topo_consistency"
    DESCRIPTION = "规则 3：dyn_stitch_edges 与 dyn_topo 双向一致"

    def check(self, ctx: RuleContext) -> List[Violation]:
        violations: List[Violation] = []

        # edges 索引：以 (seqNo, producerTaskId, consumerTaskId) 作为 key
        # seqNo 可能为 None（未解析出唯一归属），这种 edge 不参与双向一致性检查
        edge_set: Set[Tuple[int, int, int]] = set()
        edges_by_pair: Dict[Tuple[int, int, int], List] = defaultdict(list)
        unresolved = 0
        for e in ctx.stitch_edges:
            if e.inferred_seq_no is None:
                unresolved += 1
                continue
            key = (e.inferred_seq_no, e.producer_task_id, e.consumer_task_id)
            edge_set.add(key)
            edges_by_pair[key].append(e)

        if unresolved:
            violations.append(Violation(
                rule_id=self.RULE_ID,
                severity=Severity.INFO,
                message=f"有 {unresolved} 条 stitch 边 seqNo 无法唯一确定，已跳过一致性检查",
                details={"unresolvedEdgeCount": unresolved},
            ))

        # ---------------- 方向 A：edges → topo ----------------
        for key, es in edges_by_pair.items():
            seq_no, prod_id, cons_id = key
            prod_task = ctx.task_by_id.get((seq_no, prod_id))
            if prod_task is None:
                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    severity=Severity.ERROR,
                    message="edge 的 producerTaskId 在 dyn_topo 中不存在",
                    details={
                        "seqNo": seq_no,
                        "producerTaskId": prod_id,
                        "consumerTaskId": cons_id,
                    },
                ))
                continue
            if cons_id not in prod_task.stitch_successors:
                # 若落在静态段，属于归类错误；若完全不在后继中，则是 dump 不一致
                if cons_id in prod_task.static_successors:
                    severity = Severity.SUSPECT_MEDIUM
                    msg = "edge 建立的后继实际是静态后继（非 stitch）"
                else:
                    severity = Severity.ERROR
                    msg = "edge 记录的后继在 dyn_topo 对应行的 stitch 段中不存在"
                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    severity=severity,
                    message=msg,
                    details={
                        "seqNo": seq_no,
                        "producerTaskId": prod_id,
                        "consumerTaskId": cons_id,
                        "dynTopoStaticSuccessors": prod_task.static_successors,
                        "dynTopoStitchSuccessors": prod_task.stitch_successors,
                        "stitchKinds": sorted({e.stitch_kind for e in es}),
                    },
                ))

        # ---------------- 方向 B：topo → edges ----------------
        for task in ctx.dyn_tasks:
            for cons_id in task.stitch_successors:
                key = (task.seq_no, task.task_id, cons_id)
                if key in edge_set:
                    continue
                # 未在 edges 中找到；可能是 reuse stitch（当前实现在 dyn_stitch_edges 中不记录）
                # 或 dump 逻辑有 bug——无法自动区分，标记为 SUSPECT-MEDIUM 供人工分辨
                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    severity=Severity.SUSPECT_MEDIUM,
                    message="dyn_topo 有 stitch 后继但 dyn_stitch_edges 未记录（可能是 reuse stitch）",
                    details={
                        "seqNo": task.seq_no,
                        "producerTaskId": task.task_id,
                        "consumerTaskId": cons_id,
                        "producerFuncKey": task.root_index,
                    },
                ))

        return violations
