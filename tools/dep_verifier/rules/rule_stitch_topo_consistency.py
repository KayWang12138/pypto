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
"""规则 3：dyn_stitch_edges 每条记录都必须在 dyn_topo 的 successors 中可查。"""
from collections import defaultdict
from typing import Dict, List, Tuple

from ..models import Violation
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


@register_rule
class StitchTopoConsistencyRule(Rule):
    RULE_ID = "rule_stitch_topo_consistency"
    DESCRIPTION = "规则 3：stitch 边与 dyn_topo 一致"

    def check(self, ctx: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        edges_by_key: Dict[Tuple[int, int, int], int] = defaultdict(int)
        for e in ctx.stitch_edges:
            if e.inferred_seq_no is None:
                continue
            if e.producer_task_id == e.consumer_task_id:
                continue
            key = (e.inferred_seq_no, e.producer_task_id, e.consumer_task_id)
            edges_by_key[key] += 1

        for (seq_no, prod_id, cons_id) in edges_by_key.keys():
            prod_task = ctx.task_by_id.get((seq_no, prod_id))
            if prod_task is None:
                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    message=f"stitch 边的 producerTaskId 未出现在 dyn_topo "
                            f"(seqNo={seq_no}, producerTaskId={prod_id}, "
                            f"consumerTaskId={cons_id})",
                ))
                continue
            if cons_id in prod_task.successors:
                continue
            violations.append(Violation(
                rule_id=self.RULE_ID,
                message=f"stitch 边未出现在 dyn_topo 对应 successors 中 "
                        f"(seqNo={seq_no}, producerTaskId={prod_id}, "
                        f"consumerTaskId={cons_id})",
            ))
        return violations
