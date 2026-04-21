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
规则 2：Stitch 边合法性。

对 dyn_stitch_edges.csv 每条记录执行三项检查：
  A. Operation 存在性：producer/consumer 的 (funcKey, opIdx) 在 static_topo 中存在
  B. Slot 合法性（reuse 除外）：slotIdx ∈ producer.outcastSlots ∩ consumer.incastSlots
  C. 自环提示：producer 与 consumer 完全相同时降级为 INFO（inplace 读改写常见场景）
"""
from typing import List

from ..models import Severity, Violation
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


REUSE_KINDS = {"reuse", "Reuse", "REUSE"}


@register_rule
class StitchLegalityRule(Rule):
    RULE_ID = "rule_stitch_legality"
    DESCRIPTION = "规则 2：每条 stitch 边需建立在合法的数据流基础上"

    def check(self, ctx: RuleContext) -> List[Violation]:
        violations: List[Violation] = []

        for edge in ctx.stitch_edges:
            prod_op = ctx.get_static_op(edge.producer_func_key, edge.producer_op_idx)
            cons_op = ctx.get_static_op(edge.consumer_func_key, edge.consumer_op_idx)

            # A. Operation 存在性
            if prod_op is None or cons_op is None:
                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    severity=Severity.ERROR,
                    message="stitch 边引用了 static_topo 中不存在的 operation",
                    details={
                        "slotIdx": edge.slot_idx,
                        "stitchKind": edge.stitch_kind,
                        "producer": {
                            "funcKey": edge.producer_func_key,
                            "opIdx": edge.producer_op_idx,
                            "found": prod_op is not None,
                        },
                        "consumer": {
                            "funcKey": edge.consumer_func_key,
                            "opIdx": edge.consumer_op_idx,
                            "found": cons_op is not None,
                        },
                    },
                ))
                continue

            # B. Slot 合法性（reuse 不校验数据流）
            is_reuse = edge.stitch_kind in REUSE_KINDS
            if not is_reuse:
                prod_has = edge.slot_idx in prod_op.outcast_slots
                cons_has = edge.slot_idx in cons_op.incast_slots
                if not (prod_has and cons_has):
                    violations.append(Violation(
                        rule_id=self.RULE_ID,
                        severity=Severity.ERROR,
                        message="stitch 边 slot 不在 producer.outcast ∩ consumer.incast 内",
                        details={
                            "slotIdx": edge.slot_idx,
                            "stitchKind": edge.stitch_kind,
                            "producerFuncKey": edge.producer_func_key,
                            "producerOpIdx": edge.producer_op_idx,
                            "producerOutcastSlots": prod_op.outcast_slots,
                            "consumerFuncKey": edge.consumer_func_key,
                            "consumerOpIdx": edge.consumer_op_idx,
                            "consumerIncastSlots": cons_op.incast_slots,
                            "producerHasSlot": prod_has,
                            "consumerHasSlot": cons_has,
                        },
                    ))

            # C. 自环：producer 与 consumer 完全相同 → INFO 提示
            if (edge.producer_task_id == edge.consumer_task_id
                    and edge.producer_func_key == edge.consumer_func_key):
                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    severity=Severity.INFO,
                    message="stitch 自环（producer == consumer），通常是 inplace 读改写",
                    details={
                        "slotIdx": edge.slot_idx,
                        "stitchKind": edge.stitch_kind,
                        "taskId": edge.producer_task_id,
                        "funcKey": edge.producer_func_key,
                        "opIdx": edge.producer_op_idx,
                    },
                ))

        return violations
