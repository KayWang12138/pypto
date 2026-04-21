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
"""规则 2：stitch 边合法性。"""
from typing import List

from ..models import Violation
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


_REUSE_KINDS = {"reuse", "Reuse", "REUSE"}


@register_rule
class StitchLegalityRule(Rule):
    RULE_ID = "rule_stitch_legality"
    DESCRIPTION = "规则 2：每条 stitch 边必须落在合法的 slot 数据流上"

    def check(self, ctx: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        for edge in ctx.stitch_edges:
            prod_op = ctx.get_static_op(edge.producer_func_key, edge.producer_op_idx)
            cons_op = ctx.get_static_op(edge.consumer_func_key, edge.consumer_op_idx)

            if prod_op is None or cons_op is None:
                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    slot_idx=edge.slot_idx,
                    message=f"stitch 边引用了 static_topo 未声明的 op "
                            f"(producer funcKey={edge.producer_func_key}/opIdx="
                            f"{edge.producer_op_idx}, consumer funcKey="
                            f"{edge.consumer_func_key}/opIdx={edge.consumer_op_idx})",
                ))
                continue

            if edge.stitch_kind in _REUSE_KINDS:
                continue

            prod_has = edge.slot_idx in prod_op.outcast_slots
            cons_has = edge.slot_idx in cons_op.incast_slots
            if not (prod_has and cons_has):
                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    slot_idx=edge.slot_idx,
                    message=f"stitch 边 slot 不在 producer.outcast ∩ consumer.incast 内 "
                            f"(producerOutcast={prod_op.outcast_slots}, "
                            f"consumerIncast={cons_op.incast_slots})",
                ))
        return violations
