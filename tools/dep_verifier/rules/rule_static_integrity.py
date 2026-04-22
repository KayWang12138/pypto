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
"""规则 1：静态依赖完整性（static_topo 与 dyn_topo 静态段必须一致）。"""
from typing import List

from ..models import Violation, encode_task_id
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


@register_rule
class StaticIntegrityRule(Rule):
    RULE_ID = "rule_static_integrity"
    DESCRIPTION = "规则 1：静态依赖完整性"

    def check(self, ctx: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        for task in ctx.dyn_tasks:
            op = ctx.get_static_op(task.root_index, task.op_idx)
            if op is None:
                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    message=f"dyn_topo 出现 static_topo 未声明的 op "
                            f"(funcKey={task.root_index}, opIdx={task.op_idx}, "
                            f"taskId={task.task_id})",
                ))
                continue
            expected = {encode_task_id(task.func_idx, o)
                        for o in op.static_successors_op_idx}
            actual = set(task.static_successors)
            if expected != actual:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    message=f"静态后继与 static_topo 不一致 "
                            f"(funcKey={task.root_index}, opIdx={task.op_idx}, "
                            f"taskId={task.task_id}, missing={missing}, extra={extra})",
                ))
        return violations
