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
规则 1：静态依赖完整性。

目的：
  编译期在 static_topo.csv 中声明的 function 内部依赖，运行时必须原样体现在
  dyn_topo.txt 每个实例的 successors[0:staticSuccCount] 中，否则说明静态依赖被丢失或篡改。
"""
from typing import List

from ..models import Severity, Violation, encode_task_id
from ..rule_base import Rule, RuleContext
from ..rule_registry import register_rule


@register_rule
class StaticIntegrityRule(Rule):
    RULE_ID = "rule_static_integrity"
    DESCRIPTION = "规则 1：静态依赖完整性（static_topo 与 dyn_topo 的静态段必须一致）"

    def check(self, ctx: RuleContext) -> List[Violation]:
        violations: List[Violation] = []

        for task in ctx.dyn_tasks:
            op = ctx.get_static_op(task.root_index, task.op_idx)
            if op is None:
                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    severity=Severity.ERROR,
                    message="dyn_topo 出现了 static_topo 中不存在的 (funcKey, opIdx)",
                    details={
                        "seqNo": task.seq_no,
                        "taskId": task.task_id,
                        "funcKey": task.root_index,
                        "funcIdx": task.func_idx,
                        "opIdx": task.op_idx,
                        "opmagic": task.opmagic,
                    },
                ))
                continue

            # 将 static_topo 的 op_idx 后继映射为本实例下的 taskId 集合
            expected = {encode_task_id(task.func_idx, o)
                        for o in op.static_successors_op_idx}
            actual = set(task.static_successors)

            if expected != actual:
                missing = expected - actual
                extra = actual - expected
                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    severity=Severity.ERROR,
                    message="静态后继在 dyn_topo 中丢失或被篡改",
                    details={
                        "seqNo": task.seq_no,
                        "taskId": task.task_id,
                        "funcKey": task.root_index,
                        "funcIdx": task.func_idx,
                        "opIdx": task.op_idx,
                        "expected": sorted(expected),
                        "actual": sorted(actual),
                        "missing": sorted(missing),
                        "extra": sorted(extra),
                    },
                ))

            # 顺带校验 staticSuccCount 是否等于实际 successors 前缀长度
            if task.static_succ_count > len(task.successors):
                violations.append(Violation(
                    rule_id=self.RULE_ID,
                    severity=Severity.ERROR,
                    message="staticSuccCount 超过 successors 总数",
                    details={
                        "seqNo": task.seq_no,
                        "taskId": task.task_id,
                        "staticSuccCount": task.static_succ_count,
                        "successors_len": len(task.successors),
                    },
                ))

        return violations
