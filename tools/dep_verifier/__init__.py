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
运行时依赖边建立正确性校验工具包。

该工具以三个运行时 dump 文件为输入：
  - static_topo.csv      编译期静态依赖快照
  - dyn_topo.txt         运行时全量依赖 topo（含 stitch 后继）
  - dyn_stitch_edges.csv 每条 stitch 边的详细记录

按照《运行时依赖边建立正确性校验方案设计》文档定义的规则 1/2/3/4 执行校验。

包结构：
  data_loader    文件解析 → 内存数据模型
  models         共享数据类
  rule_base      规则抽象基类（Rule）
  rule_registry  规则注册器（@register_rule 装饰器，支持插件式扩展）
  report         统一 Violation 聚合与报告输出
  rules/         具体规则实现，新增规则在此目录下新建 rule_*.py 并 @register_rule
"""
from .rule_registry import register_rule, get_registered_rules
from .rule_base import Rule, RuleContext, Severity
from .models import Violation
from . import rules as _rules  # noqa: F401  触发规则自动注册

__all__ = [
    "register_rule",
    "get_registered_rules",
    "Rule",
    "RuleContext",
    "Severity",
    "Violation",
]
