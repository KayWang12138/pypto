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
规则注册器。

新增规则只需：
  1) 在 rules/ 下新建 rule_xxx.py
  2) class MyRule(Rule):    RULE_ID = "rule_xxx"
         ...
  3) 在模块末尾 @register_rule 或在 __init__.py 引入

无需修改入口脚本、base 或其他规则代码——支持规则的独立增删改。
"""
import logging
from typing import Dict, List, Type

from .rule_base import Rule

logger = logging.getLogger(__name__)

_REGISTERED: Dict[str, Type[Rule]] = {}


def register_rule(cls: Type[Rule]) -> Type[Rule]:
    """装饰器：将 Rule 子类注册到全局表。"""
    if not issubclass(cls, Rule):
        raise TypeError(f"{cls.__name__} 不是 Rule 的子类")
    rule_id = cls.RULE_ID
    if not rule_id:
        raise ValueError(f"{cls.__name__}.RULE_ID 未设置")
    if rule_id in _REGISTERED:
        logger.warning("规则 id 已被注册，将覆盖: %s", rule_id)
    _REGISTERED[rule_id] = cls
    return cls


def get_registered_rules() -> List[Type[Rule]]:
    """返回已注册的全部规则类（按 RULE_ID 字典序）。"""
    return [_REGISTERED[k] for k in sorted(_REGISTERED.keys())]


def get_rule(rule_id: str) -> Type[Rule]:
    if rule_id not in _REGISTERED:
        raise KeyError(f"未注册的规则 id: {rule_id}")
    return _REGISTERED[rule_id]
