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
自动加载本目录下所有 rule_*.py 触发 @register_rule 注册。

新增规则时：
  1) 在本目录下新建 rule_xxx.py
  2) 实现 Rule 子类并用 @register_rule 装饰
  3) 本 __init__.py 无需修改（包已通过 importlib 自动发现）
"""
import importlib
import pkgutil
from pathlib import Path

_PACKAGE_DIR = Path(__file__).parent

for _mod_info in pkgutil.iter_modules([str(_PACKAGE_DIR)]):
    if _mod_info.name.startswith("rule_"):
        importlib.import_module(f"{__name__}.{_mod_info.name}")
