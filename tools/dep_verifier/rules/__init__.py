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
"""Auto-discover and load every rule_*.py in this directory.

Each rule module decorates its Rule subclass with @register_rule, which
populates the global rule registry on import. To add a new rule, drop a
file rule_xxx.py in this directory; this __init__ does not need changes.
"""
import importlib
import pkgutil
from pathlib import Path

_PACKAGE_DIR = Path(__file__).parent

for _mod_info in pkgutil.iter_modules([str(_PACKAGE_DIR)]):
    if _mod_info.name.startswith("rule_"):
        importlib.import_module(f"{__name__}.{_mod_info.name}")
