#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass

active_module: Optional['AscppModule'] = None
module_stack: list[Optional['AscppModule']] = []


def stack_in_module(mod: 'AscppModule'):
    global module_stack, active_module
    module_stack.append(active_module)
    active_module = mod


def stack_out_module():
    global module_stack, active_module
    active_module = module_stack.pop(-1)
