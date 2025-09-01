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

from typing import Optional

from .instruction import Instruction
from .. import context


class AggregationVec():
    idx: int = -1
    _previdx: list[int]

    def __init__(self, value: Optional[list] = None):
        self._previdx = []
        if context.active_module is not None:
            self.idx = context.active_module.args_counter
            context.active_module.args_counter += 1
            if value is not None:
                context.active_module.add_inst(Instruction('new_aggregation_vec', [value], self))
            else:
                context.active_module.add_inst(Instruction('new_aggregation_vec', [], self))

    def __str__(self):
        return f'aggregationVec[{self.idx}]'

    def __repr__(self):
        return str(self)
    
    def set_idx(self, idx: int):
        self.idx = idx

    def push(self):
        self._previdx.append(self.idx)

    def pop(self):
        self.idx = self._previdx.pop(-1)
