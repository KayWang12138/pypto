# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
class PipeInst:
    def __init__(self, s: str):
        self.s = s

    def __str__(self):
        return self.s

    def __repr__(self):
        return self.s

    def __eq__(self, other: 'PipeInst'):
        if self.s==other.s:
            return True
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.s)


class PIPE:
    M = PipeInst('M')
    V = PipeInst('V')
    MTE1 = PipeInst('MTE1')
    MTE2 = PipeInst('MTE2')
    MTE3 = PipeInst('MTE3')
    FIX = PipeInst('FIX')
    S = PipeInst('S')
    ALL = PipeInst('ALL')


