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
POS_TO_ASC_MAPPING: dict[str, str] = {
    'UB'  : 'VECCALC',
    'L1'  : 'A1',
    'L0A' : 'A2',
    'L0B' : 'B2',
    'L0C' : 'CO1',
}

POS_TO_CCE_MAPPING: dict[str, str] = {
    'UB'  : 'PUB',
    'L1'  : 'PL1',
    'L0A' : 'PL0A',
    'L0B' : 'PL0B',
    'L0C' : 'PL0C',
}

class Pos():
    def __init__(self, pos: str):
        self.posstr = pos

    def asc_pos(self):
        return POS_TO_ASC_MAPPING[self.posstr]

    def cce_pos(self):
        return POS_TO_CCE_MAPPING[self.posstr]

    def __eq__(self, other: 'Pos'):
        if self.posstr==other.posstr:
            return True
        return False

    def __str__(self):
        return self.posstr


class Position:
    UB = Pos('UB')
    L1 = Pos('L1')
    L0A = Pos('L0A')
    L0B = Pos('L0B')
    L0C = Pos('L0C')
