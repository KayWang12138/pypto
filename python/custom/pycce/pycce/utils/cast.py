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
from .. import context


POSTFIX_MAPPING: dict[str, str] = {
    'UNKNOWN' : '',
    'CAST_RINT' : 'r',
    'CAST_ROUND' : 'a',
    'CAST_FLOOR' : 'f',
    'CAST_CEIL' : 'c',
    'CAST_TRUNC' : 'z',
}


class RoundModeInst:
    def __init__(self, mode: str):
        self.mode = mode

    def __str__(self):
        if context.device_type.startswith('910b') and self.mode=='UNKNOWN':
            return 'CAST_NONE'
        return self.mode

    def __eq__(self, other: 'RoundModeInst'):
        return self.mode==other.mode

    @property
    def postfix(self):
        return POSTFIX_MAPPING[self.mode]


class RoundMode:
    NONE = RoundModeInst('UNKNOWN')
    TO_EVEN = RoundModeInst('CAST_RINT')              # R
    AWAY_FROM_ZERO = RoundModeInst('CAST_ROUND')      # A
    FLOOR = RoundModeInst('CAST_FLOOR')               # F
    CEIL = RoundModeInst('CAST_CEIL')                 # C
    TRUNC = RoundModeInst('CAST_TRUNC')               # Z
