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
class ReduceModeInst():
    def __init__(self, mode: str):
        self.mode = mode

    def __str__(self):
        return self.mode

    def __eq__(self, other: 'ReduceModeInst'):
        if self.mode==other.mode:
            return True
        else:
            return False


class ReduceMode():
    ONLY_VALUE = ReduceModeInst('ONLY_VALUE')
    ONLY_INDEX = ReduceModeInst('ONLY_INDEX')
    VALUE_INDEX = ReduceModeInst('VALUE_INDEX')
    INDEX_VALUE = ReduceModeInst('INDEX_VALUE')
