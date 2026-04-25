# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""multi_jit fixture: 2 个 @pypto.jit, cheat_detector 不检测 jit 数量."""
import torch
import pypto


@pypto.jit
def relu_a(x):
    return pypto.dsl.relu(x)


@pypto.jit
def relu_b(x):
    return pypto.dsl.add(x, 0.0)


def relu_wrapper(x):
    return relu_b(relu_a(x))
