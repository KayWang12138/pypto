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
import sys
import os
from pathlib import Path
import torch
from op_rsqrt import op_rsqrt, op_rsqrt_golden
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3].joinpath("framework/tests/cmake/scripts/helper")))
from pypto_test import TestBuilder


class RsqrtTest(TestBuilder):
    def __init__(self, params: tuple, kernel, kernel_golden, tiling: int):
        super().__init__(params, kernel, kernel_golden, tiling)

    def get_input_from_param(self):
        n, m = self.tiling * 1, self.tiling * 1
        a_tensor = torch.rand(n, m, dtype=torch.float32) * 100
        self.setup_inputs(a_tensor)
        self.set_tol(rtol=3e-3, atol=3e-3)
        return (a_tensor, )


def test():
    st = RsqrtTest(((16, 16), (8, 8)), op_rsqrt, op_rsqrt_golden, tiling=32)
    st()


if __name__ == "__main__":
    st = RsqrtTest(((16, 16), (8, 8)), op_rsqrt, op_rsqrt_golden, tiling=32)
    st()
