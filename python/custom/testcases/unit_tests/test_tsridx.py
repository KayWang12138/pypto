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
from pypto.stub_fun import concat, reshape
from pypto.utils import Tensor, DATATYPE, Vector, Tensor, Var, Tuple, ConfigMap, Shape
from pypto.module import AscppModule


class TsrIdxingModule(AscppModule):
    def init(self):
        ...

    def forward(self, x: Tensor = None, int_param: Var = None, int_param2: Var = None):  # 3d tensor: 128x64x32
        # range indexing
        t1 = x[35:51, :, :8]  # 16 x 64 x 8 tensor (8192 logits)
        t2 = x[124:, ...]  # 4 x 64 x 32 (8192 logits)

        # with non-ranges
        t3 = x[..., 7]  # 128 x 64 (8192 logits)
        tmp = x[23, 34, ...]  # 1d (32 logits)
        t4 = concat([tmp] * (8192 // 32), 0)

        # unsqueeze using None
        t5 = x[None, 10:26, ..., None, 10:18, None]  # 1,16,64,1,8,1

        # Negative indexing
        t6 = x[..., -3]
        t7 = x[-20:-4, :, -8:]

        # Using vars
        t8 = x[..., int_param]
        t9 = x[6:22, :, :int_param2]

        return reshape(t1, [8192]) \
            + reshape(t2, [8192]) \
            + reshape(t3, [8192]) \
            + reshape(t4, [8192]) \
            + reshape(t5, [8192]) \
            + reshape(t6, [8192]) \
            + reshape(t7, [8192]) \
            + reshape(t8, [8192]) \
            + reshape(t9, [8192])


if __name__ == "__main__":
    mod = TsrIdxingModule()
    input_tensor = Tensor(shape=(128, 64, 32), dtype=DATATYPE.fp32)
    mod(input_tensor, 17, 8)
    mod.gen_code("./generatedcpp/test_tsridx.cpp")
