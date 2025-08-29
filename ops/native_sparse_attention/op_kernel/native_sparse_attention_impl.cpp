/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/tile_shape.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/tile_fwk_op_registry.h"
#include "operation/tilefwk_op.h"

namespace npu::tile_fwk {
void DynamicDD(uint64_t configKey) {
    (void)configKey;
    int s= 32;
    int n =8;

    Tensor t0(DT_FP32, {n * s, s}, "x0");
    Tensor t1(DT_FP32, {s, s}, "x1");
    Tensor blockTable(DT_INT32, {n, 1}, "x2");
    Tensor out(DT_FP32, {n * s, s}, "y0");

    TileShape::Current().SetVecTileShapes(s, s);
    TileShape::Current().SetCubeTileShapes({s, s}, {s, s}, {s, s});
    FUNCTION("main", FunctionType::DYNAMIC, {t0, t1, blockTable}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShapeDim(t0, 0) / s)) {
            SymbolicScalar idx = GetInputDataInt32Dim2(blockTable, i, 0);
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});

            Tensor qi(DT_FP32, {s, 2*s}, "qi");
            Assemble(t1, {0, 0}, qi);
            Assemble(t0s, {0, s}, qi);

            Tensor ki(DT_FP32, {s, 2*s}, "ki");
            Assemble(t0s, {0, 0}, ki);
            Assemble(t1, {0, s}, ki);

            Tensor t2 = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, ki);
            // conat((t0s + t1, t1)) @ concat (t0s, t1)^T
            Assemble(t2, {idx * s, 0}, out);
        }
    }
}
REGISTER_OP(NativeSparseAttention).ImplFunc({{0, DynamicDD}, {1, DynamicDD}}).ImplFunc(2, DynamicDD);
}