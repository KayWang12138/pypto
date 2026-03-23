/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file random.cpp
 * \brief Random number generator implementation
 */

#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/utils/vector_error.h"
#include "interface/operation/operation_common.h"

namespace npu::tile_fwk {

void TiledRandomBuildIn(Function &function, const TileShape &tileShape, size_t cur,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, uint64_t key,
    const std::vector<int64_t> &counter, uint16_t rounds,
    const std::vector<int64_t> &shape) {
    if (cur == result->shape.size()) {
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_RANDOM, {}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "KEY", static_cast<int64_t>(key));
        op.SetAttribute(OP_ATTR_PREFIX + "COUNTER", counter);
        op.SetAttribute(OP_ATTR_PREFIX + "ROUNDS", static_cast<int64_t>(rounds));
        op.SetAttribute(OP_ATTR_PREFIX + "SHAPE", shape);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TiledRandomBuildIn(function, tileShape, cur + 1, result, resultTileInfo, key, counter, rounds, shape);
    }
}

void RandomOperationTileFunc(Function &function, const TileShape &tileShape,
    [[maybe_unused]] const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op) {
    uint64_t key = static_cast<uint64_t>(op.GetIntAttribute(OP_ATTR_PREFIX + "KEY"));
    auto counter = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "COUNTER");
    uint16_t rounds = static_cast<uint16_t>(op.GetIntAttribute(OP_ATTR_PREFIX + "ROUNDS"));
    auto shapeAttr = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "SHAPE");
    std::vector<int64_t> shape;
    for (auto dim : shapeAttr) {
        shape.push_back(static_cast<int64_t>(dim));
    }
    
    TileInfo resultTileInfo(shape.size(), shape.size());
    TiledRandomBuildIn(function, tileShape, 0, oOperand[0], resultTileInfo, key, counter, rounds, shape);
}

Tensor Random(uint64_t key, const std::vector<int64_t> &counter,
    const std::vector<int64_t> &shape, DataType dtype, uint16_t rounds) {
    DECLARE_TRACER();
    
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, counter.size() == 4)
        << "Random: counter must have 4 elements";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, rounds == 7 || rounds == 10)
        << "Random: rounds must be 7 or 10";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, dtype == DT_UINT32)
        << "Random: only DT_UINT32 is supported";
    
    auto result = Tensor(dtype, shape);
    
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &tileShape = TileShape::Current();
    
    TileInfo resultTileInfo(shape.size(), shape.size());
    TiledRandomBuildIn(function, tileShape, 0, result.GetStorage(), resultTileInfo, key, counter, rounds, shape);
    
    return result;
}

REGISTER_OPERATION_TILED_FUNC(OP_RANDOM, Opcode::OP_RANDOM, RandomOperationTileFunc);

}
