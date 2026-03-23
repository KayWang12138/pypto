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

static std::vector<int64_t> SkipCounter(const std::vector<int64_t> &counter, uint64_t offset) {
    std::vector<int64_t> result = counter;
    
    uint32_t offsetLo = static_cast<uint32_t>(offset & 0xFFFFFFFF);
    uint32_t offsetHi = static_cast<uint32_t>(offset >> 32);
    
    uint32_t c0 = static_cast<uint32_t>(result[0] & 0xFFFFFFFF);
    uint32_t c1 = static_cast<uint32_t>(result[1] & 0xFFFFFFFF);
    uint32_t c2 = static_cast<uint32_t>(result[2] & 0xFFFFFFFF);
    uint32_t c3 = static_cast<uint32_t>(result[3] & 0xFFFFFFFF);
    
    c0 += offsetLo;
    uint32_t carry = (c0 < offsetLo) ? 1 : 0;
    
    uint32_t sum1 = c1 + offsetHi + carry;
    carry = (sum1 < c1 || (carry && sum1 == c1)) ? 1 : 0;
    c1 = sum1;
    
    if (carry) {
        c2++;
        if (c2 == 0) {
            c3++;
        }
    }
    
    result[0] = c0;
    result[1] = c1;
    result[2] = c2;
    result[3] = c3;
    
    return result;
}

static uint64_t CalculateLinearOffset(const std::vector<int64_t> &offset, const std::vector<int64_t> &shape) {
    uint64_t linearOffset = 0;
    uint64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        linearOffset += static_cast<uint64_t>(offset[i]) * stride;
        stride *= static_cast<uint64_t>(shape[i]);
    }
    return linearOffset;
}

void TiledRandomBuildIn(Function &function, const TileShape &tileShape, size_t cur,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, uint64_t key,
    const std::vector<int64_t> &baseCounter, uint16_t rounds,
    const std::vector<int64_t> &shape) {
    if (cur == result->shape.size()) {
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        
        uint64_t linearOffset = CalculateLinearOffset(resultTileInfo.offset, shape);
        std::vector<int64_t> tileCounter = SkipCounter(baseCounter, linearOffset);
        
        auto &op = function.AddOperation(Opcode::OP_RANDOM, {}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "KEY", static_cast<int64_t>(key));
        op.SetAttribute(OP_ATTR_PREFIX + "COUNTER", tileCounter);
        op.SetAttribute(OP_ATTR_PREFIX + "ROUNDS", static_cast<int64_t>(rounds));
        op.SetAttribute(OP_ATTR_PREFIX + "SHAPE", shape);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TiledRandomBuildIn(function, tileShape, cur + 1, result, resultTileInfo, key, baseCounter, rounds, shape);
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
