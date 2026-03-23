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

static std::vector<uint64_t> SkipCounter(const std::vector<uint64_t> &counter, uint64_t offset) {
    ALOG_ERROR_F("--------SkipCounter start, counter size: %d, offset: %lu", counter.size(), offset);
    for (size_t i = 0; i < counter.size(); i++) {
        ALOG_ERROR_F("--------counter[%d]: %lu", i, counter[i]);
    }
    
    std::vector<uint64_t> result = counter;
    
    uint32_t offsetLo = static_cast<uint32_t>(offset);
    uint32_t offsetHi = static_cast<uint32_t>(offset >> 32);
    
    uint32_t c0 = static_cast<uint32_t>(result[0] & 0xFFFFFFFF);
    uint32_t c1 = static_cast<uint32_t>(result[0] >> 32);
    uint32_t c2 = static_cast<uint32_t>(result[1] & 0xFFFFFFFF);
    uint32_t c3 = static_cast<uint32_t>(result[1] >> 32);
    
    c0 += offsetLo;
    if (c0 < offsetLo) {
        ++offsetHi;
    }
    c1 += offsetHi;
    if (c1 < offsetHi) {
        if (++c2 == 0) {
            ++c3;
        }
    }
    
    result[0] = static_cast<uint64_t>(c0) | (static_cast<uint64_t>(c1) << 32);
    result[1] = static_cast<uint64_t>(c2) | (static_cast<uint64_t>(c3) << 32);
    
    ALOG_ERROR_F("--------SkipCounter result: [%lu, %lu]", result[0], result[1]);
    
    return result;
}

static void TiledRandomBuildIn(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, uint64_t key,
    const std::vector<uint64_t> &baseCounter, uint16_t rounds,
    const std::vector<int64_t> &shape) {
    ALOG_ERROR_F("--------TiledRandomBuildIn start, key: %lu, rounds: %d", key, rounds);
    ALOG_ERROR_F("--------shape size: %d", shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        ALOG_ERROR_F("--------shape[%d]: %ld", i, shape[i]);
    }
    ALOG_ERROR_F("--------result->shape size: %d", result->shape.size());
    for (size_t i = 0; i < result->shape.size(); i++) {
        ALOG_ERROR_F("--------result->shape[%d]: %ld", i, result->shape[i]);
    }
    
    auto &vecTile = tileShape.GetVecTile();
    ALOG_ERROR_F("--------vecTile size: %d", vecTile.size());
    for (size_t i = 0; i < vecTile.size(); i++) {
        ALOG_ERROR_F("--------vecTile[%d]: %ld", i, vecTile[i]);
    }
    
    for (int64_t i = 0; i < result->shape[0]; i += vecTile[0]) {
        ALOG_ERROR_F("--------TiledRandomBuildIn loop i: %ld", i);
        resultTileInfo.offset[0] = i;
        resultTileInfo.shape[0] = std::min(result->shape[0] - resultTileInfo.offset[0], vecTile[0]);
        ALOG_ERROR_F("--------resultTileInfo offset: %ld, shape: %ld", resultTileInfo.offset[0], resultTileInfo.shape[0]);
        
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        
        std::vector<uint64_t> tileCounter = SkipCounter(baseCounter, static_cast<uint64_t>(i));
        
        auto &op = function.AddOperation(Opcode::OP_RANDOM, {}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "KEY", static_cast<int64_t>(key));
        op.SetAttribute(OP_ATTR_PREFIX + "COUNTER", tileCounter);
        op.SetAttribute(OP_ATTR_PREFIX + "ROUNDS", static_cast<int64_t>(rounds));
        op.SetAttribute(OP_ATTR_PREFIX + "SHAPE", shape);
        ALOG_ERROR_F("--------TiledRandomBuildIn operation added");
    }
}

void RandomOperationTileFunc(Function &function, const TileShape &tileShape,
    [[maybe_unused]] const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op) {
    ALOG_ERROR_F("--------RandomOperationTileFunc start");
    uint64_t key = static_cast<uint64_t>(op.GetIntAttribute(OP_ATTR_PREFIX + "KEY"));
    ALOG_ERROR_F("--------key: %lu", key);
    
    auto counterAttr = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "COUNTER");
    ALOG_ERROR_F("--------counterAttr size: %d", counterAttr.size());
    std::vector<uint64_t> counter;
    for (auto val : counterAttr) {
        counter.push_back(static_cast<uint64_t>(val));
        ALOG_ERROR_F("--------counter val: %ld -> %lu", val, static_cast<uint64_t>(val));
    }
    
    uint16_t rounds = static_cast<uint16_t>(op.GetIntAttribute(OP_ATTR_PREFIX + "ROUNDS"));
    ALOG_ERROR_F("--------rounds: %d", rounds);
    
    auto shapeAttr = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "SHAPE");
    ALOG_ERROR_F("--------shapeAttr size: %d", shapeAttr.size());
    std::vector<int64_t> shape;
    for (auto dim : shapeAttr) {
        shape.push_back(static_cast<int64_t>(dim));
        ALOG_ERROR_F("--------shape dim: %ld", dim);
    }
    
    TileInfo resultTileInfo(shape.size(), shape.size());
    TiledRandomBuildIn(function, tileShape, oOperand[0], resultTileInfo, key, counter, rounds, shape);
    ALOG_ERROR_F("--------RandomOperationTileFunc end");
}

static Tensor RealRandom(uint64_t key, const std::vector<uint64_t> &counter,
    const std::vector<int64_t> &shape, uint16_t rounds) {
    ALOG_ERROR_F("--------RealRandom start, key: %lu, rounds: %d", key, rounds);
    ALOG_ERROR_F("--------counter size: %d", counter.size());
    for (size_t i = 0; i < counter.size(); i++) {
        ALOG_ERROR_F("--------counter[%d]: %lu", i, counter[i]);
    }
    ALOG_ERROR_F("--------shape size: %d", shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        ALOG_ERROR_F("--------shape[%d]: %ld", i, shape[i]);
    }
    
    DECLARE_TRACER();
    
    auto resTensor = Tensor(DT_UINT32, shape);
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &tileShape = TileShape::Current();
    
    auto &vecTile = tileShape.GetVecTile();
    ALOG_ERROR_F("--------vecTile size: %d", vecTile.size());
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, vecTile.size() == 1)
        << "Random: TileShape must be 1-dimensional";
    
    auto result = resTensor.GetStorage();
    ALOG_ERROR_F("--------result->shape size: %d", result->shape.size());
    for (size_t i = 0; i < result->shape.size(); i++) {
        ALOG_ERROR_F("--------result->shape[%d]: %ld", i, result->shape[i]);
    }
    
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    TiledRandomBuildIn(function, tileShape, result, resultTileInfo, key, counter, rounds, shape);
    
    ALOG_ERROR_F("--------RealRandom end");
    return resTensor;
}

Tensor Random(uint64_t key, const std::vector<uint64_t> &counter,
    const std::vector<int64_t> &shape, uint16_t rounds) {
    ALOG_ERROR_F("--------Random start, key: %lu, rounds: %d", key, rounds);
    ALOG_ERROR_F("--------counter size: %d", counter.size());
    for (size_t i = 0; i < counter.size(); i++) {
        ALOG_ERROR_F("--------counter[%d]: %lu", i, counter[i]);
    }
    ALOG_ERROR_F("--------shape size: %d", shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        ALOG_ERROR_F("--------shape[%d]: %ld", i, shape[i]);
    }
    
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, counter.size() == 2)
        << "Random: counter must have 2 elements";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, shape.size() == 1)
        << "Random: shape must be 1-dimensional";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, rounds == 7 || rounds == 10)
        << "Random: rounds must be 7 or 10";
    
    ALOG_ERROR_F("--------Random calling RealRandom");
    return RealRandom(key, counter, shape, rounds);
}

REGISTER_OPERATION_TILED_FUNC(OP_RANDOM, Opcode::OP_RANDOM, RandomOperationTileFunc);

}
