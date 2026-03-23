/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
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
#include <iostream>
#include "interface/operation/operation_common.h"

namespace npu::tile_fwk {

static std::vector<uint64_t> SkipCounter(const std::vector<uint64_t> &counter, uint64_t offset) {
    std::cout << "[Random Trace] SkipCounter() called" << std::endl;
    std::cout << "[Random Trace]   counter: [" << counter[0] << ", " << counter[1] << "]" << std::endl;
    std::cout << "[Random Trace]   offset: " << offset << std::endl;
    
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
    
    std::cout << "[Random Trace] SkipCounter() returning: [" << result[0] << ", " << result[1] << "]" << std::endl;
    
    return result;
}

LogicalTensorPtr TensorRandom(Function &function, LogicalTensorPtr &result, uint64_t key,
    const std::vector<uint64_t> &counter, uint16_t rounds, const std::vector<int64_t> &shape) {
    std::cout << "[Random Trace] TensorRandom() called" << std::endl;
    
    auto &op = function.AddOperation(Opcode::OP_RANDOM, {}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "KEY", static_cast<int64_t>(key));
    op.SetAttribute(OP_ATTR_PREFIX + "COUNTER", counter);
    op.SetAttribute(OP_ATTR_PREFIX + "ROUNDS", static_cast<int64_t>(rounds));
    op.SetAttribute(OP_ATTR_PREFIX + "SHAPE", shape);
    return result;
}

static void TiledRandomBuildIn(Function &function, const TileShape &tileShape,    const LogicalTensorPtr &result, TileInfo &resultTileInfo, uint64_t key,    const std::vector<uint64_t> &baseCounter, uint16_t rounds,    const std::vector<int64_t> &shape) {
    std::cout << "[Random Trace] TiledRandomBuildIn() called" << std::endl;
    std::cout << "[Random Trace]   key: " << key << std::endl;
    std::cout << "[Random Trace]   baseCounter: [" << baseCounter[0] << ", " << baseCounter[1] << "]" << std::endl;
    std::cout << "[Random Trace]   rounds: " << rounds << std::endl;
    std::cout << "[Random Trace]   shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    
    auto &vecTile = tileShape.GetVecTile();
    
    std::cout << "[Random Debug] result->shape[0]: " << result->shape[0] << std::endl;
    std::cout << "[Random Debug] vecTile[0]: " << vecTile[0] << std::endl;
    
    for (int64_t i = 0; i < result->shape[0]; i += vecTile[0]) {
        std::cout << "[Random Trace]   TiledRandomBuildIn() loop iteration, i = " << i << std::endl;
        resultTileInfo.offset[0] = i;
        resultTileInfo.shape[0] = std::min(result->shape[0] - resultTileInfo.offset[0], vecTile[0]);
        
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        
        std::vector<uint64_t> tileCounter = SkipCounter(baseCounter, static_cast<uint64_t>(i));
        
        auto &op = function.AddOperation(Opcode::OP_RANDOM, {}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "KEY", static_cast<int64_t>(key));
        op.SetAttribute(OP_ATTR_PREFIX + "COUNTER", tileCounter);
        op.SetAttribute(OP_ATTR_PREFIX + "ROUNDS", static_cast<int64_t>(rounds));
        op.SetAttribute(OP_ATTR_PREFIX + "SHAPE", shape);
    }
}

void RandomOperationTileFunc(Function &function, const TileShape &tileShape,
    [[maybe_unused]] const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op) {
    std::cout << "[Random Trace] RandomOperationTileFunc() called" << std::endl;
    
    uint64_t key = static_cast<uint64_t>(op.GetIntAttribute(OP_ATTR_PREFIX + "KEY"));
    
    auto counterAttr = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "COUNTER");
    std::vector<uint64_t> counter;
    for (auto val : counterAttr) {
        counter.push_back(static_cast<uint64_t>(val));
    }
    
    uint16_t rounds = static_cast<uint16_t>(op.GetIntAttribute(OP_ATTR_PREFIX + "ROUNDS"));
    
    auto shapeAttr = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "SHAPE");
    std::vector<int64_t> shape;
    for (auto dim : shapeAttr) {
        shape.push_back(static_cast<int64_t>(dim));
    }
    
    TileInfo resultTileInfo(shape.size(), shape.size());
    TiledRandomBuildIn(function, tileShape, oOperand[0], resultTileInfo, key, counter, rounds, shape);
}

static Tensor RealRandom(uint64_t key, const std::vector<uint64_t> &counter,
    const std::vector<int64_t> &shape, uint16_t rounds) {
    std::cout << "[Random Trace] RealRandom() called" << std::endl;
    std::cout << "[Random Trace]   key: " << key << std::endl;
    std::cout << "[Random Trace]   counter: [" << counter[0] << ", " << counter[1] << "]" << std::endl;
    std::cout << "[Random Trace]   rounds: " << rounds << std::endl;
    std::cout << "[Random Trace]   shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    
    DECLARE_TRACER();
    
    auto resTensor = Tensor(DT_UINT32, shape);
    RETURN_CALL(Random, *Program::GetInstance().GetCurrentFunction(), resTensor.GetStorage(), key, counter, rounds, shape);
}

Tensor Random(uint64_t key, const std::vector<uint64_t> &counter,
    const std::vector<int64_t> &shape, uint16_t rounds) {
    std::cout << "[Random Trace] Random() called" << std::endl;
    std::cout << "[Random Trace]   key: " << key << std::endl;
    std::cout << "[Random Trace]   counter: [" << counter[0] << ", " << counter[1] << "]" << std::endl;
    std::cout << "[Random Trace]   rounds: " << rounds << std::endl;
    std::cout << "[Random Trace]   shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, counter.size() == 2)
        << "Random: counter must have 2 elements";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, shape.size() == 1)
        << "Random: shape must be 1-dimensional";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, rounds == 7 || rounds == 10)
        << "Random: rounds must be 7 or 10";
    
    return RealRandom(key, counter, shape, rounds);
}

REGISTER_OPERATION_TILED_FUNC(OP_RANDOM, Opcode::OP_RANDOM, RandomOperationTileFunc);

}  // namespace npu::tile_fwk
