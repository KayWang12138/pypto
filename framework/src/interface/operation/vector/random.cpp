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

static void SkipCounter(uint64_t &counter0, uint64_t &counter1, uint64_t offset) {
    std::cout << "[Random Trace] SkipCounter() called" << std::endl;
    std::cout << "[Random Trace]   counter0: " << counter0 << ", counter1: " << counter1 << std::endl;
    std::cout << "[Random Trace]   offset: " << offset << std::endl;
    
    uint32_t offsetLo = static_cast<uint32_t>(offset);
    uint32_t offsetHi = static_cast<uint32_t>(offset >> 32);
    
    uint32_t c0 = static_cast<uint32_t>(counter0 & 0xFFFFFFFF);
    uint32_t c1 = static_cast<uint32_t>(counter0 >> 32);
    uint32_t c2 = static_cast<uint32_t>(counter1 & 0xFFFFFFFF);
    uint32_t c3 = static_cast<uint32_t>(counter1 >> 32);
    
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
    
    counter0 = static_cast<uint64_t>(c0) | (static_cast<uint64_t>(c1) << 32);
    counter1 = static_cast<uint64_t>(c2) | (static_cast<uint64_t>(c3) << 32);
    
    std::cout << "[Random Trace] SkipCounter() returning: counter0=" << counter0 << ", counter1=" << counter1 << std::endl;
}

LogicalTensorPtr TensorRandom(Function &function, LogicalTensorPtr &result, const Element &key,
    const Element &counter0, const Element &counter1, const Element &rounds, const std::vector<int64_t> &shape) {
    std::cout << "[Random Trace] TensorRandom() called" << std::endl;
    
    auto &op = function.AddOperation(Opcode::OP_RANDOM, {}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "KEY", key);
    op.SetAttribute(OP_ATTR_PREFIX + "COUNTER0", counter0);
    op.SetAttribute(OP_ATTR_PREFIX + "COUNTER1", counter1);
    op.SetAttribute(OP_ATTR_PREFIX + "ROUNDS", rounds);
    op.SetAttribute(OP_ATTR_PREFIX + "SHAPE", shape);
    return result;
}

static void TiledRandomBuildIn(Function &function, const TileShape &tileShape, const LogicalTensorPtr &result, 
    TileInfo &resultTileInfo, uint64_t key, uint64_t counter0, uint64_t counter1, uint16_t rounds, 
    const std::vector<int64_t> &shape) {
    std::cout << "[Random Trace] TiledRandomBuildIn() called" << std::endl;
    std::cout << "[Random Trace]   key: " << key << std::endl;
    std::cout << "[Random Trace]   counter0: " << counter0 << ", counter1: " << counter1 << std::endl;
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
        
        uint64_t tileCounter0 = counter0;
        uint64_t tileCounter1 = counter1;
        SkipCounter(tileCounter0, tileCounter1, static_cast<uint64_t>(i));
        
        auto &op = function.AddOperation(Opcode::OP_RANDOM, {}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "KEY", Element(DT_UINT64, key));
        op.SetAttribute(OP_ATTR_PREFIX + "COUNTER0", Element(DT_UINT64, tileCounter0));
        op.SetAttribute(OP_ATTR_PREFIX + "COUNTER1", Element(DT_UINT64, tileCounter1));
        op.SetAttribute(OP_ATTR_PREFIX + "ROUNDS", Element(DT_UINT16, rounds));
        op.SetAttribute(OP_ATTR_PREFIX + "SHAPE", shape);
    }
}

void RandomOperationTileFunc(Function &function, const TileShape &tileShape,
    [[maybe_unused]] const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op) {
    std::cout << "[Random Trace] RandomOperationTileFunc() called" << std::endl;
    
    uint64_t key = 0;
    if (op.HasAttr(OP_ATTR_PREFIX + "KEY")) {
        auto keyAttr = op.GetAttribute(OP_ATTR_PREFIX + "KEY");
        if (keyAttr.HasValue()) {
            key = AnyCast<Element>(keyAttr).Cast<uint64_t>();
        }
    }
    
    uint64_t counter0 = 0;
    if (op.HasAttr(OP_ATTR_PREFIX + "COUNTER0")) {
        auto counter0Attr = op.GetAttribute(OP_ATTR_PREFIX + "COUNTER0");
        if (counter0Attr.HasValue()) {
            counter0 = AnyCast<Element>(counter0Attr).Cast<uint64_t>();
        }
    }
    
    uint64_t counter1 = 0;
    if (op.HasAttr(OP_ATTR_PREFIX + "COUNTER1")) {
        auto counter1Attr = op.GetAttribute(OP_ATTR_PREFIX + "COUNTER1");
        if (counter1Attr.HasValue()) {
            counter1 = AnyCast<Element>(counter1Attr).Cast<uint64_t>();
        }
    }
    
    uint16_t rounds = 10;
    if (op.HasAttr(OP_ATTR_PREFIX + "ROUNDS")) {
        auto roundsAttr = op.GetAttribute(OP_ATTR_PREFIX + "ROUNDS");
        if (roundsAttr.HasValue()) {
            rounds = AnyCast<Element>(roundsAttr).Cast<uint16_t>();
        }
    }
    
    auto shapeAttr = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "SHAPE");
    std::vector<int64_t> shape;
    for (auto dim : shapeAttr) {
        shape.push_back(static_cast<int64_t>(dim));
    }
    
    TileInfo resultTileInfo(shape.size(), shape.size());
    TiledRandomBuildIn(function, tileShape, oOperand[0], resultTileInfo, key, counter0, counter1, rounds, shape);
}

static Tensor RealRandom(uint64_t key, uint64_t counter0, uint64_t counter1,
    const std::vector<int64_t> &shape, uint16_t rounds) {
    std::cout << "[Random Trace] RealRandom() called" << std::endl;
    std::cout << "[Random Trace]   key: " << key << std::endl;
    std::cout << "[Random Trace]   counter0: " << counter0 << ", counter1: " << counter1 << std::endl;
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
    RETURN_CALL(Random, *Program::GetInstance().GetCurrentFunction(), resTensor.GetStorage(), 
                Element(DT_UINT64, key), Element(DT_UINT64, counter0), Element(DT_UINT64, counter1), 
                Element(DT_UINT16, rounds), shape);
}

Tensor Random(const Element &key, const Element &counter0, const Element &counter1,
    const std::vector<int64_t> &shape, const Element &rounds) {
    std::cout << "[Random Trace] Random() called" << std::endl;
    std::cout << "[Random Trace]   key: " << key.Cast<uint64_t>() << std::endl;
    std::cout << "[Random Trace]   counter0: " << counter0.Cast<uint64_t>() << ", counter1: " << counter1.Cast<uint64_t>() << std::endl;
    std::cout << "[Random Trace]   rounds: " << rounds.Cast<uint16_t>() << std::endl;
    std::cout << "[Random Trace]   shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    
    uint16_t roundsVal = rounds.Cast<uint16_t>();
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, shape.size() == 1)
        << "Random: shape must be 1-dimensional";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, roundsVal == 7 || roundsVal == 10)
        << "Random: rounds must be 7 or 10";
    
    return RealRandom(key.Cast<uint64_t>(), counter0.Cast<uint64_t>(), counter1.Cast<uint64_t>(), shape, roundsVal);
}

REGISTER_OPERATION_TILED_FUNC(OP_RANDOM, Opcode::OP_RANDOM, RandomOperationTileFunc);

}  // namespace npu::tile_fwk
