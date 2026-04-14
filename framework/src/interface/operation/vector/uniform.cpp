/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file uniform.cpp
 * \brief Uniform random number generator implementation
 */

#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/utils/vector_error.h"
#include "interface/operation/operation_common.h"

namespace npu::tile_fwk {

LogicalTensorPtr TensorUniform(Function &function, LogicalTensorPtr &result, const Element &key,
    const SymbolicScalar& counter0, const Element &counter1, const Element &rounds, const std::vector<int64_t> &shape,
    DataType dtype) {
    auto &op = function.AddOperation(Opcode::OP_UNIFORM, {}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "KEY", key);
    op.SetAttribute(OpAttributeKey::dynScalar, counter0);
    op.SetAttribute(OP_ATTR_PREFIX + "COUNTER1", counter1);
    op.SetAttribute(OP_ATTR_PREFIX + "ROUNDS", rounds);
    op.SetAttribute(OP_ATTR_PREFIX + "SHAPE", shape);
    op.SetAttribute(OP_ATTR_PREFIX + "DTYPE", Element(DT_INT32, static_cast<int32_t>(dtype)));
    return result;
}

static void TiledUniformBuildIn(Function &function, const TileShape &tileShape, const LogicalTensorPtr &result, 
    TileInfo &resultTileInfo, uint64_t key, const SymbolicScalar& counter0, uint64_t counter1, uint16_t rounds, 
    DataType dtype) {
    auto &vecTile = tileShape.GetVecTile();
    
    const size_t ALIGN_SIZE = 32;
    int64_t uint32BufferSize = vecTile[0];
    int64_t uint32BufferBytes = ((uint32BufferSize * BytesOf(DT_UINT32) + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
    int64_t tempByteSize = uint32BufferBytes;
    
    if (dtype == DT_FP16 || dtype == DT_BF16) {
        int64_t uint32BufferLowBytes = ((uint32BufferSize * BytesOf(DT_UINT32) + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        tempByteSize += uint32BufferLowBytes;
    }
    
    for (int64_t i = 0; i < result->shape[0]; i += vecTile[0]) {
        resultTileInfo.offset[0] = i;
        resultTileInfo.shape[0] = std::min(result->shape[0] - resultTileInfo.offset[0], vecTile[0]);
        
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        
        std::vector<int64_t> tempShape({static_cast<int64_t>(tempByteSize)});
        auto tempTensor = std::make_shared<LogicalTensor>(function, DT_UINT8, tempShape);
        
        SymbolicScalar tileCounter0 = counter0 + i / 4;
        uint64_t tileCounter1 = counter1;
        
        auto &op = function.AddOperation(Opcode::OP_UNIFORM, {}, {resultTile, tempTensor});
        op.SetAttribute(OP_ATTR_PREFIX + "KEY", Element(DT_UINT64, key));
        op.SetAttribute(OpAttributeKey::dynScalar, tileCounter0);
        op.SetAttribute(OP_ATTR_PREFIX + "COUNTER1", Element(DT_UINT64, tileCounter1));
        op.SetAttribute(OP_ATTR_PREFIX + "ROUNDS", Element(DT_UINT16, rounds));
        op.SetAttribute(OP_ATTR_PREFIX + "SHAPE", resultTileInfo.shape);
        op.SetAttribute(OP_ATTR_PREFIX + "DTYPE", Element(DT_INT32, static_cast<int32_t>(dtype)));
    }
}

void UniformOperationTileFunc(Function &function, const TileShape &tileShape,
    [[maybe_unused]] const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op) {
    auto &vecTile = tileShape.GetVecTile();
    for (size_t i = 0; i < vecTile.size(); ++i) {
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, vecTile[i] % 4 == 0)
            << "Uniform: tileShape[" << i << "] must be a multiple of 4, but got " << vecTile[i];
    }
    
    uint64_t key = 0;
    if (op.HasAttr(OP_ATTR_PREFIX + "KEY")) {
        key = op.GetElementAttribute(OP_ATTR_PREFIX + "KEY").Cast<uint64_t>();
    }
    
    SymbolicScalar counter0 = op.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
    
    uint64_t counter1 = 0;
    if (op.HasAttr(OP_ATTR_PREFIX + "COUNTER1")) {
        counter1 = op.GetElementAttribute(OP_ATTR_PREFIX + "COUNTER1").Cast<uint64_t>();
    }
    
    uint16_t rounds = 10;
    if (op.HasAttr(OP_ATTR_PREFIX + "ROUNDS")) {
        rounds = op.GetElementAttribute(OP_ATTR_PREFIX + "ROUNDS").Cast<uint16_t>();
    }
    
    DataType dtype = DT_FP32;
    if (op.HasAttr(OP_ATTR_PREFIX + "DTYPE")) {
        dtype = static_cast<DataType>(op.GetElementAttribute(OP_ATTR_PREFIX + "DTYPE").Cast<int32_t>());
    }
    
    auto shapeAttr = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "SHAPE");
    std::vector<int64_t> shape;
    for (auto dim : shapeAttr) {
        shape.push_back(static_cast<int64_t>(dim));
    }
    
    TileInfo resultTileInfo(shape.size(), shape.size());
    TiledUniformBuildIn(function, tileShape, oOperand[0], resultTileInfo, key, counter0, counter1, rounds, dtype);
}

static Tensor RealUniform(uint64_t key, const SymbolicScalar& counter0, uint64_t counter1,
    const std::vector<int64_t> &shape, uint16_t rounds, DataType dtype) {
    DECLARE_TRACER();
    
    auto resTensor = Tensor(dtype, shape);
    RETURN_CALL(Uniform, *Program::GetInstance().GetCurrentFunction(), resTensor.GetStorage(), 
                Element(DT_UINT64, key), counter0, Element(DT_UINT64, counter1), 
                Element(DT_UINT16, rounds), shape, dtype);
}

Tensor Uniform(const Element &key, const SymbolicScalar& counter0, const Element &counter1,
    const std::vector<int64_t> &shape, const Element &rounds, DataType dtype) {
    uint16_t roundsVal = rounds.Cast<uint16_t>();
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, shape.size() == 1)
        << "Uniform: shape must be 1-dimensional";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, roundsVal == 7 || roundsVal == 10)
        << "Uniform: rounds must be 7 or 10";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BF16)
        << "Uniform: dtype must be DT_FP32, DT_FP16 or DT_BF16";
    
    return RealUniform(key.Cast<uint64_t>(), counter0, counter1.Cast<uint64_t>(), shape, roundsVal, dtype);
}

REGISTER_OPERATION_TILED_FUNC(OP_UNIFORM, Opcode::OP_UNIFORM, UniformOperationTileFunc);

}  // namespace npu::tile_fwk
