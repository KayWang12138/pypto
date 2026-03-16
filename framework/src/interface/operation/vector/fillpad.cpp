/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fillpad.cpp
 * \brief FillPad operator implementation
 */

#include <cmath>
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {

void TiledFillPadImpl(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, const Element &padValue,
    const std::vector<SymbolicScalar> &inputValidShape) {
    size_t ndim = result->shape.size();
    auto &vecTile = tileShape.GetVecTile();
    if (cur == ndim) {
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        
        if (ndim == 1) {
            auto lastInputShape = inputValidShape[0];
            auto lastResultShape = result->shape[0];
            auto lastShape = input.tileInfo.shape[0];
            auto lastOffset = input.tileInfo.offset[0];
            
            if (lastOffset >= lastInputShape) {
                auto &op = function.AddOperation("TILE_VEC_DUP", {}, {resultTile});
                op.SetAttribute(OpAttributeKey::scalar, padValue);
                op.SetAttribute(OP_ATTR_PREFIX + "shape", resultTileInfo.shape);
                op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
            } else if (lastOffset + vecTile[0] > lastInputShape) {
                auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
                auto &op = function.AddOperation(Opcode::OP_FILLPAD, {inputTile}, {resultTile});
                op.SetAttribute(OpAttributeKey::scalar, padValue);
            } else {
                auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
                function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {resultTile});
            }
        } else {
            auto lastInputShape = inputValidShape[ndim - 1];
            auto lastResultShape = result->shape[ndim - 1];
            auto lastShape = input.tileInfo.shape[ndim - 1];
            auto lastOffset = input.tileInfo.offset[ndim - 1];
            auto preInputShape = inputValidShape[ndim - 2];
            auto preResultShape = result->shape[ndim - 2];
            auto preShape = input.tileInfo.shape[ndim - 2];
            auto preOffset = input.tileInfo.offset[ndim - 2];
            
            if (lastOffset >= lastInputShape || preOffset >= preInputShape) {
                auto &op = function.AddOperation("TILE_VEC_DUP", {}, {resultTile});
                op.SetAttribute(OpAttributeKey::scalar, padValue);
                op.SetAttribute(OP_ATTR_PREFIX + "shape", resultTileInfo.shape);
                op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
            } else if (lastOffset + vecTile[ndim - 1] > lastInputShape ||
                       preOffset + vecTile[ndim - 2] > preInputShape) {
                auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
                auto &op = function.AddOperation(Opcode::OP_FILLPAD, {inputTile}, {resultTile});
                op.SetAttribute(OpAttributeKey::scalar, padValue);
            } else {
                auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
                function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {resultTile});
            }
        }
        return;
    }

    for (int64_t i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        input.tileInfo.shape[cur] = std::min(result->shape[cur] - i, vecTile[cur]);
        TiledFillPadImpl(function, tileShape, cur + 1, input, result, resultTileInfo, padValue, inputValidShape);
    }
}

void TiledFillPadOperation(Function &function, const TileShape &tileShape, const LogicalTensorPtr &input,
    const LogicalTensorPtr &result, const Element &padValue, const std::vector<SymbolicScalar> &inputValidShape) {
    size_t ndim = result->shape.size();
    TileInfo resultTileInfo(ndim, ndim);
    TileInfo inputTileInfo(ndim, ndim);
    Input fillPadInput{input, inputTileInfo};
    TiledFillPadImpl(function, tileShape, 0, fillPadInput, result, resultTileInfo, padValue, inputValidShape);
}

LogicalTensorPtr TensorFillPadOperation(
    Function &function, const Tensor &self, const std::string &mode, float value) {
    ASSERT(mode == "constant") << "FillPad: only 'constant' mode is supported.";
    ASSERT(std::isinf(value) || (std::abs(value) < 1e-6)) << "FillPad: pad value must be -inf, inf, or 0.";

    auto operand = self.GetStorage();
    std::vector<int64_t> outputShape = operand->shape;
    size_t ndim = operand->shape.size();
    
    std::vector<SymbolicScalar> resultValidShape;
    const auto &inputValidShape = operand->GetDynValidShape();
    if (!inputValidShape.empty()) {
        resultValidShape = inputValidShape;
    }
    
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), outputShape, resultValidShape);
    auto &op = function.AddOperation(Opcode::OP_FILLPAD, {operand}, {result});
    op.SetAttribute(OpAttributeKey::scalar, Element(self.GetDataType(), value));
    
    return result;
}

Tensor FillPad(const Tensor &self, std::string mode, float value) {
    DECLARE_TRACER();
    RETURN_CALL(FillPadOperation, *Program::GetInstance().GetCurrentFunction(), self, mode, value);
}

void FillPadOperationTileFunc(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    Element padValue = op.GetElementAttribute(OpAttributeKey::scalar);
    
    size_t ndim = oOperand[0]->shape.size();
    std::vector<SymbolicScalar> inputValidShape;
    const auto &dynValidShape = iOperand[0]->GetDynValidShape();
    
    if (!dynValidShape.empty()) {
        inputValidShape = dynValidShape;
    } else {
        inputValidShape.resize(ndim);
        for (size_t i = 0; i < ndim; i++) {
            inputValidShape[i] = oOperand[0]->shape[i];
        }
    }
    
    TiledFillPadOperation(function, tileShape, iOperand[0], oOperand[0], padValue, inputValidShape);
}

REGISTER_OPERATION_TILED_FUNC(OP_FILLPAD, Opcode::OP_FILLPAD, FillPadOperationTileFunc);

} // namespace npu::tile_fwk
