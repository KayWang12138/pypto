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

/*
有效区域：[0, inputValidShape)
tile 范围：[offset, offset + tileSize)

判断逻辑：
1. offset >= inputValidShape
   → tile 完全在有效区域右侧/下侧
   
2. offset + tileSize > inputValidShape
   → tile 跨越边界（部分在内，部分在外）
   
3. offset < inputValidShape 且 offset + tileSize <= inputValidShape
   → tile 完全在有效区域内
*/

void TiledFillPadImpl(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, const Element &padValue,
    const std::vector<SymbolicScalar> &inputValidShape) {
    size_t ndim = result->shape.size();
    auto &vecTile = tileShape.GetVecTile();
    if (cur == ndim) {
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        
        if (ndim == 1) {
            auto lastInputShape = inputValidShape[0];//第 0 维（唯一维度）的有效数据大小
            auto lastResultShape = result->shape[0];//输出 tensor 第 0 维的总大小
            auto lastShape = input.tileInfo.shape[0];//当前 tile 在第 0 维的大小
            auto lastOffset = input.tileInfo.offset[0];//当前 tile 在第 0 维的起始位置
            
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
            auto lastInputShape = inputValidShape[ndim - 1];//最后一维（列）的有效数据大小
            auto lastResultShape = result->shape[ndim - 1];//输出 tensor 最后一维（列）的总大小
            auto lastShape = input.tileInfo.shape[ndim - 1];
            auto lastOffset = input.tileInfo.offset[ndim - 1];
            auto preInputShape = inputValidShape[ndim - 2];
            auto preResultShape = result->shape[ndim - 2];
            auto preShape = input.tileInfo.shape[ndim - 2];
            auto preOffset = input.tileInfo.offset[ndim - 2];
            
            if (lastOffset >= lastInputShape || preOffset >= preInputShape) {//完全在有效区域外
                auto &op = function.AddOperation("TILE_VEC_DUP", {}, {resultTile});
                op.SetAttribute(OpAttributeKey::scalar, padValue);
                op.SetAttribute(OP_ATTR_PREFIX + "shape", resultTileInfo.shape);
                op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
            } else if (lastOffset + vecTile[ndim - 1] > lastInputShape ||    //行跨越边界 || 列跨越边界
                       preOffset + vecTile[ndim - 2] > preInputShape) {
                auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
                auto &op = function.AddOperation(Opcode::OP_FILLPAD, {inputTile}, {resultTile});
                op.SetAttribute(OpAttributeKey::scalar, padValue);
            } else {  //// 完全在有效区域内
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
        inputValidShape.resize(ndim);  //如果输入tensor没有动态有效形状  那么假设整个tensor都是有效数据
        for (size_t i = 0; i < ndim; i++) {
            inputValidShape[i] = oOperand[0]->shape[i];
        }
    }
    
    TiledFillPadOperation(function, tileShape, iOperand[0], oOperand[0], padValue, inputValidShape);
}

REGISTER_OPERATION_TILED_FUNC(OP_FILLPAD, Opcode::OP_FILLPAD, FillPadOperationTileFunc);

} // namespace npu::tile_fwk
