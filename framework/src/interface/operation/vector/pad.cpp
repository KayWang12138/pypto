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
 * \file pad.cpp
 * \brief
 */

#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {

struct PadInput {
    LogicalTensorPtr tensor;
    TileInfo tileInfo;
};

void TiledPadImpl(Function &function, const TileShape &tileShape, size_t cur, PadInput &input,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, int64_t padRight, int64_t padBottom,
    const Element &padValue) {
    size_t ndim = result->shape.size();

    if (cur == ndim) {
        int64_t srcValidRow = input.tileInfo.shape[ndim - 2];
        int64_t srcValidCol = input.tileInfo.shape[ndim - 1];

        std::vector<int64_t> viewShape = input.tileInfo.shape;
        viewShape[ndim - 2] = std::max(viewShape[ndim - 2], static_cast<int64_t>(1));
        viewShape[ndim - 1] = std::max(viewShape[ndim - 1], static_cast<int64_t>(1));

        auto inputTile = input.tensor->View(function, viewShape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        auto &op = function.AddOperation(Opcode::OP_PAD, {inputTile}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "src_valid_row", static_cast<int64_t>(srcValidRow));
        op.SetAttribute(OP_ATTR_PREFIX + "src_valid_col", static_cast<int64_t>(srcValidCol));
        op.SetAttribute(OpAttributeKey::scalar, padValue);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    int64_t step = vecTile[cur];
    int64_t inputDimSize = input.tensor->shape[cur];
    int64_t outputDimSize = result->shape[cur];

    for (int64_t i = 0; i < outputDimSize; i += step) {
        int64_t outTileSize = std::min(outputDimSize - i, step);
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = outTileSize;

        int64_t srcValid = std::max(static_cast<int64_t>(0), std::min(inputDimSize - i, outTileSize));
        if (srcValid > 0) {
            input.tileInfo.offset[cur] = i;
            input.tileInfo.shape[cur] = srcValid;
        } else {
            input.tileInfo.offset[cur] = 0;
            input.tileInfo.shape[cur] = 0;
        }
        TiledPadImpl(function, tileShape, cur + 1, input, result, resultTileInfo, padRight, padBottom, padValue);
    }
}

void TiledPadOperation(Function &function, const TileShape &tileShape, const LogicalTensorPtr &input,
    const LogicalTensorPtr &result, int64_t padRight, int64_t padBottom, const Element &padValue) {
    size_t ndim = result->shape.size();
    TileInfo resultTileInfo(ndim, ndim);
    TileInfo inputTileInfo(ndim, ndim);
    PadInput padInput{input, inputTileInfo};
    TiledPadImpl(function, tileShape, 0, padInput, result, resultTileInfo, padRight, padBottom, padValue);
}

LogicalTensorPtr TensorPadOperation(
    Function &function, const Tensor &self, const std::vector<int64_t> &padding, const std::string &mode, float value) {
    ASSERT(mode == "constant") << "Pad: only 'constant' mode is supported.";
    ASSERT(padding.size() == 4) << "Pad: only support last 2 axis pad.";
    ASSERT(padding[0] == 0 && padding[2] == 0) << "Pad: only support bottom and right pad.";

    auto operand = self.GetStorage();
    std::vector<int64_t> outputShape = operand->shape;
    size_t ndim = operand->shape.size();
    int64_t padRight = padding[1];
    int64_t padBottom = padding[3];
    outputShape[ndim - 1] += padRight;
    outputShape[ndim - 2] += padBottom;

    std::vector<SymbolicScalar> resultValidShape;
    const auto &inputValidShape = operand->GetDynValidShape();
    if (!inputValidShape.empty()) {
        resultValidShape = inputValidShape;
        resultValidShape[ndim - 1] = resultValidShape[ndim - 1] + padRight;
        resultValidShape[ndim - 2] = resultValidShape[ndim - 2] + padBottom;
    }
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), outputShape, resultValidShape);
    auto &op = function.AddOperation(Opcode::OP_PAD, {operand}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "pad_right", static_cast<int64_t>(padRight));
    op.SetAttribute(OP_ATTR_PREFIX + "pad_bottom", static_cast<int64_t>(padBottom));
    op.SetAttribute(OpAttributeKey::scalar, Element(self.GetDataType(), value));
    return result;
}

Tensor Pad(const Tensor &self, const std::vector<int64_t> &padding, std::string mode, float value) {
    DECLARE_TRACER();
    RETURN_CALL(PadOperation, *Program::GetInstance().GetCurrentFunction(), self, padding, mode, value);
}

void PadOperationTileFunc(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    int64_t padRight = op.GetIntAttribute(OP_ATTR_PREFIX + "pad_right");
    int64_t padBottom = op.GetIntAttribute(OP_ATTR_PREFIX + "pad_bottom");
    Element padValue = op.GetElementAttribute(OpAttributeKey::scalar);
    TiledPadOperation(function, tileShape, iOperand[0], oOperand[0], padRight, padBottom, padValue);
}

REGISTER_OPERATION_TILED_FUNC(OP_PAD, Opcode::OP_PAD, PadOperationTileFunc);

} // namespace npu::tile_fwk
