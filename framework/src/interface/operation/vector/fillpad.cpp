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

void TiledFillPadOperation(
    Function &function, const TileShape &tileShape, size_t cur, Input &input, const LogicalTensorPtr &result, const Element &padValue) {
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_FILLPAD, {tile}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, padValue);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledFillPadOperation(function, tileShape, cur + 1, input, result, padValue);
    }
}

void TiledFillPadOperation(
    Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand, const LogicalTensorPtr &result, const Element &padValue) {
    ASSERT(operand->shape.size() == operand->offset.size()) << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledFillPadOperation(function, tileShape, 0, input, result, padValue);
}

LogicalTensorPtr TensorFillPadOperation(
    Function &function, const Tensor &self, const std::string &mode, float value) {
    ASSERT(mode == "constant") << "FillPad: only 'constant' mode is supported.";
    ASSERT(std::isinf(value) || (std::abs(value) < 1e-6)) << "FillPad: pad value must be -inf, inf, or 0.";

    auto operand = self.GetStorage();
    std::vector<int64_t> outputShape = operand->shape;
    std::vector<SymbolicScalar> resultValidShape = operand->GetDynValidShape();
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
    return TiledFillPadOperation(function, tileShape, iOperand[0], oOperand[0], padValue);
}

REGISTER_OPERATION_TILED_FUNC(OP_FILLPAD, Opcode::OP_FILLPAD, FillPadOperationTileFunc);

} // namespace npu::tile_fwk
