/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file permute.cpp
 * \brief
 */

#include "unary.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/operator_tracer.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {

LogicalTensorPtr TensorPermuteOperation(Function &function, const LogicalTensorPtr &self, const std::vector<int> &dims);
Tensor Permute(const Tensor &self, const std::vector<int> &dims);

static void ValidatePermuteDims(const Tensor &self, const std::vector<int> &dims) {
    ASSERT(!dims.empty()) << "Permute dims cannot be empty";
    ASSERT(dims.size() == self.GetShape().size()) << "Permute dims size must match tensor dimension size";

    std::vector<bool> used(dims.size(), false);
    for (auto dim : dims) {
        int normalizedDim = dim < 0 ? dim + dims.size() : dim;
        ASSERT(normalizedDim >= 0 && normalizedDim < static_cast<int>(dims.size()))
            << "Permute dim " << dim << " is out of range";
        ASSERT(!used[normalizedDim]) << "Permute dim " << dim << " is duplicated";
        used[normalizedDim] = true;
    }
}

static std::vector<int64_t> ComputeOutputShape(const Tensor &self, const std::vector<int> &dims) {
    std::vector<int64_t> outputShape;
    outputShape.reserve(dims.size());
    for (auto dim : dims) {
        int normalizedDim = dim < 0 ? dim + dims.size() : dim;
        outputShape.push_back(self.GetShape()[normalizedDim]);
    }
    return outputShape;
}

static std::vector<SymbolicScalar> ComputeOutputValidShape(const Tensor &self, const std::vector<int> &dims) {
    auto inputValidShape = self.GetStorage()->GetDynValidShape();
    if (inputValidShape.empty()) {
        inputValidShape = SymbolicScalar::FromConcrete(self.GetShape());
    }

    std::vector<SymbolicScalar> outputValidShape;
    outputValidShape.reserve(dims.size());
    for (auto dim : dims) {
        int normalizedDim = dim < 0 ? dim + dims.size() : dim;
        outputValidShape.push_back(inputValidShape[normalizedDim]);
    }
    return outputValidShape;
}

void TiledPermute(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, const std::vector<int> &dims) {
    if (cur == result->shape.size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        auto &op = function.AddOperation("TILE_REGISTER_COPY", {inputTile}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "dims", dims);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(result->shape[cur] - i, static_cast<int64_t>(vecTile[cur]));
        input.tileInfo.offset[cur] = i;
        TiledPermute(function, tileShape, cur + 1, input, result, dims);
    }
}

void TiledPermute(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const std::vector<int> &dims) {
    ASSERT(operand->shape.size() == operand->offset.size()) << "The shape size of operand and offset should be equal";
    ASSERT(operand->shape.size() == dims.size()) << "The dims size must match tensor dimension size";

    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    auto input = Input{operand, tileInfo};
    TiledPermute(function, tileShape, 0, input, result, dims);
}

LogicalTensorPtr TensorPermuteOperation(
    Function &function, const LogicalTensorPtr &self, const std::vector<int> &dims) {
    ASSERT(!dims.empty()) << "Permute dims cannot be empty";
    ASSERT(dims.size() == self->shape.size()) << "Permute dims size must match tensor dimension size";

    std::vector<int64_t> outputShape;
    outputShape.reserve(dims.size());
    for (auto dim : dims) {
        int normalizedDim = dim < 0 ? dim + dims.size() : dim;
        ASSERT(normalizedDim >= 0 && normalizedDim < static_cast<int>(dims.size()))
            << "Permute dim " << dim << " is out of range";
        outputShape.push_back(self->shape[normalizedDim]);
    }

    auto outputValidShape = ComputeOutputValidShape(Tensor(self), dims);
    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), outputShape, outputValidShape);

    auto &op = function.AddOperation(Opcode::OP_PERMUTE, {self}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "dims", dims);
    function.UpdateTensorDataUsage(op);

    return result;
}

Tensor Permute(const Tensor &self, const std::vector<int> &dims) {
    DECLARE_TRACER();
    ValidatePermuteDims(self, dims);

    auto outputShape = ComputeOutputShape(self, dims);
    auto outputValidShape = ComputeOutputValidShape(self, dims);

    Tensor result(self.GetStorage()->tensor->datatype, outputShape);
    result.GetStorage()->UpdateDynValidShape(outputValidShape);

    CALL(PermuteOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), dims);

    return result;
}

void PermuteOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto dims = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "dims");
    TiledPermute(function, tileShape, iOperand[0], oOperand[0], dims);
}

REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE, Opcode::OP_PERMUTE, PermuteOperationTileFunc);

} // namespace npu::tile_fwk