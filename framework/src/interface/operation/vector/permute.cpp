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
 * \file permute.cpp
 * \brief Permute operation implementation
 */

#include "interface/utils/operator_tracer.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "tensor_transformation.h"
#include "permute.h"
#include <algorithm>
#include <sstream>

namespace npu::tile_fwk {

std::vector<int64_t> PermuteTileVector(const std::vector<int64_t>& values, const std::vector<int>& perm)
{
    std::vector<int64_t> result;
    result.reserve(perm.size());
    for (int axis : perm) {
        result.push_back(values[axis]);
    }
    return result;
}

[[maybe_unused]] std::vector<SymbolicScalar> PermuteTileVector(
    const std::vector<SymbolicScalar>& values, const std::vector<int>& perm)
{
    std::vector<SymbolicScalar> result;
    result.reserve(perm.size());
    for (int axis : perm) {
        result.push_back(values[axis]);
    }
    return result;
}

void PermuteOperationOperandCheck(
    const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 1) << "Permute input operand count should be 1";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 1) << "Permute output operand count should be 1";
}

std::vector<int64_t> PermuteResultShape(const std::vector<int64_t>& inputShape, const std::vector<int>& perm)
{
    std::vector<int64_t> resultShape;
    resultShape.reserve(perm.size());
    for (int p : perm) {
        resultShape.push_back(inputShape[p]);
    }
    return resultShape;
}

bool IsIdentityPermutation(const std::vector<int>& perm)
{
    if (perm.size() <= 1) {
        return true;
    }
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int>(i)) {
            return false;
        }
    }
    return true;
}

void NormalizePermutation(std::vector<int>& perm, int shapeSize)
{
    for (int& p : perm) {
        if (p < 0) {
            p += shapeSize;
        }
    }
}

void ValidatePermutation(const std::vector<int>& perm, int shapeSize)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, perm.size() == static_cast<size_t>(shapeSize))
        << "Permute dim num should match input dim num. Expected: " << shapeSize << ", Got: " << perm.size();

    std::vector<bool> used(shapeSize, false);
    for (int p : perm) {
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, p >= 0 && p < shapeSize)
            << "Permute dim is invalid: " << p << ". Should be in range [0, " << shapeSize << ")";
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, !used[p]) << "Permute dims contain duplicate values at index " << p;
        used[p] = true;
    }
}

static LogicalTensorPtr MakePermutedLogicalTensor(
    Function& function, const LogicalTensorPtr& self, const std::vector<int>& perm)
{
    std::vector<int64_t> resultShape = PermuteResultShape(self->shape, perm);
    std::vector<SymbolicScalar> resultValidShape;
    if (!self->GetDynValidShape().empty()) {
        for (int p : perm) {
            resultValidShape.push_back(self->GetDynValidShape()[p]);
        }
    } else {
        resultValidShape = SymbolicScalar::FromConcrete(resultShape);
    }
    return std::make_shared<LogicalTensor>(function, self->tensor->datatype, resultShape, resultValidShape);
}

void TiledPermuteOperation(
    Function& function, const TileShape& tileShape, size_t cur, Input& input, const LogicalTensorPtr& result,
    const std::vector<int>& perm);

Tensor TensorPermuteOperation(Function& function, LogicalTensorPtr self, const std::vector<int>& perm)
{
    auto result = MakePermutedLogicalTensor(function, self, perm);
    auto& op = function.AddOperation(Opcode::OP_PERMUTE, {self}, {result});
    op.SetAttribute(OpAttributeKey::perm, perm);
    function.UpdateTensorDataUsage(op);
    return result;
}

void TiledPermuteOperation(
    Function& function, const TileShape& tileShape, size_t cur, Input& input, const LogicalTensorPtr& result,
    const std::vector<int>& perm)
{
    int shapeSize = static_cast<int>(input.tensor.GetShape().size());
    if (cur == static_cast<size_t>(shapeSize)) {
        auto srcTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTileShape = PermuteTileVector(input.tileInfo.shape, perm);
        auto resultTileOffset = PermuteTileVector(input.tileInfo.offset, perm);
        auto resultTile = result->View(function, resultTileShape, resultTileOffset);

        auto& op = function.AddOperation(Opcode::OP_PERMUTE, {srcTile}, {resultTile});
        op.SetAttribute(OpAttributeKey::perm, perm);
        op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledPermuteOperation(function, tileShape, cur + 1, input, result, perm);
    }
}

void PermuteOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    PermuteOperationOperandCheck(iOperand, oOperand);

    std::vector<int> perm = op.GetVectorIntAttribute<int>(OpAttributeKey::perm);

    TileInfo tileInfo(iOperand[0]->shape.size(), iOperand[0]->offset.size());
    Input input{iOperand[0], tileInfo};
    TiledPermuteOperation(function, tileShape, 0, input, oOperand[0], perm);
}

std::vector<int64_t> PermuteElementTileVector(const std::vector<int64_t>& values, const std::vector<int>& perm)
{
    std::vector<int64_t> result;
    result.reserve(perm.size());
    for (int axis : perm) {
        result.push_back(values[axis]);
    }
    return result;
}

Tensor TensorElementPermuteOperation(Function& function, LogicalTensorPtr self, const std::vector<int>& perm)
{
    auto result = MakePermutedLogicalTensor(function, self, perm);
    auto& op = function.AddOperation(Opcode::OP_PERMUTE_ELEMENT, {self}, {result});
    op.SetAttribute(OpAttributeKey::perm, perm);
    function.UpdateTensorDataUsage(op);
    return result;
}

void TiledPermuteElementOperation(
    Function& function, const TileShape& tileShape, size_t cur, Input& input, const LogicalTensorPtr& result,
    const std::vector<int>& perm)
{
    int shapeSize = static_cast<int>(input.tensor.GetShape().size());
    if (cur == static_cast<size_t>(shapeSize)) {
        auto srcTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTileShape = PermuteElementTileVector(input.tileInfo.shape, perm);
        auto resultTileOffset = PermuteElementTileVector(input.tileInfo.offset, perm);
        auto resultTile = result->View(function, resultTileShape, resultTileOffset);

        auto& op = function.AddOperation(Opcode::OP_PERMUTE_ELEMENT, {srcTile}, {resultTile});
        op.SetAttribute(OpAttributeKey::perm, perm);
        op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledPermuteElementOperation(function, tileShape, cur + 1, input, result, perm);
    }
}

void PermuteElementOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    PermuteOperationOperandCheck(iOperand, oOperand);

    std::vector<int> perm = op.GetVectorIntAttribute<int>(OpAttributeKey::perm);

    TileInfo tileInfo(iOperand[0]->shape.size(), iOperand[0]->offset.size());
    Input input{iOperand[0], tileInfo};
    TiledPermuteElementOperation(function, tileShape, 0, input, oOperand[0], perm);
}

int FindTargetPosition(const std::vector<int>& invPerm, int targetIndex, int startSearch)
{
    for (int j = startSearch; j < static_cast<int>(invPerm.size()); ++j) {
        if (invPerm[j] == targetIndex) {
            return j;
        }
    }
    return -1;
}

int CalculateTransposeCount(const std::vector<int>& perm)
{
    const int shapeSize = static_cast<int>(perm.size());
    std::vector<int> invPerm(shapeSize);
    for (int i = 0; i < shapeSize; ++i) {
        invPerm[perm[i]] = i;
    }
    int count = 0;
    for (int i = 0; i < shapeSize; ++i) {
        int targetPos = FindTargetPosition(invPerm, i, i);
        if (targetPos != i && targetPos != -1) {
            std::swap(invPerm[i], invPerm[targetPos]);
            count++;
        }
    }
    return count;
}

Tensor PermuteWithTranspose(const Tensor& self, const std::vector<int>& perm)
{
    const int shapeSize = static_cast<int>(perm.size());
    std::vector<int> invPerm(shapeSize);
    for (int i = 0; i < shapeSize; ++i) {
        invPerm[perm[i]] = i;
    }
    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    Tensor result = self;
    for (int i = 0; i < shapeSize; ++i) {
        int targetPos = FindTargetPosition(invPerm, i, i);
        if (targetPos != i && targetPos != -1) {
            result = Transpose(result, {i, targetPos});
            
            auto curVecTileShapes = TileShape::Current().GetVecTile();
            std::swap(curVecTileShapes[i], curVecTileShapes[targetPos]);
            TileShape::Current().SetVecTile(curVecTileShapes);
            
            std::swap(invPerm[i], invPerm[targetPos]);
        }
    }
    TileShape::Current().SetVecTile(oldVecTileShapes);
    return result;
}

Tensor PermuteWithCurrentStrategy(const Tensor& self, const std::vector<int>& perm)
{
    const int shapeSize = static_cast<int>(perm.size());
    bool lastAxisInvolved = (perm[shapeSize - 1] != shapeSize - 1);
    if (!lastAxisInvolved) {
        RETURN_CALL(PermuteOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), perm);
    }
    int tailOutputAxis = perm[shapeSize - 1];
    int tailInputAxis = shapeSize - 1;
    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    Tensor transposed = Transpose(self, {tailOutputAxis, tailInputAxis});
    auto newVecTileShapes = oldVecTileShapes;
    std::swap(newVecTileShapes[tailOutputAxis], newVecTileShapes[tailInputAxis]);
    TileShape::Current().SetVecTile(newVecTileShapes);
    std::vector<int> newPerm(perm);
    for (auto& p : newPerm) {
        if (p == tailOutputAxis) {
            p = tailInputAxis;
        } else if (p == tailInputAxis) {
            p = tailOutputAxis;
        }
    }
    if (IsIdentityPermutation(newPerm)) {
        TileShape::Current().SetVecTile(oldVecTileShapes);
        return transposed;
    }
    Tensor result = CALL(PermuteOperation, *Program::GetInstance().GetCurrentFunction(), 
                         transposed.GetStorage(), newPerm);
    TileShape::Current().SetVecTile(oldVecTileShapes);
    return result;
}

Tensor Permute(const Tensor& self, std::vector<int> perm)
{
    DECLARE_TRACER();
    const int shapeSize = static_cast<int>(self.GetShape().size());
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, perm.size() == static_cast<size_t>(shapeSize))
        << "Permute dim num should match input dim num. Expected: " << shapeSize << ", Got: " << perm.size();
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, shapeSize >= 1 && shapeSize <= 5)
        << "Permute only supports 1D to 5D tensors. Got: " << shapeSize << "D";

    if (IsIdentityPermutation(perm) || shapeSize == 1) {
        return self;
    }
    NormalizePermutation(perm, shapeSize);
    ValidatePermutation(perm, shapeSize);
    if (shapeSize <= 3) {
        return PermuteWithTranspose(self, perm);
    }
    int transposeCount = CalculateTransposeCount(perm);
    if (transposeCount == 1) {
        return PermuteWithTranspose(self, perm);
    }
    return PermuteWithCurrentStrategy(self, perm);
}

REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE, Opcode::OP_PERMUTE, PermuteOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE_ELEMENT, Opcode::OP_PERMUTE_ELEMENT, PermuteElementOperationTileFunc);

} // namespace npu::tile_fwk
