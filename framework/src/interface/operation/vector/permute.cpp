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
    auto result = std::make_shared<LogicalTensor>(function, self->tensor->datatype, resultShape, resultValidShape);
    result->CopyMemoryType(self);
    return result;
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

Tensor TensorPermuteMoveOutOperation(Function& function, LogicalTensorPtr self, const std::vector<int>& perm)
{
    auto result = MakePermutedLogicalTensor(function, self, perm);
    auto& op = function.AddOperation(Opcode::OP_PERMUTE_MOVEOUT, {self}, {result});
    op.SetAttribute(OpAttributeKey::perm, perm);
    function.UpdateTensorDataUsage(op);
    return result;
}

void TiledPermuteMoveOutOperation(
    Function& function, const TileShape& tileShape, size_t cur, Input& input, const LogicalTensorPtr& result,
    const std::vector<int>& perm)
{
    int shapeSize = static_cast<int>(input.tensor.GetShape().size());
    if (cur == static_cast<size_t>(shapeSize)) {
        auto srcTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTileShape = PermuteTileVector(input.tileInfo.shape, perm);
        auto resultTileOffset = PermuteTileVector(input.tileInfo.offset, perm);
        auto resultTile = result->View(function, resultTileShape, resultTileOffset);

        auto& op = function.AddOperation(Opcode::OP_PERMUTE_MOVEOUT, {srcTile}, {resultTile});
        op.SetAttribute(OpAttributeKey::perm, perm);
        op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledPermuteMoveOutOperation(function, tileShape, cur + 1, input, result, perm);
    }
}

void PermuteMoveOutOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    PermuteOperationOperandCheck(iOperand, oOperand);

    std::vector<int> perm = op.GetVectorIntAttribute<int>(OpAttributeKey::perm);

    TileInfo tileInfo(iOperand[0]->shape.size(), iOperand[0]->offset.size());
    Input input{iOperand[0], tileInfo};
    TiledPermuteMoveOutOperation(function, tileShape, 0, input, oOperand[0], perm);
}

static void ComputeStep1Step3Perm(
    const std::vector<int>& perm, std::vector<int>& step1Perm, std::vector<int>& step3Perm)
{
    int n = static_cast<int>(perm.size());
    int tailInputAxis = n - 1;
    int tailOutputAxis = perm[n - 1];

    step1Perm.resize(n);
    std::vector<bool> used(n, false);

    used[tailInputAxis] = true;
    step1Perm[n - 1] = tailInputAxis;

    used[tailOutputAxis] = true;
    step1Perm[n - 2] = tailOutputAxis;

    std::vector<int> remaining;
    for (int p : perm) {
        if (p != tailOutputAxis && p != tailInputAxis) {
            remaining.push_back(p);
        }
    }

    int rIdx = 0;
    for (int i = 0; i < n - 2; ++i) {
        step1Perm[i] = remaining[rIdx++];
    }

    std::vector<int> step2Result = step1Perm;
    std::swap(step2Result[n - 2], step2Result[n - 1]);

    step3Perm.resize(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (step2Result[j] == perm[i]) {
                step3Perm[i] = j;
                break;
            }
        }
    }
}

static Tensor TensorTailAxisPermute(
    Function& function, const LogicalTensorPtr& selfLogical, const std::vector<int>& perm)
{
    auto resultLogical = MakePermutedLogicalTensor(function, selfLogical, perm);
    resultLogical->isSubGraphBoundary = true;
    int n = static_cast<int>(perm.size());

    std::vector<int> step1Perm;
    std::vector<int> step3Perm;
    ComputeStep1Step3Perm(perm, step1Perm, step3Perm);

    std::vector<int64_t> step1Shape = PermuteResultShape(selfLogical->shape, step1Perm);
    constexpr int64_t UB_BLOCK_SIZE = 32;
    int64_t bytesPerElement = selfLogical->Datatype() == DT_FP32 ? 4 : 2;
    int64_t elementAlign = UB_BLOCK_SIZE / bytesPerElement;
    step1Shape[n - 2] = (step1Shape[n - 2] + elementAlign - 1) / elementAlign * elementAlign;
    auto ubBuf1 = std::make_shared<LogicalTensor>(
        function, selfLogical->Datatype(), step1Shape, SymbolicScalar::FromConcrete(step1Shape));
    ubBuf1->SetMemoryTypeBoth(MemoryType::MEM_UB);

    auto ubBuf2Shape = step1Shape;
    std::swap(ubBuf2Shape[n - 2], ubBuf2Shape[n - 1]);
    auto ubBuf2 = std::make_shared<LogicalTensor>(
        function, selfLogical->Datatype(), ubBuf2Shape, SymbolicScalar::FromConcrete(ubBuf2Shape));
    ubBuf2->SetMemoryTypeBoth(MemoryType::MEM_UB);

    auto tmpBufShape = ubBuf2Shape;
    tmpBufShape[n - 1] = (tmpBufShape[n - 1] + elementAlign - 1) / elementAlign * elementAlign;
    auto tmpBuf = std::make_shared<LogicalTensor>(
        function, selfLogical->Datatype(), tmpBufShape, SymbolicScalar::FromConcrete(tmpBufShape));
    tmpBuf->SetMemoryTypeBoth(MemoryType::MEM_UB);

    LogicalTensors outOp = {resultLogical, ubBuf1, ubBuf2, tmpBuf};
    auto& op = function.AddOperation(Opcode::OP_TAIL_AXIS_PERMUTE, {selfLogical}, outOp);
    std::vector<int64_t> step1Perm64(step1Perm.begin(), step1Perm.end());
    std::vector<int64_t> step3Perm64(step3Perm.begin(), step3Perm.end());
    op.SetAttribute(OP_ATTR_PREFIX + "step1Perm", step1Perm64);
    op.SetAttribute(OP_ATTR_PREFIX + "step3Perm", step3Perm64);
    op.SetAttribute(OP_ATTR_PREFIX + "dimCount", static_cast<int64_t>(n));
    op.SetAttribute(OpAttributeKey::perm, perm);
    op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultLogical->GetDynValidShape());
    function.UpdateTensorDataUsage(op);

    return Tensor(resultLogical);
}

namespace {
inline std::vector<int> VectorInt64ToInt(const std::vector<int64_t>& v)
{
    std::vector<int> result;
    result.reserve(v.size());
    for (auto val : v) {
        result.push_back(static_cast<int>(val));
    }
    return result;
}
} // anonymous namespace

void TiledTailAxisPermuteOperation(
    Function& function, const TileShape& tileShape, size_t cur, Input& input, const LogicalTensorPtr& result,
    const LogicalTensorPtr& ubBuf1, const LogicalTensorPtr& ubBuf2, const LogicalTensorPtr& tmpBuf,
    const std::vector<int64_t>& step1Perm, const std::vector<int64_t>& step3Perm, int64_t dimCount)
{
    int shapeSize = static_cast<int>(input.tensor.GetShape().size());
    if (cur == static_cast<size_t>(shapeSize)) {
        auto srcTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        srcTile->CopyMemoryType(input.tensor.GetStorage());
        auto step1PermInt = VectorInt64ToInt(step1Perm);
        auto step3PermInt = VectorInt64ToInt(step3Perm);
        auto step1TileShape = PermuteTileVector(input.tileInfo.shape, step1PermInt);
        auto step1TileOffset = PermuteTileVector(input.tileInfo.offset, step1PermInt);
        auto ubBuf1Tile = ubBuf1->View(function, step1TileShape, step1TileOffset);
        ubBuf1Tile->CopyMemoryType(ubBuf1);

        auto ubBuf2TileShape = step1TileShape;
        std::swap(ubBuf2TileShape[shapeSize - 2], ubBuf2TileShape[shapeSize - 1]);
        auto ubBuf2TileOffset = step1TileOffset;
        std::swap(ubBuf2TileOffset[shapeSize - 2], ubBuf2TileOffset[shapeSize - 1]);
        auto ubBuf2Tile = ubBuf2->View(function, ubBuf2TileShape, ubBuf2TileOffset);
        ubBuf2Tile->CopyMemoryType(ubBuf2);

        auto tmpBufTileShape = ubBuf2TileShape;
        auto tmpBufTile = tmpBuf->View(function, tmpBufTileShape, ubBuf2TileOffset);
        tmpBufTile->CopyMemoryType(tmpBuf);

        auto step2TileShape = step1TileShape;
        std::swap(step2TileShape[shapeSize - 2], step2TileShape[shapeSize - 1]);
        auto resultTileShape = PermuteTileVector(step2TileShape, step3PermInt);
        auto resultTileOffset = PermuteTileVector(ubBuf2TileOffset, step3PermInt);
        auto resultTile = result->View(function, resultTileShape, resultTileOffset);
        resultTile->CopyMemoryType(result);
        resultTile->isSubGraphBoundary = true;

        LogicalTensors outOp = {resultTile, ubBuf1Tile, ubBuf2Tile, tmpBufTile};
        auto& op = function.AddOperation(Opcode::OP_TAIL_AXIS_PERMUTE, {srcTile}, outOp);
        op.SetAttribute(OP_ATTR_PREFIX + "step1Perm", step1Perm);
        op.SetAttribute(OP_ATTR_PREFIX + "step3Perm", step3Perm);
        op.SetAttribute(OP_ATTR_PREFIX + "dimCount", dimCount);
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledTailAxisPermuteOperation(
            function, tileShape, cur + 1, input, result, ubBuf1, ubBuf2, tmpBuf, step1Perm, step3Perm, dimCount);
    }
}

void TailAxisPermuteOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    PermuteOperationOperandCheck({iOperand[0]}, {oOperand[0]});

    auto step1Perm = op.GetVectorIntAttribute<int64_t>(OP_ATTR_PREFIX + "step1Perm");
    auto step3Perm = op.GetVectorIntAttribute<int64_t>(OP_ATTR_PREFIX + "step3Perm");
    int64_t dimCount = 0;
    op.GetAttr(OP_ATTR_PREFIX + "dimCount", dimCount);

    TileInfo tileInfo(iOperand[0]->shape.size(), iOperand[0]->offset.size());
    Input input{iOperand[0], tileInfo};
    TiledTailAxisPermuteOperation(
        function, tileShape, 0, input, oOperand[0], oOperand[1], oOperand[2], oOperand[3], step1Perm, step3Perm,
        dimCount);
}

Tensor Permute(const Tensor& self, std::vector<int> perm)
{
    DECLARE_TRACER();
    CheckTensorShapeSize(self.GetStorage(), "PERMUTE");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "PERMUTE");
    CheckTensorDimRange(self.GetStorage(), 1, 5, "PERMUTE");

    const int shapeSize = static_cast<int>(self.GetShape().size());

    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, perm.size() == static_cast<size_t>(shapeSize))
        << "Permute dim num should match input dim num. Expected: " << shapeSize << ", Got: " << perm.size();

    if (shapeSize == 1) {
        return self;
    }

    NormalizePermutation(perm, shapeSize);
    ValidatePermutation(perm, shapeSize);

    if (IsIdentityPermutation(perm)) {
        return self;
    }

    bool lastAxisInvolved = (perm[shapeSize - 1] != shapeSize - 1);
    if (lastAxisInvolved) {
        return TensorTailAxisPermute(*Program::GetInstance().GetCurrentFunction(), self.GetStorage(), perm);
    }

    RETURN_CALL(PermuteOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), perm);
}

REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE, Opcode::OP_PERMUTE, PermuteOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE_ELEMENT, Opcode::OP_PERMUTE_ELEMENT, PermuteElementOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE_MOVEOUT, Opcode::OP_PERMUTE_MOVEOUT, PermuteMoveOutOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TAIL_AXIS_PERMUTE, Opcode::OP_TAIL_AXIS_PERMUTE, TailAxisPermuteOperationTileFunc);

} // namespace npu::tile_fwk
