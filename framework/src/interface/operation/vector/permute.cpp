/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "permute.h"
#include "tensor_transformation.h"
#include <algorithm>
#include <sstream>

namespace npu::tile_fwk {

namespace {

std::vector<int64_t> PermuteTileVector(const std::vector<int64_t> &values, const std::vector<int> &perm) {
    std::vector<int64_t> result;
    result.reserve(perm.size());
    for (int axis : perm) {
        result.push_back(values[axis]);
    }
    return result;
}

[[maybe_unused]] std::vector<SymbolicScalar> PermuteTileVector(
    const std::vector<SymbolicScalar> &values, const std::vector<int> &perm) {
    std::vector<SymbolicScalar> result;
    result.reserve(perm.size());
    for (int axis : perm) {
        result.push_back(values[axis]);
    }
    return result;
}

}

void PermuteOperationOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand) {
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 1) << "Permute input operand count should be 1";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 1) << "Permute output operand count should be 1";
}

std::vector<int64_t> PermuteResultShape(const std::vector<int64_t> &inputShape, const std::vector<int> &perm) {
    std::vector<int64_t> resultShape;
    resultShape.reserve(perm.size());
    for (int p : perm) {
        resultShape.push_back(inputShape[p]);
    }
    return resultShape;
}

bool IsIdentityPermutation(const std::vector<int> &perm) {
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

void NormalizePermutation(std::vector<int> &perm, int shapeSize) {
    for (int &p : perm) {
        if (p < 0) {
            p += shapeSize;
        }
    }
}

void ValidatePermutation(const std::vector<int> &perm, int shapeSize) {
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

void TiledPermuteOperation(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, const std::vector<int> &perm);

LogicalTensorPtr TensorPermuteOperation(Function &function, LogicalTensorPtr self, const std::vector<int> &perm,
    const std::vector<int64_t> &vecTileOverride, bool hasTransposeLastAxis) {
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

    auto &op = function.AddOperation(Opcode::OP_PERMUTE, {self}, {result});

    int axis0 = perm.size() > 0 ? perm[0] : -1;
    int axis1 = perm.size() > 1 ? perm[1] : -1;
    int axis2 = perm.size() > 2 ? perm[2] : -1;
    int axis3 = perm.size() > 3 ? perm[3] : -1;
    int axis4 = perm.size() > 4 ? perm[4] : -1;
    int dimCount = static_cast<int>(perm.size());

    op.SetAttribute(OP_ATTR_PREFIX + "axis0", axis0);
    op.SetAttribute(OP_ATTR_PREFIX + "axis1", axis1);
    op.SetAttribute(OP_ATTR_PREFIX + "axis2", axis2);
    op.SetAttribute(OP_ATTR_PREFIX + "axis3", axis3);
    op.SetAttribute(OP_ATTR_PREFIX + "axis4", axis4);
    op.SetAttribute(OP_ATTR_PREFIX + "dimCount", dimCount);

    if (hasTransposeLastAxis && !vecTileOverride.empty()) {
        op.SetAttribute(OP_ATTR_PREFIX + "vecTileOverride", vecTileOverride);
    }

    return result;
}

void TiledPermuteOperation(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, const std::vector<int> &perm) {
    int shapeSize = static_cast<int>(input.tensor.GetShape().size());

    if (cur == static_cast<size_t>(shapeSize)) {
        auto srcTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTileShape = PermuteTileVector(input.tileInfo.shape, perm);
        auto resultTileOffset = PermuteTileVector(input.tileInfo.offset, perm);
        auto resultTile = result->View(function, resultTileShape, resultTileOffset);
        
        auto &op = function.AddOperation(Opcode::OP_PERMUTE, {srcTile}, {resultTile});

        int axis0 = perm.size() > 0 ? perm[0] : -1;
        int axis1 = perm.size() > 1 ? perm[1] : -1;
        int axis2 = perm.size() > 2 ? perm[2] : -1;
        int axis3 = perm.size() > 3 ? perm[3] : -1;
        int axis4 = perm.size() > 4 ? perm[4] : -1;
        int dimCount = static_cast<int>(perm.size());

        op.SetAttribute(OP_ATTR_PREFIX + "axis0", axis0);
        op.SetAttribute(OP_ATTR_PREFIX + "axis1", axis1);
        op.SetAttribute(OP_ATTR_PREFIX + "axis2", axis2);
        op.SetAttribute(OP_ATTR_PREFIX + "axis3", axis3);
        op.SetAttribute(OP_ATTR_PREFIX + "axis4", axis4);
        op.SetAttribute(OP_ATTR_PREFIX + "dimCount", dimCount);
        op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledPermuteOperation(function, tileShape, cur + 1, input, result, perm);
    }
}

void PermuteOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    PermuteOperationOperandCheck(iOperand, oOperand);

    int dimCount = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "dimCount"));
    int axis0 = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis0"));
    int axis1 = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis1"));
    int axis2 = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis2"));
    int axis3 = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis3"));
    int axis4 = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis4"));

    std::vector<int> perm;
    if (dimCount > 0)
        perm.push_back(axis0);
    if (dimCount > 1)
        perm.push_back(axis1);
    if (dimCount > 2)
        perm.push_back(axis2);
    if (dimCount > 3)
        perm.push_back(axis3);
    if (dimCount > 4)
        perm.push_back(axis4);

    std::vector<int64_t> vecTileOverride;
    bool hasOverride = op.GetAttr(OP_ATTR_PREFIX + "vecTileOverride", vecTileOverride);
    if (hasOverride && vecTileOverride.empty()) {
        hasOverride = false;
    }

    std::cerr << "[PERMUTE TILE] hasOverride=" << hasOverride
              << " tileShape.vec=";
    for (auto d : tileShape.GetVecTile().tile) std::cerr << d << ",";
    std::cerr << " override=";
    for (auto d : vecTileOverride) std::cerr << d << ",";
    std::cerr << " iOperand[0].shape=";
    for (auto d : iOperand[0]->shape) std::cerr << d << ",";
    std::cerr << " perm=";
    for (auto d : perm) std::cerr << d << ",";
    std::cerr << std::endl;

    TileShape useTileShape = tileShape;
    if (hasOverride) {
        useTileShape.SetVecTile(vecTileOverride);
        std::cerr << "[PERMUTE TILE] after override useTileShape.vec=";
        for (auto d : useTileShape.GetVecTile().tile) std::cerr << d << ",";
        std::cerr << std::endl;
    } else {
        std::cerr << "[PERMUTE TILE] NO override, using tileShape directly" << std::endl;
    }

    TileInfo tileInfo(iOperand[0]->shape.size(), iOperand[0]->offset.size());
    Input input{iOperand[0], tileInfo};
    TiledPermuteOperation(function, useTileShape, 0, input, oOperand[0], perm);
}

Tensor Permute(const Tensor &self, std::vector<int> perm) {
    DECLARE_TRACER();

    const int shapeSize = static_cast<int>(self.GetShape().size());

    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, perm.size() == static_cast<size_t>(shapeSize))
        << "Permute dim num should match input dim num. Expected: " << shapeSize << ", Got: " << perm.size();

    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, shapeSize >= 1 && shapeSize <= 5)
        << "Permute only supports 2D to 5D tensors. Got: " << shapeSize << "D";
    
    if (shapeSize == 1) {
        return self;
    }

    NormalizePermutation(perm, shapeSize);
    ValidatePermutation(perm, shapeSize);

    if (IsIdentityPermutation(perm)) {
        return self;
    }

    if (shapeSize == 2) {
        return Transpose(self, {0, 1});
    }

    bool lastAxisInvolved = (perm[shapeSize - 1] != shapeSize - 1);

    if (lastAxisInvolved) {
        int innerAxis = perm[shapeSize - 1];

        Tensor transposed = Transpose(self, {innerAxis, shapeSize - 1});

        auto oldVecTileShapes = TileShape::Current().GetVecTile();

        std::cerr << "[PERMUTE] perm=";
        for (auto d : perm) std::cerr << d << ",";
        std::cerr << " innerAxis=" << innerAxis << " shapeSize=" << shapeSize;
        std::cerr << " oldVecTile=";
        for (auto d : oldVecTileShapes.tile) std::cerr << d << ",";
        std::cerr << std::endl;

        auto newVecTileShapes = oldVecTileShapes;
        std::swap(newVecTileShapes.tile[innerAxis], newVecTileShapes.tile[shapeSize - 1]);

        std::cerr << "[PERMUTE] newVecTile=";
        for (auto d : newVecTileShapes.tile) std::cerr << d << ",";
        std::cerr << std::endl;

        std::vector<int> newPerm(perm);
        for (auto &p : newPerm) {
            if (p == innerAxis) {
                p = shapeSize - 1;
            } else if (p == shapeSize - 1) {
                p = innerAxis;
            }
        }

        if (IsIdentityPermutation(newPerm)) {
            TileShape::Current().SetVecTile(oldVecTileShapes.tile);
            return transposed;
        }

        std::cerr << "[PERMUTE] newPerm=";
        for (auto d : newPerm) std::cerr << d << ",";
        std::cerr << std::endl;

        auto transposedStorage = transposed.GetStorage();
        auto result = Tensor(TensorPermuteOperation(
            *Program::GetInstance().GetCurrentFunction(), transposedStorage, newPerm,
            newVecTileShapes.tile, true));

        TileShape::Current().SetVecTile(oldVecTileShapes.tile);

        return result;
    }

    return Tensor(TensorPermuteOperation(
        *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), perm));
}

REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE, Opcode::OP_PERMUTE, PermuteOperationTileFunc);

} // namespace npu::tile_fwk
