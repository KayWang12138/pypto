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

void PermuteOperationOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand) {
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 1) << "Permute input operand count should be 1";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 1) << "Permute output operand count should be 1";
}

std::vector<int64_t> PermuteResultShape(
    const std::vector<int64_t> &inputShape, const std::vector<int> &perm) {
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
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, !used[p])
            << "Permute dims contain duplicate values at index " << p;
        used[p] = true;
    }
}

void TiledPermuteOperation(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, const std::vector<int> &perm);

LogicalTensorPtr TensorPermuteOperation(
    Function &function, LogicalTensorPtr self, const std::vector<int> &perm) {
    const int shapeSize = static_cast<int>(self->shape.size());
    
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
    
    return result;
}

void TiledPermuteOperation(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, const std::vector<int> &perm) {
    int shapeSize = static_cast<int>(input.tensor.GetShape().size());
    
    if (cur == static_cast<size_t>(shapeSize)) {
        auto srcTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        
        std::vector<int64_t> resultTileShape = input.tileInfo.shape;
        std::vector<int64_t> resultTileOffset = input.tileInfo.offset;
        
        auto &op = function.AddOperation(Opcode::OP_PERMUTE, {srcTile}, {result});
        
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
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    PermuteOperationOperandCheck(iOperand, oOperand);
    
    auto axis0Attr = op.GetAttribute(OP_ATTR_PREFIX + "axis0");
    auto axis1Attr = op.GetAttribute(OP_ATTR_PREFIX + "axis1");
    auto axis2Attr = op.GetAttribute(OP_ATTR_PREFIX + "axis2");
    auto axis3Attr = op.GetAttribute(OP_ATTR_PREFIX + "axis3");
    auto axis4Attr = op.GetAttribute(OP_ATTR_PREFIX + "axis4");
    auto dimCountAttr = op.GetAttribute(OP_ATTR_PREFIX + "dimCount");
    
    std::vector<int> perm;
    if (dimCountAttr.has_value()) {
        int dimCount = std::get<int64_t>(*dimCountAttr);
        if (dimCount > 0 && axis0Attr.has_value()) perm.push_back(std::get<int64_t>(*axis0Attr));
        if (dimCount > 1 && axis1Attr.has_value()) perm.push_back(std::get<int64_t>(*axis1Attr));
        if (dimCount > 2 && axis2Attr.has_value()) perm.push_back(std::get<int64_t>(*axis2Attr));
        if (dimCount > 3 && axis3Attr.has_value()) perm.push_back(std::get<int64_t>(*axis3Attr));
        if (dimCount > 4 && axis4Attr.has_value()) perm.push_back(std::get<int64_t>(*axis4Attr));
    }
    
    TileInfo tileInfo(iOperand[0]->shape.size(), iOperand[0]->offset.size());
    Input input{iOperand[0], tileInfo};
    TiledPermuteOperation(function, tileShape, 0, input, oOperand[0], perm);
}

Tensor Permute(const Tensor &self, std::vector<int> perm) {
    DECLARE_TRACER();
    
    const int shapeSize = static_cast<int>(self.GetShape().size());
    
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, perm.size() == static_cast<size_t>(shapeSize))
        << "Permute dim num should match input dim num. Expected: " << shapeSize << ", Got: " << perm.size();
    
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, shapeSize >= 2 && shapeSize <= 5)
        << "Permute only supports 2D to 5D tensors. Got: " << shapeSize << "D";
    
    NormalizePermutation(perm, shapeSize);
    ValidatePermutation(perm, shapeSize);
    
    if (IsIdentityPermutation(perm)) {
        return self;
    }
    
    if (shapeSize == 2 && perm[0] == 1 && perm[1] == 0) {
        return Transpose(self, {0, 1});
    }
    
    RETURN_CALL(TensorPermuteOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), perm);
}

REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE, Opcode::OP_PERMUTE, PermuteOperationTileFunc);

} // namespace npu::tile_fwk
