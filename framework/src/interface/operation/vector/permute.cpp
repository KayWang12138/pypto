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
 * \brief
 */

#include <cmath>
#include <algorithm>
#include <set>
#include "unary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"

namespace npu::tile_fwk {

static constexpr int8_t UB_MAX_BRW_NUM = 3;

struct UbPermInfo {
    std::vector<int64_t> perm;
    int8_t cnt = 0;
};

struct PermuteShapeInfo {
    std::vector<int64_t> shape;
    std::vector<int64_t> perm;
    std::vector<int64_t> reducedShape;
    std::vector<int64_t> reducedPerm;
    std::vector<std::set<int64_t>> origAxisMap;
};

void ReduceShapeForPermute(PermuteShapeInfo &shapeInfo) {
    int8_t dim = static_cast<int8_t>(shapeInfo.shape.size());
    shapeInfo.origAxisMap.resize(dim);
    
    for (int8_t i = 0; i < dim; ++i) {
        shapeInfo.origAxisMap[i].insert(i);
    }
    
    std::vector<bool> merged(dim, false);
    for (int8_t i = dim - 1; i > 0; --i) {
        if (merged[i]) {
            continue;
        }
        int64_t prevAxis = shapeInfo.perm[i];
        int64_t currAxis = shapeInfo.perm[i - 1];
        if (prevAxis == currAxis + 1 && shapeInfo.shape[currAxis] == 1) {
            for (auto axis : shapeInfo.origAxisMap[currAxis]) {
                shapeInfo.origAxisMap[prevAxis].insert(axis);
            }
            merged[currAxis] = true;
        }
    }
    
    std::vector<int64_t> axisMap(dim, -1);
    int8_t reducedIdx = 0;
    for (int8_t i = 0; i < dim; ++i) {
        if (!merged[i]) {
            axisMap[i] = reducedIdx++;
            int64_t size = 1;
            for (auto axis : shapeInfo.origAxisMap[i]) {
                size *= shapeInfo.shape[axis];
            }
            shapeInfo.reducedShape.push_back(size);
        }
    }
    
    for (int8_t i = 0; i < dim; ++i) {
        if (!merged[i]) {
            shapeInfo.reducedPerm.push_back(axisMap[i]);
        }
    }
}

void CalcInUbPermForPermute(const PermuteShapeInfo &shapeInfo, UbPermInfo &inUbPerm,
                             std::set<int64_t> &inUbPermSet, std::set<int64_t> &allUbPerm,
                             int64_t sqrtedTensor) {
    int8_t dim = static_cast<int8_t>(shapeInfo.reducedShape.size());
    int64_t remainSize = 1;
    for (int8_t i = 0; i < dim; ++i) {
        remainSize *= shapeInfo.reducedShape[i];
    }
    
    for (int8_t i = dim - 1; i >= 0 && inUbPerm.cnt < UB_MAX_BRW_NUM; --i) {
        if (remainSize <= sqrtedTensor * sqrtedTensor) {
            inUbPerm.perm[inUbPerm.cnt++] = i;
            inUbPermSet.insert(i);
            allUbPerm.insert(i);
            remainSize /= shapeInfo.reducedShape[i];
        } else {
            break;
        }
    }
}

void CalcOutUbPermForPermute(const PermuteShapeInfo &shapeInfo, UbPermInfo &outUbPerm,
                              const UbPermInfo &inUbPerm, std::set<int64_t> &outUbPermSet,
                              std::set<int64_t> &allUbPerm, int64_t sqrtedTensor) {
    int8_t dim = static_cast<int8_t>(shapeInfo.reducedShape.size());
    int64_t remainSize = 1;
    for (int8_t i = 0; i < dim; ++i) {
        remainSize *= shapeInfo.reducedShape[shapeInfo.reducedPerm[i]];
    }
    
    for (int8_t i = dim - 1; i >= 0 && outUbPerm.cnt < UB_MAX_BRW_NUM; --i) {
        int64_t origAxis = shapeInfo.reducedPerm[i];
        if (remainSize <= sqrtedTensor * sqrtedTensor) {
            outUbPerm.perm[outUbPerm.cnt++] = origAxis;
            outUbPermSet.insert(origAxis);
            allUbPerm.insert(origAxis);
            remainSize /= shapeInfo.reducedShape[origAxis];
        } else {
            break;
        }
    }
}

int64_t CalcSqrtedTensorForPermute(const std::vector<int64_t> &shape, DataType dtype) {
    int64_t elemSize = BytesOf(dtype);
    int64_t ubSize = 256 * 1024;
    int64_t pingPongNum = 2;
    int64_t maxTensorSize = ubSize / (elemSize * pingPongNum);
    return static_cast<int64_t>(std::sqrt(maxTensorSize));
}

std::vector<int64_t> CalcVecTileForPermute(const std::vector<int64_t> &shape,
                                             const UbPermInfo &inUbPerm,
                                             const UbPermInfo &outUbPerm,
                                             const std::set<int64_t> &inUbPermSet,
                                             const std::set<int64_t> &outUbPermSet) {
    std::vector<int64_t> vecTile(shape.size(), 1);
    int8_t dim = static_cast<int8_t>(shape.size());
    
    for (int8_t i = 0; i < dim; ++i) {
        bool isInUb = inUbPermSet.count(i) > 0;
        bool isOutUb = outUbPermSet.count(i) > 0;
        
        if (isInUb || isOutUb) {
            vecTile[i] = shape[i];
        } else {
            vecTile[i] = 1;
        }
    }
    
    return vecTile;
}

void TiledPermuteInner(Function &function, const TileShape &tileShape, int cur,
                       const std::shared_ptr<LogicalTensor> &operand, const std::shared_ptr<LogicalTensor> &result,
                       const std::vector<int64_t> &perm, std::vector<int64_t> &offset) {
    int shapeSize = operand->shape.size();
    if (cur == shapeSize) {
        auto tile = operand->View(function, tileShape.GetVecTile(), offset);
        
        std::vector<int64_t> resultTileShape(tileShape.GetVecTile().size());
        std::vector<int64_t> resultTileOffset(tileShape.GetVecTile().size());
        for (size_t i = 0; i < perm.size(); ++i) {
            resultTileShape[i] = tileShape.GetVecTile()[perm[i]];
            resultTileOffset[i] = offset[perm[i]];
        }
        
        auto resultTile = result->View(function, resultTileShape, resultTileOffset);
        auto &op = function.AddOperation(Opcode::OP_PERMUTE, {tile}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "perm", perm);
        return;
    }
    
    auto &vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < operand->shape[cur]; i += vecTile[cur]) {
        offset[cur] = i;
        TiledPermuteInner(function, tileShape, cur + 1, operand, result, perm, offset);
    }
}

void TiledPermute(Function &function, const TileShape &tileShape, const std::shared_ptr<LogicalTensor> &operand,
                  const std::shared_ptr<LogicalTensor> &result, const std::vector<int64_t> &perm) {
    std::vector<int64_t> offset(operand->shape.size(), 0);
    TiledPermuteInner(function, tileShape, 0, operand, result, perm, offset);
}

void InnerPermute(Function &function, const std::shared_ptr<LogicalTensor> &operand,
                  const std::shared_ptr<LogicalTensor> &result, const std::vector<int64_t> &perm) {
    PermuteShapeInfo shapeInfo;
    shapeInfo.shape = operand->shape;
    shapeInfo.perm = perm;
    
    ReduceShapeForPermute(shapeInfo);
    
    int64_t sqrtedTensor = CalcSqrtedTensorForPermute(shapeInfo.reducedShape, operand->Datatype());
    
    UbPermInfo inUbPerm;
    UbPermInfo outUbPerm;
    std::set<int64_t> inUbPermSet;
    std::set<int64_t> outUbPermSet;
    std::set<int64_t> allUbPerm;
    
    inUbPerm.perm.resize(shapeInfo.reducedShape.size());
    outUbPerm.perm.resize(shapeInfo.reducedShape.size());
    
    CalcInUbPermForPermute(shapeInfo, inUbPerm, inUbPermSet, allUbPerm, sqrtedTensor);
    CalcOutUbPermForPermute(shapeInfo, outUbPerm, inUbPerm, outUbPermSet, allUbPerm, sqrtedTensor);
    
    auto oldVecTile = TileShape::Current().GetVecTile();
    auto newVecTile = CalcVecTileForPermute(shapeInfo.reducedShape, inUbPerm, outUbPerm, inUbPermSet, outUbPermSet);
    
    std::vector<int64_t> expandedVecTile(operand->shape.size(), 1);
    std::vector<int64_t> axisToReducedIdx(operand->shape.size(), -1);
    for (size_t reducedIdx = 0; reducedIdx < shapeInfo.origAxisMap.size(); ++reducedIdx) {
        for (auto origAxis : shapeInfo.origAxisMap[reducedIdx]) {
            axisToReducedIdx[origAxis] = reducedIdx;
        }
    }
    
    for (size_t origAxis = 0; origAxis < operand->shape.size(); ++origAxis) {
        int64_t reducedIdx = axisToReducedIdx[origAxis];
        if (reducedIdx >= 0 && reducedIdx < static_cast<int64_t>(newVecTile.size())) {
            expandedVecTile[origAxis] = newVecTile[reducedIdx];
        }
    }
    
    TileShape::Current().SetVecTile(expandedVecTile);
    TiledPermute(function, TileShape::Current(), operand, result, perm);
    TileShape::Current().SetVecTile(oldVecTile);
}

Tensor Permute(const Tensor &self, std::vector<int64_t> perm) {
    DECLARE_TRACER();
    
    int shapeSize = self.GetShape().size();
    ASSERT(static_cast<int>(perm.size()) == shapeSize) << "Permute dim num should be same as input.";
    
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] < 0) {
            perm[i] += shapeSize;
        }
        ASSERT(perm[i] >= 0 && perm[i] < shapeSize) << "Permute dim " << i << " is invalid.";
    }
    
    std::vector<int64_t> checkPerm(perm);
    std::sort(checkPerm.begin(), checkPerm.end());
    for (size_t i = 0; i < checkPerm.size(); ++i) {
        ASSERT(checkPerm[i] == static_cast<int64_t>(i)) << "Permutation must contain all dimensions exactly once.";
    }
    
    bool isIdentity = true;
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int64_t>(i)) {
            isIdentity = false;
            break;
        }
    }
    if (isIdentity) {
        return self;
    }
    
    std::vector<int64_t> resultShape(shapeSize);
    for (size_t i = 0; i < perm.size(); ++i) {
        resultShape[i] = self.GetShape()[perm[i]];
    }
    
    auto oldValidShapes = self.GetStorage()->GetDynValidShape();
    std::vector<SymbolicScalar> resultValidShapes(resultShape.size());
    if (!oldValidShapes.empty()) {
        for (size_t i = 0; i < perm.size(); ++i) {
            resultValidShapes[i] = oldValidShapes[perm[i]];
        }
    } else {
        resultValidShapes = SymbolicScalar::FromConcrete(resultShape);
    }
    
    Tensor result(self.GetStorage()->Datatype(), resultShape);
    result.GetStorage()->UpdateDynValidShape(resultValidShapes);
    
    CALL(InnerPermute, *Program::GetInstance().GetCurrentFunction(),
         self.GetStorage(), result.GetStorage(), perm);
    
    return result;
}

void PermuteOperationTileFunc(Function &function, const TileShape &tileShape,
                               const std::vector<LogicalTensorPtr> &iOperand,
                               const std::vector<LogicalTensorPtr> &oOperand,
                               const Operation &op) {
    auto perm = op.GetVectorIntAttribute<int64_t>(OP_ATTR_PREFIX + "perm");
    TiledPermute(function, tileShape, iOperand[0], oOperand[0], perm);
}

REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE, Opcode::OP_PERMUTE, PermuteOperationTileFunc);

} // namespace npu::tile_fwk
