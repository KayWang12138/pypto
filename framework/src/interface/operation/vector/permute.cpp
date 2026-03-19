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
 * 
 * 框架层次说明（参考Transpose架构）：
 * 1. TensorInnerPermute：通过AddOperation添加图节点
 * 2. TileFunc：框架自动调用，内部调用TiledInnerPermute
 * 3. TiledInnerPermute：递归切分，内部调用TileOp
 * 4. TileOp：具体的二维Tile块处理（含外抛for循环）
 */

#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include "interface/utils/common.h"
#include "interface/utils/operator_tracer.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/tileop/vector/permute.h"
#include "interface/tileop/vector/mte.h"

namespace npu::tile_fwk {

constexpr int PERMUTE_MIN_DIM = 2;
constexpr int PERMUTE_MAX_DIM = 5;
constexpr int PERMUTE_MAX_AXIS_NUM = 8;

struct PermuteShapeInfo {
    std::vector<int64_t> inShape;
    std::vector<int64_t> outShape;
    std::vector<int64_t> perm;
    std::vector<int64_t> reducedInShape;
    std::vector<int64_t> reducedOutShape;
    std::vector<int64_t> reducedPerm;
    
    int64_t inShapeSize = 0;
    int64_t outShapeSize = 0;
    int64_t permSize = 0;
    int64_t origDim = 0;
    int64_t dim = 0;
    int64_t totalVolume = 0;
    int64_t lastAxisLen = 0;
    int64_t eleLenInBytes = 0;
    bool isLastAxisTranspose = false;

    PermuteShapeInfo() {
        inShape.resize(PERMUTE_MAX_AXIS_NUM, 0);
        outShape.resize(PERMUTE_MAX_AXIS_NUM, 0);
        perm.resize(PERMUTE_MAX_AXIS_NUM, 0);
        reducedInShape.resize(PERMUTE_MAX_AXIS_NUM, 0);
        reducedOutShape.resize(PERMUTE_MAX_AXIS_NUM, 0);
        reducedPerm.resize(PERMUTE_MAX_AXIS_NUM, 0);
    }

    void Reset() {
        inShapeSize = 0;
        outShapeSize = 0;
        permSize = 0;
        origDim = 0;
        dim = 0;
        totalVolume = 0;
        lastAxisLen = 0;
        eleLenInBytes = 0;
        isLastAxisTranspose = false;
        std::fill(inShape.begin(), inShape.end(), 0);
        std::fill(outShape.begin(), outShape.end(), 0);
        std::fill(perm.begin(), perm.end(), 0);
        std::fill(reducedInShape.begin(), reducedInShape.end(), 0);
        std::fill(reducedOutShape.begin(), reducedOutShape.end(), 0);
        std::fill(reducedPerm.begin(), reducedPerm.end(), 0);
    }
};

static void DuplicateArray(const std::vector<int64_t>& src, std::vector<int64_t>& dst, int64_t len) {
    for (int64_t i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

static bool IsAllOne(const PermuteShapeInfo& shapeInfo) {
    return std::all_of(shapeInfo.inShape.begin(), 
                       shapeInfo.inShape.begin() + shapeInfo.dim, 
                       [](const int64_t& item) { return item == 1; });
}

static int DecreaseCompare(const void* a, const void* b) {
    return (*(int64_t*)b - *(int64_t*)a);
}

static void CalcOutShape(PermuteShapeInfo& shapeInfo) {
    const std::vector<int64_t>& inShape = shapeInfo.reducedInShape;
    const std::vector<int64_t>& perm = shapeInfo.reducedPerm;
    std::vector<int64_t>& outShape = shapeInfo.reducedOutShape;
    for (int64_t i = 0; i < shapeInfo.dim; i++) {
        outShape[i] = inShape[perm[i]];
    }
}

bool GetShapeInfo(const Tensor& self, const std::vector<int64_t>& dims, PermuteShapeInfo& shapeInfo) {
    shapeInfo.Reset();
    
    const auto& inputShape = self.GetShape();
    int64_t inputDims = static_cast<int64_t>(inputShape.size());
    
    if (inputDims < PERMUTE_MIN_DIM || inputDims > PERMUTE_MAX_DIM) {
        return false;
    }
    
    if (static_cast<int64_t>(dims.size()) != inputDims) {
        return false;
    }
    
    shapeInfo.inShapeSize = inputDims;
    shapeInfo.permSize = static_cast<int64_t>(dims.size());
    shapeInfo.origDim = inputDims;
    shapeInfo.dim = inputDims;
    
    for (int64_t i = 0; i < inputDims; i++) {
        int64_t permVal = dims[i];
        if (permVal < 0) {
            permVal += inputDims;
        }
        if (permVal < 0 || permVal >= inputDims) {
            return false;
        }
        shapeInfo.perm[i] = permVal;
        shapeInfo.inShape[i] = inputShape[i];
    }
    
    for (int64_t i = 0; i < inputDims; i++) {
        shapeInfo.outShape[i] = shapeInfo.inShape[shapeInfo.perm[i]];
    }
    shapeInfo.outShapeSize = inputDims;
    
    DuplicateArray(shapeInfo.inShape, shapeInfo.reducedInShape, shapeInfo.dim);
    DuplicateArray(shapeInfo.perm, shapeInfo.reducedPerm, shapeInfo.dim);
    DuplicateArray(shapeInfo.outShape, shapeInfo.reducedOutShape, shapeInfo.dim);
    
    shapeInfo.totalVolume = 1;
    for (int64_t i = 0; i < shapeInfo.dim; i++) {
        shapeInfo.totalVolume *= shapeInfo.inShape[i];
    }
    
    shapeInfo.lastAxisLen = shapeInfo.inShape[shapeInfo.dim - 1];
    
    return true;
}

void RemoveAxisV2(PermuteShapeInfo& shapeInfo) {
    int64_t dim = shapeInfo.dim;
    if (dim == 1) {
        DuplicateArray(shapeInfo.inShape, shapeInfo.reducedInShape, dim);
        DuplicateArray(shapeInfo.perm, shapeInfo.reducedPerm, dim);
        DuplicateArray(shapeInfo.outShape, shapeInfo.reducedOutShape, dim);
        return;
    }

    if (IsAllOne(shapeInfo)) {
        shapeInfo.reducedInShape[0] = 1;
        shapeInfo.reducedPerm[0] = 0;
        shapeInfo.reducedOutShape[0] = 1;
        shapeInfo.dim = 1;
        return;
    }

    std::vector<int64_t>& shape = shapeInfo.reducedInShape;
    int64_t delPerm[PERMUTE_MAX_AXIS_NUM];
    int64_t newPerm[PERMUTE_MAX_AXIS_NUM];
    int64_t shapeSize = 0;
    int64_t delPermSize = 0;
    int64_t newPermSize = 0;

    for (int64_t i = 0; i < dim; i++) {
        if (shapeInfo.inShape[i] != 1) {
            shape[shapeSize++] = shapeInfo.inShape[i];
        } else {
            for (int64_t j = 0; j < dim; j++) {
                if (shapeInfo.perm[j] == i) {
                    delPerm[delPermSize++] = shapeInfo.perm[j];
                }
            }
        }
    }

    qsort(reinterpret_cast<void*>(&delPerm[0]), delPermSize, sizeof(int64_t), DecreaseCompare);

    for (int64_t i = 0; i < dim; i++) {
        bool delFlag = false;
        for (int64_t j = 0; j < delPermSize; j++) {
            if (shapeInfo.perm[i] == delPerm[j]) {
                delFlag = true;
                break;
            }
        }
        if (!delFlag) {
            newPerm[newPermSize++] = shapeInfo.perm[i];
        }
    }

    for (int64_t i = 0; i < delPermSize; i++) {
        for (int64_t j = 0; j < newPermSize; j++) {
            if (newPerm[j] > delPerm[i]) {
                newPerm[j] = newPerm[j] - 1;
            }
        }
    }

    DuplicateArray(newPerm, shapeInfo.reducedPerm, newPermSize);
    shapeInfo.dim = newPermSize;
    CalcOutShape(shapeInfo);
}

void MergeAxisV2(PermuteShapeInfo& shapeInfo) {
    int64_t dim = shapeInfo.dim;
    if (dim == 1) {
        return;
    }
    
    int64_t perm[PERMUTE_MAX_AXIS_NUM];
    int64_t shape[PERMUTE_MAX_AXIS_NUM];
    int64_t newPerm[PERMUTE_MAX_AXIS_NUM];
    int64_t newShape[PERMUTE_MAX_AXIS_NUM];
    int64_t newDimPosition[PERMUTE_MAX_AXIS_NUM];
    int64_t mergedShape[PERMUTE_MAX_AXIS_NUM] = {0};
    
    DuplicateArray(shapeInfo.reducedPerm, perm, dim);
    DuplicateArray(shapeInfo.reducedInShape, shape, dim);
    
    for (int i = 0; i < PERMUTE_MAX_AXIS_NUM; i++) {
        newDimPosition[i] = -1;
    }

    int64_t curHead = shapeInfo.reducedPerm[0];
    newDimPosition[curHead] = 0;
    mergedShape[0] = shape[curHead];
    int dimIndex = 0;
    
    for (int permIndex = 1; permIndex < dim; ++permIndex) {
        if (curHead + 1 == perm[permIndex]) {
            curHead = perm[permIndex];
            mergedShape[dimIndex] *= shape[curHead];
        } else {
            curHead = perm[permIndex];
            dimIndex++;
            newDimPosition[curHead] = dimIndex;
            mergedShape[dimIndex] = shape[curHead];
        }
    }

    shapeInfo.dim = dimIndex + 1;

    dimIndex = 0;
    for (int i = 0; i < dim; i++) {
        if (newDimPosition[i] >= 0) {
            newDimPosition[dimIndex++] = newDimPosition[i];
        }
    }

    dimIndex = 0;
    for (int64_t i = 0; i < dim; ++i) {
        if (newDimPosition[i] >= 0) {
            for (int64_t j = 0; j < dim; j++) {
                if (newDimPosition[j] == i) {
                    newPerm[dimIndex] = j;
                    break;
                }
            }
            newShape[dimIndex] = mergedShape[newDimPosition[i]];
            dimIndex++;
        }
    }

    DuplicateArray(newShape, shapeInfo.reducedInShape, dimIndex);
    DuplicateArray(newPerm, shapeInfo.reducedPerm, dimIndex);
    shapeInfo.lastAxisLen = shapeInfo.reducedInShape[shapeInfo.dim - 1];
    CalcOutShape(shapeInfo);
}

static bool IsIdentityPermutation(const std::vector<int64_t>& perm, int64_t dim) {
    for (int64_t i = 0; i < dim; i++) {
        if (perm[i] != i) {
            return false;
        }
    }
    return true;
}

struct Input {
    LogicalTensorPtr tensor;
    TileInfo tileInfo;
};

void TiledInnerPermuteComputeIdx(Function& function, const TileShape& tileShape, int cur, Input& input,
                                 const LogicalTensorPtr& idxTmpTensor,
                                 const std::vector<int64_t>& reducedPerm,
                                 const std::vector<int64_t>& reducedInShape);

void TiledInnerPermuteComputeIdx(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                                 const LogicalTensorPtr& idxTmpTensor,
                                 const std::vector<int64_t>& reducedPerm,
                                 const std::vector<int64_t>& reducedInShape);

void TiledInnerPermuteGather(Function& function, const TileShape& tileShape, int cur, Input& input,
                             const LogicalTensorPtr& result, const LogicalTensorPtr& idxTmpTensor,
                             const std::vector<int64_t>& reducedPerm);

void TiledInnerPermuteGather(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                             const LogicalTensorPtr& result, const LogicalTensorPtr& idxTmpTensor,
                             const std::vector<int64_t>& reducedPerm);

void PermuteComputeIdxTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand,
                               const Operation& op);

void PermuteGatherTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand,
                           const std::vector<LogicalTensorPtr>& oOperand,
                           const Operation& op);

void TiledInnerPermuteComputeIdx(Function& function, const TileShape& tileShape, int cur, Input& input,
                                 const LogicalTensorPtr& idxTmpTensor,
                                 const std::vector<int64_t>& reducedPerm,
                                 const std::vector<int64_t>& reducedInShape) {
    int shapeSize = input.tensor->shape.size();
    
    if (cur == shapeSize) {
        auto tile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        
        int64_t totalElements = 1;
        for (size_t i = 0; i < input.tileInfo.shape.size(); i++) {
            totalElements *= input.tileInfo.shape[i];
        }
        
        std::vector<int64_t> tmpShape = {totalElements};
        auto idxTile = idxTmpTensor->View(function, tmpShape, {0});
        
        TPermuteComputeIdx(tile, idxTile, reducedPerm, reducedInShape, 
                           input.tileInfo.shape, input.tileInfo.offset, static_cast<int64_t>(reducedPerm.size()));
        return;
    }
    
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor->shape[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledInnerPermuteComputeIdx(function, tileShape, cur + 1, input, idxTmpTensor,
                                    reducedPerm, reducedInShape);
    }
}

void TiledInnerPermuteComputeIdx(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                                 const LogicalTensorPtr& idxTmpTensor,
                                 const std::vector<int64_t>& reducedPerm,
                                 const std::vector<int64_t>& reducedInShape) {
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    auto input = Input{operand, tileInfo};
    TiledInnerPermuteComputeIdx(function, tileShape, 0, input, idxTmpTensor,
                                reducedPerm, reducedInShape);
}

void TiledInnerPermuteGather(Function& function, const TileShape& tileShape, int cur, Input& input,
                             const LogicalTensorPtr& result, const LogicalTensorPtr& idxTmpTensor,
                             const std::vector<int64_t>& reducedPerm) {
    int shapeSize = input.tensor->shape.size();
    
    if (cur == shapeSize) {
        auto tile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        
        std::vector<int64_t> resultTileShape(input.tileInfo.shape);
        std::vector<int64_t> resultTileOfs(input.tileInfo.offset);
        
        for (size_t i = 0; i < reducedPerm.size(); i++) {
            resultTileShape[i] = input.tileInfo.shape[reducedPerm[i]];
            resultTileOfs[i] = input.tileInfo.offset[reducedPerm[i]];
        }
        
        auto resultTile = result->View(function, resultTileShape, resultTileOfs);
        
        int64_t totalElements = 1;
        for (size_t i = 0; i < input.tileInfo.shape.size(); i++) {
            totalElements *= input.tileInfo.shape[i];
        }
        
        std::vector<int64_t> tmpShape = {totalElements};
        auto idxTile = idxTmpTensor->View(function, tmpShape, {0});
        
        TPermuteGather(resultTile, tile, idxTile);
        return;
    }
    
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor->shape[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledInnerPermuteGather(function, tileShape, cur + 1, input, result, idxTmpTensor, reducedPerm);
    }
}

void TiledInnerPermuteGather(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                             const LogicalTensorPtr& result, const LogicalTensorPtr& idxTmpTensor,
                             const std::vector<int64_t>& reducedPerm) {
    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledInnerPermuteGather(function, tileShape, 0, input, result, idxTmpTensor, reducedPerm);
}

void PermuteComputeIdxTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand,
                               const Operation& op) {
    auto perm = op.GetVectorIntAttribute<int64_t>(OP_ATTR_PREFIX + "perm");
    auto srcShape = op.GetVectorIntAttribute<int64_t>(OP_ATTR_PREFIX + "srcShape");
    
    const auto& srcTensor = iOperand[0];
    const auto& idxTensor = iOperand[1];
    
    TiledInnerPermuteComputeIdx(function, tileShape, srcTensor, idxTensor, perm, srcShape);
}

void PermuteGatherTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand,
                           const std::vector<LogicalTensorPtr>& oOperand,
                           const Operation& op) {
    auto perm = op.GetVectorIntAttribute<int64_t>(OP_ATTR_PREFIX + "perm");
    
    const auto& srcTensor = iOperand[0];
    const auto& idxTensor = iOperand[1];
    const auto& dstTensor = oOperand[0];
    
    TiledInnerPermuteGather(function, tileShape, srcTensor, dstTensor, idxTensor, perm);
}

REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE_COMPUTE_IDX, Opcode::OP_PERMUTE_COMPUTE_IDX, PermuteComputeIdxTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE_GATHER, Opcode::OP_PERMUTE_GATHER, PermuteGatherTileFunc);

void TensorInnerPermute(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& result,
                        const LogicalTensorPtr& idxTmpTensor,
                        const std::vector<int64_t>& reducedPerm,
                        const std::vector<int64_t>& reducedInShape,
                        const std::vector<int64_t>& reducedOutShape) {
    auto& tileShape = TileShape::Current();
    auto oldVecTileShapes = tileShape.GetVecTile();
    
    std::vector<int64_t> newVecTileShape = oldVecTileShapes;
    for (size_t i = 0; i < reducedPerm.size(); i++) {
        newVecTileShape[i] = oldVecTileShapes[reducedPerm[i]];
    }
    tileShape.SetVecTile(newVecTileShape);
    
    auto& computeIdxOp = function.AddOperation(Opcode::OP_PERMUTE_COMPUTE_IDX, {self, idxTmpTensor}, {});
    computeIdxOp.SetAttribute(OP_ATTR_PREFIX + "perm", reducedPerm);
    computeIdxOp.SetAttribute(OP_ATTR_PREFIX + "srcShape", reducedInShape);
    
    auto& gatherOp = function.AddOperation(Opcode::OP_PERMUTE_GATHER, {self, idxTmpTensor}, {result});
    gatherOp.SetAttribute(OP_ATTR_PREFIX + "perm", reducedPerm);
    
    tileShape.SetVecTile(oldVecTileShapes);
}

Tensor TensorPermuteOperation(Function& function, const LogicalTensorPtr& operand, 
                              const std::vector<int64_t>& reducedPerm,
                              const std::vector<int64_t>& reducedInShape,
                              const std::vector<int64_t>& reducedOutShape,
                              const std::vector<SymbolicScalar>& validShape) {
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), 
                                                   reducedOutShape, 
                                                   validShape, operand->Format());
    
    int64_t dim = static_cast<int64_t>(reducedPerm.size());
    
    int64_t totalElements = 1;
    for (int64_t i = 0; i < dim; i++) {
        totalElements *= reducedInShape[i];
    }
    
    std::vector<int64_t> tmpShape = {totalElements};
    auto idxTmpTensor = std::make_shared<LogicalTensor>(function, DataType::DT_UINT32, tmpShape);
    idxTmpTensor->dynValidShape_ = SymbolicScalar::FromConcrete(tmpShape);
    
    TensorInnerPermute(function, operand, result, idxTmpTensor,
                       reducedPerm, reducedInShape, reducedOutShape);
    
    return Tensor(result);
}

Tensor Permute(const Tensor& self, const std::vector<int64_t>& dims) {
    DECLARE_TRACER();
    
    PermuteShapeInfo shapeInfo;
    
    if (!GetShapeInfo(self, dims, shapeInfo)) {
        ASSERT(false) << "Permute: Invalid input dimensions or permutation";
        return self;
    }
    
    RemoveAxisV2(shapeInfo);
    
    MergeAxisV2(shapeInfo);
    
    if (IsIdentityPermutation(shapeInfo.reducedPerm, shapeInfo.dim)) {
        return self;
    }
    
    std::vector<int64_t> reducedPerm(shapeInfo.dim);
    std::vector<int64_t> reducedInShape(shapeInfo.dim);
    std::vector<int64_t> reducedOutShape(shapeInfo.dim);
    for (int64_t i = 0; i < shapeInfo.dim; i++) {
        reducedPerm[i] = shapeInfo.reducedPerm[i];
        reducedInShape[i] = shapeInfo.reducedInShape[i];
        reducedOutShape[i] = shapeInfo.reducedOutShape[i];
    }
    
    auto oldValidShapes = self.GetStorage()->GetDynValidShape();
    if (oldValidShapes.empty()) {
        oldValidShapes = SymbolicScalar::FromConcrete(self.GetShape());
    }
    
    std::vector<SymbolicScalar> outputValidShape(shapeInfo.dim);
    for (int64_t i = 0; i < shapeInfo.dim; i++) {
        outputValidShape[i] = oldValidShapes[reducedPerm[i]];
    }
    
    RETURN_CALL(TensorPermuteOperation, *Program::GetInstance().GetCurrentFunction(), 
                self.GetStorage(), reducedPerm, reducedInShape, reducedOutShape, outputValidShape);
}

Tensor Permute(const Tensor& self, const std::initializer_list<int64_t>& dims) {
    return Permute(self, std::vector<int64_t>(dims));
}

} // namespace npu::tile_fwk
