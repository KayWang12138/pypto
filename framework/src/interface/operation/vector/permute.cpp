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
 * 
 * Permute两步流程：
 * 1. ComputeIdx: 遍历输出tile，计算每个输出位置对应的输入线性索引
 * 2. Gather: 遍历输出tile，用索引从输入gather数据
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

static void DuplicateArray(const std::vector<int64_t>& src, int64_t* dst, int64_t len) {
    for (int64_t i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

static void DuplicateArray(const int64_t* src, std::vector<int64_t>& dst, int64_t len) {
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
    return (*static_cast<const int64_t*>(b) - *static_cast<const int64_t*>(a));
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

struct OutputTileInfo {
    LogicalTensorPtr tensor;
    TileInfo tileInfo;

    OutputTileInfo() : tensor(nullptr), tileInfo(0, 0) {}
};

void TiledInnerPermute(Function& function, const TileShape& tileShape, int cur,
                             OutputTileInfo& outputTile,
                             const LogicalTensorPtr& srcTensor,
                             const std::vector<int64_t>& reducedPerm,
                             const std::vector<int64_t>& reducedInShape,
                             const std::vector<int64_t>& reducedOutShape);

void TiledInnerPermute(Function& function, const TileShape& tileShape,
                             const LogicalTensorPtr& srcTensor,
                             const LogicalTensorPtr& resultTensor,
                             const std::vector<int64_t>& reducedPerm,
                             const std::vector<int64_t>& reducedInShape,
                             const std::vector<int64_t>& reducedOutShape);

void TiledInnerPermute(Function& function, const TileShape& tileShape, int cur,
                             OutputTileInfo& outputTile,
                             const LogicalTensorPtr& srcTensor,
                             const std::vector<int64_t>& reducedPerm,
                             const std::vector<int64_t>& reducedInShape,
                             const std::vector<int64_t>& reducedOutShape) {
    int shapeSize = outputTile.tensor->shape.size();
    
    if (cur == shapeSize) {
        auto resultTile = outputTile.tensor->View(function, outputTile.tileInfo.shape, outputTile.tileInfo.offset);
        std::vector<int64_t> tmpShape(outputTile.tileInfo.shape);
        auto alignSize = BLOCK_SIZE / BytesOf(outputTile.tensor->Datatype());
        tmpShape[outputTile.tileInfo.shape.size() - 1] = 
            AlignUp(tmpShape[outputTile.tileInfo.shape.size() - 1], alignSize);
        auto tempTensor = std::make_shared<LogicalTensor>(function, outputTile.tensor->Datatype(), tmpShape);
        auto& op = function.AddOperation(Opcode::OP_PERMUTE, {srcTensor, tempTensor}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "perm", reducedPerm);
        op.SetAttribute(OP_ATTR_PREFIX + "srcShape", reducedInShape);
        op.SetAttribute(OP_ATTR_PREFIX + "dstShape", reducedOutShape);
        return;
    }
    
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < outputTile.tensor->shape[cur]; i += vecTile[cur]) {
        outputTile.tileInfo.shape[cur] = std::min(outputTile.tensor->shape[cur] - i, vecTile[cur]);
        outputTile.tileInfo.offset[cur] = i;
        TiledInnerPermute(function, tileShape, cur + 1, outputTile, srcTensor,
                                reducedPerm, reducedInShape, reducedOutShape);
    }
}

void TiledInnerPermute(Function& function, const TileShape& tileShape,
                             const LogicalTensorPtr& srcTensor,
                             const LogicalTensorPtr& resultTensor,
                             const std::vector<int64_t>& reducedPerm,
                             const std::vector<int64_t>& reducedInShape,
                             const std::vector<int64_t>& reducedOutShape) {
    OutputTileInfo outputTile;
    outputTile.tensor = resultTensor;
    outputTile.tileInfo = TileInfo(resultTensor->shape.size(), resultTensor->offset.size());
    TiledInnerPermute(function, tileShape, 0, outputTile, srcTensor,    
                            reducedPerm, reducedInShape, reducedOutShape);
}

void PermuteOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand,
                           const std::vector<LogicalTensorPtr>& oOperand,
                           const Operation& op) {
    auto perm = op.GetVectorIntAttribute<int64_t>(OP_ATTR_PREFIX + "perm");
    auto srcShape = op.GetVectorIntAttribute<int64_t>(OP_ATTR_PREFIX + "srcShape");
    auto dstShape = op.GetVectorIntAttribute<int64_t>(OP_ATTR_PREFIX + "dstShape");
    
    TiledInnerPermute(function, tileShape, iOperand[0], oOperand[0], perm, srcShape, dstShape);
}

void TensorInnerPermute(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& result,
                        const std::vector<int64_t>& reducedPerm,
                        const std::vector<int64_t>& reducedInShape,
                        const std::vector<int64_t>& reducedOutShape) {
    auto &operation = function.AddOperation(Opcode::OP_PERMUTE, {self}, {result});
    operation.SetAttribute(OP_ATTR_PREFIX + "perm", reducedPerm);
    operation.SetAttribute(OP_ATTR_PREFIX + "srcShape", reducedInShape);
    operation.SetAttribute(OP_ATTR_PREFIX + "dstShape", reducedOutShape);
}

Tensor TensorPermuteOperation(Function& function, const LogicalTensorPtr& operand, 
                              const std::vector<int64_t>& reducedPerm,
                              const std::vector<int64_t>& reducedInShape,
                              const std::vector<int64_t>& reducedOutShape,
                              const std::vector<SymbolicScalar>& validShape) {
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), 
                                                   reducedOutShape, 
                                                   validShape, operand->Format());
    
    TensorInnerPermute(function, operand, result,
                       reducedPerm, reducedInShape, reducedOutShape);
    
    return Tensor(result);
}

REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE, Opcode::OP_PERMUTE, PermuteOperationTileFunc);

Tensor Permute(const Tensor& self, std::vector<int> dims) {
    DECLARE_TRACER();
    
    PermuteShapeInfo shapeInfo;
    
    std::vector<int64_t> dims64(dims.begin(), dims.end());
    if (!GetShapeInfo(self, dims64, shapeInfo)) {
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
    
    std::vector<SymbolicScalar> reducedValidShape(shapeInfo.dim);
    for (int64_t i = 0; i < shapeInfo.dim; i++) {
        reducedValidShape[i] = oldValidShapes[shapeInfo.reducedPerm[i]];
    }
    
    auto& tileShape = TileShape::Current();
    auto oldVecTileShapes = tileShape.GetVecTile();
    
    std::vector<int64_t> newVecTileShape(oldVecTileShapes.size(), 1);
    for (size_t i = 0; i < reducedPerm.size(); i++) {
        newVecTileShape[i] = oldVecTileShapes[reducedPerm[i]];
    }
    
    auto tmpInputTensor = Reshape(self, reducedInShape, oldValidShapes);
    tileShape.SetVecTile(newVecTileShape);
    
    std::vector<SymbolicScalar> outputValidShape(shapeInfo.dim);
    for (int64_t i = 0; i < shapeInfo.dim; i++) {
        outputValidShape[i] = oldValidShapes[reducedPerm[i]];
    }
    
    auto tmpOutputTensor = TensorPermuteOperation(*Program::GetInstance().GetCurrentFunction(),
                                                  tmpInputTensor.GetStorage(),
                                                  reducedPerm, reducedInShape, reducedOutShape,
                                                  outputValidShape);
    
    tileShape.SetVecTile(oldVecTileShapes);
    
    return Reshape(tmpOutputTensor, shapeInfo.outShape, oldValidShapes);    
}

}
