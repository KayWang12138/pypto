/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file op_infer_shape_impl.cpp
 * \brief
 */

#include "op_infer_shape_impl.h"
#include "interface/operation/attr_holder.h"
#include "interface/operation/operation.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {
const std::string COPY_OUT_FORCE_INFER_SHAPE = "copy_out_force_infer_shape";

void ElewiseInferFunc(Operation* op,
                      std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto inputNum = op->GetIOperands().size();
    auto shapeDimNum = op->GetIOperands()[0]->GetDynValidShape().size();
    // 将每个输入的同一维shape值填充到一个vector中，便于后续对每一维进行筛选
    std::vector<std::vector<SymbolicScalar>> dimValidShape(shapeDimNum, std::vector<SymbolicScalar>(inputNum, SymbolicScalar()));
    for (size_t i = 0; i < op->GetIOperands().size(); ++i) {
        auto validShape = op->GetIOperands()[i]->GetDynValidShape();
        for (size_t dimIdx = 0; dimIdx < validShape.size(); ++dimIdx) {
            dimValidShape[dimIdx][i] = validShape[dimIdx];
        }
    }
    std::vector<SymbolicScalar> inputValidShape;
    for (size_t i = 0; i < shapeDimNum; ++i) {
        auto flag = false;
        auto minDim = SymbolicScalar();
        for (auto dim : dimValidShape[i]) {
            if (!(dim.IsImmediate())) {
                inputValidShape.push_back(dim);
                flag = true;
                break;
            } else {
                minDim = minDim.ConcreteValid() ? std::min(minDim.Concrete(), dim.Concrete()) : dim.Concrete();
            }
        }
        // 全部都是Immediate值，取用最小的
        if (!flag) {
            inputValidShape.push_back(minDim);
        }
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(inputValidShape);
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_ADD, Opcode::OP_ADD, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ADDS, Opcode::OP_ADDS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MULS, Opcode::OP_MULS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_DIVS, Opcode::OP_S_DIVS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_PAIRMAX, Opcode::OP_PAIRMAX, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SUB, Opcode::OP_SUB, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_EXP, Opcode::OP_EXP, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SQRT, Opcode::OP_SQRT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ABS, Opcode::OP_ABS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MAXIMUM, Opcode::OP_MAXIMUM, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_CAST, Opcode::OP_CAST, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_PAIRSUM, Opcode::OP_PAIRSUM, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MUL, Opcode::OP_MUL, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_DIV, Opcode::OP_DIV, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_DIVS, Opcode::OP_DIVS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_RECIPROCAL, Opcode::OP_RECIPROCAL, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SUBS, Opcode::OP_SUBS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_ADDS, Opcode::OP_S_ADDS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_SUBS, Opcode::OP_S_SUBS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MULS, Opcode::OP_S_MULS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MAXS, Opcode::OP_S_MAXS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MINS, Opcode::OP_S_MINS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_ADD, Opcode::OP_S_ADD, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_SUB, Opcode::OP_S_SUB, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MUL, Opcode::OP_S_MUL, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_DIV, Opcode::OP_S_DIV, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MAX, Opcode::OP_S_MAX, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MIN, Opcode::OP_S_MIN, ElewiseInferFunc);

// elewise brc infer shape func
void ElewiseBrcInferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<SymbolicScalar> outValidShape;

    // elewisebrc dim is immediate and dim is 1, another dim is dst shape dim
    for (size_t i = 0; i < op->GetIOperands()[0]->GetDynValidShape().size(); ++i) {
        auto leftIShapeDim = op->GetIOperands()[0]->GetDynValidShape()[i];
        if (leftIShapeDim.IsImmediate() && leftIShapeDim.Concrete() == 1) {
            outValidShape.push_back(op->GetIOperands()[1]->GetDynValidShape()[i]);
        } else {
            outValidShape.push_back(leftIShapeDim);
        }
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_ADD_BRC, Opcode::OP_ADD_BRC, ElewiseBrcInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SUB_BRC, Opcode::OP_SUB_BRC, ElewiseBrcInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MUL_BRC, Opcode::OP_MUL_BRC, ElewiseBrcInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_DIV_BRC, Opcode::OP_DIV_BRC, ElewiseBrcInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MAX_BRC, Opcode::OP_MAX_BRC, ElewiseBrcInferFunc);

// broadcast infer shape func
void BroadcastInferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<SymbolicScalar> outValidShape;
    if (op->GetAttr(OP_ATTR_PREFIX + "validShape", outValidShape)) {
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(outValidShape);
        }
        return;
    }
    auto outTensor = op->GetOOperands()[0]; // one in, one out
    // broadcast 1对应的维度采用tileshap
    for (size_t i = 0; i < op->GetIOperands()[0]->GetDynValidShape().size(); ++i) {
        if (op->GetIOperands()[0]->oriShape[i] != 1) {
            outValidShape.push_back(op->GetIOperands()[0]->GetDynValidShape()[i]);
        } else {
            outValidShape.push_back(SymbolicScalar(op->GetOOperands()[0]->GetShape()[i]));
        }
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_EXPAND, Opcode::OP_EXPAND, BroadcastInferFunc);

// reduce infer shape func
void ReduceInferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    auto outValidShape = inputValidShapes[0];
    int axis = op->GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    outValidShape[axis] = SymbolicScalar(1);
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_ROWSUMLINE, Opcode::OP_ROWSUMLINE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWMAXLINE, Opcode::OP_ROWMAXLINE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWMAX_SINGLE, Opcode::OP_ROWMAX_SINGLE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWSUM_SINGLE, Opcode::OP_ROWSUM_SINGLE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWMAX_COMBINE_AXIS_SINGLE, Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWSUM_COMBINE_AXIS_SINGLE, Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE, ReduceInferFunc);

// Gather infer shape func
void InferFunc4Gather(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto iOperands = op->GetIOperands();
    assert(iOperands.size() >= NUM2);
    int axis = op->GetIntAttribute(OP_ATTR_PREFIX + "axis");
    int src0Rank = iOperands[0]->GetShape().size();
    if (axis < 0) {
        axis = axis + src0Rank;
    }
    assert((axis >= 0 && axis < src0Rank) && "InferFunc4Gather, axis is invalid");

    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : iOperands) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    // output shape: input0.shape[:aixs] + input1.shape + input0.shape[axis+1:]
    std::vector<SymbolicScalar> outValidShape = inputValidShapes[0];
    outValidShape.erase(outValidShape.begin() + axis);
    outValidShape.insert(outValidShape.begin() + axis, inputValidShapes[1].begin(), inputValidShapes[1].end());

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_GATHER, Opcode::OP_GATHER, InferFunc4Gather);

// matmul infer shape func
void MatmulInferFunc(Operation* op,
                     std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<SymbolicScalar> outValidShape;
    for (auto inputTensor : op->GetIOperands()) {
        auto inputValidShape = inputTensor->GetDynValidShape();
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
            outValidShape.push_back(inputValidShape[0]);
        } else if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            outValidShape.push_back(inputValidShape[1]);
        }
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_A_MUL_B, Opcode::OP_A_MUL_B, MatmulInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_A_MUL_BT, Opcode::OP_A_MUL_BT, MatmulInferFunc);

void MatmulACCInferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<SymbolicScalar> outValidShape;
    for (auto inputTensor : op->GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
            outValidShape = inputTensor->GetDynValidShape();
            break;
        }
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_A_MULACC_B, Opcode::OP_A_MULACC_B, MatmulACCInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_A_MULACC_BT, Opcode::OP_A_MULACC_BT, MatmulACCInferFunc);

// MTE infer shape func
template <bool isTrans = false>
void LoadL0InferFunc(Operation* op,
                     std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    for (auto output : op->GetOOperands()) {
        if (isTrans) {
            auto outValidShape = {inputValidShapes[0][1], inputValidShapes[0][0]};
            outValidShapes.push_back(outValidShape);
        } else {
            outValidShapes.push_back(inputValidShapes[0]);
        }
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, LoadL0InferFunc<false>);
REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, LoadL0InferFunc<false>);
REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_L0_BT, Opcode::OP_L1_TO_L0_BT, LoadL0InferFunc<true>);

void CopyInInferFunc(Operation* op,
                     std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto copyOpAttribute = dynamic_cast<CopyOpAttribute *>(op->GetOpAttribute().get());
    if (!(op->GetOOperands()[0]->GetDynValidShape().empty())) {
        outValidShapes.push_back(op->GetOOperands()[0]->GetDynValidShape());
        if (copyOpAttribute != nullptr && (copyOpAttribute->GetToDynValidShape()).empty()) {
            auto toDynShape = OpImmediate::Specified(op->GetOOperands()[0]->GetDynValidShape());
            copyOpAttribute->SetToDynValidShape(toDynShape);
        }
        return;
    }

    // 连接incast
    auto toValidShape = copyOpAttribute->GetToDynValidShape();
    std::vector<SymbolicScalar> toValidShapeSym(toValidShape.size());
    OpImmediate::NormalizeValue(toValidShapeSym, 0, toValidShape, 0, false);
    auto toValidShapeValue = SymbolicScalar::Concrete(toValidShapeSym, -1);
    auto tileShape = copyOpAttribute->GetShape();
    std::vector<SymbolicScalar> tileShapeSym(tileShape.size());
    OpImmediate::NormalizeValue(tileShapeSym, 0, tileShape, 0, false);
    if (!toValidShape.empty()) {
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(toValidShapeSym);
        }
        return;
    }
    if (!(op->GetOOperands()[0]->GetDynValidShape().empty())) {
        outValidShapes.push_back(op->GetOOperands()[0]->GetDynValidShape());
        auto toDynShape = OpImmediate::Specified(op->GetOOperands()[0]->GetDynValidShape());
        copyOpAttribute->SetToDynValidShape(toDynShape);
        return;
    }
    // 临时空间，固定大小
    if (op->GetIOperands()[0]->GetProducers().empty()) {
        if (toValidShape.empty()) {
            std::vector<SymbolicScalar> toValidShapeVec;
            for (auto dim : copyOpAttribute->GetShape()) {
                toValidShapeVec.push_back(dim.GetSpecifiedValue());
            }
            for (auto output : op->GetOOperands()) {
                outValidShapes.push_back(toValidShapeVec);
            }
        }
        auto toDynShape = OpImmediate::Specified(outValidShapes[0]);
        copyOpAttribute->SetToDynValidShape(toDynShape);
        return;
    }
    // 子图边界, 需要重新推导
    std::vector<std::vector<SymbolicScalar>> inputShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputShapes.push_back(inputTensor->GetDynValidShape());
    }
    auto offset = copyOpAttribute->GetFromOffset();
    std::vector<SymbolicScalar> oriOffset;
    for (auto offsetValue : offset) {
        oriOffset.push_back(offsetValue.GetSpecifiedValue());
    }
    std::vector<SymbolicScalar> outputShape;
    for (size_t i = 0U; i < inputShapes[0].size(); i++) {
        SymbolicScalar actualDim = std::max(0, std::min((inputShapes[0][i] - oriOffset[i]), tileShapeSym[i]));
        outputShape.push_back(actualDim);
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outputShape);
    }
    // 设置validshape到copyin的toDynvalidshape
    auto toDynShape = OpImmediate::Specified(outputShape);
    copyOpAttribute->SetToDynValidShape(toDynShape);
}
REGISTER_INFER_SHAPE_FUNC(OP_COPY_IN, Opcode::OP_COPY_IN, CopyInInferFunc);

void CopyOutInferFunc(Operation* op,
                      std::vector<std::vector<SymbolicScalar>>& outValisShapes)
{
    auto copyOpAttribute = dynamic_cast<CopyOpAttribute *>(op->GetOpAttribute().get());
    if (copyOpAttribute != nullptr) {
        copyOpAttribute->SetFromDynValidShape(OpImmediate::Specified(op->GetIOperands()[0]->GetDynValidShape()));
    } else {
        ALOG_WARN_F("Copyout [%d] has no copy out attr.", op->GetOpMagic());
        outValisShapes.push_back(op->GetIOperands()[0]->GetDynValidShape());
        return;
    }

    // 多个tile块copyout到同一个tensor时， 每一个tile都需要推导
    bool needInferShape = false;
    if (!(op->GetOOperands()[0]->GetDynValidShape().empty()) && !op->GetOOperands()[0]->GetAttr(COPY_OUT_FORCE_INFER_SHAPE, needInferShape)) {
        outValisShapes.push_back(op->GetOOperands()[0]->GetDynValidShape());
        return;
    }

    op->GetOOperands()[0]->SetAttr(COPY_OUT_FORCE_INFER_SHAPE, true);

    auto offset = copyOpAttribute->GetToOffset();
    std::vector<SymbolicScalar> oriOffset;
    for (auto offsetValue : offset) {
        oriOffset.push_back(offsetValue.GetSpecifiedValue());
    }

    std::vector<std::vector<SymbolicScalar>> inputShapes;
    std::vector<std::vector<int>> staticInputShapes;

    for (auto inputTensor : op->GetIOperands()) {
        inputShapes.push_back(inputTensor->GetDynValidShape());
        staticInputShapes.push_back(inputTensor->GetShape());
    }
    std::vector<SymbolicScalar> outDynShape = op->GetOOperands()[0]->GetDynValidShape();
    if (outDynShape.empty()) {
        for (size_t i = 0; i < op->GetOOperands()[0]->GetShape().size(); ++i) {
            outDynShape.push_back(SymbolicScalar(0));
        }
    }
    std::vector<SymbolicScalar> outShape;
    for (size_t i = 0U; i < inputShapes[0].size(); i++) {
        SymbolicScalar actualDim;
        if (staticInputShapes[0][i] == op->GetOOperands()[0]->GetShape()[i]) { //src的该维度没有被切分，assmble后该维度大小不变
            actualDim = std::max(SymbolicScalar(0), inputShapes[0][i] + oriOffset[i]);
        } else {
            actualDim = std::max(outDynShape[i], (inputShapes[0][i] + oriOffset[i]));
        }
        outShape.push_back(actualDim);
    }
    for (auto output : op->GetOOperands()) {
        outValisShapes.push_back(outShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_COPY_OUT, Opcode::OP_COPY_OUT, CopyOutInferFunc);

// MTE infer shape func
void TransposeInferFunc(Operation* op,
    std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    for (auto output : op->GetOOperands()) {
        std::vector<SymbolicScalar> res;
        res.insert(res.end(), inputValidShapes[0].begin(), inputValidShapes[0].end());
        auto axises = op->GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
        size_t index0 = axises[0];
        size_t index1 = axises[1];
        if (index0 < res.size() && index1 < res.size()) {
            std::swap(res[index0], res[index1]);
        }
        outValidShapes.push_back(res);
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_TRANSPOSE_VNCHWCONV, Opcode::OP_TRANSPOSE_VNCHWCONV, TransposeInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_TRANSPOSE_DATAMOVE, Opcode::OP_TRANSPOSE_DATAMOVE, TransposeInferFunc);

void ViewInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op->GetOpAttribute().get());
    if (viewOpAttribute == nullptr) {
        ALOG_WARN_F("View [%d] has no view attr.", op->GetOpMagic());
        outValidShapes.push_back(op->GetIOperands()[0]->GetDynValidShape());
        return;
    }
    // view的toDynValidShape是前端已经预设好，直接使用即可
    auto toValidShape = viewOpAttribute->GetToDynValidShape();
    if (!toValidShape.empty()) {
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(toValidShape);
        }
    } else {
        auto inputValidShape = op->GetIOperands()[0]->GetDynValidShape();
        if (inputValidShape.empty()) {
            auto shapeImm = OpImmediate::Specified(op->GetIOperands()[0]->GetShape());
            inputValidShape.resize(shapeImm.size());
            OpImmediate::NormalizeValue(inputValidShape, 0, shapeImm, 0, false);
        }
        auto newDynValidShape = GetViewValidShape(inputValidShape, viewOpAttribute->GetFromOffset(),
                                                    viewOpAttribute->GetFromDynOffset(), op->GetOOperands()[0]->GetShape());
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(newDynValidShape);
        }
        viewOpAttribute->SetToDynValidShape(newDynValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_VIEW, Opcode::OP_VIEW, ViewInferFunc);

void AssembleInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(op->GetOpAttribute().get());
    if (assembleOpAttribute != nullptr) {
        auto fromValidShape = op->GetIOperands()[0]->GetDynValidShape();
        assembleOpAttribute->SetFromDynValidShape(fromValidShape);
    } else {
        ALOG_WARN_F("Copyout [%d] has no copy out attr.", op->GetOpMagic());
        outValidShapes.push_back(op->GetIOperands()[0]->GetDynValidShape());
        return;
    }
    auto offset = assembleOpAttribute->GetToOffset();
    auto inputShapes = op->GetIOperands()[0]->GetDynValidShape();
    std::vector<SymbolicScalar> outDynShape = op->GetOOperands()[0]->GetDynValidShape();
    if (outDynShape.empty()) {
        for (size_t i = 0; i < op->GetOOperands()[0]->GetShape().size(); ++i) {
            outDynShape.push_back(SymbolicScalar(0));
        }
    }
    std::vector<SymbolicScalar> outShape;
    for (size_t i = 0U; i < inputShapes.size(); i++) {
        SymbolicScalar actualDim = std::max(outDynShape[i], (inputShapes[i] + offset[i]));
        outShape.push_back(actualDim);
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_ASSEMBLE, Opcode::OP_ASSEMBLE, AssembleInferFunc);

const std::string TOPK_AXIS = OP_ATTR_PREFIX + "axis";
const std::string TOPK_ORDER = OP_ATTR_PREFIX + "order";
const std::string TOPK_KVALUE = OP_ATTR_PREFIX + "kvalue";
const std::string EXTRACT_MASKMODE = OP_ATTR_PREFIX + "makeMode";
constexpr int32_t blockSize = 32;
constexpr int32_t kFactorSize = 4;
constexpr int32_t kBlockFpNum = 8;

// m,n -> m,4*n align32
void BitSortFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &outValidShapes) {
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    std::vector<SymbolicScalar> res(inputValidShapes[0]);
    auto topk_axis = op->GetIntAttribute(TOPK_AXIS);
    res[topk_axis] = (res[topk_axis] + blockSize - 1) / blockSize * blockSize;
    res[topk_axis] = res[topk_axis] * kFactorSize; // todo topk_axis -1  what happen?
    outValidShapes.push_back(res);
}

REGISTER_INFER_SHAPE_FUNC(OP_BITSORT, Opcode::OP_BITSORT, BitSortFunc);

// m,4 *n align32byte -> m, 2 * k align8
void MrgSortFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &outValidShapes) {
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    std::vector<SymbolicScalar> res(inputValidShapes[0]);
    auto topk_axis = op->GetIntAttribute(TOPK_AXIS);
    auto topk_kvalue = op->GetIntAttribute(TOPK_KVALUE);
    res[topk_axis] = (topk_kvalue + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum * NUM2;
    outValidShapes.push_back(res);
}

REGISTER_INFER_SHAPE_FUNC(OP_MRGSORT, Opcode::OP_MRGSORT, MrgSortFunc);

// m, 2 * k align8 -> m, k
void ExtractFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &outValidShapes) {
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    std::vector<SymbolicScalar> res(inputValidShapes[0]);
    res.back() = op->GetIntAttribute(TOPK_KVALUE);
    outValidShapes.push_back(res);
}

REGISTER_INFER_SHAPE_FUNC(OP_EXTRACT, Opcode::OP_EXTRACT, ExtractFunc);

void VecDupInferFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &validShapes) {
    std::vector<SymbolicScalar> validShape;
    op->GetAttr(OP_ATTR_PREFIX + "validShape", validShape);
    validShapes.push_back(validShape);
}
REGISTER_INFER_SHAPE_FUNC(OP_VEC_DUP, Opcode::OP_VEC_DUP, VecDupInferFunc);

void ReshapeInferFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &validShapes) {
    std::vector<SymbolicScalar> validShape;
    if (op->GetAttr(OP_ATTR_PREFIX + "validShape", validShape) && validShape.size() != 0) {
        validShapes.push_back(validShape);
    } else {
        auto dstShape = op->GetOOperands()[0]->GetShape();
        validShapes.push_back(SymbolicScalar::FromConcrete(dstShape));
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_RESHAPE, Opcode::OP_RESHAPE, ReshapeInferFunc);
}  // namespace npu::tile_fwk
