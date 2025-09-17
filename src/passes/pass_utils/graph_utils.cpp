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
 * \file graph_utils.cpp
 * \brief
 */

#include "graph_utils.h"

namespace npu {
namespace tile_fwk {
void GraphUtils::SetDynShape(Operation *newOp, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    if (outDynShape.empty()) {
        InferShapeRegistry::GetInstance().CallInferShapeFunc(newOp);
    } else {
        for (size_t i = 0; i < newOp->GetOOperands().size(); ++i) {
            newOp->GetOOperands()[i]->UpdateDynValidShape(outDynShape[i]);
        }
    }
}

Operation &GraphUtils::AddDynOperation(Function &function, const Opcode opCode, LogicalTensors iOperands,
                                       const LogicalTensors &oOperands, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = function.AddOperation(opCode, iOperands, oOperands);
    SetDynShape(&newOp, outDynShape);
    return newOp;
}

Operation &GraphUtils::AddDynRawOperation(Function &function, const Opcode opCode, LogicalTensors iOperands,
                                          const LogicalTensors &oOperands, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = function.AddRawOperation(opCode, iOperands, oOperands);
    SetDynShape(&newOp, outDynShape);
    return newOp;
}

Operation &GraphUtils::AddViewOperation(Function &function, const ViewOp &view, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = AddDynOperation(function, Opcode::OP_VIEW, {view.input}, {view.output}, outDynShape);
    std::vector<SymbolicScalar> toDynShape = view.output->GetDynValidShape();
    auto viewAttribute =std::make_shared<ViewOpAttribute>(view.fromOffset);
    viewAttribute->SetToDynValidShape(toDynShape);
    viewAttribute->SetToType(view.toType);
    newOp.SetOpAttribute(viewAttribute);
    UpdateViewAttr(function, newOp);
    return newOp;
}

Operation &GraphUtils::AddAssembleOperation(Function &function, const AssembleOp &assemble, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = function.AddRawOperation(Opcode::OP_ASSEMBLE, {assemble.input}, {assemble.output});
    auto assembleOpAttribute = std::make_shared<AssembleOpAttribute>(assemble.from, assemble.toOffset);
    auto fromValidShape = assemble.input->GetDynValidShape();
    assembleOpAttribute->SetFromDynValidShape(fromValidShape);
    newOp.SetOpAttribute(assembleOpAttribute);
    SetDynShape(&newOp, outDynShape);
    return newOp;
}

Operation &GraphUtils::AddReshapeOperation(Function &function, LogicalTensorPtr iOperand, const LogicalTensorPtr &oOperand, const std::vector<SymbolicScalar> &outDynShape) {
    auto &newOp = function.AddOperation(Opcode::OP_RESHAPE, {iOperand}, {oOperand});
    if (outDynShape.empty()) {
        InferShapeRegistry::GetInstance().CallInferShapeFunc(&newOp);
        std::vector<SymbolicScalar> validShape;
        if (!newOp.GetAttr(OP_ATTR_PREFIX + "validShape", validShape) || validShape.empty()) {
            newOp.SetAttribute(OP_ATTR_PREFIX + "validShape", oOperand->GetDynValidShape());
        }
    } else {
        newOp.SetAttribute(OP_ATTR_PREFIX + "validShape", outDynShape);
        oOperand->UpdateDynValidShape(outDynShape);
    }
    return newOp;
}

void GraphUtils::SetCopyInAttr(Operation *newOp, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto copyAttr = std::make_shared<CopyOpAttribute>(copy.Offset, copy.from, copy.shape, copy.rawShape, copy.fromDynValidShape);
    newOp->SetOpAttribute(copyAttr);
    SetDynShape(newOp, outDynShape);
    newOp->UpdateSubgraphID(copy.output->subGraphID);
}

void GraphUtils::SetCopyOutAttr(Operation *newOp, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto copyAttr = std::make_shared<CopyOpAttribute>(copy.from, copy.Offset, copy.shape, copy.rawShape, copy.fromDynValidShape);
    newOp->SetOpAttribute(copyAttr);
    SetDynShape(newOp, outDynShape);
    newOp->UpdateSubgraphID(copy.input->subGraphID);
}

Operation &GraphUtils::AddCopyInOperation(Function &function, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = function.AddOperation(Opcode::OP_COPY_IN, {copy.input}, {copy.output});
    SetCopyInAttr(&newOp, copy, outDynShape);
    return newOp;
}

Operation &GraphUtils::AddCopyInRawOperation(Function &function, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = function.AddRawOperation(Opcode::OP_COPY_IN, {copy.input}, {copy.output});
    SetCopyInAttr(&newOp, copy, outDynShape);
    return newOp;
}

Operation &GraphUtils::AddCopyOutOperation(Function &function, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = function.AddOperation(Opcode::OP_COPY_OUT, {copy.input}, {copy.output});
    SetCopyOutAttr(&newOp, copy, outDynShape);
    return newOp;
}

Operation &GraphUtils::AddCopyOutRawOperation(Function &function, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = function.AddRawOperation(Opcode::OP_COPY_OUT, {copy.input}, {copy.output});
    SetCopyOutAttr(&newOp, copy, outDynShape);
    return newOp;
}

void GraphUtils::CopyDynStatus(const LogicalTensorPtr &dstTensor, const LogicalTensorPtr &srcTensor) {
    dstTensor->UpdateDynValidShape(srcTensor->GetDynValidShape());
}

void GraphUtils::UpdateViewAttr(Function &function, Operation &op) {
    LogicalTensorPtr input = op.GetIOperands().front();
    LogicalTensorPtr output = op.GetIOperands().front();
    auto viewAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (function.IsFromInCast(input) || function.IsFromOutCast(output)) {
        if (viewAttribute->GetFromDynOffset().empty()) {
            std::vector<int64_t> fromOffset = viewAttribute->GetFromOffset();
            std::vector<SymbolicScalar> fromDynOffset = SymbolicScalar::FromConcrete(fromOffset);
            viewAttribute->SetFromOffset(fromOffset, fromDynOffset);
        }
    }
}
}
}