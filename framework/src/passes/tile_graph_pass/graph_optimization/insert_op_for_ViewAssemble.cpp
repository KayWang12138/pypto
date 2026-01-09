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
 * \file insert_op_for_ViewAssemble.cpp
 * \brief
 */
#include "insert_op_for_ViewAssemble.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "InsertCopyForViewAssemble"

namespace npu {
namespace tile_fwk {
void InsertCopyForViewAssemble::InsertViewAssemble(Function &function, Operation *viewOp, Operation *assembleOp) {
    auto &moveOutTensorPtr = viewOp->GetOOperands()[0];
    LogicalTensor ddrTensor(function, moveOutTensorPtr->Datatype(), moveOutTensorPtr->GetShape());
    ddrTensor.SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    LogicalTensor moveInTensor(function, moveOutTensorPtr->Datatype(), moveOutTensorPtr->GetShape());
    moveInTensor.SetMemoryTypeBoth(moveOutTensorPtr->GetMemoryTypeOriginal());
    LogicalTensorPtr ddrTensorPtr = std::make_shared<LogicalTensor>(std::move(ddrTensor));
    LogicalTensorPtr moveInTensorPtr = std::make_shared<LogicalTensor>(std::move(moveInTensor));
    std::vector<int64_t> offset(moveOutTensorPtr->GetShape().size(), 0);
    std::vector<SymbolicScalar> dynOffset(moveOutTensorPtr->GetShape().size(), 0);
    Operation &assemble = function.AddRawOperation(Opcode::OP_ASSEMBLE, {moveOutTensorPtr}, {ddrTensorPtr});
    assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(moveOutTensorPtr->GetMemoryTypeOriginal(), 
                                                                  offset, 
                                                                  dynOffset, 
                                                                  moveOutTensorPtr->GetDynValidShape()));
    Operation &view = function.AddRawOperation(Opcode::OP_VIEW, {ddrTensorPtr}, {moveInTensorPtr});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, 
                                                          moveOutTensorPtr->GetMemoryTypeOriginal(), 
                                                          dynOffset, 
                                                          moveOutTensorPtr->GetDynValidShape()));
    assembleOp->ReplaceInput(moveInTensorPtr, moveOutTensorPtr);
    moveOutTensorPtr->RemoveConsumer(assembleOp);
    assembleOp->EraseInput(moveOutTensorPtr);
}

Status InsertCopyForViewAssemble::InsertCopy(Function &function, std::pair<Operation *, Operation *> &opPair) {
    auto viewOp = opPair.first;
    auto assOp = opPair.second;
    if (assOp->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        assOp->GetIOperands()[0]->SetMemoryTypeBoth(MemoryType::MEM_UB);
    } else if (assOp->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1 ||
               assOp->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        InsertViewAssemble(function, viewOp, assOp);
    } else {
        APASS_LOG_ERROR_F(Elements::Operation, "Assemble outTensor %d memory type is unexpected, InsertCopy failed.",
                          assOp->GetOOperands()[0]->GetMagic());
        return FAILED;
    }
    return SUCCESS;
}

bool InsertCopyForViewAssemble::NeedInsertCopy(LogicalTensorPtr &assembleOut) {
    bool isNeedInsert = false;
    for (auto &assOp : assembleOut->GetProducers()) {
        auto &prodOp = *assOp->GetIOperands()[0]->GetProducers().begin();
        if (prodOp->GetOpcode() != Opcode::OP_VIEW) {
            isNeedInsert = true;
            continue;
        }
        recordOpPair.push_back(std::make_pair(prodOp, assOp));
        auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(assOp->GetOpAttribute());
        auto viewAttr = std::static_pointer_cast<ViewOpAttribute>(prodOp->GetOpAttribute());
        if (assembleAttr == nullptr || viewAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "View or Assemble attribute is nullptr, NeedInsertCopy Failed.");
            return FAILED;
        }
        if (assembleAttr->GetToOffset() != viewAttr->GetFromOffset()) {
            isNeedInsert = true;
            continue;
        }
        auto viewIn = prodOp->GetIOperands()[0];
        auto inShape = viewIn->GetShape();
        auto outShape = assembleOut->GetShape();
        if (inShape.size() != outShape.size()) {
            isNeedInsert = true;
            continue;
        }
        for (size_t i = 0; i < inShape.size(); i++) {
            if (inShape[i] != outShape[i]) {
                isNeedInsert = true;
                break;
            }
        }
    }
    return isNeedInsert;
}

Status InsertCopyForViewAssemble::JudgedViewAssemble(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto &prodOp = *op.GetIOperands()[0]->GetProducers().begin();
        if (prodOp->GetOpcode() == Opcode::OP_VIEW && assembleOutSet.find(op.GetOOperands()[0]) == assembleOutSet.end()) {
            assembleOutSet.insert(op.GetOOperands()[0]);
        }
    }
    for (auto assembleOut : assembleOutSet) {
        recordOpPair.clear();
        if (NeedInsertCopy(assembleOut)) {
            for (auto &opPair : recordOpPair) {
                if (InsertCopy(function, opPair) == FAILED) {
                    return FAILED;
                }
            }
        }
    }
    return SUCCESS;
}

Status InsertCopyForViewAssemble::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start InsertCopyForViewAssemble");
    if (JudgedViewAssemble(function) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "JudgedViewAssemble Failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End InsertCopyForViewAssemble");
    return SUCCESS;
}
}
}