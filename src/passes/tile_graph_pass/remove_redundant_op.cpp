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
 * \file remove_redundant_op.cpp
 * \brief
 */

#include "remove_redundant_op.h"
#include "passes/pass_check/remove_redundant_op_checker.h"

using namespace npu::tile_fwk;
namespace npu::tile_fwk {
namespace {
Status ProcessRegCopy(const Operation &op, const Function &function, bool &needToDelete) {
    auto regCopyIn = op.iOperand.front();
    if (regCopyIn == nullptr) {return FAILED;}
    auto regCopyOut= op.oOperand.front();
    if (regCopyOut == nullptr) {return FAILED;}
    if (regCopyIn->shape == regCopyOut->shape && regCopyIn->GetMemoryTypeOriginal() == regCopyOut->GetMemoryTypeOriginal()) {
        /*
        register copy 输入和输出且memtype相同，无拷贝意义
        view -> ub -> register copy -> ub -> op
        view -> ub  -> op
        */
        auto consumerOps = function.FindConsumers(op);
        /* register copy 一定有后继op*/
        if (consumerOps.empty()) {return FAILED;}
        for (auto &consumerOp : consumerOps) {
            if (consumerOp == nullptr) {return FAILED;}
            consumerOp->ReplaceInput(regCopyIn, regCopyOut);
        }
        needToDelete = true;
        ALOG_DEBUG_F("[RemoveRedundantOp] Delete Redundant OP_REGISTER_COPY opmagic: %d", op.opmagic);
    }
    return SUCCESS;
}

Status ProcessAssembleDDR(const Operation &op, const LogicalTensorPtr &assembleIn, const LogicalTensorPtr &assembleOut,
    Function &function, bool &needToDelete) {
    auto consumerOps = function.FindConsumers(op);
    if (!consumerOps.empty()) {
        for (auto &consumerOp : consumerOps) {
            consumerOp->ReplaceInput(assembleOut, assembleIn);
        }
        needToDelete = true;
        ALOG_DEBUG_F("[RemoveRedundantOp] Delete Redundant OP_ASSEMBLE on DDR opmagic: %d", op.opmagic);
        return SUCCESS;
    }
    /* DDR --> Assemble --> OUTCAST */
    if (assembleOut->nodetype != NodeType::OUTCAST) {return FAILED;}
    if (!function.IsFromOutCast(assembleOut)) {return FAILED;}
    ALOG_DEBUG_F("[RemoveRedundantOp] OP_ASSEMBLE has no consumers, opmagic: %d", op.opmagic);
    auto childOpsBackup = assembleIn->GetConsumers();
    for (auto &childOp : childOpsBackup) {
        if (childOp->GetOpMagic() == op.GetOpMagic()) {
            continue;
        }
        childOp->ReplaceInput(assembleOut, assembleIn);
        ALOG_DEBUG_F("Repalce input of %s opmagic: %d, tensor %d --> tensor %d", childOp->GetOpcodeStr().c_str(),
            childOp->GetOpMagic(), assembleIn->magic, assembleOut->magic);
    }
    auto producerOps = op.ProducerOps();
    for (auto &producerOp : producerOps) {
        producerOp->ReplaceOutput(assembleOut, assembleIn);
        ALOG_DEBUG_F("Repalce output of %s opmagic: %d, tensor %d --> tensor %d",
            producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic(), assembleIn->magic, assembleOut->magic);
    }
    needToDelete = true;
    ALOG_DEBUG_F("[RemoveRedundantOp] Delete Redundant OP_ASSEMBLE on DDR opmagic: %d", op.opmagic);
    return SUCCESS;
}

Status ProcessAssembleUB(const Operation &op, const LogicalTensorPtr &ASSEMBLE_in, const LogicalTensorPtr &ASSEMBLE_out,
    Function &function, bool &needToDelete) {
    /*
    assemble 输入和输出相同，无意义
    */
    auto consumerOps = function.FindConsumers(op);
    /* UB 上的 ASSEMBLE 一定有后继op*/
    if (consumerOps.empty()) {return FAILED;}
    for (auto &consumerOp : consumerOps) {
        if (consumerOp == nullptr) {return FAILED;}
        consumerOp->ReplaceInput(ASSEMBLE_in, ASSEMBLE_out);
    }
    needToDelete = true;
    ALOG_DEBUG_F("[RemoveRedundantOp] Delete Redundant OP_ASSEMBLE on UB opmagic: %d", op.opmagic);
    return SUCCESS;
}

Status ProcessAssemble(const Operation &op, Function &function, bool &needToDelete) {
    auto ASSEMBLE_in = op.iOperand.front();
    if (ASSEMBLE_in == nullptr) {
        return FAILED;
    }
    auto ASSEMBLE_out = op.oOperand.front();
    if (ASSEMBLE_out == nullptr) {
        return FAILED;
    }
    if (ASSEMBLE_in->shape != ASSEMBLE_out->shape) {
        return SUCCESS;
    }
    if (ASSEMBLE_in->GetMemoryTypeOriginal() != ASSEMBLE_out->GetMemoryTypeOriginal()) {
        return SUCCESS;
    }
    if (ASSEMBLE_in->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
        ProcessAssembleDDR(op, ASSEMBLE_in, ASSEMBLE_out, function, needToDelete)) {
        return FAILED;
    }
    if (ASSEMBLE_in->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
        ProcessAssembleUB(op, ASSEMBLE_in, ASSEMBLE_out, function, needToDelete)) {
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessView(const Operation &op, Function &function, bool &needToDelete) {
    auto in = op.iOperand.front();
    if (in == nullptr) {return FAILED;}
    auto out = op.oOperand.front();
    if (out == nullptr) {return FAILED;}
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (viewOpAttribute) {
        auto newDynValidShape = viewOpAttribute->GetToDynValidShape();
        std::vector<int64_t> validShape;
        for (auto validSym : newDynValidShape) {
            if (!validSym.ConcreteValid()) {needToDelete = false; return SUCCESS;}
            validShape.push_back(validSym.Concrete());
        }
        if (!newDynValidShape.empty() && out->shape != validShape) { needToDelete = false; return SUCCESS; }
    }
    if (in->shape == out->shape && in->GetMemoryTypeOriginal() == out->GetMemoryTypeOriginal()) {
        auto consumerOps = function.FindConsumers(op);
        if (consumerOps.empty()) {
            auto producerOps = op.ProducerOps();
            for (auto &producerOp : producerOps) {
                if (producerOp == nullptr) {return FAILED;}
                producerOp->ReplaceOutput(out, in);
            }
        } else {
            for (auto &consumerOp : consumerOps) {
                if (consumerOp == nullptr) {return FAILED;}
                consumerOp->ReplaceInput(in, out);
            }
        }
        needToDelete = true;
        ALOG_DEBUG_F("[RemoveRedundantOp] Delete Redundant OP_VIEW opmagic: %d", op.opmagic);
    }
    if (out->GetConsumers().size() == 1) {
        auto childOp = *(out->GetConsumers().begin());
        if (childOp == nullptr) {return FAILED;}
        if (childOp->GetOpcode() == Opcode::OP_COMM_WAIT_FLAG) {
            childOp->ReplaceInput(in, out);
            needToDelete = true;
        }
    }
    return SUCCESS;
}

/*
before:
                                            / --> child1
                                            /
inputTensor -> op (OP_EXPAND) -> outputTensor  --> child2
                                            \
                                            \ --> child3

after:
            / --> child1
            /
inputTensor    --> child2
            \
            \ --> child3
*/
Status ProcessExpand(const Operation &op, bool &needToDelete) {
    if (op.GetIOperands().size() != 1 || op.GetOOperands().size() != 1) {return FAILED;}
    auto inputTensor = op.GetIOperands().front();
    if (inputTensor == nullptr) {return FAILED;}
    auto outputTensor = op.GetOOperands().front();
    if (outputTensor == nullptr) {return FAILED;}
    if (inputTensor->shape.size() != outputTensor->shape.size()) {return SUCCESS;}
    needToDelete = true;
    for (size_t dimIdx = 0; dimIdx < inputTensor->shape.size(); dimIdx++) {
        if (inputTensor->shape[dimIdx] != outputTensor->shape[dimIdx]) {
            needToDelete = false;
        }
    }
    return SUCCESS;
}
}

Status RemoveRedundantOp::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start RemoveRedundantOp");
    if (DeleteRedundantOps(function) != SUCCESS) {return FAILED;}
    if (RemoveDummyExpand(function) != SUCCESS) {return FAILED;}
    ALOG_INFO_F("===> End RemoveRedundantOp");
    return SUCCESS;
}

Status RemoveRedundantOp::PreCheck(Function &function) {
    RemoveRedundantOpChecker checker;
    return checker.DoPreCheck(function);
}

Status RemoveRedundantOp::PostCheck(Function &function) {
    RemoveRedundantOpChecker checker;
    return checker.DoPostCheck(function);
}

Status RemoveRedundantOp::RemoveDummyExpand(Function &function) const {
    std::vector<Operation *> dummyOp;
    bool needToDelete;
    for (auto &op: function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_EXPAND) {
            if (ProcessExpand(op, needToDelete) != SUCCESS) {return FAILED;}
            if (needToDelete) {
                dummyOp.push_back(&op);
                ALOG_INFO_F("Delete OP_EXPAND opmagic: %d.", op.opmagic);
            }
        }
    }
    for (const auto &op : dummyOp) {
        function.UpdateOperandBeforeRemoveOp(*op, false);
    }
    for (auto op : dummyOp) {
        if (op->IsDeleted()) {return FAILED;}
        op->SetAsDeleted();
    }
    function.EraseOperations(true);
    return SUCCESS;
}

Status RemoveRedundantOp::NeedToDelete(const Operation &op, Function &function, bool &needToDelete) const {
    needToDelete = false;
    auto opcode = op.GetOpcode();
    switch (opcode) {
        case Opcode::OP_REGISTER_COPY: {
            if (ProcessRegCopy(op, function, needToDelete)) {return FAILED;}
            break;
        }
        case Opcode::OP_ASSEMBLE: {
            if (ProcessAssemble(op, function, needToDelete)) {return FAILED;}
            break;
        }
        case Opcode::OP_VIEW: {
            if (ProcessView(op, function, needToDelete)) {return FAILED;}
            break;
        }
        default:
            break;
    }
    return SUCCESS;
}

Status RemoveRedundantOp::DeleteRedundantOps(Function &function) const {
    std::vector<Operation *> redundantOp;
    bool needToDelete;
    for (auto &op : function.Operations()) {
        if (NeedToDelete(op, function, needToDelete) != SUCCESS) {return FAILED;}
        if (needToDelete) {
            redundantOp.push_back(&op);
        }
    }
    for (const auto &op : redundantOp) {
        function.HandleControlOps(*op, redundantOp);
        function.UpdateOperandBeforeRemoveOp(*op, false);
    }
    for (auto op : redundantOp) {
        if (op->IsDeleted()) {return FAILED;}
        op->SetAsDeleted();
    }
    function.EraseOperations(true);
    return SUCCESS;
}
} // namespace npu::tile_fwk