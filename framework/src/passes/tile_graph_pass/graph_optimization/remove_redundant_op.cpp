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
bool EqualShapeInOut(const Operation &op) {
    auto in = op.GetIOperands().front();
    auto out = op.GetOOperands().front();
    // 比较静态shape
    bool equalShape = (in->GetShape() == out->GetShape());
    // 比较动态dynValidShape_
    bool equalDynValidShape = true;
    if (!in->GetDynValidShape().empty() && !out->GetDynValidShape().empty()) {
        auto inDynValidShape = in->GetDynValidShape();
        auto outDynValidShape = out->GetDynValidShape();
        for (size_t i = 0; i < inDynValidShape.size(); i++) {
            // 比较SymbolicScalar dump后的string是否相等
            // 可能是 1.concrete value; 2.symbol; 3.expression
            if (inDynValidShape[i].Dump() == outDynValidShape[i].Dump()) {
                continue;
            } else {
                equalDynValidShape = false;
                break;
            }
        }
    } else if (in->GetDynValidShape().empty() && out->GetDynValidShape().empty()) {
        // 输入和输出同时没有 dynamic valid shape
        equalDynValidShape = true;
    } else {
        // 输入和输出必须同时有 dynamic valid shape
        equalDynValidShape = false;
    }
    return (equalShape && equalDynValidShape);
}

Status ProcessRegCopy(const Operation &op, const Function &function, bool &needToDelete) {
    auto regCopyIn = op.iOperand.front();
    auto regCopyOut= op.oOperand.front();
    if (regCopyIn->shape == regCopyOut->shape && regCopyIn->GetMemoryTypeOriginal() == regCopyOut->GetMemoryTypeOriginal()) {
        /*
        register copy 输入和输出且memtype相同，无拷贝意义
        view -> ub -> register copy -> ub -> op
        view -> ub  -> op
        */
        auto consumerOps = function.FindConsumers(op);
        /* register copy 一定有后继op*/
        if (consumerOps.empty()) {
            APASS_LOG_ERROR_F("RemoveRedundantOp", "Operation", 
            "OP_REG_COPY[%d]'s output has no consumer; OP_REG_COPY[%d]'s output must have consumer.", op.opmagic, op.opmagic);
            return FAILED;
        }
        for (auto &consumerOp : consumerOps) {
            consumerOp->ReplaceInput(regCopyIn, regCopyOut);
        }
        needToDelete = true;
        APASS_LOG_DEBUG_F("RemoveRedundantOp", "Operation", "Delete Redundant OP_REGISTER_COPY opmagic: %d.", op.opmagic);
    }
    return SUCCESS;
}

Status ProcessAssembleDDR(const Operation &op, const LogicalTensorPtr &assembleIn, const LogicalTensorPtr &assembleOut,
    Function &function, bool &needToDelete) {
    bool allProdView{true};
    for (auto &prod : function.FindProducers(op)) {
        if (prod->GetOpcode() != Opcode::OP_VIEW) {
            allProdView = false;
        }
    }
    if (allProdView) { return SUCCESS; }
    auto consumerOps = function.FindConsumers(op);
    if (!consumerOps.empty()) {
        for (auto &consumerOp : consumerOps) {
            consumerOp->ReplaceInput(assembleOut, assembleIn);
        }
        needToDelete = true;
        APASS_LOG_DEBUG_F("RemoveRedundantOp", "Operation", "Delete Redundant OP_ASSEMBLE on DDR opmagic: %d.", op.opmagic);
        return SUCCESS;
    }
    /* DDR --> Assemble --> OUTCAST */
    if (assembleOut->nodetype != NodeType::OUTCAST || !function.IsFromOutCast(assembleOut)) {
        APASS_LOG_ERROR_F("RemoveRedundantOp", "Operation", 
        "OP_ASSEMBLE[%d]'s output has no consumer but is not outcast; Please check if the OP_ASSEMBLE[%d]'s output is outcast.", op.opmagic, op.opmagic);
        return FAILED;
    }
    APASS_LOG_DEBUG_F("RemoveRedundantOp", "Operation", "OP_ASSEMBLE has no consumers, opmagic: %d.", op.opmagic);
    auto childOpsBackup = assembleIn->GetConsumers();
    for (auto &childOp : childOpsBackup) {
        if (childOp->GetOpMagic() == op.GetOpMagic()) {
            continue;
        }
        childOp->ReplaceInput(assembleOut, assembleIn);
        APASS_LOG_DEBUG_F("RemoveRedundantOp", "Tensor", "Repalce input of %s opmagic: %d, tensor %d --> tensor %d.", 
        childOp->GetOpcodeStr().c_str(), childOp->GetOpMagic(), assembleIn->magic, assembleOut->magic);
    }
    auto producerOps = op.ProducerOps();
    for (auto &producerOp : producerOps) {
        producerOp->ReplaceOutput(assembleOut, assembleIn);
        APASS_LOG_DEBUG_F("RemoveRedundantOp", "Tensor", "Repalce output of %s opmagic: %d, tensor %d --> tensor %d",
            producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic(), assembleIn->magic, assembleOut->magic);
    }
    needToDelete = true;
    APASS_LOG_DEBUG_F("RemoveRedundantOp", "Operation", "Delete Redundant OP_ASSEMBLE on DDR opmagic: %d.", op.opmagic);
    return SUCCESS;
}

Status ProcessAssembleUB(const Operation &op, const LogicalTensorPtr &ASSEMBLE_in, const LogicalTensorPtr &ASSEMBLE_out,
    Function &function, bool &needToDelete) {
    /*
    assemble 输入和输出相同，无意义
    */
    auto consumerOps = function.FindConsumers(op);
    /* UB 上的 ASSEMBLE 一定有后继op*/
    if (consumerOps.empty()) {
        APASS_LOG_ERROR_F("RemoveRedundantOp", "Operation", 
        "OP_ASSEMBLE[%d]'s output is empty; OP_ASSEMBLE[%d]'s output for ub must have consumer.", op.opmagic, op.opmagic);
        return FAILED;
    }
    for (auto &consumerOp : consumerOps) {
        consumerOp->ReplaceInput(ASSEMBLE_in, ASSEMBLE_out);
    }
    needToDelete = true;
    APASS_LOG_DEBUG_F("RemoveRedundantOp", "Operation", "Delete Redundant OP_ASSEMBLE on UB opmagic: %d", op.opmagic);
    return SUCCESS;
}

Status ProcessAssemble(const Operation &op, Function &function, bool &needToDelete) {
    auto ASSEMBLE_in = op.iOperand.front();
    auto ASSEMBLE_out = op.oOperand.front();
    if (ASSEMBLE_in->shape != ASSEMBLE_out->shape) {
        return SUCCESS;
    }
    if (ASSEMBLE_in->GetMemoryTypeOriginal() != ASSEMBLE_out->GetMemoryTypeOriginal()) {
        return SUCCESS;
    }
    if (ASSEMBLE_in->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
        ProcessAssembleDDR(op, ASSEMBLE_in, ASSEMBLE_out, function, needToDelete)) {
        APASS_LOG_ERROR_F("RemoveRedundantOp", "Operation", "ProcessAssembleDDR failed.");
        return FAILED;
    }
    if (ASSEMBLE_in->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
        ProcessAssembleUB(op, ASSEMBLE_in, ASSEMBLE_out, function, needToDelete)) {
        APASS_LOG_ERROR_F("RemoveRedundantOp", "Operation", "ProcessAssembleUB failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessView(const Operation &op, Function &function, bool &needToDelete) {
    auto in = op.iOperand.front();
    auto out = op.oOperand.front();
    if (EqualShapeInOut(op) && in->GetMemoryTypeOriginal() == out->GetMemoryTypeOriginal()) {
        auto consumerOps = function.FindConsumers(op);
        if (consumerOps.empty()) {
            auto producerOps = op.ProducerOps();
            for (auto &producerOp : producerOps) {
                producerOp->ReplaceOutput(out, in);
            }
        } else {
            for (auto &consumerOp : consumerOps) {
                consumerOp->ReplaceInput(in, out);
            }
        }
        needToDelete = true;
        APASS_LOG_DEBUG_F("RemoveRedundantOp", "Operation", "Delete Redundant OP_VIEW opmagic: %d", op.opmagic);
    }
    if (out->GetConsumers().size() == 1) {
        auto childOp = *(out->GetConsumers().begin());
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
    if (op.GetIOperands().size() != 1 || op.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F("RemoveRedundantOp", "Operation", 
        "Expand[%d] has incorrect input/output num; Please check the Expand[%d]'s input/output num.", op.opmagic, op.opmagic);
        return FAILED;
    }
    needToDelete = false;
    if (EqualShapeInOut(op)) {
        needToDelete = true;
    }
    return SUCCESS;
}
}

Status RemoveRedundantOp::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "===> Start RemoveRedundantOp");
    if (DeleteRedundantOps(function) != SUCCESS) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "DeleteRedundantOps failed.");
        return FAILED;
    }
    if (RemoveDummyExpand(function) != SUCCESS) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "RemoveDummyExpand failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "===> End RemoveRedundantOp");
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
            if (ProcessExpand(op, needToDelete) != SUCCESS) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "ProcessExpand failed.");
                return FAILED;
            }
            if (needToDelete) {
                dummyOp.push_back(&op);
                APASS_LOG_INFO_F(GetName().c_str(), "Operation", "Delete OP_EXPAND opmagic: %d.", op.opmagic);
            }
        }
    }
    for (const auto &op : dummyOp) {
        function.UpdateOperandBeforeRemoveOp(*op, false);
    }
    for (auto op : dummyOp) {
        if (op->IsDeleted()) {
            APASS_LOG_ERROR_F(GetName().c_str(), "Operation", 
            "Found invalid op[%d]; Please check the op[%d] is not deleted (RemoveDummyExpand).", op->opmagic, op->opmagic);
            return FAILED;
        }
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
            if (ProcessRegCopy(op, function, needToDelete)) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "ProcessRegCopy failed.");
                return FAILED;
            }
            break;
        }
        case Opcode::OP_ASSEMBLE: {
            if (ProcessAssemble(op, function, needToDelete)) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "ProcessAssemble failed.");
                return FAILED;
            }
            break;
        }
        case Opcode::OP_VIEW: {
            if (ProcessView(op, function, needToDelete)) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "ProcessView failed.");
                return FAILED;
            }
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
        if (NeedToDelete(op, function, needToDelete) != SUCCESS) {
            APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "NeedToDelete failed.");
            return FAILED;
        }
        if (needToDelete) {
            redundantOp.push_back(&op);
        }
    }
    for (const auto &op : redundantOp) {
        function.HandleControlOps(*op, redundantOp);
        function.UpdateOperandBeforeRemoveOp(*op, false);
    }
    for (auto op : redundantOp) {
        if (op->IsDeleted()) {
            APASS_LOG_ERROR_F(GetName().c_str(), "Operation", 
            "Found invalid op[%d]; Please check the op[%d] is not deleted (DeleteRedundantOps).", op->opmagic, op->opmagic);
            return FAILED;
        }
        op->SetAsDeleted();
    }
    function.EraseOperations(true);
    return SUCCESS;
}
} // namespace npu::tile_fwk