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
 * \file remove_redundent_op.cpp
 * \brief
 */

#include "remove_redundent_op.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
namespace {
// Precheck for assemble
// in->assemble->out1 (will be removed)
//   ->assemble->out2 (will be removed)
//   ->op->...
Status PreCheckAssemble(const Operation &op, const LogicalTensorPtr &in) {
    uint32_t assembleRemoveNum = 0;
    uint32_t otherOpNum = 0;
    for (auto &childOp : in->GetConsumers()) {
        if (childOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
            auto child_in = op.iOperand.front();
            if (child_in == nullptr) {return FAILED;}
            auto child_out = op.oOperand.front();
            if (child_out == nullptr) {return FAILED;}
            if (child_out->GetConsumers().empty()) {
                if (child_in->shape == child_out->shape &&
                    child_in->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                    child_out->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                    ++assembleRemoveNum;
                }
            } else {
                ++otherOpNum;
            }
        } else {
            ++otherOpNum;
        }
    }
    if (assembleRemoveNum > 1 && otherOpNum > 0) {
        ALOG_ERROR_F("More than one assemble ddr op without consumer!");
        return FAILED;
    }
    return SUCCESS;
}

// Precheck for view
// in->op->...
//   ->view->out (removed with no consumer)
Status PreCheckView(const Operation &op, const LogicalTensorPtr &in) {
    auto out = op.oOperand.front();
    if (out == nullptr) {return FAILED;}
    if (in->shape == out->shape && op.ConsumerOps().empty() && in->GetConsumers().size() > 1) {
        ALOG_ERROR_F("There is another op consumes the input of a view op without consumer!");
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessPreCheck(const Operation &op) {
    if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
        auto assemble_in = op.iOperand.front();
        if (assemble_in == nullptr) {return FAILED;}
        if (PreCheckAssemble(op, assemble_in) != SUCCESS) {
            ALOG_ERROR_F("PreCheck for assemble op[%d] failed!", op.GetOpMagic());
            return FAILED;
        }
    } else if (op.GetOpcode() == Opcode::OP_VIEW) {
        auto view_in = op.iOperand.front();
        if (view_in == nullptr) {return FAILED;}
        if (PreCheckView(op, view_in) != SUCCESS) {
            ALOG_ERROR_F("PreCheck for view op[%d] failed!", op.GetOpMagic());
            return FAILED;
        }  
    }
    return SUCCESS;
}

Status ProcessPostCheckAssemble(const Operation &op) {
    auto assemble_in = op.iOperand.front();
    if (assemble_in == nullptr) {return FAILED;}
    auto assemble_out = op.oOperand.front();
    if (assemble_out == nullptr) {return FAILED;}
    auto parentOp = *assemble_in->GetProducers().begin();
    if (parentOp == nullptr) {return FAILED;}
    if (assemble_in->shape == assemble_out->shape) {
        if (assemble_in->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
            assemble_out->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            ALOG_ERROR_F("PostCheck for assembleDDR op[%d] failed!", op.GetOpMagic());
            return FAILED;
        } else if (assemble_in->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
            assemble_out->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            ALOG_ERROR_F("PostCheck for assembleUB op[%d] failed!", op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status ProcessPostCheckView(const Operation &op) {
    auto view_in = op.iOperand.front();
    if (view_in == nullptr) {return FAILED;}
    auto view_out = op.oOperand.front();
    if (view_out == nullptr) {return FAILED;}
    if (view_in->shape == view_out->shape && view_in->GetMemoryTypeOriginal() == view_out->GetMemoryTypeOriginal()) {
        ALOG_ERROR_F("PostCheck for view op[%d] failed!", op.GetOpMagic());
        return FAILED;
    } else if (view_out->GetConsumers().size() == 1) {
        auto childOp = *(view_out->GetConsumers().begin());
        if (childOp == nullptr) {return FAILED;}
        if (childOp->GetOpcode() == Opcode::OP_COMM_WAIT_FLAG) {return FAILED;}
    }
    return SUCCESS;
}

Status ProcessPostRegCopy(const Operation &op) {
    auto regcopy_in = op.iOperand.front();
    if (regcopy_in == nullptr) {return FAILED;}
    auto regcopy_out = op.oOperand.front();
    if (regcopy_out == nullptr) {return FAILED;}
    if (regcopy_in->shape == regcopy_out->shape) {
        ALOG_ERROR_F("PostCheck for regcopy op[%d] failed!", op.GetOpMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessPostCopyIn(const Operation &op) {
    auto copy_in = op.iOperand.front();
    if (copy_in == nullptr) {return FAILED;}
    auto copy_out = op.oOperand.front();
    if (copy_out == nullptr) {return FAILED;}
    if (copy_in->shape == copy_out->shape && copy_out->GetMemoryTypeOriginal() == npu::tile_fwk::MEM_L1) {
        bool isRedundant = true;
        for (auto &producerOp : op.ProducerOps()) {
            if (producerOp == nullptr) {return FAILED;}
            if (producerOp->GetOpcode() != Opcode::OP_VIEW) {
                isRedundant = false;
                break;
            }
        }
        if (isRedundant) {
            ALOG_ERROR_F("PostCheck for copyin op[%d] failed!", op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status ProcessPostExpand(const Operation &op) {
    auto expand_in = op.iOperand.front();
    if (expand_in == nullptr) {return FAILED;}
    auto expand_out = op.oOperand.front();
    if (expand_out == nullptr) {return FAILED;}
    if (expand_in->shape == expand_out->shape) {
        ALOG_ERROR_F("PostCheck for expand op[%d] failed!", op.GetOpMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessPostCheck(const Operation &op) {
    if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
        if (ProcessPostCheckAssemble(op) != SUCCESS) {return FAILED;}
    } else if (op.GetOpcode() == Opcode::OP_VIEW) {
        if (ProcessPostCheckView(op) != SUCCESS) {return FAILED;}
    } else if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
        if (ProcessPostRegCopy(op) != SUCCESS) {return FAILED;}
    } else if (op.GetOpcode() == Opcode::OP_COPY_IN) {
        if (ProcessPostCopyIn(op) != SUCCESS) {return FAILED;}
    } else if (op.GetOpcode() == Opcode::OP_EXPAND) {
        if (ProcessPostExpand(op) != SUCCESS) {return FAILED;}
    }
    return SUCCESS;
}

Status ProcessRegCopy(const Operation &op, const Function &function, bool &needToDelete) {
    auto regCopyIn = op.iOperand.front();
    if (regCopyIn == nullptr) {return FAILED;}
    auto regCopyOut= op.oOperand.front();
    if (regCopyOut == nullptr) {return FAILED;}
    if (regCopyIn->shape == regCopyOut->shape) {
        /*
        register copy 输入和输出相同，无拷贝意义
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
        ALOG_DEBUG_F("[RemoveRedundentOp] Delete Redundent OP_REGISTER_COPY opmagic: %d", op.opmagic);
    }
    return SUCCESS;
}

Status ProcessAssembleDDR(const Operation &op, const LogicalTensorPtr &assembleIn, 
                          const LogicalTensorPtr &assembleOut, Function &function, 
                          bool &needToDelete) {
    auto consumerOps = function.FindConsumers(op);
    if (consumerOps.empty()) {
        /* DDR --> Assemble --> OUTCAST */
        if (assembleOut->nodetype != NodeType::OUTCAST) {return FAILED;}
        if (!function.IsFromOutCast(assembleOut)) {return FAILED;}
        ALOG_DEBUG_F("[RemoveRedundentOp] OP_ASSEMBLE has no consumers, opmagic: %d", op.opmagic);
        auto childOpsBackup = assembleIn->GetConsumers();
        for (auto &childOp : childOpsBackup) {
            if (childOp->GetOpMagic() == op.GetOpMagic()) {
                continue;
            }
            childOp->ReplaceInput(assembleOut, assembleIn);
            ALOG_DEBUG_F("Repalce input of %s opmagic: %d, tensor %d --> tensor %d", childOp->GetOpcodeStr(), 
                childOp->GetOpMagic(), assembleIn->magic, assembleOut->magic);
        }
        auto producerOps = op.ProducerOps();
        for (auto &producerOp : producerOps) {
            producerOp->ReplaceOutput(assembleOut, assembleIn);
            ALOG_DEBUG_F("Repalce output of %s opmagic: %d, tensor %d --> tensor %d", producerOp->GetOpcodeStr(), 
                producerOp->GetOpMagic(), assembleIn->magic, assembleOut->magic);
        }
    } else {
        for (auto &consumerOp : consumerOps) {
            consumerOp->ReplaceInput(assembleOut, assembleIn);
        }
    }
    needToDelete = true;
    ALOG_DEBUG_F("[RemoveRedundentOp] Delete Redundent OP_ASSEMBLE on DDR opmagic: %d", op.opmagic);
    return SUCCESS;
}

Status ProcessAssembleUB(const Operation &op, const LogicalTensorPtr &ASSEMBLE_in, 
                          const LogicalTensorPtr &ASSEMBLE_out, Function &function, 
                          bool &needToDelete) {
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
    ALOG_DEBUG_F("[RemoveRedundentOp] Delete Redundent OP_ASSEMBLE on UB opmagic: %d", op.opmagic);
    return SUCCESS;
}

Status ProcessAssemble(const Operation &op, Function &function, bool &needToDelete) {
    auto ASSEMBLE_in = op.iOperand.front();
    if (ASSEMBLE_in == nullptr) {return FAILED;}
    auto ASSEMBLE_out = op.oOperand.front();
    if (ASSEMBLE_out == nullptr) {return FAILED;}
    if (ASSEMBLE_in->shape == ASSEMBLE_out->shape && ASSEMBLE_in->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
        ASSEMBLE_out->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        if (ProcessAssembleDDR(op, ASSEMBLE_in, ASSEMBLE_out, function, needToDelete)) {return FAILED;}
    } else if (ASSEMBLE_in->shape == ASSEMBLE_out->shape && ASSEMBLE_in->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
               ASSEMBLE_out->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        if (ProcessAssembleUB(op, ASSEMBLE_in, ASSEMBLE_out, function, needToDelete)) {return FAILED;}
    }
    return SUCCESS;
}

Status ProcessView(const Operation &op, Function &function, bool &needToDelete) {
    auto in = op.iOperand.front();
    if (in == nullptr) {return FAILED;}
    auto out = op.oOperand.front();
    if (out == nullptr) {return FAILED;}
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
        ALOG_DEBUG_F("[RemoveRedundentOp] Delete Redundent OP_VIEW opmagic: %d", op.opmagic);
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
isRedundant = true
view --> in --> L1_COPY_IN --> out --> view'/L1_TO_L0A

isRedundant = false
op --> in --> L1_COPY_IN --> out --> view'/L1_TO_L0A
*/    
Status ProccessCopyIn(Operation &op, Function &function, bool &needToDelete) {
    auto in = op.iOperand.front();
    if (in == nullptr) {return FAILED;}
    auto out = op.oOperand.front();
    if (out == nullptr) {return FAILED;}
    if (in->shape == out->shape && out->GetMemoryTypeOriginal() == npu::tile_fwk::MEM_L1) {
        auto consumerOps = function.FindConsumers(op);
        auto producerOps = op.ProducerOps();
        bool isRedundant = true;
        for (auto &producerOp : producerOps) {
            if (producerOp == nullptr) {return FAILED;}
            if (producerOp->GetOpcode() != Opcode::OP_VIEW) {
                isRedundant = false;
                break;
            }
            in->SetMemoryTypeToBe(MemoryType::MEM_L1);
            in->SetMemoryTypeOriginal(MemoryType::MEM_L1, true);
            std::shared_ptr<ViewOpAttribute> attr = std::static_pointer_cast<ViewOpAttribute>(producerOp->GetOpAttribute());
            if (attr == nullptr) {return FAILED;}
            attr->SetToType(MemoryType::MEM_L1);
        }
        if (!isRedundant) {
            std::vector<OpImmediate> newOffset;
            auto inputOffset = op.GetIOperands().front()->GetOffset();
            for (size_t i = 0; i < op.oOperand.front()->shape.size(); i++) {
                newOffset.push_back(OpImmediate::Specified(SymbolicScalar(inputOffset[i])));
            }
            std::shared_ptr<CopyOpAttribute> cur_attr = std::make_shared<CopyOpAttribute>(newOffset, MemoryType::MEM_L1,
                OpImmediate::Specified(in->shape),
                OpImmediate::Specified(in->tensor->GetDynRawShape()));
            if (cur_attr == nullptr) {return FAILED;}
            op.SetOpAttribute(cur_attr);
            return SUCCESS;
        }
        for (auto &consumerOp : consumerOps) {
            consumerOp->ReplaceInput(in, out);
        }
        needToDelete = true;
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

Status RemoveRedundentOp::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start RemoveRedundentOp");
    if (DeleteRedundantOps(function) != SUCCESS) {return FAILED;}
    if (RemoveDummyExpand(function) != SUCCESS) {return FAILED;}
    ALOG_INFO_F("===> End RemoveRedundentOp");
    return SUCCESS;
}

Status RemoveRedundentOp::PreCheck(Function &function) {
    ALOG_INFO_F("PreCheck for RemoveRedundentOp");
    for (const auto &op : function.Operations().DuplicatedOpList()) {
        if (op == nullptr) {return FAILED;}
    }
    if (!function.LoopCheck().empty()) {return FAILED;}
    for (auto &op : function.Operations()) {
        if (ProcessPreCheck(op) != SUCCESS) {
            ALOG_ERROR_F("PreCheck for RemoveRedundentOp failed!");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundentOp::PostCheck(Function &function) {
    ALOG_INFO_F("PostCheck for RemoveRedundentOp");
    if (!function.LoopCheck().empty()) {return FAILED;}
    for (auto &op : function.Operations()) {
        if (ProcessPostCheck(op) != SUCCESS) {
            ALOG_ERROR_F("PostCheck for RemoveRedundentOp failed!");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundentOp::RemoveDummyExpand(Function &function) const {
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

Status RemoveRedundentOp::NeedToDelete(const Operation &op, Function &function, bool &needToDelete) const {
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

Status RemoveRedundentOp::DeleteCopyIn(Operation &op, Function &function, bool &needToDelete) const {
    needToDelete = false;
    switch (op.GetOpcode()) {
        case Opcode::OP_COPY_IN:
            if (ProccessCopyIn(op, function, needToDelete)) {return FAILED;}
            break;
        default:
            break;
    }
    return SUCCESS;
}

Status RemoveRedundentOp::DeleteRedundantOps(Function &function) const {
    std::vector<Operation *> redundentOp;
    bool needToDelete;
    for (auto &op : function.Operations()) {
        if (NeedToDelete(op, function, needToDelete) != SUCCESS) {return FAILED;}
        if (needToDelete) {
            redundentOp.push_back(&op);
        }
    }
    for (const auto &op : redundentOp) {
        function.HandleControlOps(*op, redundentOp);
        function.UpdateOperandBeforeRemoveOp(*op, false);
    }
    for (auto &op : function.Operations()) {
        if (DeleteCopyIn(op, function, needToDelete) != SUCCESS) {return FAILED;}
        if (needToDelete) {
            redundentOp.push_back(&op);
        }
    }
    for (auto op : redundentOp) {
        if (op->IsDeleted()) {return FAILED;}
        op->SetAsDeleted();
    }
    function.EraseOperations(true);
    return SUCCESS;
}
} // namespace npu::tile_fwk