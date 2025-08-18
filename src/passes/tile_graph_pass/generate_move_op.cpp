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
 * \file generate_move_op.cpp
 * \brief
 */

#include "passes/tile_graph_pass/generate_move_op.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/tile_graph_pass/dead_operation_eliminate.h"

namespace npu::tile_fwk {
Status GenerateMoveOp::RunOnFunction(Function &function) {
    ASLOGI("===> Start GenerateMoveOp");
    CreateMoveOp(function);
    MergeMoveOp(function);
    EraseRedundantCopyOut(function);
    DeadOperationEliminator::EliminateDeadOperation(function);

    ASLOGI("===> End GenerateMoveOp");
    return SUCCESS;
}

bool GenerateMoveOp::ValidViewOp(const Operation &op) const{
    //校验view单输入单输出，指针非空
    if((op.GetOpAttribute().get() == nullptr) ||
       (op.GetIOperands().size() != 1) || (op.GetOOperands().size() != 1) ||
       (op.GetIOperands().front() == nullptr) || (op.GetOOperands().front() == nullptr) ||
       (*(op.oOperand[0]->GetConsumers().begin()) == nullptr)){
        ALOG_ERROR_F("View op [%d] check failed.", op.GetOpMagic());
        return false;
    }
    //校验view输出tensor内存是否合理
    if (op.GetOOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        auto consumerOps = op.oOperand[0]->GetConsumers(); 
        for(auto childOp : consumerOps) {
            if(childOp == nullptr) {
                ALOG_ERROR_F("View op [%d] output has null consumers.",op.GetOpMagic());
                return false;
            }
            auto opcode = childOp->GetOpcode();
            const auto &inputsMemType = OpcodeManager::Inst().GetInputsMemType(opcode);
            bool hasDDRinput = std::find(inputsMemType.begin(),inputsMemType.end(),MemoryType::MEM_DEVICE_DDR) != inputsMemType.end();
            if(opcode == Opcode::OP_RESHAPE || hasDDRinput) {
                continue;
            }else if(opcode == Opcode::OP_CONVERT) {
                auto convertOpAttribute = dynamic_cast<ConvertOpAttribute *>(op.GetOpAttribute().get());
                auto convertPath = convertOpAttribute->GetConvertPath();
                if(convertPath.first != MemoryType::MEM_DEVICE_DDR){
                    ALOG_ERROR_F("View op [%d] consumer %s[%d] has invalid convert path.", op.GetOpMagic(),childOp->GetOpcodeStr().c_str(),childOp->GetOpMagic());
                    return false;
                }
            }else {
                ALOG_ERROR_F("View op [%d] consumer %s[%d] does not support DDR input.", op.GetOpMagic(),childOp->GetOpcodeStr().c_str(),childOp->GetOpMagic());
                return false;
            }
        }    
    }
    return true;
}
bool GenerateMoveOp::ValidAssembleOp(const Operation &op) const{
    //校验assemble单输入单输出，指针非空
    bool valid = true;
    if((op.GetOpAttribute().get() == nullptr) ||
       (op.GetIOperands().size() != 1) || (op.GetOOperands().size() != 1) ||
       (op.GetIOperands().front() == nullptr) || (op.GetOOperands().front() == nullptr)){
        ALOG_ERROR_F("Assemble op [%d] check failed.", op.GetOpMagic());
        valid = false;
    }
    return valid;
}
bool GenerateMoveOp::ValidConvertOp(const Operation &op) const{
    //校验convert单输入单输出，指针非空，输入输出内存类型不同，且存在DDR类型
    bool valid = true;
    if((op.GetOpAttribute().get() == nullptr) || (op.GetIOperands().size() != 1) || (op.GetOOperands().size() != 1) ||
       (op.GetIOperands().front() == nullptr) || (op.GetOOperands().front() == nullptr) ||
       (op.GetIOperands().front()->GetMemoryTypeOriginal() == op.GetOOperands().front()->GetMemoryTypeOriginal()) ||
       (op.GetIOperands().front()->GetShape() != op.GetOOperands().front()->GetShape())){
        ALOG_ERROR_F("Convert op [%d] check failed.",op.GetOpMagic());
        valid = false;
    }
    return valid;
}

Status GenerateMoveOp::PreCheck(Function &function) {
    ALOG_INFO_F("===> Start Precheck.");
    Status baseStatus = Pass::PreCheck(function);
    if (baseStatus != SUCCESS) {
        return FAILED;
    }
    auto operations = function.Operations();
    // Check iOperand and oOperand of OP_CONVERT
    for (auto &operation : operations) {
        const auto opcode = operation.GetOpcode();
        bool isValid = true;
        switch(opcode) {
            case Opcode :: OP_CONVERT:
                isValid = ValidConvertOp(operation);
                break;
            case Opcode :: OP_VIEW:
                isValid = ValidViewOp(operation);
                break;
            case Opcode :: OP_ASSEMBLE:
                isValid = ValidAssembleOp(operation);
                break;
            default:
                continue;
        }
        if(!isValid) {
            ALOG_ERROR_F("Operation validation failed.");
            return FAILED;
        }
    }
    ALOG_INFO_F("===> End Precheck.");
    return SUCCESS;
}

Status GenerateMoveOp::PostCheck(Function &function) {
    ALOG_INFO_F("===> Start Postcheck.");
    Status baseStatus = Pass::PostCheck(function);
    if (baseStatus != SUCCESS) {
        return baseStatus;
    }
    auto operations = function.Operations();
    for (auto &operation : operations) {
        auto op = operation.GetOpcode();
        bool isValid =true;
        if (((op == Opcode::OP_ASSEMBLE || op == Opcode::OP_VIEW)
            &&((operation.GetIOperands().size() != 1) ||(operation.GetOOperands().size() != 1)
            ||(operation.GetIOperands().front()->GetMemoryTypeOriginal() != operation.GetOOperands().front()->GetMemoryTypeOriginal())))
            ||(op == Opcode::OP_DUPLICATE || op == Opcode::OP_CONVERT)) {
                isValid = false;
            }
        if(!isValid) {
            ALOG_ERROR_F("Operation validation failed : op [%d] check failed.",operation.GetOpMagic());
            return FAILED;
        }
    }
    ALOG_INFO_F("===> End Postcheck.");
    return SUCCESS;
}

bool GenerateMoveOp::HasGmInput(Operation &op) const {
    return op.GetOpcode() == Opcode::OP_INDEX_OUTCAST || op.GetOpcode() == Opcode::OP_COMM_WAIT_FLAG;
}

void GenerateMoveOp::CreateMoveOpForView(Operation &op) const {
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    bool needInsertCopyIn = op.iOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
    if (needInsertCopyIn) {
        auto nextOp = *(op.oOperand[0]->GetConsumers().begin());
        auto viewResult = op.GetOOperands()[0];
        if (viewResult->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            return;
        }
        op.SetOpCode(Opcode::OP_COPY_IN); // 将view转化为copyin
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(viewOpAttribute->GetFromTensorOffset()),
            viewOpAttribute->GetTo(), OpImmediate::Specified(op.oOperand.front()->shape),
            OpImmediate::Specified(op.iOperand.front()->tensor->GetDynRawShape()),
            OpImmediate::Specified(viewOpAttribute->GetToDynValidShape())));
        if (nextOp->HasAttr(OpAttributeKey::tag)) {
            op.SetAttribute(OpAttributeKey::tag, nextOp->GetStringAttribute(OpAttributeKey::tag));
        }
    } else if (op.HasAttr(OpAttributeKey::isGlobalInput) && op.GetBoolAttribute(OpAttributeKey::isGlobalInput)) {
        auto viewResult = op.GetOOperands()[0];
        auto consumersCopy = viewResult->GetConsumers();
        for (auto childOp : consumersCopy) {
            for (size_t j = 0; j < childOp->iOperand.size(); j++) {
                if (childOp->iOperand[j] == viewResult) {
                    Tensor newTensor(viewResult->Datatype(), viewResult->shape);
                    newTensor.GetStorage()->UpdateOffset(dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get())->GetFrom());
                    newTensor.GetStorage()->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
                    newTensor.GetStorage()->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
                    newTensor.GetStorage()->isSubGraphBoundary = false;
                    // update consumer of oOperand
                    newTensor.GetStorage()->AddConsumer(childOp);
                    childOp->iOperand[j] = newTensor.GetStorage();
                    newTensor.GetStorage()->tensor = op.GetIOperands()[0]->tensor;
                }
            }
        }
        op.SetAsDeleted();
    }
}

void GenerateMoveOp::CreateMoveOpForAssemble(Operation &op) const {
    auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(op.GetOpAttribute().get());
    auto ASSEMBLE_in = op.iOperand.front();
    auto parentOp = *ASSEMBLE_in->GetProducers().begin();
    bool needCreate = true;
    if (op.iOperand.front()->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR &&
        op.oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
        parentOp->GetOpcode() != Opcode::OP_TRANSPOSE_DATAMOVE && parentOp->GetOpcode() != Opcode::OP_INDEX_OUTCAST) {
        op.SetOpCode(Opcode::OP_COPY_OUT);
    } else {
        needCreate = false;
    }
    if (!needCreate) {
        return;
    }
    auto preOp = *(op.iOperand[0]->GetProducers().begin());
    if (preOp->HasAttr(OpAttributeKey::tag)) {
        op.SetAttribute(OpAttributeKey::tag, preOp->GetStringAttribute(OpAttributeKey::tag));
    }
    if (assembleOpAttribute->GetFrom() != ASSEMBLE_in->GetMemoryTypeOriginal()) {
        ASLOGE(" Assemble op from Attr is different from iOperand, opmagic: %d, do force setting.", op.opmagic);
    }
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(ASSEMBLE_in->GetMemoryTypeOriginal(),
        OpImmediate::Specified(assembleOpAttribute->GetToTensorOffset()),
        OpImmediate::Specified(op.iOperand.front()->shape),
        OpImmediate::Specified(op.oOperand.front()->tensor->GetDynRawShape())));
}

void GenerateMoveOp::CreateMoveOpForConvert(Operation &op) const {
    auto convertOpAttribute = dynamic_cast<ConvertOpAttribute *>(op.GetOpAttribute().get());
    auto [from, to] = convertOpAttribute->GetConvertPath();
    if (from == MemoryType::MEM_DEVICE_DDR) {
        op.SetOpCode(Opcode::OP_COPY_IN); //将convert根据memorytype转化为copyin和copyout
        std::vector<OpImmediate> newOffset;
        auto inputOffset = op.GetIOperands().front()->GetOffset();
        for (size_t i = 0; i < op.oOperand.front()->shape.size(); i++) {
            newOffset.push_back(OpImmediate::Specified(SymbolicScalar(inputOffset[i])));
        }
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(newOffset, to,
            OpImmediate::Specified(op.oOperand.front()->shape),
            OpImmediate::Specified(op.iOperand.front()->tensor->GetDynRawShape()),
            OpImmediate::Specified(op.iOperand.front()->GetDynValidShape())));
        auto childOp = *op.oOperand.front()->GetConsumers().begin();
        op.UpdateSubgraphID(childOp->GetSubgraphID());
    } else if (to == MemoryType::MEM_DEVICE_DDR) {
        op.SetOpCode(Opcode::OP_COPY_OUT);
        std::vector<OpImmediate> newOffset;
        auto inputOffset = op.GetIOperands().front()->GetOffset();
        for (size_t i = 0; i < op.iOperand.front()->shape.size(); i++) {
            newOffset.push_back(OpImmediate::Specified(SymbolicScalar(inputOffset[i])));
        }
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(from, newOffset,
            OpImmediate::Specified(op.iOperand.front()->shape),
            OpImmediate::Specified(op.oOperand.front()->tensor->GetDynRawShape())));
        auto parentOp = *op.iOperand.front()->GetProducers().begin();
        op.UpdateSubgraphID(parentOp->GetSubgraphID());
    }else if ((from == MemoryType::MEM_L1) && (to == MemoryType::MEM_L0A)) {
        op.SetOpCode(Opcode::OP_L1_TO_L0A);
        auto childOp = *op.oOperand.front()->GetConsumers().begin();
        op.UpdateSubgraphID(childOp->GetSubgraphID());
    }else if ((from == MemoryType::MEM_L1) && (to == MemoryType::MEM_L0B)) {
        op.SetOpCode(Opcode::OP_L1_TO_L0B);
        auto childOp = *op.oOperand.front()->GetConsumers().begin();
        op.UpdateSubgraphID(childOp->GetSubgraphID());
    }
}

void GenerateMoveOp::CreateMoveOp(Function &function) const {
    for (auto &op : function.Operations()) {
        switch (op.GetOpcode()) {
            case Opcode::OP_ASSEMBLE: {
                CreateMoveOpForAssemble(op);
                break;
            }
            case Opcode::OP_VIEW: {
                CreateMoveOpForView(op);
                break;
            }
            case Opcode::OP_CONVERT: {
                CreateMoveOpForConvert(op);
                break;
            }
            case Opcode::OP_DUPLICATE: {
                op.SetOpCode(Opcode::OP_COPY_OUT); //将duplicate转化为copyout
                std::vector<OpImmediate> newOffset;
                for (size_t i = 0; i < op.iOperand.front()->shape.size(); i++) {
                    newOffset.push_back(OpImmediate::Specified(SymbolicScalar(0)));
                }
                op.SetOpAttribute(std::make_shared<CopyOpAttribute>(op.iOperand.front()->GetMemoryTypeOriginal(),
                    newOffset, OpImmediate::Specified(op.iOperand.front()->shape),
                    OpImmediate::Specified(op.oOperand.front()->tensor->GetDynRawShape())));
                break;
            }
            default: break;
        }
    }
}

void GenerateMoveOp::MergeMoveOp(Function &function) const{
    for (auto &op : function.Operations()) {
        switch (op.GetOpcode()) {
            case Opcode::OP_COPY_IN: {
                MergeCopyInCopyOut(function, op);
                break;
            }
            default:
                break;
        }
    }
}

// copyin直接连接copyout的场景，如果copyout的输入和输出的大小相同 则将copyout节点删除
void GenerateMoveOp::MergeCopyInCopyOut(Function &function, Operation &operation) const {
    auto consumers = function.FindConsumers(operation);
    for (auto &op : consumers) {
        if (op->GetOpcode() == Opcode::OP_COPY_OUT) {
            auto &startTensor = operation.iOperand.front();
            auto &endTensor = op->oOperand.front();
            if (startTensor->shape == endTensor->shape && startTensor->offset == endTensor->offset) {
                // Skip the CopyOut of OCAST
                if (endTensor->GetConsumers().size() == 0) {
                    continue;
                }
                for (auto &consumer : endTensor->GetConsumers()) {
                    consumer->iOperand = {startTensor};
                    startTensor->AddConsumer(consumer);
                }
                endTensor->GetConsumers().clear();
                op->oOperand.clear();
                function.GetTensorMap().Erase(endTensor);
            }
        }
    }
}

// 将输入tensor的producers为空的copyout节点删除
void GenerateMoveOp::EraseRedundantCopyOut(Function &function) const {
    std::vector<Operation *> redundantCopyOuts;
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() !=  Opcode::OP_COPY_OUT) {
            continue;
        }

        if (op.iOperand.front()->GetProducers().empty()) {
            redundantCopyOuts.push_back(&op);
        }
    }
    for (const auto &op : redundantCopyOuts) {
        function.HandleControlOps(*op, redundantCopyOuts);
        function.UpdateOperandBeforeRemoveOp(*op, false);
    }

    for (auto op : redundantCopyOuts) {
        ASSERT(!op->IsDeleted());
        op->SetAsDeleted();
    }
    function.EraseOperations(false);
}
} // namespace npu::tile_fwk
