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
 * \file inplace_process.cpp
 * \brief
 */

#include "inplace_process.h"

namespace npu::tile_fwk {

Status InplaceProcess::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start InplaceProcess.");
    auto opList = function.Operations();
    for (auto &op : opList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            if (!ValidMeaninglessOp(op)) {
                return FAILED;
            }
            ProcessView(op);
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (!ValidMeaninglessOp(op)) {
                return FAILED;
            }
            auto assembleOut = op.GetOOperands().front();
            // 校验Assemble输出的汇聚后tensor大小是否超过UB上限
            const int UB_SIZE = PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_UB);
            if (assembleOut->GetMemoryTypeOriginal() == MemoryType::MEM_UB && (assembleOut->tensor->GetRawDataSize() > UB_SIZE)) {
                ALOG_ERROR_F(" Local Buffer Assemble Result Oversized, %d, tensor: %d, size: %ld B.", op.opmagic,
                    assembleOut->magic, assembleOut->tensor->GetRawDataSize());
                return FAILED;
            }
            ProcessAssemble(op);
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            if (!ValidMeaninglessOp(op)) {
                return FAILED;
            }
            ProcessReshape(function, op);
        } else {
            if (ProcessInplaceOp(function, op) != SUCCESS) {
                ALOG_ERROR_F(" Processing inplace op %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
        }
    }
    ALOG_INFO_F("===> End InplaceProcess.");
    return SUCCESS;
}

bool InplaceProcess::ValidMeaninglessOp(const Operation &op) const {
    // 校验单输入单输出，且输入输出mem类型相同
    bool valid = true;
    if ((op.GetIOperands().size() != 1) || (op.GetOOperands().size() != 1) ||
        (op.GetIOperands().front() == nullptr) || (op.GetOOperands().front() == nullptr) ||
        (op.GetIOperands().front()->GetMemoryTypeOriginal() != op.GetOOperands().front()->GetMemoryTypeOriginal())) {
        ALOG_INFO_F(
            "InplaceProcess %s[%d] Invalid: IOperands.size is %d; OOperands.size is %d; "
            "IOperands.front is nullptr (%d); OOperands.front is nullptr (%d); IOperands.front.MemoryType is %d; "
            "OOperands.front.MemoryType is %d.",
            (op.GetOpcodeStr().c_str()), (op.GetOpMagic()), (op.GetIOperands().size()), (op.GetOOperands().size()),
            (op.GetIOperands().front() == nullptr), (op.GetOOperands().front() == nullptr),
            (op.GetIOperands().front()->GetMemoryTypeOriginal()), (op.GetOOperands().front()->GetMemoryTypeOriginal()));
        valid = false;
    }
    return valid;
}

void InplaceProcess::ProcessView(Operation &op) const {
    ALOG_DEBUG_F("Find Internal View %d.", op.opmagic);
    std::vector<int> inputOffset = op.GetIOperands()[0]->GetOffset();
    for (auto &consumer : op.GetIOperands()[0]->GetConsumers()) {
        if ((consumer->GetOpcode() != Opcode::OP_VIEW) || (consumer->GetOpMagic() != op.GetOpMagic())) {
            continue;
        }
        auto viewAttr = dynamic_cast<ViewOpAttribute *>(consumer->GetOpAttribute().get());
        if (viewAttr != nullptr) {
            std::vector<int> viewOpOffset = viewAttr->GetFrom();
            // 增加校验: input --> View --> ouput 三者的offset size 相同
            for (size_t i = 0; i < inputOffset.size(); i++) {
                viewOpOffset[i] = inputOffset[i] + viewOpOffset[i];
            }
            viewAttr->SetFromOffset(viewOpOffset);
            consumer->oOperand[0]->tensor = op.GetIOperands()[0]->tensor;
            consumer->oOperand[0]->UpdateOffset(viewOpOffset);
        }
    }
}

void InplaceProcess::AlignCopyInConsumer(std::shared_ptr<LogicalTensor> tensorGm) const {
    if (tensorGm->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        return;
    }
    ALOG_DEBUG_F("InplaceProcess::AlignCopyInConsumer tensor[%d].", tensorGm->magic);
    for (auto &consumerOp : tensorGm->GetConsumers()) {
        if (consumerOp->GetOpcode() == Opcode::OP_COPY_IN) {
            std::shared_ptr<CopyOpAttribute> opAttr = std::static_pointer_cast<CopyOpAttribute>(consumerOp->GetOpAttribute());
            std::vector<OpImmediate> newFromOffset;
            for (size_t i = 0; i < opAttr->GetFromOffset().size(); i++) {
                newFromOffset.push_back(opAttr->GetFromOffset()[i] + OpImmediate::Specified(SymbolicScalar(tensorGm->offset[i])));
            }
            opAttr->SetFromOffset(newFromOffset);
            opAttr->SetRawShape(OpImmediate::Specified(tensorGm->tensor->GetRawShape()));
        }
    }
}

void InplaceProcess::AlignCopyOutProducer(std::shared_ptr<LogicalTensor> tensorGm) const {
    if (tensorGm->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        return;
    }
    ALOG_DEBUG_F("InplaceProcess::AlignCopyOutProducer tensor[%d].", tensorGm->magic);
    for (auto &producerOp : tensorGm->GetProducers()) {
        if (producerOp->GetOpcode() == Opcode::OP_COPY_OUT) {
            std::shared_ptr<CopyOpAttribute> opAttr = std::static_pointer_cast<CopyOpAttribute>(producerOp->GetOpAttribute());
            std::vector<OpImmediate> newToOffset;
            for (size_t i = 0; i < opAttr->GetToOffset().size(); i++) {
                newToOffset.push_back(opAttr->GetToOffset()[i] + OpImmediate::Specified(SymbolicScalar(tensorGm->offset[i])));
            }
            opAttr->SetToOffset(newToOffset);
            opAttr->SetRawShape(OpImmediate::Specified(tensorGm->tensor->GetRawShape()));
            ALOG_DEBUG_F("InplaceProcess::AlignCopyOutProducer update Attr for %s[%d].", producerOp->GetOpcodeStr().c_str(), 
                producerOp->GetOpMagic());
        }
    }
}

void InplaceProcess::ReplaceRawTensor(std::shared_ptr<LogicalTensor> logicalTensor, 
    const std::shared_ptr<LogicalTensor> targetTensor, const Operation &op) {
    logicalTensor->tensor = targetTensor->tensor;
    logicalTensor->UpdateOffset(dynamic_cast<AssembleOpAttribute *>(op.GetOpAttribute().get())->GetToOffset());
    ALOG_DEBUG_F("update the offset for Tensor %d.", logicalTensor->magic);
}

void InplaceProcess::ProcessAssemble(Operation &op) {
    auto assembleIn = op.GetIOperands().front();
    auto assembleOut = op.GetOOperands().front();
    if (op.iOperand[0]->tensor->GetRawDataSize() > assembleOut->tensor->GetRawDataSize()) {
        if (assembleOut->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            ALOG_DEBUG_F(" Invalid Assemble case, opmagic: %d.", op.opmagic);
            return;
        }
        std::vector<LogicalTensorPtr> assembleInputList;
        for (auto &producer : assembleOut->GetProducers()) {
            assembleInputList.push_back(producer->iOperand[0]);
        }
        bool allAssembleInputContinuous = FunctionUtils::IsContinuous(assembleInputList);
        ALOG_DEBUG_F("Check all the %s[%d] input %d is continuous: %d.", op.GetOpcodeStr().c_str(), op.GetOpMagic(),
            assembleOut->magic, allAssembleInputContinuous);
        if (!allAssembleInputContinuous) {
            return;
        }
        std::vector<int> newOffset(op.iOperand[0]->offset.size(), INT_MAX);
        for (auto &assembleOp : assembleOut->GetProducers()) {
            auto tempOffset = assembleOp->iOperand[0]->offset;
            for (size_t m = 0; m < op.iOperand[0]->offset.size(); m++) {
                newOffset[m] = std::min(newOffset[m], tempOffset[m]);
            }
        }
        assembleOut->UpdateOffset(newOffset);
        for (auto &assembleOp : assembleOut->GetProducers()) {
            // Update all other assemble's offset, their offset = their input's offset
            auto &inputOffset = assembleOp->iOperand[0]->offset;
            dynamic_cast<AssembleOpAttribute *>(assembleOp->GetOpAttribute().get())->SetToOffset(inputOffset);
        }
        assembleOut->tensor = op.iOperand[0]->tensor;
        ALOG_DEBUG_F("Success update %s[%d] oOperand: %d.", op.GetOpcodeStr().c_str(), op.GetOpMagic(), assembleOut->magic);
        // if the DDR assembleOut is consumed by Copy_In, the Copy_In's from offset should be algined
        AlignCopyInConsumer(assembleOut);
    } else {
        // check each producer of the assem_result
        for (auto &producer : assembleOut->GetProducers()) {
            if ((producer->GetOpcode() != Opcode::OP_ASSEMBLE) || 
                std::find(visitedAssembleOp.begin(), visitedAssembleOp.end(), producer->GetOpMagic()) != visitedAssembleOp.end()) {
                continue;
            }
            /*
            producer->iOperand[0] 可能来自一个被复用过的op
            raw tensor 与 producer->iOperand[0] 的 raw tensor 相同的所有logical tensor 都应该update
            */
            ReplaceRawTensor(producer->iOperand[0], assembleOut, *producer);
            visitedAssembleOp.push_back(producer->GetOpMagic());
        }
    }
}

void InplaceProcess::ProcessReshape(Function &function, Operation &op) const {
    auto reshapeIn = op.GetIOperands()[0];
    auto reshapeOut = op.GetOOperands()[0];
    ALOG_DEBUG_F(" %s[%d] on %s.", op.GetOpcodeStr().c_str(), op.GetOpMagic(), BriefMemoryTypeToString(reshapeIn->GetMemoryTypeOriginal()).c_str());
    if ((reshapeOut->tensor->actualRawmagic == -1) && (!function.IsFromOutCast(reshapeOut))) {
        reshapeOut->tensor->actualRawmagic = reshapeIn->GetRawMagic();
        ALOG_DEBUG_F(" update reshape opmagic %d, output's actualRaw: %d.", op.opmagic, reshapeOut->GetRawMagic());
    }
}

Status InplaceProcess::ProcessInplaceOp(Function &function, Operation &op) const {
    auto opcode = op.GetOpcode();
    if (inplaceOpMap.find(opcode) == inplaceOpMap.end()) {
        return SUCCESS;
    }
    for (auto &reusePair : inplaceOpMap.at(opcode)) {
        auto inputIdx = reusePair.first;
        auto outputIdx = reusePair.second;
        if (inputIdx >= op.GetIOperands().size() || outputIdx >= op.GetOOperands().size()) {
            ALOG_ERROR_F("Invalid inplace op info for %s[%d].", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
        auto tensorIn = op.GetIOperands()[inputIdx];
        auto tensorOut = op.GetOOperands()[outputIdx];
        if (tensorIn == nullptr || tensorOut == nullptr) {
            ALOG_ERROR_F("%s[%d] inplace input or output is nullptr.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
        if (function.IsFromOutCast(tensorOut)) {
            tensorIn->tensor = tensorOut->tensor;
            tensorIn->UpdateOffset(tensorOut->GetOffset());
        } else {
            tensorOut->tensor = tensorIn->tensor;
            tensorOut->UpdateOffset(tensorIn->GetOffset());
        }
        ALOG_DEBUG_F("%s[%d] output %d reuses input %d.", op.GetOpcodeStr().c_str(), op.GetOpMagic(), outputIdx, inputIdx);
        ALOG_DEBUG_F("output magic: %d, raw maigc: %d.", tensorOut->magic, tensorOut->tensor->GetRawMagic());
        ALOG_DEBUG_F("input magic: %d, raw maigc: %d.", tensorIn->magic, tensorIn->tensor->GetRawMagic());
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk