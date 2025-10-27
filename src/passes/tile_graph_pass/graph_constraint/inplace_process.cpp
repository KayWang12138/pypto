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

namespace npu {
namespace tile_fwk {
bool InplaceProcess::HasSameConsecutive(Operation &op) {
    for (auto &nextOp : op.ConsumerOps()) {
        if (nextOp->GetOpcode() == op.GetOpcode()) {
            return true;
        }
    }
    return false;
}
Status InplaceProcess::PreCheck(Function &function) {
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "PreCheck for InplaceProcess.");
    if (!function.LoopCheck().empty()) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Loopcheck failed before PreGraph; Please check whether there is a loop.");
        return FAILED;
    }
    for (auto &op : function.Operations()) {
        if (op.GetSubgraphID() == NOT_IN_SUBGRAPH) {
            APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "%s[%d] is not partitioned; Please check subGraphIDs.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
        if ((op.GetOpcode() != Opcode::OP_ASSEMBLE) && (op.GetOpcode() != Opcode::OP_VIEW) && 
            (op.GetOpcode() != Opcode::OP_RESHAPE)) {
            continue;
        }
        if (HasSameConsecutive(op)) {
            APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "%s[%d] has the same Opcode child op; Plese check child ops.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
        auto tensorIn = op.GetIOperands().front();
        auto tensorOut = op.GetOOperands().front();
        if (tensorIn->GetMemoryTypeOriginal() != tensorOut->GetMemoryTypeOriginal()) {
            APASS_LOG_ERROR_F(GetName().c_str(), "Tensor", "unmatched input output memory type for reshape opmagic: %d, input mem type: %s, output mem type: %s; Please check the input ans output.", 
                op.opmagic,
                MemoryTypeToString(tensorIn->GetMemoryTypeOriginal()).c_str(),
                MemoryTypeToString(tensorOut->GetMemoryTypeOriginal()).c_str());
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "PreCheck for InplaceProcess success.");
    return SUCCESS;
}

Status InplaceProcess::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "===> Start InplaceProcess.");
    auto opList = function.Operations();
    for (auto &op : opList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            if (ValidMeaninglessOp(op) != SUCCESS) {
                return FAILED;
            }
            ProcessView(function, op);
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (ValidMeaninglessOp(op) != SUCCESS) {
                return FAILED;
            }
            auto assembleOut = op.GetOOperands().front();
            // 校验Assemble输出的汇聚后tensor大小是否超过UB上限
            const int UB_SIZE = PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_UB);
            if (assembleOut->GetMemoryTypeOriginal() == MemoryType::MEM_UB && (assembleOut->tensor->GetRawDataSize() > UB_SIZE)) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Tensor", "Local Buffer Assemble Result Oversized, %d, tensor: %d, size: %ld B; Please check the result size.", op.opmagic,
                    assembleOut->magic, assembleOut->tensor->GetRawDataSize());
                return FAILED;
            }
            ProcessAssemble(function, op);
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            if (ValidMeaninglessOp(op) != SUCCESS) {
                return FAILED;
            }
            ProcessReshape(function, op);
            continue;
        }
        if (ProcessInplaceOp(function, op) != SUCCESS) {
            APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Processing inplace op %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "===> End InplaceProcess.");
    return SUCCESS;
}

Status InplaceProcess::ValidMeaninglessOp(const Operation &op) const {
    // 校验单输入单输出，且输入输出mem类型相同
    if ((op.GetIOperands().size() != 1) || (op.GetOOperands().size() != 1) ||
        (op.GetIOperands().front() == nullptr) || (op.GetOOperands().front() == nullptr) ||
        (op.GetIOperands().front()->GetMemoryTypeOriginal() != op.GetOOperands().front()->GetMemoryTypeOriginal())) {
        APASS_LOG_ERROR_F(
            GetName().c_str(), "Operation", "InplaceProcess %s[%d] Invalid: IOperands.size is %d; OOperands.size is %d; "
            "IOperands.front is nullptr (%d); OOperands.front is nullptr (%d); IOperands.front.MemoryType is %d; "
            "OOperands.front.MemoryType is %d.",
            (op.GetOpcodeStr().c_str()), (op.GetOpMagic()), (op.GetIOperands().size()), (op.GetOOperands().size()),
            (op.GetIOperands().front() == nullptr), (op.GetOOperands().front() == nullptr),
            (op.GetIOperands().front()->GetMemoryTypeOriginal()), (op.GetOOperands().front()->GetMemoryTypeOriginal()));
        return FAILED;
    }
    return SUCCESS;
}

void InplaceProcess::ProcessView(Function &function, Operation &op) const {
    APASS_LOG_DEBUG_F(GetName().c_str(), "Operation", "Find Internal View %d.", op.opmagic);
    std::vector<int64_t> inputOffset = op.GetIOperands()[0]->GetOffset();
    for (auto &consumer : op.GetIOperands()[0]->GetConsumers()) {
        if ((consumer->GetOpcode() != Opcode::OP_VIEW) || (consumer->GetOpMagic() != op.GetOpMagic())) {
            continue;
        }
        if (function.IsFromOutCast(consumer->oOperand[0])) {
            APASS_LOG_WARN_F(GetName().c_str(), "Operation", "InplaceProcess::ProcessView: OP_VIEW oOperand tensor[%d] is outCast.",
                consumer->oOperand[0]->GetMagic());
            continue;
        }
        auto viewAttr = dynamic_cast<ViewOpAttribute *>(consumer->GetOpAttribute().get());
        if (viewAttr == nullptr) {
            continue;
        }
        std::vector<int64_t> viewOpOffset = viewAttr->GetFrom();
        auto inputDynOffset = op.GetIOperands()[0]->GetDynOffset();
        if (inputDynOffset.empty()) {
            inputDynOffset = std::vector<SymbolicScalar>(inputOffset.size(), 0);
        }
        auto attrDynOffset = viewAttr->GetFromDynOffset();
        std::vector<SymbolicScalar> outTensorOffset;
        // 增加校验: input --> View --> ouput 三者的offset size 相同
        for (size_t i = 0; i < inputOffset.size(); i++) {
            viewOpOffset[i] = inputOffset[i] + viewOpOffset[i];
            if (attrDynOffset.size() == inputOffset.size()) {
                attrDynOffset[i] = inputDynOffset[i] + attrDynOffset[i];
            }
        }
        viewAttr->SetFromOffset(viewOpOffset, viewAttr->GetFromDynOffset());
        consumer->oOperand[0]->tensor = op.GetIOperands()[0]->tensor;
        TensorOffset newOffset(viewOpOffset, attrDynOffset);
        consumer->oOperand[0]->UpdateOffset(newOffset);
    }
}

void InplaceProcess::AlignCopyInConsumer(std::shared_ptr<LogicalTensor> tensorGm) const {
    if (tensorGm->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        return;
    }
    APASS_LOG_DEBUG_F(GetName().c_str(), "Tensor", "InplaceProcess::AlignCopyInConsumer tensor[%d].", tensorGm->magic);
    for (auto &consumerOp : tensorGm->GetConsumers()) {
        if (consumerOp->GetOpcode() == Opcode::OP_COPY_IN) {
            std::shared_ptr<CopyOpAttribute> opAttr = std::static_pointer_cast<CopyOpAttribute>(consumerOp->GetOpAttribute());
            std::vector<OpImmediate> newFromOffset;
            for (size_t i = 0; i < opAttr->GetFromOffset().size(); i++) {
                newFromOffset.push_back(opAttr->GetFromOffset()[i] + OpImmediate::Specified(SymbolicScalar(tensorGm->offset[i])));
            }
            opAttr->SetFromOffset(newFromOffset);
            opAttr->SetRawShape(OpImmediate::Specified(tensorGm->tensor->GetDynRawShape()));
        }
    }
}

void InplaceProcess::AlignCopyOutProducer(std::shared_ptr<LogicalTensor> tensorGm) const {
    if (tensorGm->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        return;
    }
    APASS_LOG_DEBUG_F(GetName().c_str(), "Tensor", "InplaceProcess::AlignCopyOutProducer tensor[%d].", tensorGm->magic);
    for (auto &producerOp : tensorGm->GetProducers()) {
        if (producerOp->GetOpcode() == Opcode::OP_COPY_OUT) {
            std::shared_ptr<CopyOpAttribute> opAttr = std::static_pointer_cast<CopyOpAttribute>(producerOp->GetOpAttribute());
            std::vector<OpImmediate> newToOffset;
            for (size_t i = 0; i < opAttr->GetToOffset().size(); i++) {
                newToOffset.push_back(opAttr->GetToOffset()[i] + OpImmediate::Specified(SymbolicScalar(tensorGm->offset[i])));
            }
            opAttr->SetToOffset(newToOffset);
            opAttr->SetRawShape(OpImmediate::Specified(tensorGm->tensor->GetDynRawShape()));
            APASS_LOG_DEBUG_F(GetName().c_str(), "Operation", "InplaceProcess::AlignCopyOutProducer update Attr for %s[%d].",
                producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic());
        }
    }
}

void InplaceProcess::ReplaceRawTensor(Function &function, std::shared_ptr<LogicalTensor> logicalTensor,
    const std::shared_ptr<LogicalTensor> targetTensor, const Operation &op) {
    if (function.IsFromInCast(logicalTensor)) {
        APASS_LOG_WARN_F(GetName().c_str(), "Tensor", "InplaceProcess::ProcessAssemble: OP_ASSEMBLE iOperand tensor[%d] is inCast.",
            logicalTensor->GetMagic());
        return;
    }
    logicalTensor->tensor = targetTensor->tensor;
    logicalTensor->UpdateOffset(dynamic_cast<AssembleOpAttribute *>(op.GetOpAttribute().get())->GetToOffset());
    APASS_LOG_DEBUG_F(GetName().c_str(), "Tensor", "update the offset for Tensor %d.", logicalTensor->magic);
}

void InplaceProcess::ProcessAssemble(Function &function, Operation &op) {
    auto assembleIn = op.GetIOperands().front();
    auto assembleOut = op.GetOOperands().front();
    bool fromIncast = function.IsFromInCast(assembleIn);
    APASS_LOG_DEBUG_F(GetName().c_str(), "Operation", "assembleIn from Incast: %d.", fromIncast);

    // check each producer of the assem_result
    for (auto &producer : assembleOut->GetProducers()) {
        if ((producer->GetOpcode() != Opcode::OP_ASSEMBLE) ||
            std::find(visitedAssembleOp.begin(), visitedAssembleOp.end(), producer->GetOpMagic()) !=
                visitedAssembleOp.end()) {
            continue;
        }
        /*
        producer->iOperand[0] 可能来自一个被复用过的op
        raw tensor 与 producer->iOperand[0] 的 raw tensor 相同的所有logical tensor 都应该update
        */
        ReplaceRawTensor(function, producer->iOperand[0], assembleOut, *producer);
        visitedAssembleOp.push_back(producer->GetOpMagic());
    }
}

void InplaceProcess::ProcessReshape(Function &function, Operation &op) const {
    auto reshapeIn = op.GetIOperands()[0];
    auto reshapeOut = op.GetOOperands()[0];
    APASS_LOG_DEBUG_F(GetName().c_str(), "Operation", " %s[%d] on %s.", op.GetOpcodeStr().c_str(), op.GetOpMagic(),
        BriefMemoryTypeToString(reshapeIn->GetMemoryTypeOriginal()).c_str());
    if (reshapeOut->tensor->actualRawmagic != -1) {
        return;
    }
    if (function.IsFromOutCast(reshapeOut)) {
        APASS_LOG_WARN_F(
            GetName().c_str(), "Operation", "InplaceProcess::ProcessReshape: OP_RESHAPE oOperand tensor[%d] is outCast.", reshapeOut->GetMagic());
        return;
    }
    reshapeOut->tensor->actualRawmagic = reshapeIn->GetRawMagic();
    APASS_LOG_DEBUG_F(GetName().c_str(), "Operation", "Update reshape opmagic %d, output's actualRaw: %d.", op.opmagic, reshapeOut->GetRawMagic());
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
            APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Invalid inplace op info for %s[%d]. Please check op inputs&outputs, supported inplace info "
                    "can be found in inplace_process.h."
                    "\n|----detect input size: %d, recorded inplace input idx: %d."
                    "\n|----detect output size: %d, recorded inplace output idx: %d.",
                op.GetOpcodeStr().c_str(), op.GetOpMagic(), op.GetIOperands().size(), inputIdx,
                op.GetOOperands().size(), outputIdx);
            return FAILED;
        }
        auto tensorIn = op.GetIOperands()[inputIdx];
        auto tensorOut = op.GetOOperands()[outputIdx];
        if (tensorIn == nullptr || tensorOut == nullptr) {
            APASS_LOG_ERROR_F(GetName().c_str(), "Tensor", "%s[%d] inplace input or output is nullptr.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
        if (function.IsFromOutCast(tensorOut) && function.IsFromInCast(tensorIn)) {
            APASS_LOG_WARN_F(GetName().c_str(), "Tensor", "InplaceProcess::ProcessInplaceOp: inplaceOp iOperand tensor[%d] is inCast and oOperand "
                        "tensor[%d] is outCast.", tensorIn->GetMagic(), tensorOut->GetMagic());
            continue;
        }
        APASS_LOG_DEBUG_F(GetName().c_str(), "Operation", "%s[%d] output %d reuses input %d.", op.GetOpcodeStr().c_str(), op.GetOpMagic(), outputIdx, inputIdx);
        if (function.IsFromOutCast(tensorOut)) {
            if ((tensorIn->tensor->symbol != "") && (tensorOut->tensor->symbol == "")) {
                tensorOut->tensor->symbol = tensorIn->tensor->symbol;
            }
            tensorIn->tensor = tensorOut->tensor;
            tensorIn->UpdateOffset(tensorOut->GetOffset());
            APASS_LOG_DEBUG_F(GetName().c_str(), "Tensor", "Output magic: %d, raw maigc: %d.", tensorOut->magic, tensorOut->tensor->GetRawMagic());
            APASS_LOG_DEBUG_F(GetName().c_str(), "Tensor", "Input magic: %d, raw maigc: %d.", tensorIn->magic, tensorIn->tensor->GetRawMagic());
            continue;
        }
        if ((tensorIn->tensor->symbol == "") && (tensorOut->tensor->symbol != "")) {
            tensorIn->tensor->symbol = tensorOut->tensor->symbol;
        }
        tensorOut->tensor = tensorIn->tensor;
        tensorOut->UpdateOffset(tensorIn->GetOffset());
        APASS_LOG_DEBUG_F(GetName().c_str(), "Tensor", "Output magic: %d, raw maigc: %d.", tensorOut->magic, tensorOut->tensor->GetRawMagic());
        APASS_LOG_DEBUG_F(GetName().c_str(), "Tensor", "Input magic: %d, raw maigc: %d.", tensorIn->magic, tensorIn->tensor->GetRawMagic());
    }
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu