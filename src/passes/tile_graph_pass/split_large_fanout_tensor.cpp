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
 * \file split_large_fanout_tensor.cpp
 * \brief
 */

#include "passes/tile_graph_pass/split_large_fanout_tensor.h"

namespace npu::tile_fwk {
Status SplitLargeFanoutTensor::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start SplitLargeFanoutTensor.");
    copyOutSources.clear();
    assembles.clear();
    CollectCopyOut(function);
    CompareWithCopyIn(function);

    for (auto &a : assembles) {
        auto &newCopyOut = function.AddOperation(Opcode::OP_ASSEMBLE, {a.input}, {a.output});
        newCopyOut.SetOpAttribute(std::make_shared<AssembleOpAttribute>(a.from, a.toOffset));
    }

    EraseRedundantCopyOut(function);
    EraseRedundantCopyIn(function);
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        ALOG_ERROR_F("Eliminate dead operation failed in CommonOperationEliminate.");
        return FAILED;
    }
    UpdateOverSizedLocalBuffer(function);
    ALOG_INFO_F("===> End SplitLargeFanoutTensor.");
    return SUCCESS;
}

void SplitLargeFanoutTensor::CollectCopyOut(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();
        if (copyOutSources.count(output->tensor->rawmagic) == 0) {
            copyOutSources.insert({output->tensor->rawmagic, {}});
        }
        auto opAttr = dynamic_cast<AssembleOpAttribute *>(op.GetOpAttribute().get());
        if (opAttr != nullptr) {
            copyOutSources[output->tensor->rawmagic].emplace_back(input, opAttr->GetToOffset());
        }
    }
}

void SplitLargeFanoutTensor::RecordMatched(Function &function, Operation &op,
    const std::shared_ptr<LogicalTensor> &targetTensor, const std::vector<std::shared_ptr<LogicalTensor>> &matchedTensors,
    const std::vector<std::shared_ptr<LogicalTensor>> &overlaps) {
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    auto &fromOffset = viewOpAttribute->GetFromOffset();
    auto status = CalcOverlap(targetTensor, matchedTensors, true);
    std::vector<int> newOffset(fromOffset.size(), 0);
    switch (status) {
        case OverlapStatus::PERFECTLY_MATCH: {
            auto overlap = overlaps.front();
            auto newInput = std::make_shared<LogicalTensor>(function, input->Datatype(), output->shape);
            if (newInput == nullptr) { break; }
            newInput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
            assembles.emplace_back(AssembleOp{overlap->GetMemoryTypeOriginal(), newOffset, overlap, newInput});
            viewOpAttribute->SetFromOffset(newOffset);
            op.ReplaceInput(newInput, input);
            break;
        }
        case OverlapStatus::BE_COVERED: {
            auto overlap = overlaps.front();
            auto newInput = std::make_shared<LogicalTensor>(function, input->Datatype(), overlap->shape);
            if (newInput == nullptr) { break; }
            newInput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
            assembles.emplace_back(AssembleOp{overlap->GetMemoryTypeOriginal(), newOffset, overlap, newInput});
            std::transform(fromOffset.begin(), fromOffset.end(), matchedTensors.front()->offset.begin(),
                newOffset.begin(), [](int a, int b) { return a - b; });
            viewOpAttribute->SetFromOffset(newOffset);
            op.ReplaceInput(newInput, input);
            break;
        }
        case OverlapStatus::PERFECTLY_MATCH_WITH_ALL: {
            auto newInput = std::make_shared<LogicalTensor>(function, input->Datatype(), output->shape);
            if (newInput == nullptr) { break; }
            newInput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
            for (size_t i = 0; i < overlaps.size(); i++) {
                auto &overlap = overlaps[i];
                std::vector<int> newAssembleOffset = matchedTensors[i]->GetOffset();
                for (size_t j = 0; j < newAssembleOffset.size(); ++j) {
                    newAssembleOffset[j] -= fromOffset[j];
                }
                assembles.emplace_back(
                    AssembleOp{overlap->GetMemoryTypeOriginal(), newAssembleOffset, overlap, newInput});
            }
            viewOpAttribute->SetFromOffset(newOffset);
            op.ReplaceInput(newInput, input);
            break;
        }
        default: break;
    }
}

void SplitLargeFanoutTensor::CompareWithCopyIn(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();
        if (input->shape == output->shape) {
            continue;
        }
        auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
        if (viewOpAttribute == nullptr) {
            continue;
        }
        auto &fromOffset = viewOpAttribute->GetFromOffset();

        std::vector<std::shared_ptr<LogicalTensor>> overlaps;
        auto inputView = std::make_shared<LogicalTensor>(function, input->tensor, fromOffset, output->shape);
        if (inputView == nullptr) {
            continue;
        }
        std::vector<std::shared_ptr<LogicalTensor>> outputOfAssemble;
        for (auto &[copyOutSource, offsetInAssemble] : copyOutSources[input->tensor->rawmagic]) {
            auto target =
                std::make_shared<LogicalTensor>(function, input->tensor, offsetInAssemble, copyOutSource->GetShape());
            if (target == nullptr) {
                continue;
            }
            auto status = CalcOverlap(inputView, target, true);
            if (status == OverlapStatus::PERFECTLY_MATCH || status == OverlapStatus::BE_COVERED) {
                outputOfAssemble.emplace_back(target);
                overlaps.push_back(copyOutSource);
                break;
            }
            if (status == OverlapStatus::COVERED) {
                outputOfAssemble.emplace_back(target);
                overlaps.push_back(copyOutSource);
                continue;
            }
        }
        if (overlaps.empty()) {
            continue;
        }
        RecordMatched(function, op, inputView, outputOfAssemble, overlaps);
    }
}

void SplitLargeFanoutTensor::RemoveOps(Function &function, std::vector<Operation *> &opList) const {
    for (const auto &op : opList) {
        function.UpdateOperandBeforeRemoveOp(*op, false);
    }
    for (auto op : opList) {
        if (!op->IsDeleted()) {
            op->SetAsDeleted();
        }
    }
    function.EraseOperations(true);
}

void SplitLargeFanoutTensor::UpdateForRedundantAssemble(Operation &op) {
    auto output = op.oOperand.front();
    auto input = op.iOperand.front();
    auto consumersBackup = output->GetConsumers();
    for (auto &childOp : consumersBackup) {
        childOp->ReplaceInput(input, output);
        if (childOp->GetOpcode() == Opcode::OP_VIEW) {
            auto tensorOffset = input->GetTensorOffset();
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(childOp->GetOpAttribute().get());
            auto viewOffset = viewOpAttribute->GetFromTensorOffset();
            auto newStaticOffset = TensorOffset::Add(viewOffset.offset_, tensorOffset.offset_);
            auto newDynOffset = TensorOffset::Add(viewOffset.dynOffset_, tensorOffset.dynOffset_);
            viewOpAttribute->SetFromOffset(newStaticOffset, newDynOffset);
            ALOG_INFO_F("update offset for OP_VIEW, opmagic: %d.", childOp->GetOpMagic());
        }
    }
}

void SplitLargeFanoutTensor::EraseRedundantCopyOut(Function &function) {
    std::vector<Operation *> redundentCopyOuts;
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        if (op.GetOutputOperand(0)->GetMemoryTypeOriginal() != op.GetInputOperand(0)->GetMemoryTypeOriginal()) {
            continue;
        }
        auto output = op.oOperand.front();
        auto input = op.iOperand.front();
        if ((input == nullptr) || (output == nullptr)) {
            ALOG_ERROR_F("%s[%d] has nullptr input/output.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            continue;
        }
        if (output->nodetype == NodeType::LOCAL && output->GetConsumers().empty()) {
            redundentCopyOuts.push_back(&op);
        }
        if (output->GetProducers().size() == 1 && output->GetConsumers().size() == 1) {
            auto consumerOp = *(output->GetConsumers().begin());
            bool requireCopy = (input->tensor->GetRawShapeSize() != output->tensor->GetRawShapeSize());
            if (consumerOp->GetOpcode() == Opcode::OP_VIEW && !requireCopy) {
                redundentCopyOuts.push_back(&op);
            } else if (input->shape == output->shape && input->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                       output->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                UpdateForRedundantAssemble(op);
                redundentCopyOuts.push_back(&op);
            }
        }
    }
    if (!redundentCopyOuts.empty()) {
        RemoveOps(function, redundentCopyOuts);
    }
}

void SplitLargeFanoutTensor::UpdateForRedundantView(Operation &op, Operation &consumer) {
    auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    auto newOffset = viewAttr->GetFromOffset();
    auto nextViewAttr = dynamic_cast<ViewOpAttribute *>(consumer.GetOpAttribute().get());
    auto viewDynShape = GetViewValidShape(viewAttr->GetToDynValidShape(), nextViewAttr->GetFromOffset(),
        nextViewAttr->GetFromDynOffset(), consumer.oOperand.front()->GetShape());
    nextViewAttr->SetToDynValidShape(viewDynShape);
    auto nextViewOffset = nextViewAttr->GetFromOffset();
    auto newDynOffset = viewAttr->GetFromDynOffset();
    for (size_t i = 0; i < newOffset.size(); ++i) {
        newOffset[i] = newOffset[i] + nextViewOffset[i];
    }
    for (size_t i = 0; i < newDynOffset.size(); i++) {
        newDynOffset[i] = newDynOffset[i] + nextViewOffset[i];
    }
    nextViewAttr->SetFromOffset(newOffset, newDynOffset);
    if (newDynOffset.size() != 0) {
        auto consumerOOperand = consumer.oOperand.front();
        std::vector<SymbolicScalar> dynValidShape;
        if (!nextViewAttr->GetToDynValidShape().empty()) {
            dynValidShape = nextViewAttr->GetToDynValidShape();
        } else if (!consumerOOperand->GetDynValidShape().empty()) {
            dynValidShape = consumerOOperand->GetDynValidShape();
        } else {
            dynValidShape = SymbolicScalar::FromConcrete(consumerOOperand->GetShape());
        }
        if (nextViewAttr->GetToDynValidShape().empty()) {
            nextViewAttr->SetToDynValidShape(dynValidShape);
        }
        if (consumerOOperand->GetDynValidShape().empty()) {
            consumerOOperand->UpdateDynValidShape(dynValidShape);
        }
    }
}

/*
before:
tensor -> View1 -> tensor1 -> View2 -> tesnor2

after:
tensor -> View2_new -> tensor2
*/
void SplitLargeFanoutTensor::EraseRedundantCopyIn(Function &function) {
    std::vector<Operation *> redundentView;
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        auto consumers = op.oOperand.front()->GetConsumers();
        bool allChildrenView = std::all_of(consumers.begin(), consumers.end(),
            [](const Operation *opNext) { return opNext->GetOpcode() == Opcode::OP_VIEW; });
        if (allChildrenView) {
            for (auto &consumer : consumers) {
                UpdateForRedundantView(op, *consumer);
            }
            redundentView.push_back(&op);
        }
    }
    if (!redundentView.empty()) {
        RemoveOps(function, redundentView);
    }
}

void SplitLargeFanoutTensor::UpdateOverSizedLocalBuffer(Function &function) {
    const int UB_SIZE_THRESHOLD = static_cast<int>(PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_UB) * 0.5);
    const int L1_SIZE_THRESHOLD = static_cast<int>(PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_L1) * 0.5);
    ALOG_INFO_F("UB buffer size threshold %d", UB_SIZE_THRESHOLD);
    ALOG_INFO_F("L1 buffer size threshold %d", L1_SIZE_THRESHOLD);
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto assembleOut = op.GetOOperands().front();
        auto memType = assembleOut->GetMemoryTypeOriginal();
        bool oversized = false;
        if ((memType == MemoryType::MEM_UB) && (assembleOut->GetDataSize() > UB_SIZE_THRESHOLD)) {
            oversized = true;
        } else if ((memType == MemoryType::MEM_L1) && (assembleOut->GetDataSize() > L1_SIZE_THRESHOLD)) {
            oversized = true;
        }
        if (oversized) {
            assembleOut->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
            ALOG_INFO_F("%s[%d] output %d is oversized, set as MEM_DEVICE_DDR", op.GetOpcodeStr().c_str(), op.GetOpMagic(), assembleOut->magic);
        }
    }
}

} // namespace npu::tile_fwk
