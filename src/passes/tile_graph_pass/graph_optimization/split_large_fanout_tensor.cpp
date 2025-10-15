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

#include "split_large_fanout_tensor.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {
Status SplitLargeFanoutTensor::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start SplitLargeFanoutTensor.");
    copyOutSources.clear();
    assembles.clear();
    CollectCopyOut(function);
    CompareWithCopyIn(function);
    for (auto &a : assembles) {
        GraphUtils::AddAssembleOperation(function, a);
    }
    EraseRedundantCopyOut(function);
    EraseRedundantCopyIn(function);
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        ALOG_ERROR_F("Eliminate dead operation failed in CommonOperationEliminate.");
        return FAILED;
    }
    ALOG_INFO_F("===> End SplitLargeFanoutTensor.");
    return SUCCESS;
}

/*
收集所有Assemble Op的信息，按输出的raw tensor id进行分类

tensor1 --> Assemble1(toOffset=[0,0])  -->\
                                           \                   /--> view1(fromOffset=[0,0])  --> tensor6
tensor2 --> Assemble2(toOffset=[0,32]) ---->\                 /
                                              tesnor5(raw=10) 
tensor3 --> Assemble3(toOffset=[32,0]) ---->/                 \
                                           /                   \--> view2(fromOffset=[0,32]) --> tensor7
tensor4 --> Assemble4(toOffset=[32,32])-->/

after:
key = 10
val = [(tensor1Ptr, [0,0]), (tensor2Ptr, [0,32]), (tensor3Ptr, [32,0]), (tensor4Ptr, [32,32])], 共计4个元素

*/
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
    std::vector<int64_t> newOffset(fromOffset.size(), 0);
    switch (status) {
        case OverlapStatus::PERFECTLY_MATCH: {
            // 一个Assemble对一个View
            auto overlap = overlaps.front();
            auto newInput = std::make_shared<LogicalTensor>(function, input->Datatype(), output->shape);
            if (newInput == nullptr) { break; }
            newInput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
            assembles.emplace_back(AssembleOp{overlap->GetMemoryTypeOriginal(), newOffset, overlap, newInput});
            viewOpAttribute->SetFromOffset(newOffset);
            GraphUtils::UpdateViewAttr(function, op);
            op.ReplaceInput(newInput, input);
            break;
        }
        case OverlapStatus::BE_COVERED: {
            // Assemble输入大于View的输出
            auto overlap = overlaps.front();
            auto newInput = std::make_shared<LogicalTensor>(function, input->Datatype(), overlap->shape);
            if (newInput == nullptr) { break; }
            newInput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
            assembles.emplace_back(AssembleOp{overlap->GetMemoryTypeOriginal(), newOffset, overlap, newInput});
            std::transform(fromOffset.begin(), fromOffset.end(), matchedTensors.front()->offset.begin(), newOffset.begin(), [](int a, int b) { return a - b; });
            viewOpAttribute->SetFromOffset(newOffset);
            GraphUtils::UpdateViewAttr(function, op);
            op.ReplaceInput(newInput, input);
            break;
        }
        case OverlapStatus::PERFECTLY_MATCH_WITH_ALL: {
            // 多个Assemble对一个View
            auto newInput = std::make_shared<LogicalTensor>(function, input->Datatype(), output->shape);
            if (newInput == nullptr) { break; }
            newInput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
            for (size_t i = 0; i < overlaps.size(); i++) {
                auto &overlap = overlaps[i];
                std::vector<int64_t> newAssembleOffset = matchedTensors[i]->GetOffset();
                for (size_t j = 0; j < newAssembleOffset.size(); ++j) {
                    newAssembleOffset[j] -= fromOffset[j];
                }
                assembles.emplace_back(AssembleOp{overlap->GetMemoryTypeOriginal(), newAssembleOffset, overlap, newInput});
            }
            viewOpAttribute->SetFromOffset(newOffset);
            GraphUtils::UpdateViewAttr(function, op);
            op.ReplaceInput(newInput, input);
            break;
        }
        default: break;
    }
}

/*
在收集完所有Assemble的信息后，遍历所有的View Op
判断是否存在View所需的输出tensor与已收集的Assemble的输入间是否存在
*/
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
        /*
        创建一块与View的输出shape相同, offset与属性记录相同，但是指向其输入的raw tensor的一块等效tensor
        代表了前View 需要的tensor
        */
        auto inputView = std::make_shared<LogicalTensor>(function, input->tensor, fromOffset, output->shape);
        if (inputView == nullptr) {
            continue;
        }
        std::vector<std::shared_ptr<LogicalTensor>> outputOfAssemble;
        /* View的输入raw 与 Assemble的输出raw 一致 */
        for (auto &[copyOutSource, offsetInAssemble] : copyOutSources[input->tensor->rawmagic]) {
            /*
            创建一块与Assemble输入shape相同, offset与属性记录相同，但是指向View的输入的raw tensor的一块目标tensor
            代表目前Assemble可以提供的
            */
            auto target =
                std::make_shared<LogicalTensor>(function, input->tensor, offsetInAssemble, copyOutSource->GetShape());
            if (target == nullptr) {
                continue;
            }
            /*
            1. 此时 inputView 和 target 都指向了同一个raw tensor;
            2. 判断view所需要的tensor 和 所有Assemble可提供的tensor间的overlap关系
            */
            auto status = CalcOverlap(inputView, target, true);
            ALOG_DEBUG_F("Assemble In %d(raw %d) versus View Out %d(raw %d), commom tensor %d(raw %d), status: %s",
                copyOutSource->magic, copyOutSource->tensor->rawmagic,
                output->magic, output->tensor->rawmagic,
                input->magic, input->tensor->rawmagic,
                OverlapStatusString(status).c_str());
            if (status == OverlapStatus::PERFECTLY_MATCH || status == OverlapStatus::BE_COVERED) {
                // 1. inputView 与 target 完全match; 2. inputView 被 target 覆盖
                outputOfAssemble.emplace_back(target);
                overlaps.push_back(copyOutSource);
                break;
            }
            if (status == OverlapStatus::COVERED) {
                // inputView 覆盖了 target, 需要继续遍历
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
    std::vector<Operation *> redundantCopyOuts;
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
        if (!function.IsFromOutCast(output) && output->GetConsumers().empty()) {
            /* input --> Assemble --> output(非OCAST, 且没有consumer) */
            redundantCopyOuts.push_back(&op);
        }
        if (output->GetProducers().size() != 1 || output->GetConsumers().size() != 1) {
            continue;
        }
        auto consumerOp = *(output->GetConsumers().begin());
        // Assemble输入和输出的raw tensor大小不相等，意味着要做拷贝
        bool requireCopy = (input->tensor->GetRawShapeSize() != output->tensor->GetRawShapeSize());
        if (consumerOp->GetOpcode() == Opcode::OP_VIEW && !requireCopy) {
            /*
            Before: input --> Assmeble --> output --> View
            After:  input --> View
            因为input和output的raw shape相同，所以View上的offset不需要修改
            */
            redundantCopyOuts.push_back(&op);
            continue;
        }
        if (input->shape == output->shape && input->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                    output->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            /* 因为input和output raw shape size不同，但shape相同，因此删除前需要重新计算View的offset */
            UpdateForRedundantAssemble(op);
            redundantCopyOuts.push_back(&op);
        }
    }
    if (!redundantCopyOuts.empty()) {
        RemoveOps(function, redundantCopyOuts);
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
    if (newDynOffset.size() == 0) {
        return;
    }
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

/*
before:
tensor -> View1 -> tensor1 -> View2 -> tesnor2

after:
tensor -> View2_new -> tensor2
*/
void SplitLargeFanoutTensor::EraseRedundantCopyIn(Function &function) {
    std::vector<Operation *> redundantView;
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        /*
        case1. split_large_fanout_tensor 在AssignmemType 之前，view op GetOpAttribute()->GetTo() == MemoryType::MEM_L1的view op一定是tile op展开时插入的，不能删；
        case2. 框架在tile 展开插入的view之前还插入了一个view，目前不删除，后续优化可以考虑删除。
        */
        bool isViewToL1 = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get())->GetTo() == MemoryType::MEM_L1;
        auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
        auto consumers = op.oOperand.front()->GetConsumers();
        if (consumers.empty()) {
            continue;
        }
        bool allChildrenView = std::all_of(consumers.begin(), consumers.end(),
            [=](const Operation *opNext) {
                if (opNext->GetOpcode() != Opcode::OP_VIEW) {
                    return false;
                }
                auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(opNext->GetOpAttribute().get());
                if(isViewToL1 || viewOpAttribute->GetTo() == MemoryType::MEM_L1) {
                    return false;
                }
                return true;
            });
        if (allChildrenView) {
            GraphUtils::UpdateViewAttr(function, op);
            for (auto &consumer : consumers) {
                UpdateForRedundantView(op, *consumer);
            }
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            ALOG_DEBUG_F("Found redundant view and remove it, opmagic: %d, to: %s. Input mem: %s, Output mem: %s.",
                op.GetOpMagic(),BriefMemoryTypeToString(viewAttr->GetTo()).c_str(),
                BriefMemoryTypeToString(input->GetMemoryTypeOriginal()).c_str(),
                BriefMemoryTypeToString(output->GetMemoryTypeOriginal()).c_str());
            redundantView.push_back(&op);
        }
    }
    if (!redundantView.empty()) {
        RemoveOps(function, redundantView);
    }
}

} // namespace npu::tile_fwk
