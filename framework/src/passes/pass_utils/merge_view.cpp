/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dead_operation_eliminate.cpp
 * \brief
 */

#include "merge_view.h"
#include "interface/operation/attribute.h"
#include "interface/tensor/logical_tensor.h"
#include "dead_operation_eliminate.h"

namespace npu::tile_fwk {

Status MergeView::MergeViewOp(Function &function) {
    Status status = Initialize();
    if (status != SUCCESS) {
        // APASS_LOG_ERROR_F(Elements::Function, "MergeView initialization failed.");
        return status;
    }
    for (auto &op : function.Operations()) {
        if (visitedOp_.count(op.GetOpMagic()) != 0) {
            continue;
        }
        Status processStatus = SUCCESS;
        std::vector<Operation *> chain;
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            processStatus = MergeViewChain(function, op, chain);
        }
        if (processStatus != SUCCESS) {
            // APASS_LOG_ERROR_F(Elements::Operation, "ProcessView failed for operation %s[%d].%s",
            //     op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return processStatus;
        }
    }
    status = AppendMergedViewOperations(function);
    if (status != SUCCESS) {
        // APASS_LOG_ERROR_F(Elements::Function, "AppendMergedViewOperations phase failed.");
        return status;
    }
    status = CleanUp(function);
    if (status != SUCCESS)
    {
        // APASS_LOG_ERROR_F(Elements::Function, "Cleanup phase failed.");
        return status;
    }
    return SUCCESS;
}

Status MergeView::Initialize() {
    visitedOp_.clear();
    // APASS_LOG_INFO_F(Elements::Function, "===> Start MergeView.");
    viewOpToAppend_.clear();
    return SUCCESS;
}

Status MergeView::MergeViewChain(
    Function &function, Operation &operation, std::vector<Operation *> &chain) {
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    // APASS_LOG_DEBUG_F(Elements::Operation, "Processing View operation %d, memory_to: %d, chain size: %zu.",
    //             operation.GetOpMagic(),
    //             viewOpAttribute ? static_cast<int>(viewOpAttribute->GetTo()) : -1,
    //             chain.size());
    // 1. 初始化操作链
    InitOperationChain(operation, chain);

    // 2. 处理消费者链
    auto consumers = function.FindConsumers(operation);
    bool chainEnd = true;
    Status status = ProcessConsumerChain(function, consumers, chain, chainEnd);
    if (status != SUCCESS) {
        return status;
    }

    // 3. 处理链尾情况
    if (chainEnd && chain.size() > 1) {
        return ProcessChainEnd(function, chain);
    }

    return SUCCESS;
}

void MergeView::InitOperationChain(Operation &operation, std::vector<Operation *> &chain)
{
    visitedOp_.insert(operation.opmagic);
    chain.emplace_back(&operation);
}

Status MergeView::ProcessConsumerChain(
    Function &function,
    const std::set<Operation*, LogicalTensor::CompareOp>& consumers,
    std::vector<Operation *> &chain,
    bool &chainEnd)
{
    if (consumers.empty()) {
        return SUCCESS;
    }
    Operation *currentOp = chain.back();
    auto currentViewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(currentOp->GetOpAttribute());
    if (!currentViewAttr) {
        // APASS_LOG_ERROR_F(Elements::Operation, "Failed to get current view attribute.%s", GetFormatBacktrace(*currentOp).c_str());
        return FAILED;
    }
    MemoryType currentMemType = currentViewAttr->GetTo();
    for (auto &op : consumers) {
        if (!op) {
            // APASS_LOG_ERROR_F(Elements::Operation, "Null consumer operation found.%s", GetFormatBacktrace(*currentOp).c_str());
            return FAILED;
        }
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
            if (viewOpAttribute == nullptr) {
                // APASS_LOG_ERROR_F(Elements::Operation, "View operation %d has null viewOpAttribute.", op->GetOpMagic());
                return FAILED;
            }
            auto memory_to = viewOpAttribute->GetTo();
            // 根据新的合并原则判断是否可以合并
            bool canMerge = false;
            if (currentMemType == MemoryType::MEM_UNKNOWN) {
                // unknown memType 可以向它之后的view合并
                canMerge = true;
                // APASS_LOG_DEBUG_F(Elements::Operation, "Current memType is UNKNOWN, can merge with the next view (memType: %d).", 
                        //    static_cast<int>(memory_to));
            } else if (currentMemType == memory_to) {
                // 相同memType的view可以合并
                canMerge = true;
                // APASS_LOG_DEBUG_F(Elements::Operation, "Same memType (%d), can merge view operations.", static_cast<int>(currentMemType));
            } else {
                // 不相同memType的不能合并
                // APASS_LOG_DEBUG_F(Elements::Operation, "Cannot merge view operations with different memory types: current=%d, next=%d.", 
                        //    static_cast<int>(currentMemType), static_cast<int>(memory_to));
            }
            if (canMerge) {
                chainEnd = false;
                Status status = MergeViewChain(function, *op, chain);
                if (status != SUCCESS) {
                    return status;
                }
                chain.pop_back();
            } else {
                chainEnd = true;
            }
        }
    }
    return SUCCESS;
}

Status MergeView::ProcessChainEnd(
    Function &function,
    std::vector<Operation *> &chain)
{
    // 1. 验证链的有效性
    Operation *startOp = chain.front();
    Operation *endOp = chain.back();
    if (startOp->iOperand.empty()) {
        // APASS_LOG_ERROR_F(Elements::Operation, "First operation in chain (opmagic: %d) has no input operands.%s",
        //     startOp->GetOpMagic(), GetFormatBacktrace(*startOp).c_str());
        return FAILED;
    }
    if (endOp->oOperand.empty()) {
        // APASS_LOG_ERROR_F(Elements::Operation, "Last operation in chain (opmagic: %d) has no output operands.%s",
        //     endOp->GetOpMagic(), GetFormatBacktrace(*endOp).c_str());
        return FAILED;
    }
    auto &startTensor = startOp->iOperand.front();
    auto &endTensor = endOp->oOperand.front();
    if (!startTensor) {
        // APASS_LOG_ERROR_F(Elements::Operation, "Null input tensor found for first operation in chain (opmagic: %d).%s",
        //     startOp->GetOpMagic(), GetFormatBacktrace(*startOp).c_str());
        return FAILED;
    }
    if (!endTensor) {
        // APASS_LOG_ERROR_F(Elements::Operation, "Null output tensor found for last operation in chain (opmagic: %d).%s",
        //     endOp->GetOpMagic(), GetFormatBacktrace(*endOp).c_str());
        return FAILED;
    }
    std::vector<int64_t> newOffset;
    std::vector<SymbolicScalar> newDynOffset;
    std::vector<SymbolicScalar> newDynValidShape;
    Status status = CalculateMergedOffsets(chain, newOffset, newDynOffset, newDynValidShape);
    if (status != SUCCESS) {
        return status;
    }
    // 记录合并操作
    RecordMergedViewOperation(endOp, startTensor, endTensor, newOffset, newDynOffset, newDynValidShape);

    // 清理链尾
    endOp->oOperand.clear();
    function.GetTensorMap().Erase(endTensor);
    // APASS_LOG_DEBUG_F(Elements::Operation, "Successfully processed view chain ending with opmagic: %d, chain size: %zu.", 
    //             endOp->GetOpMagic(), chain.size());
    return SUCCESS;
}

Status MergeView::CalculateMergedOffsets(const std::vector<Operation *> &chain, std::vector<int64_t> &newOffset,
    std::vector<SymbolicScalar> &newDynOffset, std::vector<SymbolicScalar> &newDynValidShape) {
    for (size_t i = 0; i < chain.size(); ++i) {
        const auto &view = chain[i];
        if (!view) {
            // APASS_LOG_ERROR_F(Elements::Operation, "Null view operation in chain.");
            return FAILED;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(view->GetOpAttribute());
        if (!viewOpAttribute) {
            // APASS_LOG_ERROR_F(Elements::Operation, "Failed to get ViewOpAttribute.%s", GetFormatBacktrace(*view).c_str());
            return FAILED;
        }
        if (i == 0) {
            newOffset = viewOpAttribute->GetFromOffset();
            newDynOffset = viewOpAttribute->GetFromDynOffset();
            if (!viewOpAttribute->GetToDynValidShape().empty()) {
                newDynValidShape = viewOpAttribute->GetToDynValidShape();
            }
            continue;
        }
        auto ret = TensorOffset::Add(newOffset, newDynOffset, viewOpAttribute->GetFromOffset(), viewOpAttribute->GetFromDynOffset());
        if (!ret.first.empty()) {
            newOffset = ret.first;
            newDynOffset = ret.second;
        }
        if (!viewOpAttribute->GetToDynValidShape().empty()) {
            newDynValidShape = viewOpAttribute->GetToDynValidShape();
            continue;
        }
        newDynValidShape = GetViewValidShape(newDynValidShape, viewOpAttribute->GetFromOffset(),
            viewOpAttribute->GetFromDynOffset(), view->GetOOperands()[0]->GetShape());
    }
    return SUCCESS;
}

void MergeView::RecordMergedViewOperation(Operation* lastViewOp, const std::shared_ptr<LogicalTensor> &startTensor,
    const std::shared_ptr<LogicalTensor> &endTensor, const std::vector<int64_t> &newOffset,
    const std::vector<SymbolicScalar> &newDynOffset, const std::vector<SymbolicScalar> &newDynValidShape) {
    // 获取最后一个VIEW的属性
    auto lastViewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(lastViewOp->GetOpAttribute());
    if (!lastViewAttr) {
        // APASS_LOG_ERROR_F(Elements::Operation, "Failed to get last ViewOpAttribute for opmagic: %d.%s", 
        //             lastViewOp->GetOpMagic(), GetFormatBacktrace(*lastViewOp).c_str());
        return;
    }
    // 获取特定的 op_attr_copy_in_mode 属性
    int64_t copyInModeValue = 0;
    bool hasCopyInMode = lastViewOp->GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue);
    // 清理消费者关系
    endTensor->GetProducers().clear();
    // 记录合并op
    viewOpToAppend_.emplace_back(ViewOp{startTensor, endTensor, newOffset, newDynOffset, newDynValidShape, lastViewAttr->GetTo(), hasCopyInMode, std::move(copyInModeValue)});
    // APASS_LOG_DEBUG_F(Elements::Operation, "Recorded merged view operation from opmagic: %d, hasCopyInMode: %d.", 
    //             lastViewOp->GetOpMagic(), hasCopyInMode);
}

Status MergeView::AppendMergedViewOperations(Function &function) {
    /* Process View ops first to avoid View output being cleared in View-Assemble scenarios */
    for (auto &viewOp : viewOpToAppend_) {
        auto attr = std::make_shared<ViewOpAttribute>(viewOp.offset, viewOp.toType, viewOp.dynOffset,
                     viewOp.dynValidShape);
        if (!attr) {
            // APASS_LOG_ERROR_F(Elements::Operation, "Failed to create ViewOpAttribute.");
            return FAILED;
        }
        auto &mergedViewOp = function.AddRawOperation(Opcode::OP_VIEW, {viewOp.input}, {viewOp.output});
        mergedViewOp.SetOpAttribute(attr);
        // 继承op_attr_copy_in_mode属性
        if (viewOp.hasCopyInMode) {
            mergedViewOp.SetAttr("op_attr_copy_in_mode", viewOp.copyInModeValue); 
            // APASS_LOG_DEBUG_F(Elements::Operation, "Inherited op_attr_copy_in_mode attribute for merged view operation.");
        }   
        viewOp.output->UpdateDynValidShape(viewOp.dynValidShape);    
    }
    // APASS_LOG_DEBUG_F(Elements::Operation, "Appended %zu merged view operations.", viewOpToAppend_.size());
    return SUCCESS;
}

Status MergeView::EraseRedundantAssemble(Function &function) const {
    std::unordered_set<Operation *> redundantAssembles;
    for (auto &op : function.Operations(false)) {
        if (op.GetOpcode() !=  Opcode::OP_ASSEMBLE) {
            continue;
        }
        if (op.iOperand.empty()) {
            // APASS_LOG_ERROR_F(Elements::Operation, "Assemble operation with no input operands.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if (op.iOperand.front()->GetProducers().empty()) {
            redundantAssembles.emplace(&op);
        }
    }
    for (auto &ele : redundantAssembles) {
        if (!ele) {
            // APASS_LOG_ERROR_F(Elements::Operation, "Null operation in redundantAssembles.");
            continue;
        }
        ele->SetAsDeleted();
    }
    function.EraseOperations(true, false);
    return SUCCESS;
}

Status MergeView::CleanUp(Function &function) {
    Status status = EraseRedundantAssemble(function);
    if (status != SUCCESS)
    {
        // APASS_LOG_ERROR_F(Elements::Function, "EraseRedundantAssemble failed.");
        return status;
    }
    DeadOperationEliminator eliminator;
    eliminator.EliminateOperation(function, false);
    // APASS_LOG_INFO_F(Elements::Function, "===> End MergeViewAssemble.");
    return SUCCESS;
}
} // namespace