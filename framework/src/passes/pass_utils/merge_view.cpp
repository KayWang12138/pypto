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
            return processStatus;
        }
    }
    status = AppendMergedViewOperations(function);
    if (status != SUCCESS) {
        return status;
    }
    status = CleanUp(function);
    if (status != SUCCESS)
    {
        return status;
    }
    return SUCCESS;
}

Status MergeView::Initialize() {
    visitedOp_.clear();
    viewOpToAppend_.clear();
    return SUCCESS;
}

Status MergeView::MergeViewChain(
    Function &function, Operation &operation, std::vector<Operation *> &chain) {
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
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
        return FAILED;
    }
    MemoryType currentMemType = currentViewAttr->GetTo();
    for (auto &op : consumers) {
        if (!op) {
            return FAILED;
        }
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
            if (viewOpAttribute == nullptr) {
                return FAILED;
            }
            auto memory_to = viewOpAttribute->GetTo();
            // 根据新的合并原则判断是否可以合并
            bool canMerge = false;
            if (currentMemType == MemoryType::MEM_UNKNOWN) {
                // unknown memType 可以向它之后的view合并
                canMerge = true;
            } else (currentMemType == memory_to) {
                // 相同memType的view可以合并
                canMerge = true;
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
        return FAILED;
    }
    if (endOp->oOperand.empty()) {
        return FAILED;
    }
    auto &startTensor = startOp->iOperand.front();
    auto &endTensor = endOp->oOperand.front();
    if (!startTensor) {
        return FAILED;
    }
    if (!endTensor) {
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
    return SUCCESS;
}

Status MergeView::CalculateMergedOffsets(const std::vector<Operation *> &chain, std::vector<int64_t> &newOffset,
    std::vector<SymbolicScalar> &newDynOffset, std::vector<SymbolicScalar> &newDynValidShape) {
    for (size_t i = 0; i < chain.size(); ++i) {
        const auto &view = chain[i];
        if (!view) {
            return FAILED;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(view->GetOpAttribute());
        if (!viewOpAttribute) {
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
        return;
    }
    // 获取特定的 op_attr_copy_in_mode 属性
    int64_t copyInModeValue = 0;
    bool hasCopyInMode = lastViewOp->GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue);
    // 清理消费者关系
    endTensor->GetProducers().clear();
    // 记录合并op
    viewOpToAppend_.emplace_back(ViewOp{startTensor, endTensor, newOffset, newDynOffset, newDynValidShape, lastViewAttr->GetTo(), hasCopyInMode, std::move(copyInModeValue)});
}

Status MergeView::AppendMergedViewOperations(Function &function) {
    /* Process View ops first to avoid View output being cleared in View-Assemble scenarios */
    for (auto &viewOp : viewOpToAppend_) {
        auto attr = std::make_shared<ViewOpAttribute>(viewOp.offset, viewOp.toType, viewOp.dynOffset,
                     viewOp.dynValidShape);
        if (!attr) {
            return FAILED;
        }
        auto &mergedViewOp = function.AddRawOperation(Opcode::OP_VIEW, {viewOp.input}, {viewOp.output});
        mergedViewOp.SetOpAttribute(attr);
        // 继承op_attr_copy_in_mode属性
        if (viewOp.hasCopyInMode) {
            mergedViewOp.SetAttr("op_attr_copy_in_mode", viewOp.copyInModeValue); 
        }   
        viewOp.output->UpdateDynValidShape(viewOp.dynValidShape);    
    }
    return SUCCESS;
}

Status MergeView::EraseRedundantAssemble(Function &function) const {
    std::unordered_set<Operation *> redundantAssembles;
    for (auto &op : function.Operations(false)) {
        if (op.GetOpcode() !=  Opcode::OP_ASSEMBLE) {
            continue;
        }
        if (op.iOperand.empty()) {
            return FAILED;
        }
        if (op.iOperand.front()->GetProducers().empty()) {
            redundantAssembles.emplace(&op);
        }
    }
    for (auto &ele : redundantAssembles) {
        if (!ele) {
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
        return status;
    }
    DeadOperationEliminator eliminator;
    eliminator.EliminateOperation(function, false);
    return SUCCESS;
}
} // namespace