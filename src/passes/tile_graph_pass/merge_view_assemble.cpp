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
 * \file merge_view_assemble.cpp
 * \brief Implementation of view and assemble operation merging pass
 */

#include "merge_view_assemble.h"
#include "interface/operation/attribute.h"
#include "passes/tile_graph_pass/dead_operation_eliminate.h"

namespace npu::tile_fwk {
Status MergeViewAssemble::RunOnFunction(Function &function) {
    Status status = Initialize();
    if (status != SUCCESS) 
    {
        ALOG_ERROR_F("MergeViewAssemble initialization failed"); 
        return status; 
    }
    status = ProcessViewOperations(function);
    if (status != SUCCESS) 
    {
        ALOG_ERROR_F("Processing view operations failed"); 
        return status; 
    }
    status = ProcessAssembleOperations(function);
    if (status != SUCCESS) 
    { 
        ALOG_ERROR_F("Processing assemble operations failed"); 
        return status; 
    }
    status = CleanUp(function);
    if (status != SUCCESS) 
    { 
        ALOG_ERROR_F("Cleanup phase failed"); 
        return status; 
    }
    return SUCCESS;
}

Status MergeViewAssemble::Initialize() {
    visitedOp_.clear();
    ALOG_INFO_F("===> Start MergeViewAssemble.");
    viewOpToAppend_.clear();
    assembleOpToAppend_.clear();
    return SUCCESS;
}

Status MergeViewAssemble::ProcessViewOperations(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            std::vector<Operation *> chain;
            if (visitedOp_.count(op.opmagic) == 0) {
                Status status = MergeViewChain(function, op, chain);
                if (status != SUCCESS) { ALOG_ERROR_F("MergeViewChain failed for operation %d", op.opmagic); return status; }
            }
        }
    }
    return AppendMergedViewOperations(function);
}

Status MergeViewAssemble::ProcessAssembleOperations(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && visitedOp_.count(op.GetOpMagic()) == 0) {
            std::vector<Operation *> chain;
            Status status = MergeAssembleChain(function, op, chain);
            if (status != SUCCESS) { ALOG_ERROR_F("MergeAssembleChain failed for operation %d", op.GetOpMagic()); return status; }
        }
    }
    return AppendMergedAssembleOperations(function);
}

Status MergeViewAssemble::AppendMergedViewOperations(Function &function) {
    /* Process View ops first to avoid View output being cleared in View-Assemble scenarios */
    for (auto &viewOp : viewOpToAppend_) {
        auto attr = std::make_shared<ViewOpAttribute>(viewOp.offset, viewOp.dynOffset, 
                     viewOp.dynValidShape);
        if (!attr) { ALOG_ERROR_F("Failed to create ViewOpAttribute"); return FAILED; }
        auto &mergedViewOp = function.AddRawOperation(Opcode::OP_VIEW, {viewOp.input}, {viewOp.output});
        mergedViewOp.SetOpAttribute(attr);
        viewOp.output->UpdateDynValidShape(viewOp.dynValidShape);
    }
    return SUCCESS;
}

Status MergeViewAssemble::AppendMergedAssembleOperations(Function &function) {
    for (auto &assembleOp : assembleOpToAppend_) {
        auto attr = std::make_shared<AssembleOpAttribute>(assembleOp.offset, assembleOp.dynOffset);
        if (!attr) { ALOG_ERROR_F("Failed to create AssembleOpAttribute"); return FAILED; }
        auto &mergedAssembleOp = function.AddRawOperation(Opcode::OP_ASSEMBLE, {assembleOp.input}, {assembleOp.output});
        mergedAssembleOp.SetOpAttribute(attr);
    }
    return SUCCESS;
}

Status MergeViewAssemble::CleanUp(Function &function) {
    Status status = EraseRedundantAssemble(function);
    if (status != SUCCESS) 
    { 
        ALOG_ERROR_F("EraseRedundantAssemble failed"); 
        return status; 
    }
    DeadOperationEliminator eliminator;
    eliminator.EliminateDeadOperationBackward(function);
    ALOG_INFO_F("===> End MergeViewAssemble.");
    return SUCCESS;
}

Status MergeViewAssemble::MergeViewChain(
    Function &function, Operation &operation, std::vector<Operation *> &chain) {
    // 1. 初始化操作链
    InitOperationChain(operation, chain);
    
    // 2. 处理消费者链
    auto consumers = function.FindConsumers(operation);
    bool chainEnd = true;
    Status status = ProcessConsumerChain(function, consumers, chain, chainEnd);
    if (status != SUCCESS) { return status; }

    // 3. 处理链尾情况
    if (chainEnd && chain.size() > 1) {
        return ProcessChainEnd(function, chain);
    }

    return SUCCESS;
}

void MergeViewAssemble::InitOperationChain(Operation &operation, std::vector<Operation *> &chain) 
{
    visitedOp_.insert(operation.opmagic);
    chain.emplace_back(&operation);
}

Status MergeViewAssemble::ProcessConsumerChain(
    Function &function,
    const std::set<Operation*, LogicalTensor::CompareOp>& consumers,
    std::vector<Operation *> &chain,
    bool &chainEnd)
{
    if (consumers.empty()) { return SUCCESS; }
    for (auto &op : consumers) {
        if (!op) { ALOG_ERROR_F("Null consumer operation found"); return FAILED; }
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            chainEnd = false;
            Status status = MergeViewChain(function, *op, chain);
            if (status != SUCCESS) { return status; }
            chain.pop_back();
        }
    }
    return SUCCESS;
}

Status MergeViewAssemble::ProcessChainEnd(
    Function &function,
    std::vector<Operation *> &chain)
{
    // 1. 验证链的有效性
    if (chain.front()->iOperand.empty() || chain.back()->oOperand.empty()) { ALOG_ERROR_F("Invalid chain operations"); return FAILED; }
    auto &startTensor = chain.front()->iOperand.front();
    auto &endTensor = chain.back()->oOperand.front();
    if (!startTensor || !endTensor) { ALOG_ERROR_F("Null tensor found in chain"); return FAILED; }
    std::vector<int32_t> newOffset;
    std::vector<SymbolicScalar> newDynOffset;
    std::vector<SymbolicScalar> newDynValidShape;
    Status status = CalculateMergedOffsets(chain, newOffset, newDynOffset, newDynValidShape);
    if (status != SUCCESS) { return status; }

    // 4. 记录合并操作
    RecordMergedViewOperation(startTensor, endTensor, newOffset, newDynOffset, newDynValidShape);
    
    // 5. 清理链尾
    chain.back()->oOperand.clear();
    function.GetTensorMap().Erase(endTensor);
    
    return SUCCESS;
}

Status MergeViewAssemble::CalculateMergedOffsets(
    const std::vector<Operation *> &chain,
    std::vector<int32_t> &newOffset,
    std::vector<SymbolicScalar> &newDynOffset,
    std::vector<SymbolicScalar> &newDynValidShape)
{
    for (size_t i = 0; i < chain.size(); ++i) {
        const auto &view = chain[i];
        if (!view) { ALOG_ERROR_F("Null view operation in chain"); return FAILED; }
        auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view->GetOpAttribute().get());
        if (!viewOpAttribute) { ALOG_ERROR_F("Failed to get ViewOpAttribute"); return FAILED; }
        if (i == 0) {
            newOffset = viewOpAttribute->GetFromOffset();
            newDynOffset = viewOpAttribute->GetFromDynOffset();
            if (newDynValidShape.empty() && !viewOpAttribute->GetToDynValidShape().empty()) {
                newDynValidShape = viewOpAttribute->GetToDynValidShape();
            }
        } else {
            auto ret = TensorOffset::Add(newOffset, newDynOffset, viewOpAttribute->GetFromOffset(), viewOpAttribute->GetFromDynOffset());
            if (!ret.first.empty()) {
                newOffset = ret.first;
                newDynOffset = ret.second;
            }
            if (newDynValidShape.empty() && !viewOpAttribute->GetToDynValidShape().empty()) {
                newDynValidShape = viewOpAttribute->GetToDynValidShape();
            } else {
                newDynValidShape = GetViewValidShape(newDynValidShape, viewOpAttribute->GetFromOffset(),
                    viewOpAttribute->GetFromDynOffset(), view->GetOOperands()[0]->GetShape());
            }
        }
    }
    return SUCCESS;
}

void MergeViewAssemble::RecordMergedViewOperation(
    const std::shared_ptr<LogicalTensor> &startTensor,
    const std::shared_ptr<LogicalTensor> &endTensor,
    const std::vector<int32_t> &newOffset,
    const std::vector<SymbolicScalar> &newDynOffset,
    const std::vector<SymbolicScalar> &newDynValidShape)
{
    endTensor->GetProducers().clear();
    viewOpToAppend_.emplace_back(ViewOp{startTensor, endTensor, newOffset, newDynOffset, newDynValidShape});
}

Status MergeViewAssemble::MergeAssembleChain(Function &function, Operation &operation, std::vector<Operation *> &chain) {
    // 1. 初始化操作链
    InitAssembleChain(operation, chain);
    
    // 2. 处理消费者
    auto consumers = function.FindConsumers(operation);
    bool chainEnd = consumers.empty();
    Status status = ProcessAssembleConsumers(function, consumers, chain, chainEnd);
    if (status != SUCCESS) { return status; }

    // 3. 处理链尾情况
    if (chainEnd && chain.size() > 1) {
        status = ProcessAssembleChainEnd(function, chain, operation);
        if (status != SUCCESS) { return status; }
    }

    chain.pop_back();
    return SUCCESS;
}

void MergeViewAssemble::InitAssembleChain(
    Operation &operation, 
    std::vector<Operation *> &chain) 
{
    visitedOp_.insert(operation.opmagic);
    chain.emplace_back(&operation);
}

Status MergeViewAssemble::ProcessAssembleConsumers(
    Function &function,
    const std::set<Operation*, LogicalTensor::CompareOp>& consumers,
    std::vector<Operation *> &chain,
    bool &chainEnd)
{
    if (consumers.empty()) { return SUCCESS; }
    for (auto &op : consumers) {
        if (!op) { ALOG_ERROR_F("Null consumer operation found"); return FAILED; }
        if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            Status status = MergeAssembleChain(function, *op, chain);
            if (status != SUCCESS) { return status; }
        } else {
            chainEnd = true;
        }
    }
    return SUCCESS;
}

Status MergeViewAssemble::ProcessAssembleChainEnd(
    Function &function,
    std::vector<Operation *> &chain,
    Operation &operation)
{
    // 验证链有效性
    if (chain.front()->iOperand.empty() || chain.back()->oOperand.empty()) { ALOG_ERROR_F("Invalid chain operations"); return FAILED; }
    auto &startTensor = chain.front()->iOperand.front();
    auto &endTensor = chain.back()->oOperand.front();
    if (!startTensor || !endTensor) { ALOG_ERROR_F("Null tensor found in chain"); return FAILED; }
    // 计算合并offset
    auto [newOffset, newDynOffset] = CalculateAssembleOffsets(chain, startTensor->offset.size());
    // 4. 记录并清理
    RecordAssembleOperation(startTensor, endTensor, newOffset, newDynOffset);
    function.GetTensorMap().Erase(endTensor);
    operation.SetAsDeleted();

    return SUCCESS;
}

std::pair<std::vector<int32_t>, std::vector<SymbolicScalar>> 
MergeViewAssemble::CalculateAssembleOffsets(
    const std::vector<Operation *> &chain,
    size_t offsetSize)
{
    std::vector<int32_t> newOffset(offsetSize, 0);
    std::vector<SymbolicScalar> newDynOffset;
    for (size_t i = 0; i < chain.size(); ++i) {
        const auto &assemble = chain[i];
        if (!assemble) { ALOG_ERROR_F("Null assemble operation in chain"); return {}; }
        auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(assemble->GetOpAttribute().get());
        if (!assembleOpAttribute) { ALOG_ERROR_F("Failed to get AssembleOpAttribute"); return {}; }
        if (i == 0) {
            newOffset = assembleOpAttribute->GetToOffset();
            newDynOffset = assembleOpAttribute->GetToDynOffset();
        } else {
            auto ret = TensorOffset::Add(newOffset, newDynOffset, assembleOpAttribute->GetToOffset(), assembleOpAttribute->GetToDynOffset());
            if (!ret.first.empty()) {
                newOffset = ret.first;
                newDynOffset = ret.second;
            }
        }
    }
    return {newOffset, newDynOffset};
}

void MergeViewAssemble::RecordAssembleOperation(
    const std::shared_ptr<LogicalTensor> &input,
    const std::shared_ptr<LogicalTensor> &output,
    const std::vector<int32_t> &offset,
    const std::vector<SymbolicScalar> &dynOffset)
{
    assembleOpToAppend_.emplace_back(AssembleOp{input, output, offset, dynOffset});
}

Status MergeViewAssemble::EraseRedundantAssemble(Function &function) const {
    std::unordered_set<Operation *> redundantAssembles;
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() !=  Opcode::OP_ASSEMBLE) {
            continue;
        }
        if (op.iOperand.empty()) { ALOG_ERROR_F("Assemble operation with no input operands"); return FAILED; }
        if (op.iOperand.front()->GetProducers().empty()) {
            redundantAssembles.emplace(&op);
        }
    }
    for (auto &ele : redundantAssembles) {
        if (!ele) {
            ALOG_ERROR_F("Null operation in redundantAssembles");
            continue;
        }
        ele->SetAsDeleted();
    }
    function.EraseOperations(true);
    return SUCCESS;
}
} // namespace npu::tile_fwk
