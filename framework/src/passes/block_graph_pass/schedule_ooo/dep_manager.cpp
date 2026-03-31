/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dep_manager.cpp
 * \brief Dependency manager implementation
 */

#include "passes/block_graph_pass/schedule_ooo/dep_manager.h"
#include "passes/pass_log/pass_log.h"

#ifndef MODULE_NAME
#define MODULE_NAME "DependencyManager"
#endif

namespace npu::tile_fwk {

void DependencyManager::RegisterOp(Operation *op) {
    if (op == nullptr) {
        return;
    }
    if (inGraph_.find(op) == inGraph_.end()) {
        inGraph_[op] = std::unordered_set<Operation *>();
    }
    if (outGraph_.find(op) == outGraph_.end()) {
        outGraph_[op] = std::unordered_set<Operation *>();
    }
}

void DependencyManager::Clear() {
    inGraph_.clear();
    outGraph_.clear();
}

void DependencyManager::ClearDependencies() {
    for (auto &[op, preds] : inGraph_) {
        preds.clear();
    }
    for (auto &[op, succs] : outGraph_) {
        succs.clear();
    }
}

bool DependencyManager::IsOpAlloc(Operation *op) {
    if (op == nullptr) {
        return false;
    }
    return op->GetOpcodeStr().find("ALLOC") != std::string::npos;
}

void DependencyManager::AddDependency(Operation *preOp, Operation *postOp) {
    if (preOp == nullptr || postOp == nullptr) {
        return;
    }
    if (!IsOpAlloc(preOp) && !IsOpAlloc(postOp)) {
        outGraph_[preOp].insert(postOp);
        inGraph_[postOp].insert(preOp);
    }
}

void DependencyManager::AddAllocDependency(Operation *preOp, Operation *postOp) {
    if (preOp == nullptr || postOp == nullptr) {
        return;
    }
    outGraph_[preOp].insert(postOp);
    inGraph_[postOp].insert(preOp);
}

bool DependencyManager::RemoveDependency(Operation *preOp, Operation *postOp) {
    if (preOp == nullptr || postOp == nullptr) {
        return false;
    }
    bool removedFromSucc = false;
    bool removedFromPred = false;

    if (outGraph_.find(preOp) != outGraph_.end()) {
        removedFromSucc = outGraph_[preOp].erase(postOp) > 0;
    }
    if (inGraph_.find(postOp) != inGraph_.end()) {
        removedFromPred = inGraph_[postOp].erase(preOp) > 0;
    }

    return removedFromSucc && removedFromPred;
}

std::unordered_set<Operation *> &DependencyManager::GetSuccessors(Operation *op) {
    return outGraph_[op];
}

std::unordered_set<Operation *> &DependencyManager::GetPredecessors(Operation *op) {
    return inGraph_[op];
}

bool DependencyManager::HasOp(Operation *op) const {
    return inGraph_.find(op) != inGraph_.end();
}

Status DependencyManager::TransferSuccessorsByMemId(Operation *opA, Operation *opB, int memId,
    const std::function<bool(Operation *)> &isRetired,
    const std::function<const std::vector<int> &(Operation *)> &getReqMemIds) {
    if (opA == nullptr || opB == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "TransferSuccessorsByMemId: opA or opB is nullptr");
        return FAILED;
    }

    auto &successors = outGraph_[opA];
    std::vector<Operation *> toTransfer;

    for (auto succOp : successors) {
        if (!isRetired(succOp)) {
            const auto &reqMemIds = getReqMemIds(succOp);
            if (std::find(reqMemIds.begin(), reqMemIds.end(), memId) != reqMemIds.end()) {
                toTransfer.push_back(succOp);
            }
        }
    }

    for (auto succOp : toTransfer) {
        outGraph_[opA].erase(succOp);
        if (inGraph_[succOp].erase(opA) == 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "TransferSuccessorsByMemId: erase opA from succOp failed");
            return FAILED;
        }
        inGraph_[succOp].insert(opB);
        outGraph_[opB].insert(succOp);
    }

    return SUCCESS;
}

void DependencyManager::ReplaceAllocPredecessor(const std::vector<Operation *> &ops, int memId, Operation *newPre,
    const std::function<bool(Operation *)> &isRetired,
    const std::function<const std::vector<int> &(Operation *)> &getReqMemIds) {
    if (newPre == nullptr) {
        return;
    }

    for (auto op : ops) {
        if (isRetired(op) || IsOpAlloc(op)) {
            continue;
        }
        auto predecessors = inGraph_[op];
        for (auto predOp : predecessors) {
            if (IsOpAlloc(predOp)) {
                const auto &predReqMemIds = getReqMemIds(predOp);
                if (std::find(predReqMemIds.begin(), predReqMemIds.end(), memId) != predReqMemIds.end()) {
                    inGraph_[op].erase(predOp);
                    inGraph_[op].insert(newPre);
                }
            }
        }
    }
}

std::string DependencyManager::PrintOp(Operation *op) {
    return op->GetOpcodeStr() + "[" + std::to_string(op->GetOpMagic()) + "]";
}

void DependencyManager::PrintDependencies(const std::vector<Operation *> &ops) {
    if (static_cast<int>(LoggerManager::GetManager().level) > static_cast<int>(LoggerLevel::DEBUG)) {
        return;
    }
    for (const auto &op : ops) {
        if (inGraph_.find(op) == inGraph_.end() || outGraph_.find(op) == outGraph_.end()) {
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "%s", PrintOp(op).c_str());
        for (const auto &preOp : inGraph_.at(op)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Predecessors:");
            APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", PrintOp(preOp).c_str());
        }
        for (const auto &succOp : outGraph_.at(op)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Successors:");
            APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", PrintOp(succOp).c_str());
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "\n");
    }
}

Operation *DependencyManager::SkipViewChain(Operation *start, bool followProducers) {
    if (start == nullptr)
        return nullptr;
    Operation *op = start;
    Operation *lastView = nullptr;
    while (op != nullptr && IsViewOp(*op)) {
        lastView = op;
        if (followProducers) {
            const auto &nextOps = op->GetInputOperand(0)->GetProducers();
            if (nextOps.size() != 1)
                break;
            op = *nextOps.begin();
        } else {
            const auto &nextOps = op->GetOutputOperand(0)->GetConsumers();
            if (nextOps.size() != 1)
                break;
            op = *nextOps.begin();
        }
    }
    return lastView;
}

Status DependencyManager::InitAllocDependencies(Operation* op, std::unordered_map<int, Operation*> &tensor2AllocOpMap) {
    for (auto &tensor : op->GetOOperands()) {
        int memId = tensor->memoryrange.memId;
        if (tensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
            if (tensor2AllocOpMap.find(memId) == tensor2AllocOpMap.end()) {
                APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d] must have alloc. magic: %d, op: %s", memId, tensor->GetMagic(), PrintOp(op).c_str());
                return FAILED;
            }
            AddAllocDependency(tensor2AllocOpMap[memId], op);
        }
    }
    return SUCCESS;
}

void DependencyManager::FindDependencies(Operation *op) {
    if (op == nullptr)
        return;

    if (op->GetOpcode() == Opcode::OP_L1_TO_L0A_SCALE) {
        auto matmulOp = *(op->GetOOperands()[0])->GetConsumers().begin();
        for (auto &input : matmulOp->GetIOperands()) {
            if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
                auto prodOp = *input->GetProducers().begin();
                AddDependency(prodOp, op);
            }
        }
    }
    if (op->GetOpcode() == Opcode::OP_L1_TO_L0B_SCALE) {
        auto matmulOp = *(op->GetOOperands()[0])->GetConsumers().begin();
        for (auto &input : matmulOp->GetIOperands()) {
            if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
                auto prodOp = *input->GetProducers().begin();
                AddDependency(prodOp, op);
            }
        }
    }

    for (auto &producer : op->ProducerOps()) {
        if (IsViewOp(*producer)) {
            for (auto viewProducer : producer->ProducerOps()) {
                Operation *lastView = SkipViewChain(viewProducer, true);
                Operation *realProd = (lastView != nullptr) ? *lastView->ProducerOps().begin() : viewProducer;
                AddDependency(realProd, op);
            }
        } else {
            AddDependency(producer, op);
        }
    }

    for (auto &consumer : op->ConsumerOps()) {
        if (IsViewOp(*consumer)) {
            for (auto viewConsumer : consumer->ConsumerOps()) {
                Operation *lastView = SkipViewChain(viewConsumer, false);
                Operation *realCon = (lastView != nullptr) ? *lastView->ConsumerOps().begin() : viewConsumer;
                AddDependency(op, realCon);
            }
        } else {
            AddDependency(op, consumer);
        }
    }
}

Status DependencyManager::InitDependencies(const std::vector<Operation *> &ops) {
    std::unordered_map<int, Operation *> tensor2AllocOpMap;

    for (const auto &op : ops) {
        if (IsOpAlloc(op)) {
            if (op->GetOOperands().size() != 1) {
                APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] oOperand must be 1.", op->GetOpMagic());
                return FAILED;
            }
            int memId = op->GetOutputOperand(0)->memoryrange.memId;
            tensor2AllocOpMap[memId] = op;
        }
    }

    Clear();
    for (const auto &op : ops) {
        RegisterOp(op);
    }

    for (const auto &op : ops) {
        if (!IsOpAlloc(op)) {
            FindDependencies(op);
            if (InitAllocDependencies(op, tensor2AllocOpMap) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "InitAllocDependencies failed.");
                return FAILED;
            }
        }
    }

    return SUCCESS;
}

} // namespace npu::tile_fwk