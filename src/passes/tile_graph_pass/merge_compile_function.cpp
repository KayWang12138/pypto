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
 * \file merge_compile_function.cpp
 * \brief
 */

#include "merge_compile_function.h"

#include <queue>
#include "interface/tensor/raw_tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "allocator.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
Status MergeCompileFunction::RunOnFunction(Function &function) {
    // init pass
    ALOG_INFO("***** OptimizeStaticLeaf ", function.GetMagicName(), " *****: Begin");
    CheckLoop(function);
    Init(function);
    CutIslands(function);
    ALOG_INFO("***** OptimizeStaticLeaf ", function.GetMagicName(), " *****: End");
    return SUCCESS;
}

// 1. 检查是有有循环，如果有环即跳出
void MergeCompileFunction::CheckLoop(Function &function) const {
    std::map<int, std::vector<Operation *>> subgraphs;
    std::map<int, int> opToSubgraph;
    auto subGraphInCycle = function.LoopCheck();
    if (!subGraphInCycle.empty()) {
        ASSERT(false);
    }
}

// 2. 初始化,将所有op的subgraphID初始化为0， 同时将ioperand的memory设置为ddr，同时记录到ddrTensorOutDegree中
void MergeCompileFunction::Init(Function &function) {
    // Iterate through all operations
    subGraphID_ = NOT_IN_SUBGRAPH;
    auto operationsViewer = function.Operations();
    for (size_t i = 0; i < operationsViewer.size(); i++) {
        auto &op = operationsViewer[i];
        operationIndex_[&op] = i;
        if (config::UseTIG()) {
            ASSERT(op.GetSubgraphID() == NOT_IN_SUBGRAPH);
        } else {
            subGraphID_ = std::max(subGraphID_, op.GetSubgraphID());
        }
        for (auto &&operand : op.GetIOperands()) {
            if (operand->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            ddrTensorOutDegree_[operand]++;
        }
    }
    ALOG_ERROR("subGraphID: ", subGraphID_);
    subGraphID_++;
}

// 3. 进行切分，将所有的切分子图进行排序，并进行分配
void MergeCompileFunction::CutIslands(Function &function) {
    TiFWKJointAllocator ddrAllocator{ModelID::ASCEND_910B};
    auto operationsViewer = function.Operations();
    for (size_t i = 0; i < operationsViewer.size(); i++) {
        auto &op = operationsViewer[i];
        if (op.GetOpcode() == Opcode::OP_NOP) {
            continue;
        }
        if (op.GetSubgraphID() > 0) { // already allocated
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW || op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            continue;
        }
        if (TryAllocate(function, op, subGraphID_, ddrAllocator)) {
            subGraphID_++;
        }
    }
    CheckLoop(function);
    ALOG_INFO("subGraphID=", subGraphID_);
    // Update partition number to function for building graph
    function.SetTotalSubGraphCount(subGraphID_);
}
// 4. 尝试分配，如果分配成功，返回true，否则返回false
bool MergeCompileFunction::TryAllocate(Function &function, Operation &src, int subGraphID, TiFWKJointAllocator &ddrAllocator) {
    TiFWKJointAllocator jalloc(ModelID::ASCEND_910B);
    auto cmp = [this](Operation * const a, Operation * const b) { return operationIndex_.find(a)->second > operationIndex_.find(b)->second; };
    std::priority_queue<Operation *, std::vector<Operation *>, decltype(cmp)> q(cmp);
    std::unordered_set<Operation *> visited;
    std::set<int> result;
    q.emplace(&src);
    visited.emplace(&src);
    while (!q.empty()) {
        auto cur = q.top();
        q.pop();
        if (cur->GetSubgraphID() != NOT_IN_SUBGRAPH) {
            continue;
        }
        auto needAppendOperationsIdx = FindOperationsToAppend(function, *cur);
        if (!CheckNewOperationsValid(function, needAppendOperationsIdx, subGraphID)) {
            continue;
        }
        auto solution = result;
        solution.insert(needAppendOperationsIdx.begin(), needAppendOperationsIdx.end());
        if (!TryAllocateForOneSolution(function, solution, subGraphID)) {
            continue;
        }
        result = solution;
        auto operationsViewer = function.Operations();
        for (auto opIndex : needAppendOperationsIdx) {
            auto &op = operationsViewer[opIndex];
            for (auto &&oOperand : op.GetOOperands()) {
                if (oOperand->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR) {
                    continue;
                }
                for (const auto &consumer : oOperand->GetConsumers()) {
                    if (consumer->GetSubgraphID() != NOT_IN_SUBGRAPH) {
                        continue;
                    }
                    if (visited.count(consumer) == 1) {
                        continue;
                    }
                    visited.emplace(consumer);
                    q.emplace(consumer);
                }
            }
        }
    }
    if (result.empty()) {
        return false;
    }
    ASSERT(TryAllocateForOneSolution(function, result, subGraphID));
    // 为DDR tensor分配内存
    std::vector<std::shared_ptr<LogicalTensor>> toAllocTensors;
    std::unordered_set<std::shared_ptr<LogicalTensor>> toAllocTensorsSet;
    auto operationsViewer = function.Operations();
    for (auto opIndex : result) {
        auto &op = operationsViewer[opIndex];
        for (auto &&operand : op.GetIOperands()) {
            if (operand->GetMemoryTypeToBe() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            if (toAllocTensorsSet.count(operand) == 0) {
                toAllocTensorsSet.insert(operand);
                toAllocTensors.push_back(operand);
            }
        }
        for (auto &&operand : op.GetOOperands()) {
            if (operand->GetMemoryTypeToBe() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            if (toAllocTensorsSet.count(operand) == 0) {
                toAllocTensorsSet.insert(operand);
                toAllocTensors.push_back(operand);
            }
        }
    }
    auto tryToAlloc = [&ddrAllocator](std::vector<std::shared_ptr<LogicalTensor>> &tensors) {
        std::shared_ptr<LogicalTensor> _;
        return ddrAllocator.CanAllocate(tensors, 0, _);
    };
    ASSERT(tryToAlloc(toAllocTensors));
    for (auto opIndex : result) {
        auto &op = operationsViewer[opIndex];
        for (auto &&iOperand : op.GetIOperands()) {
            if (iOperand->GetMemoryTypeToBe() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            if (--ddrTensorOutDegree_[iOperand] == 0) {
                ddrAllocator.Free(false, iOperand, 0);
            }
        }
    }
    return true;
}

bool MergeCompileFunction::CheckNewOperationsValid(Function &function, const std::set<int> &newOperations, int subGraphID) {
    std::queue<Operation *> q;
    std::unordered_set<Operation *> visited;
    auto operationsViewer = function.Operations();
    for (auto opIndex : newOperations) {
        auto &op = operationsViewer[opIndex];
        for (auto &&operand : op.GetIOperands()) {
            if (operand->GetMemoryTypeToBe() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            // DDR是增加operation的边界，目前分析来看DDR之前不能再属于当前subGraphID
            for (auto &producer : operand->GetProducers()) {
                ASSERT(producer->GetSubgraphID() != subGraphID);
                if (newOperations.count(operationIndex_[producer]) > 0) {
                    continue;
                }
                if (visited.count(producer) > 0) {
                    continue;
                }
                visited.emplace(producer);
                q.emplace(producer);
            }
        }
    }
    while (!q.empty()) {
        auto &cur = *q.front();
        q.pop();
        for (auto &&operand : cur.GetIOperands()) {
            for (auto &producer : operand->GetProducers()) {
                if (visited.count(producer) > 0) {
                    continue;
                }
                if (producer->GetSubgraphID() == subGraphID) {
                    return false;
                }
                visited.emplace(producer);
                q.emplace(producer);
            }
        }
    }
    return true;
}

std::unordered_map<std::shared_ptr<LogicalTensor>, int> MergeCompileFunction::GetTensorOutDegreeByOperations(
    Function &function, const std::set<int> &solution) const {
    std::unordered_map<std::shared_ptr<LogicalTensor>, int> tensorOutDegree;
    auto operationsViewer = function.Operations();
    for (auto opIndex : solution) {
        auto &op = operationsViewer[opIndex];
        for (auto &iOperand : op.GetIOperands()) {
            tensorOutDegree[iOperand]++;
        }
    }
    return tensorOutDegree;
}

bool MergeCompileFunction::TryAllocateForOneSolution(Function &function, const std::set<int> &solution, int subGraphID) const {
    TiFWKJointAllocator jalloc(ModelID::ASCEND_910B);
    auto tryToAlloc = [&jalloc, &subGraphID](std::vector<std::shared_ptr<LogicalTensor>> &tensors) {
        std::shared_ptr<LogicalTensor> _;
        return jalloc.CanAllocate(tensors, subGraphID, _);
    };
    std::vector<std::shared_ptr<LogicalTensor>> toAllocTensors;
    std::unordered_set<std::shared_ptr<LogicalTensor>> toAllocTensorsSet;
    std::unordered_set<std::shared_ptr<LogicalTensor>> allocatedTensorsSet;
    auto tensorOutDegree = GetTensorOutDegreeByOperations(function, solution);
    bool succeed = true;
    auto operationsViewer = function.Operations();
    for (auto opIndex : solution) {
        auto &op = operationsViewer[opIndex];
        op.UpdateSubgraphID(NOT_IN_SUBGRAPH);
        for (auto &&operand : op.GetIOperands()) {
            if (operand->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            operand->subGraphID = NOT_IN_SUBGRAPH;
            operand->memorymap.erase(subGraphID);
        }
        for (auto &&operand : op.GetOOperands()) {
            if (operand->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            operand->subGraphID = NOT_IN_SUBGRAPH;
            operand->memorymap.erase(subGraphID);
        }
    }
    for (auto opIndex : solution) {
        auto &op = operationsViewer[opIndex];
        for (auto &&operand : op.GetIOperands()) {
            if (operand->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            if (toAllocTensorsSet.count(operand) == 0) {
                toAllocTensorsSet.insert(operand);
                toAllocTensors.push_back(operand);
            }
        }
        for (auto &&operand : op.GetOOperands()) {
            if (operand->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            if (toAllocTensorsSet.count(operand) == 0) {
                toAllocTensorsSet.insert(operand);
                toAllocTensors.push_back(operand);
            }
        }
        allocatedTensorsSet.insert(toAllocTensorsSet.begin(), toAllocTensorsSet.end());
        if (!tryToAlloc(toAllocTensors)) {
            ALOG_DEBUG("Cann't Alloc for Operation ", op.Dump());
            succeed = false;
            break;
        }
        for (auto &&iOperand : op.GetIOperands()) {
            if (iOperand->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            iOperand->subGraphID = subGraphID;
            ASSERT(tensorOutDegree[iOperand] > 0);
            if (--tensorOutDegree[iOperand] == 0) {
                jalloc.Free(false, iOperand, subGraphID);
            }
        }
        for (auto &&oOperand : op.GetOOperands()) {
            if (oOperand->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            oOperand->subGraphID = subGraphID;
        }
        op.UpdateSubgraphID(subGraphID);
    }
    if (!succeed) {
        for (auto opIndex : solution) {
            auto &op = operationsViewer[opIndex];
            op.UpdateSubgraphID(NOT_IN_SUBGRAPH);
        }
        for (const auto &tensor : allocatedTensorsSet) {
            tensor->subGraphID = NOT_IN_SUBGRAPH;
            tensor->memorymap.erase(subGraphID);
        }
        for (auto opIndex : solution) {
            auto &op = operationsViewer[opIndex];
            ASSERT(op.GetSubgraphID() == NOT_IN_SUBGRAPH);
            for (auto &&operand : op.GetIOperands()) {
                ASSERT(operand->subGraphID == NOT_IN_SUBGRAPH);
                ASSERT(operand->memorymap.count(subGraphID) == 0);
            }
            for (auto &&operand : op.GetOOperands()) {
                ASSERT(operand->subGraphID == NOT_IN_SUBGRAPH);
                ASSERT(operand->memorymap.count(subGraphID) == 0);
            }
        }
    }
    return succeed;
}

std::set<int> MergeCompileFunction::FindOperationsToAppend(Function &function, Operation &src) {
    std::set<int> needAppendOperationsIdx;
    std::unordered_set<Operation *> inserted;
    std::queue<Operation *> q;
    if (src.GroupID() == NON_GROUP) {
        q.emplace(&src);
        inserted.emplace(&src);
    } else {
        for (const auto op : function.GetGroupByID(src.GroupID())) {
            q.emplace(op);
            inserted.emplace(op);
        }
    }
    while (!q.empty()) {
        auto op = q.front();
        q.pop();
        ASSERT(operationIndex_.count(op) > 0);
        ASSERT(op->GetSubgraphID() == NOT_IN_SUBGRAPH);
        needAppendOperationsIdx.emplace(operationIndex_[op]);
        for (const auto &iOperand : op->GetIOperands()) {
            if (iOperand->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            for (auto &producer : iOperand->GetProducers()) {
                if (inserted.count(producer) == 1) {
                    continue;
                }
                if (producer->GetSubgraphID() != NOT_IN_SUBGRAPH) {
                    continue;
                }
                inserted.emplace(producer);
                q.emplace(producer);
            }
        }
    }
    return needAppendOperationsIdx;
}
} // namespace
