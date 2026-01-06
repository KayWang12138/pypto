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
 * \file mix_subgraph_split_utils.cpp
 * \brief
 */

#include "passes/block_graph_pass/mix_subgraph_split_utils.h"

namespace npu {
namespace tile_fwk {
static void MixSubgraphSplitUtils::BroadcastDependencyClosure(std::set<int> &deps_i, std::set<int> &newDeps, std::unordered_map<int, std::set<int>> &closure, bool &changed, int i) {
    for (int j : deps_i) {
        // 如果j有传递依赖k，把k也加入i的依赖
        if (closure.count(j)) {
            for (int k : closure[j]) {
                if (newDeps.insert(k).second) {
                    changed = true;
                    ALOG_DEBUG_F("Iteration: %d -> %d -> %d, added %d -> %d",
                                i, j, k, i, k);
                }
            }
        }
    }
}

static void MixSubgraphSplitUtils::InitiateClosure(const std::unordered_map<int, std::vector<int>>& directDeps,
                                    std::unordered_map<int, std::set<int>> &closure) {
    // 确保所有组件都在closure中，即使没有出边
    int maxComponent = 0;
    for (const auto& [component, deps] : directDeps) {
        closure[component] = std::set<int>(deps.begin(), deps.end());
        if (component > maxComponent) {
            maxComponent = component;
        }
        for (int dep : deps) {
            if (dep > maxComponent) {
                maxComponent = dep;
            }
        }
    }
    // 确保所有组件索引都在closure中
    for (int i = 0; i <= maxComponent; i++) {
        closure[i]; // 确保存在，即使没有依赖关系
    } 
}

static void MixSubgraphSplitUtils::CalculateClosure(std::unordered_map<int, std::set<int>> &closure) {
    bool changed;
    int iteration = 0;
    do {
        changed = false;
        iteration++;
        for (auto& [i, deps_i] : closure) {
            std::set<int> newDeps = deps_i;
            // 对于i的每个直接依赖j
            BroadcastDependencyClosure(deps_i, newDeps, closure, changed, i);
            deps_i = std::move(newDeps);
        }
        ALOG_DEBUG_F("After iteration %d, changed: %s", iteration, changed ? "true" : "false");
    } while (changed);
}

static void MixSubgraphSplitUtils::UpdateOperandsForIncast(const std::vector<IncastParamPackTy> &incastParamList,
                                                            const LogicalTensors &originalIncasts, 
                                                            const LogicalTensors &originalIOperands, 
                                                            LogicalTensors &newIOperands, 
                                                            std::set<LogicalTensorPtr> &processedTensors) {
    for (const auto& param : incastParamList) {
        int tensorMagic = incastParam.tensor->magic;
        int originalIndex = FindTensorIndexInList(tensorMagic, originalIncasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalIOperands.size())) {
            newIOperands.push_back(originalIOperands[originalIndex]);
            processedTensors.insert(param.tensor);
            ALOG_DEBUG_F("  Found: tensor magic=%d -> original Operand[%d] (tensor magic=%d)",
                                tensorMagic, originalIndex, originalIOperands[originalIndex]->magic);
        } 
    }
}

static void MixSubgraphSplitUtils::UpdateOperandsForOutcast(const std::vector<OutcastParamPackTy> &outcastParamList,
                                                            const LogicalTensors &originalOutcasts, 
                                                            const LogicalTensors &originalOOperands, 
                                                            LogicalTensors &newOOperands, 
                                                            std::set<LogicalTensorPtr> &processedTensors) {
    for (const auto& param : outcastParamList) {
        int tensorMagic = incastParam.tensor->magic;
        int originalIndex = FindTensorIndexInList(tensorMagic, originalOutcasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalOOperands.size())) {
            newOOperands.push_back(originalOperands[originalIndex]);
            processedTensors.insert(param.tensor);
            ALOG_DEBUG_F("  Found: tensor magic=%d -> original Operand[%d] (tensor magic=%d)",
                                tensorMagic, originalIndex, originalOOperands[originalIndex]->magic);
        } 
    }
}

static void MixSubgraphSplitUtils::UpdateOperandsForGlobalTensor(const std::vector<TensorParamPackTy> &paramList,
                                                                const LogicalTensors &originalTensors, 
                                                                const LogicalTensors &originalOperands, 
                                                                LogicalTensors &newOperands, 
                                                                std::set<LogicalTensorPtr> &processedTensors) {
    for (const auto& tensorParam : paramList) {
        if (tensorParam.opMagic == -1 || tensorParam.tensor == nullptr || tensorParam.isOutputToGM) {
            continue;
        }
        int tensorMagic = tensorParam.tensor->magic;
        int originalIndex = FindTensorIndexInList(tensorMagic, originalTensors);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalOperands.size())) {
            newOperands.push_back(originalOperands[originalIndex]);
            processedTensors.insert(tensorParam.tensor);
            ALOG_DEBUG_F("  Found: global tensor magic=%d -> original Operand[%d]",
                            tensorMagic, originalIndex);
        } 
    }
}

static void MixSubgraphSplitUtils::UpdateBroadcastForInOutCast(const LogicalTensors &actualTensors, 
                                                                const LogicalTensors &originalTensors, 
                                                                const LogicalTensors &originalOperands, 
                                                                LogicalTensors &newOperands, 
                                                                std::set<LogicalTensorPtr> &processedTensors) {
    // 处理传播的tensor
    for (const auto& tensor : actualTensors) {       
        // 检查是否已经在之前的列表中处理过
        if (processedTensors.count(tensor) > 0) {
            ALOG_DEBUG_F("  Propagated incast tensor magic=%d already processed, skipping", incast->magic);
            continue;
        }
        int tensorMagic = tensor->magic;
        ALOG_DEBUG_F("  Checking propagated incast tensor magic=%d", tensorMagic);
        int originalIndex = FindTensorIndexInList(tensorMagic, originalTensors);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalOperands.size())) {
            newOperands.push_back(originalOperands[originalIndex]);
            processedTensors.insert(tensor);
            ALOG_DEBUG_F("    Found: propagated incast tensor magic=%d -> original iOperand[%d]",
                            tensorMagic, originalIndex);
        } 
    }
}

static Operation* MixSubgraphSplitUtils::FindFirstOpForward(Operation* startOp, Function& mixSubgraphFunc, std::function<bool(Operation*)> predicate) {
    const auto& opList = mixSubgraphFunc.Operations(false).DuplicatedOpList();
    int startIndex = GetStartIndex(opList, startOp);
    if (startIndex == -1) {
        return nullptr;
    }

    // 向后搜索（向序列结束方向）
    for (int i = startIndex + 1; i < static_cast<int>(opList.size()); ++i) {
        Operation* candidate = opList[i];
        if (predicate(candidate)) {
            ALOG_DEBUG_F("Found target op %d at index %d (searching forward from %d)", candidate->GetOpMagic(), i, startIndex);
            return candidate;
        }
    }

    ALOG_DEBUG_F("No matching op found for op %d in forward direction", startOp->GetOpMagic());
    return nullptr;
}

static Operation* MixSubgraphSplitUtils::FindFirstOpBackward(Operation* startOp, Function& mixSubgraphFunc, std::function<bool(Operation*)> predicate) {
    const auto& opList = mixSubgraphFunc.Operations(false).DuplicatedOpList();
    int startIndex = GetStartIndex(opList, startOp);
    if (startIndex == -1) {
        return nullptr;
    }

    // 向前搜索（向序列开始方向）
    for (int i = startIndex - 1; i >= 0; --i) {
        Operation* candidate = opList[i];
        if (predicate(candidate)) {
            ALOG_DEBUG_F("Found target op %d at index %d (searching backward from %d)", candidate->GetOpMagic(), i, startIndex);
            return candidate;
        }
    }

    ALOG_DEBUG_F("No matching op found for op %d in backward direction", startOp->GetOpMagic());
    return nullptr;
}

static bool MixSubgraphSplitUtils::IsSyncOperation(Operation* op) {
    if (!op) {
        return false;
    }

    Opcode opcode = op->GetOpcode();

    // 同步操作类型列表
    return opcode == Opcode::OP_SYNC_SRC ||
           opcode == Opcode::OP_SYNC_DST ||
           opcode == Opcode::OP_CV_SYNC_SRC ||
           opcode == Opcode::OP_CV_SYNC_DST ||
           opcode == Opcode::OP_PHASE1 ||
           opcode == Opcode::OP_PHASE2 ||
           opcode == Opcode::OP_BAR_V ||
           opcode == Opcode::OP_BAR_M ||
           opcode == Opcode::OP_BAR_ALL;
}

static bool MixSubgraphSplitUtils::IsMixSubgraph(Function& leafFunc) {
    auto operations = function.Operations(false);
    for (size_t idx = 0; idx < operations.size(); idx++) {
        auto& op = operations[idx];
        if (op.IsNOP()) continue;
        // 只要有一个op有有效的internalSubgraphID，就认为是Mix子图
        int internalSubgraphID = op.GetInternalSubgraphID();
        if (internalSubgraphID > 0) {
            ALOG_DEBUG_F("Function %s identified as mix subgraph: op %s has internalSubgraphID=%d",
                        function.GetRawName().c_str(),
                        op.GetOpcodeStr().c_str(),
                        internalSubgraphID);
            return true;
        }
    }
    ALOG_DEBUG_F("Function %s is not a mix subgraph: no ops with internalSubgraphID",
            function.GetRawName().c_str());
    return false;
}
}
}