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
 * \file mix_dependency_analyzer.CPP
 * \brief
 */

#include "passes/pass_utils/pass_utils.h"
#include "passes/block_graph_pass/mix_subgraph_split/mix_dependency_analyzer.h"

namespace npu {
namespace tile_fwk {
std::unordered_map<int, std::set<int>> MixDependencyAnalyzer::AnalyzeComponentDependencies(Function &mixFunc,
    std::map<std::pair<int, int>, std::vector<LogicalTensorPtr>>& crossComponentTensors) {
    std::unordered_map<int, std::set<int>> dependencies;
    // 分析子图的所有的op
    for (auto &op : mixFunc.Operations(false)) {
        if (op.IsNOP()) {
            continue;
        }
        int producerInternalID = op.GetInternalSubgraphID();
        for (size_t k = 0; k < op.GetOOperands().size(); k++) {
            auto oOperand = op.GetOOperands()[k];
            if (oOperand == nullptr) {
                continue;
            }
            // 分析该op的输出tensor的消费者
            auto consumers = oOperand->GetConsumers();
            for (auto* consumer : consumers) {
                if (consumer == nullptr) {
                    continue;
                }
                // 要求消费者也在同一个subgraph中
                if (consumer->GetSubgraphID() != op.GetSubgraphID()) {
                    continue;
                }
                // 记录进一步切分后的依赖关系
                int consumerID = consumer->GetInternalSubgraphID();
                if (producerInternalID != consumerID) {
                    dependencies[producerInternalID].insert(consumerID);
                    std::pair<int, int> edge = {producerInternalID, consumerID};
                    crossComponentTensors[edge].push_back(oOperand);
                    ALOG_DEBUG_F("Recorded cross-component tensor: raw=%d, magic=%d, %d->%d",
                            oOperand->GetRawMagic(), oOperand->magic,
                            producerInternalID, consumerID);
                }
            }
        }
    }dependencies
    // 用于记录mixsplit间的依赖关系
    return dependencies;
}

void MixDependencyAnalyzer::InitDependencies(std::unordered_map<int, std::set<int>> &dependencies) {
    // 记录最大的mixsplit id
    maxComponent = 0;
    for (const auto &pair : dependencies) {
        if (pair.first > maxComponent) {
            maxComponent = pair.first;
        }
        for (int dep : pair.second) {
            if (dep > maxComponent) {
                maxComponent = dep;
            }
        }
    }
    // 确保所有组件索引都在closure中
    for (int i = 0; i <= maxComponent; i++) {
        dependencies[i]; // 确保存在，即使没有依赖关系
    }
}

void MixDependencyAnalyzer::WarshallAlgorithm(std::vector<std::vector<bool>> &matrix) {
    size_t n = matrix.size();
    for (size_t k = 0; k < n; ++k) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                matrix[i][j] = matrix[i][j] || (matrix[i][k] && matrix[k][j]);
            }
        }
    }
}

void MixDependencyAnalyzer::UpdateDependencies(std::unordered_map<int, std::set<int>> &dependencies) {
    size_t n = dependencies.size();
    std::vector<std::vector<bool>> matrix(n, std::vector<bool>(n, false));
    for (const auto& pair : dependencies) {
        int fromId = pair.first;
        for (int toId : pair.second) {
            matrix[fromId][toId] = true;
        }
    }
    WarshallAlgorithm(matrix);
    for (size_t fromId = 0; fromId < n; ++fromId) {
        for (size_t toId = 0; toId < n; ++toId) {
            if (matrix[fromId][toId]) {
                dependencies[fromId].insert(toId);
            }
        }
    }
}

void MixDependencyAnalyzer::ComputeDependencyClosure(std::unordered_map<int, std::set<int>> &dependencies) {
    // 步骤1：初始化直接依赖
    InitDependencies(dependencies);
    // 步骤2：使用Warshall算法将邻接矩阵转换为可达矩阵，时间复杂度为O(n^3)，空间复杂度为O(n^2)
    UpdateDependencies(dependencies);
}

void MixDependencyAnalyzer::ExtractExternalDependencies() {
    if (!leafFunction_) {
        ALOG_ERROR_F("originalMixFunc_ is null, cannot extract external dependencies");
        return;
    }
    ALOG_INFO_F("Extracting external dependencies from originalMixFunc: %s", 
            leafFunction_->GetName().c_str());
    // 从 leafFunction 提取 incasts - 这些 incast 应该分配给所有scope
    const auto& incastTensors = leafFunction_->GetIncast();
    ALOG_INFO_F("Original mix function has %zu incast tensors", incastTensors.size());
    // 为每个scope分配相同的 incast tensors
    for (int compId = 0; compId <= maxComponent; compId++) {
        for (const auto& tensor : incastTensors) {
            allIncasts[compId].push_back(tensor);
            ALOG_DEBUG_F("Assigned incast tensor %d to component %d", 
                    tensor->GetRawMagic(), compId);
        }
    }
    // 从 leafFunction 获取 outcasts - 这些 outcast 应该分配给所有scope
    const auto& outcastTensors = leafFunction_->GetOutcast();
    ALOG_INFO_F("Original mix function has %zu outcast tensors", outcastTensors.size());
    
    // 为每个scope分配相同的 outcast tensors    
    for (int compId = 0; compId <= maxComponent; compId++) {
        for (const auto& tensor : outcastTensors) {
            allOutcasts[compId].push_back(tensor);
            ALOG_DEBUG_F("Assigned outcast tensor %d to component %d", 
                        tensor->GetRawMagic(), compId);
        }
    }
    
    ALOG_INFO_F("Extracted external dependencies: %zu incasts per component, %zu outcasts per component, total components=%d", 
                incastTensors.size(), outcastTensors.size(), maxComponent + 1);
}

void MixDependencyAnalyzer::CollectInternalDependencies(const std::unordered_map<int, std::set<int>> &dependencyClosure,
                                                        const std::vector<InternalComponentInfo> &components) {
    // 遍历传递闭包中的每个依赖关系
    for (const auto& [srcComp, dstComps] : dependencyClosure) {
        ComponentType srcType = components[srcComp].componentType;
        if (srcType == ComponentType::UNKNOWN) {
            continue;
        }
        for (int dstComp : dstComps) {
            // 跳过自依赖（scope依赖自己）
            if (srcComp == dstComp) {
                ALOG_DEBUG_F("Skip self-dependency: component %d -> %d", srcComp, dstComp);
                continue;    
            }
            ComponentType dstType = components[dstComp].componentType;
            // 只添加同类型scope间的依赖（C-C、V-V）   
            if (srcType == dstType && dstType != ComponentType::UNKNOWN) {
                // 添加这两个组件间的tensor依赖
                InternalDependencyInfo depInfo(srcComp, dstComp, srcType);
                internalDeps.push_back(depInfo);
                ALOG_DEBUG_F("Added internal dependency: component %d (%s) -> component %d (%s)",
                           srcComp, srcType == ComponentType::C_SCOPE ? "C" : "V",
                           dstComp, dstType == ComponentType::C_SCOPE ? "C" : "V");
            }
        }
    } 
    ALOG_INFO_F("Collected %zu internal dependencies between same-type components", 
               internalDeps.size());
}

void MixDependencyAnalyzer::ObtainMinAdjMatrix(std::vector<std::vector<bool>> &matrix) {
    size_t n = matrix.size();
    std::vector<std::vector<bool>> matrixCopy = matrix;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                matrix[i][j] = false;
                continue;
            }
            bool isRedundant = false;
            for (size_t k = 0; k < n; ++k) {
                if (k != i && k != j && matrixCopy[i][k] && matrixCopy[k][j]) {
                    isRedundant = true;
                    break;
                }
            }
            matrix[i][j] = isRedundant ? false : true;
        }
    }
}

std::vector<std::vector<bool>> MixDependencyAnalyzer::Transpose(const std::vector<std::vector<bool>> &matrix) {
    size_t n = matrix.size();
    std::vector<std::vector<bool>> ret(n, std::vector<bool>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            ret[j][i] = matrix[i][j];
        }
    }
    return ret;
}

Status MixDependencyAnalyzer::ProcessDependencyAnalyzer(const AnalyzerInput &input, AnalyzerOutput &output) {
    Reset();
    // 设置原始MixFunc
    SetOriginalMixFunc(input.originalMixFunc);
    // 分析组件间直接依赖（scope与scope之间的依赖）
    ALOG_INFO_F("Analyzing inter-component dependencies and recording cross-component tensors...");
    std::map<std::pair<int, int>, std::vector<LogicalTensorPtr>> crossComponentTensors;
    auto directDeps = AnalyzeComponentDependencies(*input.originalMixFunc, crossComponentTensors);
    ALOG_INFO_F("Found %zu cross-component tensor dependencies:", crossComponentTensors.size());
    for (const auto& [edge, tensors] : crossComponentTensors) {
        ALOG_INFO_F("  Component %d -> %d: %zu tensor(s)", 
                   edge.first, edge.second, tensors.size());
    }   
    // 计算依赖传递闭包
    ALOG_INFO_F("Computing dependency closure...");
    ComputeDependencyClosure(directDeps);    
    // 计算所有依赖（包括外部依赖和内部依赖）
    ALOG_INFO_F("Computing all dependencies...");
    // 提取外部依赖（直接从 originalMixFunc）
    ExtractExternalDependencies();
    // 验证循环依赖是否合法
    Status validationStatus = ValidateCrossComponentDependencies(input, directDeps, crossComponentTensors);
    if (validationStatus != SUCCESS) {
        ALOG_ERROR_F("Cross-component dependency validation failed, aborting ProcessDependencyAnalyzer...");
        return FAILED;  // 立即返回，不继续执行
    }
    // 添加内部同类型scope之间的依赖（只收集C-C、V-V的依赖）
    CollectInternalDependencies(directDeps, input.components);
    output.internalDeps = internalDeps;
    output.allIncasts = allIncasts;
    output.allOutcasts = allOutcasts;
    ALOG_INFO_F("Final state: all %d components share %zu incasts and %zu outcasts",
                maxComponent + 1, 
                allIncasts[0].size(), 
                allOutcasts[0].size());
    return SUCCESS;
}

void MixDependencyAnalyzer::Reset() {
    internalDeps.clear();
    allIncasts.clear();
    allOutcasts.clear();
}

Status MixDependencyAnalyzer::ValidateCrossComponentDependencies(
    const AnalyzerInput &input,
    const std::unordered_map<int, std::set<int>>& directDeps,
    const std::map<std::pair<int, int>, std::vector<LogicalTensorPtr>>& crossComponentTensors) {
    ALOG_INFO_F("=== VALIDATING CROSS-COMPONENT DEPENDENCY CYCLES ===");
    // 收集双向依赖
    std::vector<std::pair<int, int>> bidirectionalDeps;
    for (const auto& [src, dsts] : directDeps) {
        for (int dst : dsts) {
            auto it = directDeps.find(dst);      
            if (it != directDeps.end() && it->second.count(src) > 0) {  
                bidirectionalDeps.emplace_back(src, dst);
            }
        }
    }
    if (bidirectionalDeps.empty()) {
        ALOG_INFO_F("No bidirectional dependencies detected - safe");
        return SUCCESS;
    }  
    ALOG_INFO_F("=== CHECKING BIDIRECTIONAL DEPENDENCIES ===");
    for (const auto& [comp1, comp2] : bidirectionalDeps) {
        ALOG_INFO_F("Checking component %d <-> component %d:", comp1, comp2);
        // 获取两个方向的tensor
        auto it1 = crossComponentTensors.find({comp1, comp2});
        auto it2 = crossComponentTensors.find({comp2, comp1});
        bool hasValid1to2 = false;
        bool hasValid2to1 = false;
        
        if (it1 != crossComponentTensors.end()) {
            CheckDirectionAndCollectValid(it1->second, comp1, comp2, hasValid1to2);
        }
        
        if (it2 != crossComponentTensors.end()) {
            CheckDirectionAndCollectValid(it2->second, comp2, comp1, hasValid2to1);
        }
        if (hasValid1to2 && hasValid2to1) {
            LogIllegalBidirectionalDependency(comp1, comp2, input);
            return FAILED;
        }
    }
    ALOG_INFO_F("=== VALIDATION PASSED ===");   
    return SUCCESS; 
}

bool MixDependencyAnalyzer::CheckDirectionAndCollectValid(
    const std::vector<LogicalTensorPtr>& tensors, int src, int dst, bool& hasValid) const {
    
    if (tensors.empty()) return false;
    
    ALOG_INFO_F("  Component %d -> %d tensors (%zu):", src, dst, tensors.size());
    bool directionValid = false;
    for (auto& tensor : tensors) {
        bool isInIncast = IsTensorInComponentIncasts(dst, tensor);
        ALOG_INFO_F("    - tensor magic=%d (raw=%d), in comp%d.incasts=%s",
                  tensor->magic, tensor->GetRawMagic(), dst,
                  isInIncast ? "YES" : "NO");
        if (isInIncast) {
            directionValid = true;
            hasValid = true;
        }
    }
    return directionValid;
}

void MixDependencyAnalyzer::LogIllegalBidirectionalDependency(
    int comp1, int comp2, const AnalyzerInput& input) const {
    
    ALOG_ERROR_F("ILLEGAL BIDIRECTIONAL DEPENDENCY DETECTED!");
    ALOG_ERROR_F("==========================================================");
    ALOG_ERROR_F("Component %d <-> Component %d", comp1, comp2);
    ALOG_ERROR_F("Component types: %s <-> %s",
               input.components[comp1].componentType == ComponentType::C_SCOPE ? "CUBE" :
               input.components[comp1].componentType == ComponentType::V_SCOPE ? "VECTOR" : "UNKNOWN",
               input.components[comp2].componentType == ComponentType::C_SCOPE ? "CUBE" :
               input.components[comp2].componentType == ComponentType::V_SCOPE ? "VECTOR" : "UNKNOWN");
}
}
}