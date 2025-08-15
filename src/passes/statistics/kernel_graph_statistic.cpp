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
 * \file kernel_graph_statistic.cpp
 * \brief
 */

#include "kernel_graph_statistic.h"
#include <algorithm>
#include <fstream>
#include <unordered_set>

using json = nlohmann::json;

namespace npu::tile_fwk {
// 核心函数：KernelGraph同构性及子图特征分析
json KernelGraphStatistic::AnalyzeKernelGraph(const std::multimap<int, int>& psgToESgMap, const std::vector<std::vector<OperationPtr>>& subgraphGroups) {
    json report;

    // 1. 同构性分析
    AnalyzeIsomorphism(report, psgToESgMap, subgraphGroups);

    // 2. 子图内部连接性分析
    AnalyzeInternalConnectivity(report, psgToESgMap, subgraphGroups);

    return report;
}

bool KernelGraphStatistic::IsInternalTensor(
    const std::shared_ptr<LogicalTensor>& tensor,
    const std::vector<OperationPtr>& subgraphOps) const {
    if (tensor->GetProducers().empty() || tensor->GetConsumers().empty()) {
        return false;
    }

    std::unordered_set<Operation*> opSet;
    for (const auto& op : subgraphOps) {
        opSet.insert(op.get());
    }

    for (const auto& producer : tensor->GetProducers()) {
        if (opSet.find(producer) == opSet.end()) {
            return false;
        }
    }

    for (const auto& consumer : tensor->GetConsumers()) {
        if (opSet.find(consumer) == opSet.end()) {
            return false;
        }
    }

    return true;
}

int KernelGraphStatistic::CountInternalProducers(
    const std::shared_ptr<LogicalTensor>& tensor,
    const std::vector<OperationPtr>& subgraphOps) const {
    const std::unordered_set<OperationPtr> opSet(subgraphOps.begin(), subgraphOps.end());
    int count = 0;
    for (const auto& producer : tensor->GetProducers()) {
        auto it = std::find_if(opSet.begin(), opSet.end(),
            [producer](const OperationPtr& op) {
                return op.get() == producer;
            });
        if (it != opSet.end()) {
            ++count;
        }
    }
    return count;
}

int KernelGraphStatistic::CountInternalConsumers(
    const std::shared_ptr<LogicalTensor>& tensor,
    const std::vector<OperationPtr>& subgraphOps) const {
    const std::unordered_set<OperationPtr> opSet(subgraphOps.begin(), subgraphOps.end());
    int count = 0;
    for (const auto& consumer : tensor->GetConsumers()) {
        auto it = std::find_if(opSet.begin(), opSet.end(),
            [consumer](const OperationPtr& op) {
                return op.get() == consumer;
            });
        if (it != opSet.end()) {
            ++count;
        }
    }
    return count;
}

// 统计某个op的输入Tensor中有多少是来自当前子图内部的
int KernelGraphStatistic::CountInternalInputs(
    const OperationPtr& op,
    const std::vector<OperationPtr>& subgraphOps) const {
    const std::unordered_set<OperationPtr> opSet(subgraphOps.begin(), subgraphOps.end());
    int count = 0;
    for (const auto& tensor : op->GetIOperands()) {
        bool isInternal = true;
        for (const auto& producer : tensor->GetProducers()) {
            auto it = std::find_if(opSet.begin(), opSet.end(),
                [producer](const OperationPtr& innerOp) {
                    return innerOp.get() == producer;
                });
            if (it == opSet.end()) {
                isInternal = false;
                break;
            }
        }
        if (isInternal) {
            ++count;
        }
    }
    return count;
}

// 基于psgToESgMap的同构性分析
void KernelGraphStatistic::AnalyzeIsomorphism(
    json& report,
    const std::multimap<int, int>& psgToESgMap,
    const std::vector<std::vector<OperationPtr>>& subgraphGroups) {
    // 统计同构子图分布
    std::unordered_map<int, std::vector<int>> isomorphicGroups;
    for (const auto& [psgId, esgId] : psgToESgMap) {
        isomorphicGroups[psgId].push_back(esgId);
    }

    // 计算同构率
    double homogeneityRatio = subgraphGroups.empty() ? 0.0 : static_cast<double>(subgraphGroups.size()) / isomorphicGroups.size();

    report["isomorphism"] = {
        {"total_subgraphs", subgraphGroups.size()},
        {"unique_subgraph_types", isomorphicGroups.size()},
        {"homogeneity_ratio", homogeneityRatio},
        {"isomorphic_groups", json::object()}
    };

    // 记录每个同构组的详细信息
    for (const auto& [psgId, esgIds] : isomorphicGroups) {
        report["isomorphism"]["isomorphic_groups"][std::to_string(psgId)] = {
            {"template_subgraph_id", psgId},
            {"instance_count", esgIds.size()},
            {"instance_ids", esgIds}
        };
    }
}

// 子图内部连接特征分析
void KernelGraphStatistic::AnalyzeInternalConnectivity(
    json& report,
    const std::multimap<int, int>& psgToESgMap,
    const std::vector<std::vector<OperationPtr>>& subgraphGroups) {
    // 1. 首先构建同构组映射（psgId->代表性子图ID）
    std::unordered_map<int, int> templateSubgraphs;
    for (const auto& [psgId, esgId] : psgToESgMap) {
        if (templateSubgraphs.find(psgId) == templateSubgraphs.end()) {
            templateSubgraphs[psgId] = esgId; // 记录第一个实例作为代表
        }
    }

    // 2. 只计算代表性子图的连接性
    std::unordered_map<int, json> psgStats;
    for (const auto& [psgId, esgId] : templateSubgraphs) {
        if (esgId < 0 || static_cast<size_t>(esgId) >= subgraphGroups.size()) continue;
        const auto& subgraph = subgraphGroups[esgId];
        json connStats = CalculateSubgraphConnectivity(subgraph);
        psgStats[psgId] = {
            {"representative_esg_id", esgId},
            {"connectivity", connStats}
        };
    }

    // 3. 生成最终报告（按psgId组织）
    report["internal_connectivity"] = {
        {"template_stats", psgStats},
        {"instance_mapping", json::object()}
    };

    for (const auto& [psgId, esgId] : psgToESgMap) {
        report["internal_connectivity"]["instance_mapping"][std::to_string(esgId)] = psgId;
    }
}

/**
 * 计算子图的内部连接性特征
 * @param subgraph 子图的操作列表
 * @return JSON格式的统计结果，包含以下字段：
 * -max_producers: 内部Tensor的最大生产者数量
 * -max_consumers: 内部Tensor的最大消费者数量
 * -max_internal_inputs: op的最大内部输入数量
 * -max_outputs: op的最大输出数量
 * -internal_tensor_count: 完全位于子图内部的Tensor数量
 * -crossing_tensor_count: 跨子图边界的Tensor数量
 */
json KernelGraphStatistic::CalculateSubgraphConnectivity(
    const std::vector<OperationPtr>& subgraph) const
{
    const std::unordered_set<OperationPtr> opSet(subgraph.begin(), subgraph.end());
    json stats;
    std::unordered_set<std::shared_ptr<LogicalTensor>> internalTensors;
    std::unordered_set<std::shared_ptr<LogicalTensor>> crossingTensors;
    int maxProducers = 0;
    int maxConsumers = 0;
    struct OpMaxRecord { 
        int value = 0; 
        std::vector<int> nodes; 
    };
    OpMaxRecord maxInternalInputs;
    OpMaxRecord maxOutputs;

    for (const auto& op : subgraph) {
        // Operation统计
        int outputCount = op->GetOOperands().size();
        if (outputCount >= maxOutputs.value) {
            if (outputCount > maxOutputs.value) maxOutputs.nodes.clear();
            maxOutputs.value = outputCount;
            maxOutputs.nodes.push_back(op->GetOpMagic());
        }

        int internalInputCount = CountInternalInputs(op, subgraph);
        if (internalInputCount >= maxInternalInputs.value) {
            if (internalInputCount > maxInternalInputs.value) maxInternalInputs.nodes.clear();
            maxInternalInputs.value = internalInputCount;
            maxInternalInputs.nodes.push_back(op->GetOpMagic());
        }

        // Tensor统计
        for (const auto& tensor : op->GetIOperands()) {
            if (internalTensors.count(tensor) || crossingTensors.count(tensor)) continue;
            
            if (IsInternalTensor(tensor, subgraph)) {
                internalTensors.insert(tensor);
                maxProducers = std::max(maxProducers, CountInternalProducers(tensor, subgraph));
                maxConsumers = std::max(maxConsumers, CountInternalConsumers(tensor, subgraph));
            } else {
                crossingTensors.insert(tensor);
            }
        }
    }

    // 组装结果
    stats = {
        {"max_producers", maxProducers},
        {"max_consumers", maxConsumers},
        {"max_internal_inputs", {{"value", maxInternalInputs.value}, {"nodes", maxInternalInputs.nodes}}},
        {"max_outputs", {{"value", maxOutputs.value}, {"nodes", maxOutputs.nodes}}},
        {"internal_tensor_count", internalTensors.size()},
        {"crossing_tensor_count", crossingTensors.size()}
    };

    return stats;
}
} // namespace npu::tile_fwk