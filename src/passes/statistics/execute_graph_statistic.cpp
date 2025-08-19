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
 * \file execute_graph_statistic.cpp
 * \brief
 */

#include "execute_graph_statistic.h"
#include <fstream>
#include <queue>
#include <unordered_set>
#include "interface/utils/log.h"

using json = nlohmann::json;

namespace npu::tile_fwk {
// 基础分析能力
PathResult ExecutionGraphStatistic::FindLongestPath(Function& func) {
    PathResult result;
    const auto& topoInfo = func.topoInfo_.GetTopology();
    std::unordered_map<int, int> esgDepth; // 存储每个ESG的最大深度
    std::unordered_map<int, std::vector<int>> esgToSuccessors; // ESG到其后继的指针

    // 构建ESG到后继的映射关系
    for (const auto& entry : topoInfo) {
        esgToSuccessors[entry.esgId] = std::vector<int>(entry.outGraph.begin(), entry.outGraph.end());
    }

    // 深度优先搜索计算最长路径
    std::function<int(int)> dfs = [&](int esgId) {
        if (esgDepth.count(esgId)) {
            return esgDepth[esgId];
        }
        int maxDepth = 0;
        for (int succEsgId : esgToSuccessors[esgId]) {
            maxDepth = std::max(maxDepth, dfs(succEsgId));
        }
        esgDepth[esgId] = maxDepth + 1;
        result.maxLength = std::max(result.maxLength, esgDepth[esgId]);
        return esgDepth[esgId];
    };

    // 从所有readyState为READY的ESG开始
    for (const auto& entry : topoInfo) {
        if (entry.readyState == 0) {
            dfs(entry.esgId);
        }
    }

    return result;
}

ConcurrencyStats ExecutionGraphStatistic::CalculateConcurrency(Function& func) {
    ConcurrencyStats stats;
    const auto& topoInfo = func.topoInfo_.GetTopology();
    std::unordered_map<int, int> inDegree;
    std::unordered_map<int, std::vector<int>> graph;
    std::queue<int> readyQueue;

    // 构建图并计算入度
    for (const auto& entry : topoInfo) {
        inDegree[entry.esgId] = 0;
    }

    // 构建后继关系图和计算入度
    for (const auto& entry : topoInfo) {
        for (int succEsgId : entry.outGraph) {
            graph[entry.esgId].push_back(succEsgId);
            inDegree[succEsgId]++;
        }
    }

    // 初始化ready队列
    for (const auto& entry : topoInfo) {
        if (entry.readyState == 0 && inDegree[entry.esgId] == 0) {
            readyQueue.push(entry.esgId);
        }
    }

    // 模拟执行过程，计算最大并发度
    while (!readyQueue.empty()) {
        int currentSize = readyQueue.size();
        stats.maxConcurrency = std::max(stats.maxConcurrency, currentSize);

        for (int i = 0; i < currentSize; ++i) {
            int current = readyQueue.front();
            readyQueue.pop();
            for (int neighbor : graph[current]) {
                if (--inDegree[neighbor] == 0) {
                    readyQueue.push(neighbor);
                }
            }
        }
    }
    return stats;
}

template <typename tType>
uint64_t CalcTensorSize(const std::vector<tType> &curShape) {
    uint64_t res = 1;
    for (auto &dim : curShape) {
        res *= dim;
    }
    return res;
}

json ExecutionGraphStatistic::AnalyzeExecutionGraph(
    Function& func,
    const std::multimap<int, int>& psgToESgMap,
    const std::vector<std::vector<OperationPtr>>& subgraphGroups)
{
    json report;

    uint64_t peakMemoryUsage = 0;
    std::vector<int> peakMemoryUsageSubgraphs;
    std::unordered_map<int, uint64_t> callOpMemoryUsage; // key: callOp magic, value: memory usage
    Function* rootFunc = func.GetRootFunction();
    if (!rootFunc) {
        ALOG_ERROR_F("Root function is null");
        return report;
    }
    int totalSubgraphNum = func.GetTotalSubGraphCount();
    std::unordered_map<CoreType, int> coreTypeCounts;
    std::vector<uint64_t> subgraphLatencies(totalSubgraphNum, 0);
    auto operations = rootFunc->Operations();
    for (size_t i = 0; i < operations.size(); i++) {
        auto& op = operations[i];
        auto callAttr = dynamic_cast<CallOpAttribute*>(op.GetOpAttribute().get());
        if (!callAttr || !callAttr->invokeInfo_) {
            ALOG_WARN_F("Invalid CallOpAttribute at index %zu", i);
            continue;
        }
        // 统计内存使用
        uint64_t currentOpMemory = 0;
        // 统计输入tensor的内存占用
        for (auto& iOperand : op.GetIOperands()) {
            if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                uint64_t tensorSize = CalcTensorSize(iOperand->GetShape()) * BytesOf(iOperand->Datatype());
                currentOpMemory += tensorSize;
            }
        }
        // 统计输出tensor的内存占用
        for (auto& oOperand : op.GetOOperands()) {
            if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                uint64_t tensorSize = CalcTensorSize(oOperand->GetShape()) * BytesOf(oOperand->Datatype());
                currentOpMemory += tensorSize;
            }
        }
        callOpMemoryUsage[op.GetOpMagic()] = currentOpMemory;
        if (currentOpMemory > peakMemoryUsage) {
            peakMemoryUsage = currentOpMemory;
            peakMemoryUsageSubgraphs.clear();
            peakMemoryUsageSubgraphs.push_back(op.GetSubgraphID());
        } else if (currentOpMemory == peakMemoryUsage) {
            peakMemoryUsageSubgraphs.push_back(op.GetSubgraphID());
        }

        CoreType graphType = callAttr->invokeInfo_->GetGraphType();
        coreTypeCounts[graphType]++;
    }

    std::vector<int> maxLatencySubgraphs;
    std::vector<int> minLatencySubgraphs;
    // 收集每个子图的latency
    auto operationViewer = func.Operations();
    for (size_t i = 0; i < operationViewer.size(); i++) {
        int subGraphId = operationViewer[i].GetSubgraphID();
        subgraphLatencies[subGraphId] += operationViewer[i].GetLatency();
    }
    // 计算最大值、最小值和平均值
    uint64_t totalLatency = 0;
    uint64_t maxLatency = 0;
    uint64_t minLatency = UINT64_MAX;

    for (int i = 0; i < totalSubgraphNum; i++) {
        uint64_t latency = subgraphLatencies[i];
        totalLatency += latency;
        if (latency > maxLatency) {
            maxLatency = latency;
            maxLatencySubgraphs = {i};
        } else if (latency == maxLatency) {
            maxLatencySubgraphs.push_back(i);
        }
         if (latency < minLatency) {
            minLatency = latency;
            minLatencySubgraphs = {i};
        } else if (latency == minLatency) {
            minLatencySubgraphs.push_back(i);
        }
    }

    // 计算依赖关系
    auto dependencies = AnalyzeGraphDependencies(func);
    report = {
        {"totalSubgraphCount", totalSubgraphNum},
        {"maxSubgraphDepth", FindLongestPath(func).maxLength},
        {"maxSubgraphWidth", CalculateConcurrency(func).maxConcurrency},
        {"maxSubgraphFanin", dependencies["Predecessors"]["MAX"]["value"]},
        {"maxFaninSubgraphs", dependencies["Predecessors"]["MAX"]["subgraph"]},
        {"maxSubgraphFanout", dependencies["Successors"]["MAX"]["value"]},
        {"maxFanoutSubgraphs", dependencies["Successors"]["MAX"]["subgraph"]},
        {"maxSubgraphCycle", maxLatency},
        {"minSubgraphCycle", minLatency == UINT64_MAX ? 0 : minLatency},
        {"avgSubgraphCycle", totalSubgraphNum > 0 ? totalLatency / totalSubgraphNum : 0},
        {"maxCycleSubgraphs", maxLatencySubgraphs},
        {"minCycleSubgraphs", minLatencySubgraphs},
        {"peakMemoryUsage", peakMemoryUsage},
        {"peakMemoryUsageSubgraphs", peakMemoryUsageSubgraphs},
        {"aivSubgraphCount", coreTypeCounts[CoreType::AIV]},
        {"aicSubgraphCount", coreTypeCounts[CoreType::AIC]},
        {"aicpuSubgraphCount", coreTypeCounts[CoreType::AICPU]},
        {"gmatomicSubgraphCount", coreTypeCounts[CoreType::GMATOMIC]},
        {"hubSubgraphCount", coreTypeCounts[CoreType::HUB]},
        {"invalidSubgraphCount", coreTypeCounts[CoreType::INVALID]},
        {"mixSubgraphCount", coreTypeCounts[CoreType::MIX]}
    };
    AnalyzeIsomorphism(report, psgToESgMap, subgraphGroups);
    return report;
}

json ExecutionGraphStatistic::AnalyzeGraphDependencies(Function& func)
{
    const auto& topology = func.topoInfo_.topology_;
    DependencyStats stats;
    MinMaxStats pred_stats{INT_MAX, INT_MIN, {}, {}};
    MinMaxStats succ_stats{INT_MAX, INT_MIN, {}, {}};
    for (const auto& entry : topology) {
        int pred_count = -entry.readyState;
        UpdateMinMaxStats(pred_count, entry.esgId, pred_stats);
        stats.total_predecessors += pred_count;
        int succ_count = entry.outGraph.size();
        UpdateMinMaxStats(succ_count, entry.esgId, succ_stats);
        stats.total_successors += succ_count;

        stats.valid_entries++;
    }

    stats.min_predecessors = pred_stats.min_value;
    stats.max_predecessors = pred_stats.max_value;
    stats.min_pred_nodes = pred_stats.min_nodes;
    stats.max_pred_nodes = pred_stats.max_nodes;

    stats.min_successors = succ_stats.min_value;
    stats.max_successors = succ_stats.max_value;
    stats.min_succ_nodes = succ_stats.min_nodes;
    stats.max_succ_nodes = succ_stats.max_nodes;
    return FormatDependencyStats(stats);
}

void ExecutionGraphStatistic::UpdateMinMaxStats(int count, int esgId, MinMaxStats& stats) {
    if (count < stats.min_value) {
        stats.min_value = count;
        stats.min_nodes = {esgId};
    } else if (count == stats.min_value) {
        stats.min_nodes.push_back(esgId);
    }

    if (count > stats.max_value) {
        stats.max_value = count;
        stats.max_nodes = {esgId};
    } else if (count == stats.max_value) {
        stats.max_nodes.push_back(esgId);
    }
}

json ExecutionGraphStatistic::FormatDependencyStats(const DependencyStats& stats)
{
    double avg_predecessors = stats.valid_entries > 0 ? static_cast<double>(stats.total_predecessors) / static_cast<double>(stats.valid_entries) : 0.0;
    double avg_successors = stats.valid_entries > 0 ? static_cast<double>(stats.total_successors) / static_cast<double>(stats.valid_entries) : 0.0;
    return {
        {"Predecessors", {
            {"MIN", {
                {"value", stats.min_predecessors},
                {"subgraph", stats.min_pred_nodes}  // 添加最小前驱节点列表
            }},
            {"MAX", {
                {"value", stats.max_predecessors},
                {"subgraph", stats.max_pred_nodes}  // 添加最大前驱节点列表
            }},
            {"AVG", avg_predecessors}
        }},
        {"Successors", {
            {"MIN", {
                {"value", stats.min_successors},
                {"subgraph", stats.min_succ_nodes}  // 添加最小后继节点列表
            }},
            {"MAX", {
                {"value", stats.max_successors},
                {"subgraph", stats.max_succ_nodes}  // 添加最大后继节点列表
            }},
            {"AVG", avg_successors}
        }}
    };
}

// 基于psgToESgMap的同构性分析
void ExecutionGraphStatistic::AnalyzeIsomorphism(
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

    report["uniqueSubgraphTypes"] = isomorphicGroups.size();
    report["homogeneityRatio"] = homogeneityRatio;
    
    // 构建实例映射关系
    json instanceMapping = json::object();
    for (const auto& [psgId, esgIds] : isomorphicGroups) {
        instanceMapping[std::to_string(psgId)] = esgIds;
    }
    report["instanceMapping"] = instanceMapping;
}
} // namespace npu::tile_fwk