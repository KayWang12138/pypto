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

json ExecutionGraphStatistic::AnalyzeExecutionGraph(Function& func)
{
    json report;
    report["executeGraph"] = {
        {"total_operations", func.rootFunc_ ? func.rootFunc_->Operations().size() : 0},
        {"critical_path", FindLongestPath(func).maxLength},
        {"max_concurrency", CalculateConcurrency(func).maxConcurrency},
        {"dependencies", AnalyzeGraphDependencies(func)}
    };
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
} // namespace npu::tile_fwk