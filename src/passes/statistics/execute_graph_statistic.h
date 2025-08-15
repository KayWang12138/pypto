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
 * \file execute_graph_statistic.h
 * \brief Execution graph analysis and statistic reporting
 */
#ifndef EXECUTE_GRAPH_STATISTIC_H
#define EXECUTE_GRAPH_STATISTIC_H

#include <climits>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "interface/function/function.h"

using json = nlohmann::json;

namespace npu::tile_fwk {

struct PathResult {
    int maxLength = 0;  // Maximum path length found
};

struct ConcurrencyStats {
    int maxConcurrency = 0; // Maximum concurrent operations found
};

struct DependencyStats {
    size_t total_predecessors = 0;
    size_t total_successors = 0;
    int min_predecessors = INT_MAX;
    int max_predecessors = 0;
    int min_successors = INT_MAX;
    int max_successors = 0;
    size_t valid_entries = 0;
    std::vector<int> min_pred_nodes;
    std::vector<int> max_pred_nodes;
    std::vector<int> min_succ_nodes;
    std::vector<int> max_succ_nodes;
};

struct MinMaxStats {
    int min_value;
    int max_value;
    std::vector<int> min_nodes;
    std::vector<int> max_nodes;
};

class ExecutionGraphStatistic {
public:
    json AnalyzeExecutionGraph(Function& func);

private:
    PathResult FindLongestPath(Function& func);
    ConcurrencyStats CalculateConcurrency(Function& func);   
    json AnalyzeGraphDependencies(Function& func);
    void UpdateMinMaxStats(int count, int esgId, MinMaxStats& stats);
    json FormatDependencyStats(const DependencyStats& stats);
};
} // namespace npu::tile_fwk

#endif // EXECUTE_GRAPH_STATISTIC_H