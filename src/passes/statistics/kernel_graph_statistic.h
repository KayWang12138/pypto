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
 * \file kernel_graph_statistic.h
 * \brief Kernel graph analysis and statistic reporting
 */
#ifndef KERNEL_GRAPH_STATISTIC_H
#define KERNEL_GRAPH_STATISTIC_H

#include <unordered_map>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

using json = nlohmann::json;

namespace npu::tile_fwk {

class KernelGraphStatistic {
public:
    /*!
     * \brief Performs health check analysis on kernel graphs
     * \param psgToESgMap Mapping from PSG to ESG IDs
     * \param subgraphGroups Groups of operations forming subgraphs
     */
    json AnalyzeKernelGraph(
        const std::multimap<int, int>& psgToESgMap, 
        const std::vector<std::vector<OperationPtr>>& subgraphGroups);

private:
    // Helper functions
    bool IsInternalTensor(
        const std::shared_ptr<LogicalTensor>& tensor,
        const std::vector<OperationPtr>& subgraphOps) const;
    
    int CountInternalProducers(
        const std::shared_ptr<LogicalTensor>& tensor,
        const std::vector<OperationPtr>& subgraphOps) const;
    
    int CountInternalConsumers(
        const std::shared_ptr<LogicalTensor>& tensor,
        const std::vector<OperationPtr>& subgraphOps) const;
    
    int CountInternalInputs(
        const OperationPtr& op,
        const std::vector<OperationPtr>& subgraphOps) const;
    
    // Analysis functions
    void AnalyzeIsomorphism(
    nlohmann::json& report,
    const std::multimap<int, int>& psgToESgMap,
    const std::vector<std::vector<OperationPtr>>& subgraphGroups);

    void AnalyzeInternalConnectivity(
    nlohmann::json& report,
    const std::multimap<int, int>& psgToESgMap,
    const std::vector<std::vector<OperationPtr>>& subgraphGroups);

    json CalculateSubgraphConnectivity(
        const std::vector<OperationPtr>& subgraph) const;
};

} // namespace npu::tile_fwk

#endif // KERNEL_GRAPH_STATISTIC_H

