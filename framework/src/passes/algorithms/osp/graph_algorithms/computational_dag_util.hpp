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
 * \file computational_dag_util.hpp
 * \brief
 */

#ifndef OSP_COMPUTATIONAL_DAG_UTIL_HPP
#define OSP_COMPUTATIONAL_DAG_UTIL_HPP

#include <numeric>

#include "directed_graph_top_sort.hpp"
#include "passes/algorithms/osp/concepts/computational_dag_concept.hpp"

namespace npu::tile_fwk {
namespace osp {

template <typename GraphT>
VWorkwT<GraphT> CriticalPathWeight(const GraphT &graph) {
    static_assert(isDirectedGraphEdgeDescV<GraphT>, "GraphT must satisfy the directed_graph concept");
    static_assert(hasVertexWeightsV<GraphT>, "GraphT must have vertex weights");

    if (graph.NumVertices() == 0) {
        return 0;
    }

    std::vector<VWorkwT<GraphT>> topLength(graph.NumVertices(), 0);
    VWorkwT<GraphT> criticalPathWeight = 0;

    // calculating lenght of longest path
    for (const auto &node : GetTopOrder(graph)) {
        VWorkwT<GraphT> maxTemp = 0;
        for (const auto &parent : graph.Parents(node)) {
            maxTemp = std::max(maxTemp, topLength[parent]);
        }

        topLength[node] = maxTemp + graph.VertexWorkWeight(node);

        if (topLength[node] > criticalPathWeight) {
            criticalPathWeight = topLength[node];
        }
    }

    return criticalPathWeight;
}

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_COMPUTATIONAL_DAG_UTIL_HPP
