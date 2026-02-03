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
 * \file subgraph_algorithms.hpp
 * \brief
 */

#ifndef OSP_SUBGRAPH_ALGORITHMS_HPP
#define OSP_SUBGRAPH_ALGORITHMS_HPP

#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "passes/algorithms/osp/concepts/graph_traits.hpp"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#include "passes/algorithms/osp/concepts/constructable_computational_dag_concept.hpp"
#include "passes/algorithms/osp/concepts/directed_graph_concept.hpp"

namespace npu::tile_fwk {
namespace osp {

template <typename GraphTIn, typename GraphTOut>
std::unordered_map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>> CreateInducedSubgraphMap(
    const GraphTIn &dag, GraphTOut &dagOut, const std::vector<VertexIdxT<GraphTIn>> &selectedNodes) {
    static_assert(std::is_same_v<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>>,
                  "GraphTIn and out must have the same VertexIdx types");

    static_assert(isConstructableCdagVertexV<GraphTOut>, "GraphTOut must satisfy the constructable_cdag_vertex concept");

    static_assert(isConstructableCdagEdgeV<GraphTOut>, "GraphTOut must satisfy the constructable_cdag_edge concept");

    assert(dagOut.NumVertices() == 0);

    std::unordered_map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>> localIdx;
    localIdx.reserve(selectedNodes.size());

    for (const auto &node : selectedNodes) {
        localIdx[node] = dagOut.NumVertices();
        dagOut.AddVertex(dag.VertexWorkWeight(node), dag.VertexCommWeight(node), dag.VertexMemWeight(node), dag.VertexType(node));
    }

    if constexpr (hasEdgeWeightsV<GraphTIn> and hasEdgeWeightsV<GraphTOut>) {
        // add edges with edge comm weights
        for (const auto &node : selectedNodes) {
            for (const auto &inEdge : InEdges(node, dag)) {
                const auto &pred = Source(inEdge, dag);
                if (localIdx.count(pred)) {
                    dagOut.AddEdge(localIdx[pred], localIdx[node], dag.EdgeCommWeight(inEdge));
                }
            }
        }

    } else {
        // add edges without edge comm weights
        for (const auto &node : selectedNodes) {
            for (const auto &pred : dag.Parents(node)) {
                if (localIdx.count(pred)) {
                    dagOut.AddEdge(localIdx[pred], localIdx[node]);
                }
            }
        }
    }

    return localIdx;
}

// template <typename GraphTIn, typename VertT, typename EdgeT, typename WorkWeightType, typename CommWeightType, typename MemWeightType, typename VertexTypeTemplateType>
// std::unordered_map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTIn>> CreateInducedSubgraphMap(
//     const GraphTIn &dag,
//     CompactSparseGraph<true, true, true, true, true, VertT, EdgeT, WorkWeightType, CommWeightType, MemWeightType, VertexTypeTemplateType>
//         &dagOut,
//     const std::vector<VertexIdxT<GraphTIn>> &selectedNodes) {
//     using GraphTOut
//         = CompactSparseGraph<true, true, true, true, true, VertT, EdgeT, WorkWeightType, CommWeightType, MemWeightType, VertexTypeTemplateType>;

//     static_assert(std::is_same_v<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>>,
//                   "GraphTIn and out must have the same VertexIdx types");

//     const std::vector<VertexIdxT<GraphTIn>> topOrder = GetTopOrder(dag);
//     std::vector<VertexIdxT<GraphTIn>> topOrderPosition(topOrder.size());
//     for (VertexIdxT<GraphTIn> pos = 0; pos < dag.NumVertices(); ++pos) {
//         topOrderPosition[topOrder[pos]] = pos;
//     }

//     auto topCmp = [&topOrderPosition](const VertexIdxT<GraphTIn> &lhs, const VertexIdxT<GraphTIn> &rhs) {
//         return topOrderPosition[lhs] < topOrderPosition[rhs];
//     };

//     std::set<VertexIdxT<GraphTIn>, decltype(topCmp)> selectedVerticesOrdered(selectedNodes.begin(), selectedNodes.end(), topCmp);

//     std::unordered_map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTIn>> localIdx;
//     localIdx.reserve(selectedNodes.size());

//     VertexIdxT<GraphTIn> nodeCntr = 0;
//     for (const auto &node : selectedVerticesOrdered) {
//         localIdx[node] = nodeCntr++;
//     }

//     std::vector<std::pair<VertexIdxT<GraphTIn>, VertexIdxT<GraphTIn>>> edges;
//     for (const auto &node : selectedVerticesOrdered) {
//         for (const auto &chld : dag.Children(node)) {
//             if (selectedVerticesOrdered.find(chld) != selectedVerticesOrdered.end()) {
//                 edges.emplace_back(localIdx.at(node), localIdx.at(chld));
//             }
//         }
//     }

//     dagOut = GraphTOut(nodeCntr, edges);

//     for (const auto &[oriVert, outVert] : localIdx) {
//         dagOut.SetVertexWorkWeight(outVert, dag.VertexWorkWeight(oriVert));
//         dagOut.SetVertexCommWeight(outVert, dag.VertexCommWeight(oriVert));
//         dagOut.SetVertexMemWeight(outVert, dag.VertexMemWeight(oriVert));
//         dagOut.SetVertexType(outVert, dag.VertexType(oriVert));
//     }

//     return localIdx;
// }

}    // end namespace osp
} // namespace npu::tile_fwk
#endif // OSP_SUBGRAPH_ALGORITHMS_HPP
