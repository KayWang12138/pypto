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
void CreateInducedSubgraph(const GraphTIn &dag,
                           GraphTOut &dagOut,
                           const std::set<VertexIdxT<GraphTIn>> &selectedNodes,
                           const std::set<VertexIdxT<GraphTIn>> &extraSources = {}) {
    static_assert(std::is_same_v<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>>,
                  "GraphTIn and out must have the same VertexIdx types");

    static_assert(isConstructableCdagVertexV<GraphTOut>, "GraphTOut must satisfy the constructable_cdag_vertex concept");

    static_assert(isConstructableCdagEdgeV<GraphTOut>, "GraphTOut must satisfy the constructable_cdag_edge concept");

    assert(dagOut.NumVertices() == 0);

    std::map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTIn>> localIdx;

    for (const auto &node : extraSources) {
        localIdx[node] = dagOut.NumVertices();
        if constexpr (isConstructableCdagTypedVertexV<GraphTOut> and hasTypedVerticesV<GraphTIn>) {
            // add extra source with type
            dagOut.AddVertex(0, dag.VertexCommWeight(node), dag.VertexMemWeight(node), dag.VertexType(node));
        } else {
            // add extra source without type
            dagOut.AddVertex(0, dag.VertexCommWeight(node), dag.VertexMemWeight(node));
        }
    }

    for (const auto &node : selectedNodes) {
        localIdx[node] = dagOut.NumVertices();

        if constexpr (isConstructableCdagTypedVertexV<GraphTOut> and hasTypedVerticesV<GraphTIn>) {
            // add vertex with type
            dagOut.AddVertex(
                dag.VertexWorkWeight(node), dag.VertexCommWeight(node), dag.VertexMemWeight(node), dag.VertexType(node));
        } else {
            // add vertex without type
            dagOut.AddVertex(dag.VertexWorkWeight(node), dag.VertexCommWeight(node), dag.VertexMemWeight(node));
        }
    }

    if constexpr (hasEdgeWeightsV<GraphTIn> and hasEdgeWeightsV<GraphTOut>) {
        // add edges with edge comm weights
        for (const auto &node : selectedNodes) {
            for (const auto &inEdge : InEdges(node, dag)) {
                const auto &pred = Source(inEdge, dag);
                if (selectedNodes.find(pred) != selectedNodes.end() || extraSources.find(pred) != extraSources.end()) {
                    dagOut.AddEdge(localIdx[pred], localIdx[node], dag.EdgeCommWeight(inEdge));
                }
            }
        }

    } else {
        // add edges without edge comm weights
        for (const auto &node : selectedNodes) {
            for (const auto &pred : dag.Parents(node)) {
                if (selectedNodes.find(pred) != selectedNodes.end() || extraSources.find(pred) != extraSources.end()) {
                    dagOut.AddEdge(localIdx[pred], localIdx[node]);
                }
            }
        }
    }
}

template <typename GraphTIn, typename GraphTOut>
void CreateInducedSubgraph(const GraphTIn &dag, GraphTOut &dagOut, const std::vector<VertexIdxT<GraphTIn>> &selectedNodes) {
    return CreateInducedSubgraph(dag, dagOut, std::set<VertexIdxT<GraphTIn>>(selectedNodes.begin(), selectedNodes.end()));
}

template <typename GraphT>
bool CheckOrderedIsomorphism(const GraphT &first, const GraphT &second) {
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");

    if (first.NumVertices() != second.NumVertices() || first.NumEdges() != second.NumEdges()) {
        return false;
    }

    for (const auto &node : first.Vertices()) {
        if (first.VertexWorkWeight(node) != second.VertexWorkWeight(node)
            || first.VertexMemWeight(node) != second.VertexMemWeight(node)
            || first.VertexCommWeight(node) != second.VertexCommWeight(node) || first.VertexType(node) != second.VertexType(node)) {
            return false;
        }

        if (first.InDegree(node) != second.InDegree(node) || first.OutDegree(node) != second.OutDegree(node)) {
            return false;
        }

        if constexpr (hasEdgeWeightsV<GraphT>) {
            std::set<std::pair<VertexIdxT<GraphT>, ECommwT<GraphT>>> firstChildren, secondChildren;

            for (const auto &outEdge : OutEdges(node, first)) {
                firstChildren.emplace(Target(outEdge, first), first.EdgeCommWeight(outEdge));
            }

            for (const auto &outEdge : OutEdges(node, second)) {
                secondChildren.emplace(Target(outEdge, second), second.EdgeCommWeight(outEdge));
            }

            auto itr = firstChildren.begin(), secondItr = secondChildren.begin();
            for (; itr != firstChildren.end() && secondItr != secondChildren.end(); ++itr) {
                if (*itr != *secondItr) {
                    return false;
                }
                ++secondItr;
            }

        } else {
            std::set<VertexIdxT<GraphT>> firstChildren, secondChildren;

            for (const auto &child : first.Children(node)) {
                firstChildren.emplace(child);
            }

            for (const auto &child : second.Children(node)) {
                secondChildren.emplace(child);
            }

            auto itr = firstChildren.begin(), secondItr = secondChildren.begin();
            for (; itr != firstChildren.end() && secondItr != secondChildren.end(); ++itr) {
                if (*itr != *secondItr) {
                    return false;
                }
                ++secondItr;
            }
        }
    }

    return true;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<GraphTOut> CreateInducedSubgraphs(const GraphTIn &dagIn, const std::vector<unsigned> &partitionIDs) {
    // assumes that input partition IDs are consecutive and starting from 0

    static_assert(std::is_same_v<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>>,
                  "GraphTIn and out must have the same VertexIdx types");

    static_assert(isConstructableCdagVertexV<GraphTOut>, "GraphTOut must satisfy the constructable_cdag_vertex concept");

    static_assert(isConstructableCdagEdgeV<GraphTOut>, "GraphTOut must satisfy the constructable_cdag_edge concept");

    unsigned numberOfParts = 0;
    for (const auto id : partitionIDs) {
        numberOfParts = std::max(numberOfParts, id + 1);
    }

    std::vector<GraphTOut> splitDags(numberOfParts);

    std::vector<VertexIdxT<GraphTOut>> localIdx(dagIn.NumVertices());

    for (const auto node : dagIn.Vertices()) {
        localIdx[node] = splitDags[partitionIDs[node]].NumVertices();

        if constexpr (isConstructableCdagTypedVertexV<GraphTOut> and hasTypedVerticesV<GraphTIn>) {
            splitDags[partitionIDs[node]].AddVertex(
                dagIn.VertexWorkWeight(node), dagIn.VertexCommWeight(node), dagIn.VertexMemWeight(node), dagIn.VertexType(node));
        } else {
            splitDags[partitionIDs[node]].AddVertex(
                dagIn.VertexWorkWeight(node), dagIn.VertexCommWeight(node), dagIn.VertexMemWeight(node));
        }
    }

    if constexpr (hasEdgeWeightsV<GraphTIn> and hasEdgeWeightsV<GraphTOut>) {
        for (const auto node : dagIn.Vertices()) {
            for (const auto &outEdge : OutEdges(node, dagIn)) {
                auto succ = Target(outEdge, dagIn);

                if (partitionIDs[node] == partitionIDs[succ]) {
                    splitDags[partitionIDs[node]].AddEdge(localIdx[node], localIdx[succ], dagIn.EdgeCommWeight(outEdge));
                }
            }
        }
    } else {
        for (const auto node : dagIn.Vertices()) {
            for (const auto &child : dagIn.Children(node)) {
                if (partitionIDs[node] == partitionIDs[child]) {
                    splitDags[partitionIDs[node]].AddEdge(localIdx[node], localIdx[child]);
                }
            }
        }
    }

    return splitDags;
}

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

        if constexpr (isConstructableCdagTypedVertexV<GraphTOut> and hasTypedVerticesV<GraphTIn>) {
            // add vertex with type
            dagOut.AddVertex(
                dag.VertexWorkWeight(node), dag.VertexCommWeight(node), dag.VertexMemWeight(node), dag.VertexType(node));
        } else {
            // add vertex without type
            dagOut.AddVertex(dag.VertexWorkWeight(node), dag.VertexCommWeight(node), dag.VertexMemWeight(node));
        }
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
