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
* \file directed_graph_edge_desc_concept.hpp
* \brief
*/

#ifndef OSP_DIRECTED_GRAPH_EDGE_DESC_CONCEPT_HPP
#define OSP_DIRECTED_GRAPH_EDGE_DESC_CONCEPT_HPP

#include "graph_traits.hpp"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_edge_view.hpp"

/**
 * @file directed_graph_edge_desc_concept.hpp
 * @brief Concepts and default implementations for edge descriptors in directed graphs.
 *
 * This file extends the basic directed graph concepts to support edge descriptors.
 * It provides default implementations for accessing source/target vertices from edges
 * and defines the `is_directed_graph_edge_desc` concept to check if a graph type
 * properly supports edge descriptors and edge iteration.
 *
 * Note: If a graph implementation satisfies `directed_graph_concept`, OSP automatically
 * adds the edge descriptor API on top using `directed_edge` as the default edge descriptor.
 * The mechanics for this are implemented in `directed_graph_edge_view.hpp`.
 */

namespace npu::tile_fwk {
namespace osp {

/**
 * @brief Default implementation to get the source vertex of an edge.
 *
 * @tparam GraphT The graph type.
 * @param edge The edge descriptor.
 * @return The source vertex index.
 */
template <typename GraphT>
inline VertexIdxT<GraphT> Source(const DirectedEdge<GraphT> &edge, const GraphT &) {
    return edge.source_;
}

/**
 * @brief Default implementation to get the target vertex of an edge.
 *
 * @tparam GraphT The graph type.
 * @param edge The edge descriptor.
 * @return The target vertex index.
 */
template <typename GraphT>
inline VertexIdxT<GraphT> Target(const DirectedEdge<GraphT> &edge, const GraphT &) {
    return edge.target_;
}

/**
 * @brief Get a view of all edges in the graph.
 *
 * @tparam GraphT The graph type.
 * @param graph The graph instance.
 * @return An `edge_view` allowing iteration over all edges.
 */
template <typename GraphT>
inline EdgeView<GraphT> Edges(const GraphT &graph) {
    return EdgeView<GraphT>(graph);
}

/**
 * @brief Get a view of outgoing edges from a vertex.
 *
 * @tparam GraphT The graph type.
 * @param u The source vertex index.
 * @param graph The graph instance.
 * @return An `out_edge_view` allowing iteration over outgoing edges from `u`.
 */
template <typename GraphT>
inline OutEdgeView<GraphT> OutEdges(VertexIdxT<GraphT> u, const GraphT &graph) {
    return OutEdgeView<GraphT>(graph, u);
}

/**
 * @brief Get a view of incoming edges to a vertex.
 *
 * @tparam GraphT The graph type.
 * @param v The target vertex index.
 * @param graph The graph instance.
 * @return An `in_edge_view` allowing iteration over incoming edges to `v`.
 */
template <typename GraphT>
inline InEdgeView<GraphT> InEdges(VertexIdxT<GraphT> v, const GraphT &graph) {
    return InEdgeView<GraphT>(graph, v);
}

/**
 * @brief Concept check for a directed graph with edge descriptors.
 *
 * Checks if a type `T` satisfies the requirements of a directed graph that also
 * supports edge descriptors, including:
 * - Validity of `directed_graph_edge_desc_traits`.
 * - Existence of `Edges()`, `OutEdges()`, and `InEdges()` functions returning input ranges of edge descriptors.
 * - Existence of `source()` and `target()` functions mapping edge descriptors to vertex indices.
 * - Default and copy constructibility of the edge descriptor type.
 *
 * @tparam T The graph type to check.
 */
template <typename T, typename = void>
struct IsDirectedGraphEdgeDesc : std::false_type {};

template <typename T>
struct IsDirectedGraphEdgeDesc<T,
                               std::void_t<typename DirectedGraphEdgeDescTraits<T>::DirectedEdgeDescriptor,
                                           decltype(Edges(std::declval<T>())),
                                           decltype(OutEdges(std::declval<VertexIdxT<T>>(), std::declval<T>())),
                                           decltype(InEdges(std::declval<VertexIdxT<T>>(), std::declval<T>())),
                                           decltype(Source(std::declval<EdgeDescT<T>>(), std::declval<T>())),
                                           decltype(Target(std::declval<EdgeDescT<T>>(), std::declval<T>()))>>
    : std::conjunction<std::is_default_constructible<EdgeDescT<T>>,
                       std::is_copy_constructible<EdgeDescT<T>>,
                       IsInputRangeOf<decltype(Edges(std::declval<T>())), EdgeDescT<T>>,
                       IsInputRangeOf<decltype(OutEdges(std::declval<VertexIdxT<T>>(), std::declval<T>())), EdgeDescT<T>>,
                       IsInputRangeOf<decltype(InEdges(std::declval<VertexIdxT<T>>(), std::declval<T>())), EdgeDescT<T>>,
                       std::is_same<decltype(Source(std::declval<EdgeDescT<T>>(), std::declval<T>())), VertexIdxT<T>>,
                       std::is_same<decltype(Target(std::declval<EdgeDescT<T>>(), std::declval<T>())), VertexIdxT<T>>> {};

template <typename T>
inline constexpr bool isDirectedGraphEdgeDescV = IsDirectedGraphEdgeDesc<T>::value;

/**
 * @brief Specialization for graphs that define a directed_edge_descriptor that can be used as a key in a hash table.
 *
 * Compatible with STL hash tables (requires `std::hash` specialization and equality operator).
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct HasHashableEdgeDesc : std::false_type {};

template <typename T>
struct HasHashableEdgeDesc<T,
                           std::void_t<decltype(std::hash<EdgeDescT<T>>{}(std::declval<EdgeDescT<T>>())),
                                       decltype(std::declval<EdgeDescT<T>>() == std::declval<EdgeDescT<T>>()),
                                       decltype(std::declval<EdgeDescT<T>>() != std::declval<EdgeDescT<T>>())>>
    : std::conjunction<IsDirectedGraphEdgeDesc<T>,
                       std::is_default_constructible<EdgeDescT<T>>,
                       std::is_copy_constructible<EdgeDescT<T>>> {};

template <typename T>
inline constexpr bool hasHashableEdgeDescV = HasHashableEdgeDesc<T>::value;

}    // namespace osp
}    // namespace npu::tile_fwk
#endif // OSP_DIRECTED_GRAPH_EDGE_DESC_CONCEPT_HPP