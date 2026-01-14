/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file directed_graph_concept.hpp
* \brief
*/

#ifndef OSP_DIRECTED_GRAPH_CONCEPT_HPP
#define OSP_DIRECTED_GRAPH_CONCEPT_HPP

#include "graph_traits.hpp"
#include "iterator_concepts.hpp"

namespace npu::tile_fwk {
namespace osp {

/**
 * @brief Concept for a directed graph structure in OneStopParallel (OSP).
 *
 * OneStopParallel is a header-only library where directed graphs serve as the fundamental
 * data structure for all scheduling algorithms and other DAG processing algorithms, such as
 * coarsening or partitioning. The `is_directed_graph` concept defines the minimal interface
 * that a graph type must satisfy to be used within the OSP ecosystem.
 *
 * A type `T` satisfies `is_directed_graph` if it provides the following API:
 *
 * - **Vertices()**: Returns a range of all vertices in the graph.
 * - **NumVertices()**: Returns the total number of vertices as an integral type.
 * - **NumEdges()**: Returns the total number of edges as an integral type.
 * - **parents(v)**: Returns a range of parent vertices for a given vertex `v`.
 *   - `v` must be of type `VertexIdxT<T>`.
 * - **children(v)**: Returns a range of child vertices for a given vertex `v`.
 *   - `v` must be of type `VertexIdxT<T>`.
 * - **InDegree(v)**: Returns the number of incoming edges for vertex `v` as an integral type.
 * - **OutDegree(v)**: Returns the number of outgoing edges for vertex `v` as an integral type.
 *
 * This concept ensures that any graph implementation passed to OSP algorithms exposes
 * the necessary structural information for processing.
 *
 * This concept encapsulates a classic adjacency list graph structure, allowing efficient
 * iteration over the parents and children of any given node.
 *
 * @tparam T The graph type to check against the concept.
 */
template <typename T, typename = void>
struct IsDirectedGraph : std::false_type {};

template <typename T>
struct IsDirectedGraph<T,
                       std::void_t<typename DirectedGraphTraits<T>::VertexIdx,
                                   decltype(std::declval<T>().Vertices()),
                                   decltype(std::declval<T>().NumVertices()),
                                   decltype(std::declval<T>().NumEdges()),
                                   decltype(std::declval<T>().Parents(std::declval<VertexIdxT<T>>())),
                                   decltype(std::declval<T>().Children(std::declval<VertexIdxT<T>>())),
                                   decltype(std::declval<T>().InDegree(std::declval<VertexIdxT<T>>())),
                                   decltype(std::declval<T>().OutDegree(std::declval<VertexIdxT<T>>()))>>
    : std::conjunction<IsForwardRangeOf<decltype(std::declval<T>().Vertices()), VertexIdxT<T>>,
                       std::is_integral<decltype(std::declval<T>().NumVertices())>,
                       std::is_integral<decltype(std::declval<T>().NumEdges())>,
                       IsInputRangeOf<decltype(std::declval<T>().Parents(std::declval<VertexIdxT<T>>())), VertexIdxT<T>>,
                       IsInputRangeOf<decltype(std::declval<T>().Children(std::declval<VertexIdxT<T>>())), VertexIdxT<T>>,
                       std::is_integral<decltype(std::declval<T>().InDegree(std::declval<VertexIdxT<T>>()))>,
                       std::is_integral<decltype(std::declval<T>().OutDegree(std::declval<VertexIdxT<T>>()))>> {};

template <typename T>
inline constexpr bool isDirectedGraphV = IsDirectedGraph<T>::value;

}    // namespace osp
}    // namespace npu::tile_fwk
#endif // OSP_DIRECTED_GRAPH_CONCEPT_HPP
