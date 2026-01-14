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
* \file directed_graph_concept.hpp
* \brief
*/

#pragma once

#include "graph_traits.hpp"
#include "iterator_concepts.hpp"
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
 * - **vertices()**: Returns a range of all vertices in the graph.
 * - **num_vertices()**: Returns the total number of vertices as an integral type.
 * - **num_edges()**: Returns the total number of edges as an integral type.
 * - **parents(v)**: Returns a range of parent vertices for a given vertex `v`.
 *   - `v` must be of type `vertex_idx_t<T>`.
 * - **children(v)**: Returns a range of child vertices for a given vertex `v`.
 *   - `v` must be of type `vertex_idx_t<T>`.
 * - **in_degree(v)**: Returns the number of incoming edges for vertex `v` as an integral type.
 * - **out_degree(v)**: Returns the number of outgoing edges for vertex `v` as an integral type.
 *
 * This concept ensures that any graph implementation passed to OSP algorithms exposes
 * the necessary structural information for processing.
 *
 * This concept encapsulates a classic adjacency list graph structure, allowing efficient
 * iteration over the parents and children of any given node.
 *
 * @tparam T The graph type to check against the concept.
 */
template<typename T, typename = void>
struct is_directed_graph : std::false_type {};

template<typename T>
struct is_directed_graph<
    T, std::void_t<typename directed_graph_traits<T>::vertex_idx,
                   decltype(std::declval<T>().vertices()),
                   decltype(std::declval<T>().num_vertices()),
                   decltype(std::declval<T>().num_edges()),
                   decltype(std::declval<T>().parents(std::declval<vertex_idx_t<T>>())),
                   decltype(std::declval<T>().children(std::declval<vertex_idx_t<T>>())),
                   decltype(std::declval<T>().in_degree(std::declval<vertex_idx_t<T>>())),
                   decltype(std::declval<T>().out_degree(std::declval<vertex_idx_t<T>>()))>>
    : std::conjunction<
          is_forward_range_of<decltype(std::declval<T>().vertices()), vertex_idx_t<T>>,
          std::is_integral<decltype(std::declval<T>().num_vertices())>,
          std::is_integral<decltype(std::declval<T>().num_edges())>,
          is_input_range_of<decltype(std::declval<T>().parents(std::declval<vertex_idx_t<T>>())), vertex_idx_t<T>>,
          is_input_range_of<decltype(std::declval<T>().children(std::declval<vertex_idx_t<T>>())), vertex_idx_t<T>>,
          std::is_integral<decltype(std::declval<T>().in_degree(std::declval<vertex_idx_t<T>>()))>,
          std::is_integral<decltype(std::declval<T>().out_degree(std::declval<vertex_idx_t<T>>()))>> {};

template<typename T>
inline constexpr bool is_directed_graph_v = is_directed_graph<T>::value;

} // namespace osp