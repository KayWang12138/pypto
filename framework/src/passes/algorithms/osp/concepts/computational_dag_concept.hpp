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
* \file computational_dag_concept.hpp
* \brief
*/

#pragma once

#include <type_traits>

#include "directed_graph_edge_desc_concept.hpp"

/**
 * @file computational_dag_concept.hpp
 * @brief Concepts for Computational Directed Acyclic Graphs (cDAGs).
 *
 * This file defines concepts that validate whether a graph type satisfies the requirements
 * of a computational DAG.
 *
 * A Computational DAG combines:
 * - The `directed_graph_concept`.
 * - Mandatory vertex weights: work, communication, and memory.
 *
 * Optional extensions include:
 * - Vertex types (for heterogeneous systems).
 * - Edge weights (communication).
 *
 * A computational DAG serves as an input to scheduling algorithms.
 */

namespace osp {

/**
 * @brief Concept to check if a graph has vertex weights.
 *
 * Requires validation of:
 * - `vertex_work_weight(v)`: Returns arithmetic type.
 * - `vertex_comm_weight(v)`: Returns arithmetic type.
 * - `vertex_mem_weight(v)`: Returns arithmetic type.
 *
 * @tparam T The graph type.
 */
template<typename T, typename = void>
struct has_vertex_weights : std::false_type {};

template<typename T>
struct has_vertex_weights<T,
                          std::void_t<decltype(std::declval<T>().vertex_work_weight(std::declval<vertex_idx_t<T>>())),
                                      decltype(std::declval<T>().vertex_comm_weight(std::declval<vertex_idx_t<T>>())),
                                      decltype(std::declval<T>().vertex_mem_weight(std::declval<vertex_idx_t<T>>()))>>
    : std::conjunction<
          std::is_arithmetic<decltype(std::declval<T>().vertex_work_weight(std::declval<vertex_idx_t<T>>()))>,
          std::is_arithmetic<decltype(std::declval<T>().vertex_comm_weight(std::declval<vertex_idx_t<T>>()))>,
          std::is_arithmetic<decltype(std::declval<T>().vertex_mem_weight(std::declval<vertex_idx_t<T>>()))>> {};

template<typename T>
inline constexpr bool has_vertex_weights_v = has_vertex_weights<T>::value;

/**
 * @brief Concept to check if a graph has typed vertices.
 *
 * Requires validation of:
 * - `vertex_type(v)`: Returns an integral type representing the type of vertex `v`.
 * - `num_vertex_types()`: Returns the total number of distinct vertex types.
 *
 * This is useful for scheduling on heterogeneous resources where tasks (vertices)
 * may be compatible only with certain processor types.
 *
 * @tparam T The graph type.
 */
template<typename T, typename = void>
struct has_typed_vertices : std::false_type {};

template<typename T>
struct has_typed_vertices<T, std::void_t<decltype(std::declval<T>().vertex_type(std::declval<vertex_idx_t<T>>())),
                                         decltype(std::declval<T>().num_vertex_types())>>
    : std::conjunction<std::is_integral<decltype(std::declval<T>().vertex_type(std::declval<vertex_idx_t<T>>()))>,
                       std::is_integral<decltype(std::declval<T>().num_vertex_types())>> {};

template<typename T>
inline constexpr bool has_typed_vertices_v = has_typed_vertices<T>::value;

/**
 * @brief Concept to check if edges have communication weights.
 *
 * Requires:
 * - The graph must satisfy `is_directed_graph_edge_desc` (supports edge descriptors).
 * - `edge_comm_weight(e)`: Returns an arithmetic type for a given edge descriptor `e`.
 *
 * @tparam T The graph type.
 */
template<typename T, typename = void>
struct has_edge_weights : std::false_type {};

template<typename T>
struct has_edge_weights<T,
                        std::void_t<typename directed_graph_edge_desc_traits<T>::directed_edge_descriptor,
                                    decltype(std::declval<T>().edge_comm_weight(std::declval<edge_desc_t<T>>()))>>
    : std::conjunction<
          std::is_arithmetic<decltype(std::declval<T>().edge_comm_weight(std::declval<edge_desc_t<T>>()))>,
          is_directed_graph_edge_desc<T>> {};

template<typename T>
inline constexpr bool has_edge_weights_v = has_edge_weights<T>::value;

/**
 * @brief Concept for a basic computational DAG.
 *
 * A computational DAG must:
 * - Be a directed graph (`is_directed_graph`).
 * - Have mandatory vertex weights (`has_vertex_weights`): work, communication, and memory.
 *
 * @tparam T The graph type.
 */
template<typename T, typename = void>
struct is_computational_dag : std::false_type {};

template<typename T>
struct is_computational_dag<T, std::void_t<>> : std::conjunction<is_directed_graph<T>, has_vertex_weights<T>> {};

template<typename T>
inline constexpr bool is_computational_dag_v = is_computational_dag<T>::value;

/**
 * @brief Concept for a computational DAG with typed vertices.
 *
 * Extends `is_computational_dag` by also requiring `has_typed_vertices`.
 *
 * @tparam T The graph type.
 */
template<typename T, typename = void>
struct is_computational_dag_typed_vertices : std::false_type {};

template<typename T>
struct is_computational_dag_typed_vertices<T, std::void_t<>>
    : std::conjunction<is_computational_dag<T>, has_typed_vertices<T>> {};

template<typename T>
inline constexpr bool is_computational_dag_typed_vertices_v = is_computational_dag_typed_vertices<T>::value;

/**
 * @brief Concept for a computational DAG that supports explicit edge descriptors.
 *
 * Extends `is_computational_dag` by requiring `is_directed_graph_edge_desc`,
 * allowing iteration over edges using explicit descriptors.
 *
 * @tparam T The graph type.
 */
template<typename T, typename = void>
struct is_computational_dag_edge_desc : std::false_type {};

template<typename T>
struct is_computational_dag_edge_desc<T, std::void_t<>>
    : std::conjunction<is_directed_graph_edge_desc<T>, is_computational_dag<T>> {};

template<typename T>
inline constexpr bool is_computational_dag_edge_desc_v = is_computational_dag_edge_desc<T>::value;

/**
 * @brief Concept for a computational DAG with both typed vertices and edge descriptors.
 *
 * Combines `is_directed_graph_edge_desc` and `is_computational_dag_typed_vertices`.
 *
 * @tparam T The graph type.
 */
template<typename T, typename = void>
struct is_computational_dag_typed_vertices_edge_desc : std::false_type {};

template<typename T>
struct is_computational_dag_typed_vertices_edge_desc<T, std::void_t<>>
    : std::conjunction<is_directed_graph_edge_desc<T>, is_computational_dag_typed_vertices<T>> {};

template<typename T>
inline constexpr bool is_computational_dag_typed_vertices_edge_desc_v =
    is_computational_dag_typed_vertices_edge_desc<T>::value;

} // namespace osp