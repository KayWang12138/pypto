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
* \file computational_dag_concept.hpp
* \brief
*/

#ifndef OSP_COMPUTATIONAL_DAG_CONCEPT_HPP
#define OSP_COMPUTATIONAL_DAG_CONCEPT_HPP

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

namespace npu::tile_fwk {
namespace osp {

/**
 * @brief Concept to check if a graph has vertex weights.
 *
 * Requires validation of:
 * - `VertexWorkWeight(v)`: Returns arithmetic type.
 * - `VertexCommWeight(v)`: Returns arithmetic type.
 * - `VertexMemWeight(v)`: Returns arithmetic type.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct HasVertexWeights : std::false_type {};

template <typename T>
struct HasVertexWeights<T,
                        std::void_t<decltype(std::declval<T>().VertexWorkWeight(std::declval<VertexIdxT<T>>())),
                                    decltype(std::declval<T>().VertexCommWeight(std::declval<VertexIdxT<T>>())),
                                    decltype(std::declval<T>().VertexMemWeight(std::declval<VertexIdxT<T>>()))>>
    : std::conjunction<std::is_arithmetic<decltype(std::declval<T>().VertexWorkWeight(std::declval<VertexIdxT<T>>()))>,
                       std::is_arithmetic<decltype(std::declval<T>().VertexCommWeight(std::declval<VertexIdxT<T>>()))>,
                       std::is_arithmetic<decltype(std::declval<T>().VertexMemWeight(std::declval<VertexIdxT<T>>()))>> {};

template <typename T>
inline constexpr bool hasVertexWeightsV = HasVertexWeights<T>::value;

/**
 * @brief Concept to check if a graph has typed vertices.
 *
 * Requires validation of:
 * - `VertexType(v)`: Returns an integral type representing the type of vertex `v`.
 * - `NumVertexTypes()`: Returns the total number of distinct vertex types.
 *
 * This is useful for scheduling on heterogeneous resources where tasks (vertices)
 * may be compatible only with certain processor types.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct HasTypedVertices : std::false_type {};

template <typename T>
struct HasTypedVertices<
    T,
    std::void_t<decltype(std::declval<T>().VertexType(std::declval<VertexIdxT<T>>())), decltype(std::declval<T>().NumVertexTypes())>>
    : std::conjunction<std::is_integral<decltype(std::declval<T>().VertexType(std::declval<VertexIdxT<T>>()))>,
                       std::is_integral<decltype(std::declval<T>().NumVertexTypes())>> {};

template <typename T>
inline constexpr bool hasTypedVerticesV = HasTypedVertices<T>::value;

/**
 * @brief Concept to check if edges have communication weights.
 *
 * Requires:
 * - The graph must satisfy `is_directed_graph_edge_desc` (supports edge descriptors).
 * - `EdgeCommWeight(e)`: Returns an arithmetic type for a given edge descriptor `e`.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct HasEdgeWeights : std::false_type {};

template <typename T>
struct HasEdgeWeights<T,
                      std::void_t<typename DirectedGraphEdgeDescTraits<T>::DirectedEdgeDescriptor,
                                  decltype(std::declval<T>().EdgeCommWeight(std::declval<EdgeDescT<T>>()))>>
    : std::conjunction<std::is_arithmetic<decltype(std::declval<T>().EdgeCommWeight(std::declval<EdgeDescT<T>>()))>,
                       IsDirectedGraphEdgeDesc<T>> {};

template <typename T>
inline constexpr bool hasEdgeWeightsV = HasEdgeWeights<T>::value;

/**
 * @brief Concept for a basic computational DAG.
 *
 * A computational DAG must:
 * - Be a directed graph (`is_directed_graph`).
 * - Have mandatory vertex weights (`has_vertex_weights`): work, communication, and memory.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsComputationalDag : std::false_type {};

template <typename T>
struct IsComputationalDag<T, std::void_t<>> : std::conjunction<HasVertexWeights<T>> {};

template <typename T>
inline constexpr bool isComputationalDagV = IsComputationalDag<T>::value;

}    // namespace osp
}    // namespace npu::tile_fwk
#endif // OSP_COMPUTATIONAL_DAG_CONCEPT_HPP