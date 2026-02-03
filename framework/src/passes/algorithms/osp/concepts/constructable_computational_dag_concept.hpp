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
* \file constructable_computational_dag_concept.hpp
* \brief
*/

#ifndef OSP_CONSTRUCTABLE_COMPUTATIONAL_DAG_CONCEPT_HPP
#define OSP_CONSTRUCTABLE_COMPUTATIONAL_DAG_CONCEPT_HPP

#include <set>

/**
 * @file constructable_computational_dag_concept.hpp
 * @brief Concepts for Constructable and Modifiable Computational DAGs.
 *
 * This file defines concepts that validate whether a graph type supports dynamic construction
 * and modification of its structure and properties. This includes adding vertices and edges,
 * as well as setting weights and types for existing elements.
 *
 * These concepts are useful for algorithms that need to build or transform graphs,
 * such as graph generators or coarsening algorithms.
 */

namespace npu::tile_fwk {
namespace osp {

/**
 * @brief Concept to check if vertex weights are modifiable.
 *
 * Requires:
 * - `SetVertexWorkWeight(v, w)`
 * - `SetVertexCommWeight(v, w)`
 * - `SetVertexMemWeight(v, w)`
 *
 * Also requires the graph to be default constructible, copy/move constructible, and assignable.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsModifiableCdagVertex : std::false_type {};

template <typename T>
struct IsModifiableCdagVertex<
    T,
    std::void_t<decltype(std::declval<T>().SetVertexWorkWeight(std::declval<VertexIdxT<T>>(), std::declval<VWorkwT<T>>())),
                decltype(std::declval<T>().SetVertexCommWeight(std::declval<VertexIdxT<T>>(), std::declval<VCommwT<T>>())),
                decltype(std::declval<T>().SetVertexMemWeight(std::declval<VertexIdxT<T>>(), std::declval<VMemwT<T>>()))>>
    : std::conjunction<std::is_default_constructible<T>,
                       std::is_copy_constructible<T>,
                       std::is_move_constructible<T>,
                       std::is_copy_assignable<T>,
                       std::is_move_assignable<T>> {};

template <typename T>
inline constexpr bool isModifiableCdagVertexV = IsModifiableCdagVertex<T>::value;

/**
 * @brief Concept to check if vertices can be added to the graph.
 *
 * Requires:
 * - `AddVertex(work_weight, comm_weight, mem_weight)`
 * - Constructibility from `vertex_idx_t` (for reserving size).
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsConstructableCdagVertex : std::false_type {};

template <typename T>
struct IsConstructableCdagVertex<T,
                                 std::void_t<decltype(std::declval<T>().AddVertex(
                                     std::declval<VWorkwT<T>>(), std::declval<VCommwT<T>>(), std::declval<VMemwT<T>>()))>>
    : std::conjunction<IsModifiableCdagVertex<T>, std::is_constructible<T, VertexIdxT<T>>> {};

template <typename T>
inline constexpr bool isConstructableCdagVertexV = IsConstructableCdagVertex<T>::value;


/**
 * @brief Concept for a fully constructable computational DAG.
 *
 * Combines `is_constructable_cdag_vertex` and `is_constructable_cdag_edge`.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsConstructableCdag : std::false_type {};

template <typename T>
struct IsConstructableCdag<T, std::void_t<>>
    : std::conjunction<IsConstructableCdagVertex<T>> {};

template <typename T>
inline constexpr bool isConstructableCdagV = IsConstructableCdag<T>::value;

/**
 * @brief Helper trait to check if a graph can be directly constructed from a vertex count and a set of edges.
 */
template <typename T>
inline constexpr bool isDirectConstructableCdagV
    = std::is_constructible<T, VertexIdxT<T>, std::set<std::pair<VertexIdxT<T>, VertexIdxT<T>>>>::value;

}    // namespace osp
}    // namespace npu::tile_fwk
#endif // OSP_CONSTRUCTABLE_COMPUTATIONAL_DAG_CONCEPT_HPP
