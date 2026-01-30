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
* \file graph_traits.hpp
* \brief

*/

#ifndef OSP_GRAPH_TRAITS_HPP
#define OSP_GRAPH_TRAITS_HPP

#include "iterator_concepts.hpp"

#include "interface/utils/common.h"

/**
 * @file graph_traits.hpp
 * @brief Type traits and concepts for graph structures in OneStopParallel.
 *
 * This file defines the core requirements for types used by graph implementations in the library,
 * specifically for computational DAGs. It provides mechanisms
 * to extract types for vertex indices, edge descriptors, and weights,
 * ensuring that graph implementations conform to the expected interfaces.
 */

namespace npu::tile_fwk {
namespace osp {

/**
 * @brief Traits to check for the existence of specific type members.
 *
 * These structs inherit from `std::true_type` if the specified member type exists in `T`,
 * otherwise they inherit from `std::false_type`.
 */
template <typename T, typename = void>
struct HasVertexIdxTmember : std::false_type {};

template <typename T>
struct HasVertexIdxTmember<T, std::void_t<typename T::VertexIdx>> : std::true_type {};

template <typename T, typename = void>
struct HasEdgeDescTmember : std::false_type {};

template <typename T>
struct HasEdgeDescTmember<T, std::void_t<typename T::DirectedEdgeDescriptor>> : std::true_type {};

template <typename T, typename = void>
struct HasVertexWorkWeightTmember : std::false_type {};

template <typename T>
struct HasVertexWorkWeightTmember<T, std::void_t<typename T::VertexWorkWeightType>> : std::true_type {};

template <typename T, typename = void>
struct HasVertexCommWeightTmember : std::false_type {};

template <typename T>
struct HasVertexCommWeightTmember<T, std::void_t<typename T::VertexCommWeightType>> : std::true_type {};

template <typename T, typename = void>
struct HasVertexMemWeightTmember : std::false_type {};

template <typename T>
struct HasVertexMemWeightTmember<T, std::void_t<typename T::VertexMemWeightType>> : std::true_type {};

template <typename T, typename = void>
struct HasVertexTypeTmember : std::false_type {};

template <typename T>
struct HasVertexTypeTmember<T, std::void_t<typename T::VertexTypeType>> : std::true_type {};

template <typename T, typename = void>
struct HasEdgeCommWeightTmember : std::false_type {};

template <typename T>
struct HasEdgeCommWeightTmember<T, std::void_t<typename T::EdgeCommWeightType>> : std::true_type {};

/**
 * @brief Core traits for any directed graph type.
 *
 * Requires that the graph type `T` defines a `VertexIdx` type member.
 *
 * @tparam T The graph type.
 */
template <typename T>
struct DirectedGraphTraits {
    static_assert(HasVertexIdxTmember<T>::value, "graph must have VertexIdx");
    using VertexIdx = typename T::VertexIdx;
};

/**
 * @brief Alias to easily access the vertex index type of a graph.
 */
template <typename T>
using VertexIdxT = typename DirectedGraphTraits<T>::VertexIdx;

/**
 * @brief A default edge descriptor for directed graphs.
 *
 * This struct is used when the graph type does not provide its own edge descriptor.
 * It simply holds the source and target vertex indices.
 *
 * @tparam GraphT The graph type.
 */
template <typename GraphT>
struct DirectedEdge {
    VertexIdxT<GraphT> source_;
    VertexIdxT<GraphT> target_;

    bool operator==(const DirectedEdge &other) const { return source_ == other.source_ && target_ == other.target_; }

    bool operator!=(const DirectedEdge &other) const { return !(*this == other); }

    DirectedEdge() : source_(0), target_(0) {}

    DirectedEdge(const DirectedEdge &other) = default;
    DirectedEdge(DirectedEdge &&other) = default;
    DirectedEdge &operator=(const DirectedEdge &other) = default;
    DirectedEdge &operator=(DirectedEdge &&other) = default;
    ~DirectedEdge() = default;

    DirectedEdge(VertexIdxT<GraphT> src, VertexIdxT<GraphT> tgt) : source_(src), target_(tgt) {}
};

/**
 * @brief Helper struct to extract the edge descriptor type of a directed graph.
 *
 * If the graph defines `directed_edge_descriptor`, it is extracted; otherwise, `directed_edge` is used as a default implementation.
 */
template <typename T, bool hasEdge>
struct DirectedGraphEdgeDescTraitsHelper {
    using DirectedEdgeDescriptor = DirectedEdge<T>;
};

template <typename T>
struct DirectedGraphEdgeDescTraitsHelper<T, true> {
    using DirectedEdgeDescriptor = typename T::DirectedEdgeDescriptor;
};

template <typename T>
struct DirectedGraphEdgeDescTraits {
    using DirectedEdgeDescriptor =
        typename DirectedGraphEdgeDescTraitsHelper<T, HasEdgeDescTmember<T>::value>::DirectedEdgeDescriptor;
};

template <typename T>
using EdgeDescT = typename DirectedGraphEdgeDescTraits<T>::DirectedEdgeDescriptor;

/**
 * @brief Traits for computational Directed Acyclic Graphs (DAGs).
 *
 * Computational DAGs extend basic graphs by adding requirements for weight types:
 * - `VertexWorkWeightType`: Represents computational cost of a task.
 * - `VertexCommWeightType`: Represents data size/communication cost.
 * - `VertexMemWeightType`: Represents memory usage of a task.
 *
 * @tparam T The computational DAG type.
 */
template <typename T>
struct ComputationalDagTraits {
    static_assert(HasVertexWorkWeightTmember<T>::value, "cdag must have vertex work weight type");
    static_assert(HasVertexCommWeightTmember<T>::value, "cdag must have vertex comm weight type");
    static_assert(HasVertexMemWeightTmember<T>::value, "cdag must have vertex mem weight type");

    using VertexWorkWeightType = typename T::VertexWorkWeightType;
    using VertexCommWeightType = typename T::VertexCommWeightType;
    using VertexMemWeightType = typename T::VertexMemWeightType;
};

template <typename T>
using VWorkwT = typename ComputationalDagTraits<T>::VertexWorkWeightType;

template <typename T>
using VCommwT = typename ComputationalDagTraits<T>::VertexCommWeightType;

template <typename T>
using VMemwT = typename ComputationalDagTraits<T>::VertexMemWeightType;

/**
 * @brief Traits to extract the vertex type of a computational DAG, if defined.
 *
 * If the DAG defines `VertexTypeType`, it is extracted; otherwise, `void` is used.
 */
template <typename T, typename = void>
struct ComputationalDagTypedVerticesTraits {
    using VertexTypeType = void;
};

template <typename T>
struct ComputationalDagTypedVerticesTraits<T, std::void_t<typename T::VertexTypeType>> {
    using VertexTypeType = typename T::VertexTypeType;
};

template <typename T>
using VTypeT = typename ComputationalDagTypedVerticesTraits<T>::VertexTypeType;

/**
 * @brief Traits to extract the edge communication weight type of a computational DAG, if defined.
 *
 * If the DAG defines `edge_comm_weight_type`, it is extracted; otherwise, `void` is used.
 */
template <typename T, typename = void>
struct ComputationalDagEdgeDescTraits {
    using EdgeCommWeightType = void;
};

template <typename T>
struct ComputationalDagEdgeDescTraits<T, std::void_t<typename T::EdgeCommWeightType>> {
    using EdgeCommWeightType = typename T::EdgeCommWeightType;
};

template <typename T>
using ECommwT = typename ComputationalDagEdgeDescTraits<T>::EdgeCommWeightType;

// -----------------------------------------------------------------------------
// Property Traits
// -----------------------------------------------------------------------------

/**
 * @brief Check if a graph guarantees vertices are stored/iterated in topological order.
 * It allows a graph implementation to notify algorithms that vertices are stored/iterated in topological order which can be used
 * to optimize the algorithm.
 */
template <typename T, typename = void>
struct HasVerticesInTopOrderTrait : std::false_type {};

template <typename T>
struct HasVerticesInTopOrderTrait<T, std::void_t<decltype(T::verticesInTopOrder_)>>
    : std::bool_constant<std::is_same_v<decltype(T::verticesInTopOrder_), const bool> && T::verticesInTopOrder_> {};

template <typename T>
inline constexpr bool hasVerticesInTopOrderV = HasVerticesInTopOrderTrait<T>::value;

}    // namespace osp
}    // namespace npu::tile_fwk

/**
 * @brief Specialization of std::hash for osp::directed_edge.
 *
 * This specialization provides a hash function for osp::directed_edge, which is used in hash-based containers like
 * std::unordered_set and std::unordered_map.
 */
template <typename GraphT>
struct std::hash<npu::tile_fwk::osp::DirectedEdge<GraphT>> {
    std::size_t operator()(const npu::tile_fwk::osp::DirectedEdge<GraphT> &p) const noexcept {
        // Combine hashes of source and target
        std::size_t h1 = std::hash<npu::tile_fwk::osp::VertexIdxT<GraphT>>{}(p.source_);
        std::size_t h2 = std::hash<npu::tile_fwk::osp::VertexIdxT<GraphT>>{}(p.target_);
        npu::tile_fwk::HashCombine(h1, h2);
        return h1;
    }
};
#endif // OSP_GRAPH_TRAITS_HPP