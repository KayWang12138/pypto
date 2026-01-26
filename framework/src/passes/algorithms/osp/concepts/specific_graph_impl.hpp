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
 * \file specific_graph_impl.hpp
 * \brief
 */
/**
 * @file specific_graph_impl.hpp
 * @brief Type traits for specific graph implementations used in OneStopParallel.
 *
 * This file contains trait checks for specific graph implementations, such as
 * Compact Sparse Graphs (CSR-like structures), which may require specialized
 * handling or offer optimizations in certain algorithms.
 */

#ifndef OSP_SPECIFIC_GRAPH_IMPL_HPP
#define OSP_SPECIFIC_GRAPH_IMPL_HPP

namespace npu::tile_fwk {
namespace osp {

/**
 * @brief Trait to check if a graph type is a `CompactSparseGraph`.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsCompactSparseGraph : std::false_type {};

template <typename T>
inline constexpr bool isCompactSparseGraphV = IsCompactSparseGraph<T>::value;

/**
 * @brief Trait to check if a graph type is a `CompactSparseGraph` that supports reordering.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsCompactSparseGraphReorder : std::false_type {};

template <typename T>
inline constexpr bool isCompactSparseGraphReorderV = IsCompactSparseGraphReorder<T>::value;

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_SPECIFIC_GRAPH_IMPL_HPP
