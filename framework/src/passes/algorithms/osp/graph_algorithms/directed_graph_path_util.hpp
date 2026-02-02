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
 * \file directed_graph_path_util.hpp
 * \brief
 */

#ifndef PASS_OSP_DIRECTED_GRAPH_PATH_UTIL_HPP
#define PASS_OSP_DIRECTED_GRAPH_PATH_UTIL_HPP

#include <map>
#include <queue>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "passes/algorithms/osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace npu::tile_fwk {
namespace osp {

template <typename GraphT, typename T = unsigned>
std::vector<T> GetBottomNodeDistance(const GraphT &graph) {
    static_assert(std::is_integral_v<T>, "T must be of integral type");

    std::vector<T> bottomDistance(graph.NumVertices(), 0);

    const auto topOrder = GetTopOrder(graph);
    for (auto topRevIt = topOrder.crbegin(); topRevIt != topOrder.crend(); ++topRevIt) {
        T maxTemp = 0;
        for (const auto &j : graph.Children(*topRevIt)) {
            maxTemp = std::max(maxTemp, bottomDistance[j] + 1);
        }
        bottomDistance[*topRevIt] = maxTemp;
    }
    return bottomDistance;
}

template <typename GraphT, typename T = unsigned>
std::vector<T> GetTopNodeDistance(const GraphT &graph) {
    static_assert(std::is_integral_v<T>, "T must be of integral type");

    std::vector<T> topDistance(graph.NumVertices(), 0);

    for (const auto &vertex : GetTopOrder(graph)) {
        T maxTemp = 0;
        for (const auto &j : graph.Parents(vertex)) {
            maxTemp = std::max(maxTemp, topDistance[j] + 1);
        }
        topDistance[vertex] = maxTemp;
    }
    return topDistance;
}

}    // namespace osp
}    // namespace npu::tile_fwk
#endif    // PASS_OSP_DIRECTED_GRAPH_PATH_UTIL_HPP