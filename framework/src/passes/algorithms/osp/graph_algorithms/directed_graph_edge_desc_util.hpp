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
 * \file directed_graph_edge_desc_util.hpp
 * \brief
 */

#ifndef OSP_DIRECTED_GRAPH_EDGE_DESC_UTIL_HPP
#define OSP_DIRECTED_GRAPH_EDGE_DESC_UTIL_HPP

#include <queue>
#include <unordered_set>
#include <vector>

#include "passes/algorithms/osp/concepts/directed_graph_edge_desc_concept.hpp"

namespace npu::tile_fwk {
namespace osp {

template <typename GraphT>
std::pair<EdgeDescT<GraphT>, bool> EdgeDesc(const VertexIdxT<GraphT> &src, const VertexIdxT<GraphT> &dest, const GraphT &graph) {
    for (const auto &edge : OutEdges(src, graph)) {
        if (Target(edge, graph) == dest) {
            return {edge, true};
        }
    }
    return {EdgeDescT<GraphT>(), false};
}

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_DIRECTED_GRAPH_EDGE_DESC_UTIL_HPP
