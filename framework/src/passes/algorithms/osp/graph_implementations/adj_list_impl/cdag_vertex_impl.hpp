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
 * \file cdag_vertex_impl.hpp
 * \brief
 */

#ifndef PASS_OSP_CDAG_VERTEX_IMPL_HPP
#define PASS_OSP_CDAG_VERTEX_IMPL_HPP

#include <cstddef>    // for std::size_t

namespace npu::tile_fwk {
namespace osp {

/**
 * @brief Implementation of a computational DAG vertex.
 *
 * This struct holds the properties of a vertex in a computational DAG, including its ID,
 * weights (work, communication, memory), and type.
 *
 * @tparam vertex_idx_t Type for vertex indices.
 * @tparam workw_t Type for work weights.
 * @tparam commw_t Type for communication weights.
 * @tparam memw_t Type for memory weights.
 * @tparam vertex_type_t Type for vertex types.
 */
template <typename VertexIdxT, typename WorkwT, typename CommwT, typename MemwT, typename VertexTypeT>
struct CDagVertexImpl {
    using VertexIdxType = VertexIdxT;
    using WorkWeightType = WorkwT;
    using CommWeightType = CommwT;
    using MemWeightType = MemwT;
    using CDagVertexTypeType = VertexTypeT;

    CDagVertexImpl() = default;

    CDagVertexImpl(const CDagVertexImpl &other) = default;
    CDagVertexImpl(CDagVertexImpl &&other) noexcept = default;
    CDagVertexImpl &operator=(const CDagVertexImpl &other) = default;
    CDagVertexImpl &operator=(CDagVertexImpl &&other) noexcept = default;

    /**
     * @brief Constructs a vertex with specified properties.
     *
     * @param vertexIdx The unique identifier for the vertex.
     * @param workW The computational work weight.
     * @param commW The communication weight.
     * @param memW The memory weight.
     * @param vertexT The type of the vertex.
     */
    CDagVertexImpl(VertexIdxT vertexIdx, WorkwT workW, CommwT commW, MemwT memW, VertexTypeT vertexT)
        : id_(vertexIdx), workWeight_(workW), commWeight_(commW), memWeight_(memW), vertexType_(vertexT) {}

    VertexIdxT id_ = 0;

    WorkwT workWeight_ = 0;
    CommwT commWeight_ = 0;
    MemwT memWeight_ = 0;

    VertexTypeT vertexType_ = 0;
};

/**
 * @brief A vertex implementation with integer weights. Indexed by std::size_t. Node types are unsigned.
 *
 * This struct implements a vertex with integer weights for work, communication, and memory.
 */
using CDagVertexImplInt = CDagVertexImpl<std::size_t, int, int, int, unsigned>;

/**
 * @brief A vertex implementation with unsigned weights. Indexed by std::size_t. Node types are unsigned.
 *
 * This struct implements a vertex with unsigned weights for work, communication, and memory.
 */
using CDagVertexImplUnsigned = CDagVertexImpl<std::size_t, unsigned, unsigned, unsigned, unsigned>;

}    // namespace osp
}    // namespace npu::tile_fwk
#endif    // PASS_OSP_CDAG_VERTEX_IMPL_HPP