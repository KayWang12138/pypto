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
* \file hash_util.hpp
* \brief
*/

#ifndef OSP_HASH_UTIL_HPP
#define OSP_HASH_UTIL_HPP

#include <cstddef>

namespace npu::tile_fwk {
namespace osp {

template <typename VertexType, std::size_t defautlVal = 11U>
struct UniformNodeHashFunc {
    using ResultType = std::size_t;

    constexpr ResultType operator()(const VertexType &) { return defautlVal; }
};

template <typename VertexType>
struct VectorNodeHashFunc {
    const std::vector<std::size_t> &nodeHashes_;

    VectorNodeHashFunc(const std::vector<std::size_t> &nodeHashes) : nodeHashes_(nodeHashes) {}

    using ResultType = std::size_t;

    ResultType operator()(const VertexType &v) const { return nodeHashes_[v]; }
};

template<class T>
void HashCombine(std::size_t &seed, const T &v) {
    std::hash<T> hasher;
    constexpr std::size_t magicNumber_ = 0x9e3779b9;
    constexpr std::size_t magicNumberTwo_ = 2U;
    constexpr std::size_t magicNumberSix_ = 6U;
    seed ^= hasher(v) + magicNumber_ + (seed << magicNumberSix_) + (seed >> magicNumberTwo_);
}
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_HASH_UTIL_HPP
