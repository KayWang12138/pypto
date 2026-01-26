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
 * \file permute.hpp
 * \brief
 */

#ifndef PASS_OSP_PERMUTE_H
#define PASS_OSP_PERMUTE_H

#include <cassert>
#include <type_traits>
#include <utility>
#include <vector>

namespace npu::tile_fwk {
namespace osp {

template <typename T, typename Ind>
void PermuteInplace(std::vector<T> &vec, std::vector<Ind> &perm) {
    static_assert(std::is_integral_v<Ind>);
    static_assert(std::is_unsigned_v<Ind>);

    for (Ind i = 0; i < perm.size(); ++i) {
        while (perm[i] != i) {
            std::swap(vec[i], vec[perm[i]]);
            std::swap(perm[i], perm[perm[i]]);
        }
    }
}

template <typename T, typename Ind>
void InversePermuteInplace(std::vector<T> &vec, std::vector<Ind> &perm) {
    static_assert(std::is_integral_v<Ind>);
    static_assert(std::is_unsigned_v<Ind>);

    for (Ind i = 0; i < perm.size(); ++i) {
        Ind j = i;
        while (i != perm[i]) {
            std::swap(vec[j], vec[perm[i]]);
            j = perm[i];
            std::swap(perm[j], perm[i]);
        }
    }
}

}    // namespace osp
}    // namespace npu::tile_fwk
#endif    // PASS_OSP_PERMUTE_H