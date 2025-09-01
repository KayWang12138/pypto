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
 * \file tile_tensor.h
 * \brief
 */

#ifndef TILEOP_UTILS_TILE_TENSOR_H
#define TILEOP_UTILS_TILE_TENSOR_H

#include "common_type.h"

template <typename T, typename LA, Hardware FMT = Hardware::UB>
struct TileTensor
{
    using Type = T;
    using LayoutType = LA;
    static constexpr Hardware FORMAT = FMT;

    __aicore__ inline TileTensor(__ubuf__ T *addr, LA layout) : addr_(addr), layout_(layout) {}
    __aicore__ inline TileTensor(__ubuf__ T *addr) : addr_(addr) {}
    __aicore__ inline __ubuf__ T *GetAddr() { return addr_; }
    __aicore__ inline LA GetLayout() { return layout_; }
    __aicore__ inline const LA GetLayout() const { return layout_; }
    __aicore__ inline constexpr Hardware GetPhyType() { return FORMAT; }

 private:
    __ubuf__ T *addr_;
    LA layout_;
};

template <typename T, typename LA, Hardware FMT = Hardware::UB>
TILEOP TileTensor<T, LA, FMT> MakeTensor(__ubuf__ T *addr, LA layout) {
    return TileTensor<T, LA, FMT>(addr, layout);
}
#endif // TILEOP_UTILS_TILE_TENSOR_H
