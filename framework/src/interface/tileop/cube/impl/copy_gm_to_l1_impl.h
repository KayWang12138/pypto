/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_gm_to_l1_impl.h
 * \brief DDR->L1数据搬运实现（A3/A5架构）
 */

#ifndef TILEOP_TILE_OPERATOR_TLOAD__H
#define TILEOP_TILE_OPERATOR_TLOAD__H

#include "cube_tools.h"

// Copy data from DDR to L1 with ND -> NZ format
template <PaddingMode padMode, typename T, typename U>
INLINE void TLoadND2NZ(T& dst, U& src, const int64_t& offset0, const int64_t& offset1)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t srcStride0 = GetStride<0>(src);
    int64_t srcStride1 = GetStride<1>(src);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    using tileData = pto::Tile<
        pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, -1, -1,
        pto::SLayout::RowMajor>;
    int64_t gmOffset = offset1 + offset0 * srcShape1;
    constexpr bool isB4 = CheckIsB4<T>();
    if constexpr (isB4) {
        gmOffset = gmOffset >> 1;
    }
    globalData src0Global(
        (__gm__ typename U::Type*)(src.GetAddr() + gmOffset), shapeDim2(staticL1H, staticL1W),
        strideDim2(srcStride0, srcStride1));
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, src0Global);
    if constexpr (padMode != PaddingMode::NO_PADDING) {
        pto::TFILLPAD(dstL1, dstL1);
    }
    return;
}

// Copy data from DDR to L1 with NZ -> NZ format
template <PaddingMode padMode, typename T, typename U>
INLINE void TLoadNZ2NZ(
    T& dst, U& src, const int64_t& offset0, const int64_t& offset1, const int64_t& curH, const int64_t& curW)
{
    constexpr bool isB4 = CheckIsB4<T>();
    constexpr int64_t c0Size = isB4 ? FP4_BLOCK_ALIGN_BYTE : BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    int64_t srcShape0 = curH;
    int64_t srcShape1 = curW;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, -1, -1, BLOCK_CUBE_M_N, c0Size>;
    using strideDim2 = pto::Stride<-1, -1, -1, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim2, strideDim2, pto::Layout::NZ>;
    using tileData = pto::Tile<
        pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, -1, -1,
        pto::SLayout::RowMajor>;
    int64_t gmOffset = CalNZOffset(srcShape0, srcShape1, offset0, offset1, c0Size);
    if constexpr (isB4) {
        gmOffset = gmOffset >> 1;
    }
    globalData src0Global(
        (__gm__ typename U::Type*)(src.GetAddr() + gmOffset), shapeDim2(dstShape1 / c0Size, dstShape0 / BLOCK_CUBE_M_N),
        strideDim2(srcShape0 * srcShape1, srcShape0 * c0Size, BLOCK_CUBE_M_N * c0Size));
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, src0Global);
    if constexpr (padMode != PaddingMode::NO_PADDING) {
        pto::TFILLPAD(dstL1, dstL1);
    }
    return;
}

// Copy data from DDR to L1 with ND -> ND format
template <typename T, typename U>
INLINE void TLoadND2ND(T& dst, U& src, const int64_t& offset0, const int64_t& offset1)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcStride0 = GetStride<0>(src);
    int64_t srcStride1 = GetStride<1>(src);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    // 目前场景,ND2ND只搬运bias和fixpipe，大小均为1 * N，offset0默认均为0
    int64_t gmOffset = offset1 + offset0 * srcShape1;
    globalData src0Global(
        (__gm__ typename U::Type*)(src.GetAddr() + gmOffset), shapeDim2(dstShape0, dstShape1),
        strideDim2(srcStride0, srcStride1));
    using tileData =
        pto::Tile<pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::RowMajor, -1, -1>;
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, src0Global);
    return;
}


#endif // TILEOP_TILE_OPERATOR_TLOAD__H