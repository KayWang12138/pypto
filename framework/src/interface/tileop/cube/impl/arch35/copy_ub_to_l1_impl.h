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
 * \file copy_ub_to_ub_impl.h
 * \brief UB->UB数据搬运实现（仅A5架构）
 */

#ifndef TILEOP_TILE_OPERATOR_ARCH35_LOAD_UB2L1__H
#define TILEOP_TILE_OPERATOR_ARCH35_LOAD_UB2L1__H

#include "../utools.h"

template <typename Coord, typename T, typename U>
INLINE void TExtractUB2L1Impl(T& dst, U& src, const Coord& coord)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr bool isB4 = CheckIsB4<T>();
    constexpr int64_t c0Size = isB4 ? FP4_BLOCK_ALIGN_BYTE : BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t>&>(coord).GetValue();
    constexpr int64_t staticUBH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t staticUBW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    constexpr int64_t staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t staticL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;

    using tileUBTensor = pto::Tile<
        pto::TileType::Vec, typename U::Type, staticUBH, staticUBW, pto::BLayout::ColMajor, staticUBH, staticUBW,
        pto::SLayout::RowMajor>;
    using tileL1Tensor = pto::Tile<
        pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, staticL1H, staticL1W,
        pto::SLayout::RowMajor>;
    tileUBTensor UBTile;
    tileL1Tensor l1Tile;
    pto::TASSIGN(UBTile, (uint64_t)src.GetAddr());
    pto::TASSIGN(l1Tile, (uint64_t)dst.GetAddr());
    pto::TEXTRACT(l1Tile, UBTile, offset0, offset1);
    return;
}

#endif // TILEOP_TILE_OPERATOR_ARCH35_LOAD_UB2L1__H