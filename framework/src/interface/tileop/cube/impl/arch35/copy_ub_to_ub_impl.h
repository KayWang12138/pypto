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

#ifndef TILEOP_TILE_OPERATOR_ARCH35_LOAD_UB2UB__H
#define TILEOP_TILE_OPERATOR_ARCH35_LOAD_UB2UB__H

#include "../utools.h"

template <typename T, typename U>
INLINE void TMoveND2NZImpl(T& dst, U& src)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr int64_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t staticNDH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t staticNDW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    constexpr int64_t staticNZH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t staticNZW = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    using tileNDTensor = pto::Tile<
        pto::TileType::Vec, typename U::Type, staticNDH, staticNDW, pto::BLayout::RowMajor, staticNDH, staticNDW>;
    using tileNZTensor = pto::Tile<
        pto::TileType::Vec, typename T::Type, staticNZH, staticNZW, pto::BLayout::ColMajor, staticNZH, staticNZW,
        pto::SLayout::RowMajor>;
    tileNDTensor srcTile;
    tileNZTensor dstTile;
    pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
    pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
    pto::TMOV(dstTile, srcTile);
    return;
}

#endif // TILEOP_TILE_OPERATOR_ARCH35_LOAD_UB2UB__H