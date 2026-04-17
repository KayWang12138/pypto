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
 * \file copy_l1_to_l0_impl.h
 * \brief L1->L0数据搬运实现（A3/A5架构）
 */

#ifndef TILEOP_TILE_OPERATOR_EXTRACT_L12L0__H
#define TILEOP_TILE_OPERATOR_EXTRACT_L12L0__H

#include "utools.h"

template <bool isTrans, typename T, typename U>
INLINE void TExtractL1ToL0Impl(T& dst, U& src, const int64_t& offset0, const int64_t& offset1)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    constexpr auto staticL0H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr auto staticL0W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    // L1 Tile内模板参数对应含义：
    // Tile类型为Cube用于Matmul，矩阵数据类型，TileShape0，TileShape1，大分型RowMajor表明Z，ColMajor表明N,
    // validShape0, validShape1, 小分型
    using tileL1Tensor = pto::Tile<
        pto::TileType::Mat, typename U::Type, isTrans ? staticL1W : staticL1H, isTrans ? staticL1H : staticL1W,
        isTrans ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, -1, -1,
        isTrans ? pto::SLayout::ColMajor : pto::SLayout::RowMajor>;
    // L0 TileLeft为L0A的Tile，TileRight为L0B的Tile，传入的值分别为：
    // 矩阵数据类型，tileShape0，tileShape1，validShape0，validShape0（-1表明传递动态值，在声明时传入）
    using tileL0Tensor = std::conditional_t<
        T::FORMAT == Hardware::L0A, pto::TileLeftCompact<typename T::Type, staticL0H, staticL0W, -1, -1>,
        pto::TileRightCompact<typename T::Type, staticL0H, staticL0W, -1, -1>>;
    tileL1Tensor l1Tile(srcShape0, srcShape1);
    tileL0Tensor l0Tile(dstShape0, dstShape1);
    if (std::is_same<typename tileL0Tensor::DType, float>::value && T::FORMAT == Hardware::L0A) {
        l0Tile.SetKAligned(true);
    }
    pto::TASSIGN(l1Tile, (uint64_t)src.GetAddr());
    pto::TASSIGN(l0Tile, (uint64_t)dst.GetAddr());
    pto::TEXTRACT(l0Tile, l1Tile, isTrans ? offset1 : offset0, isTrans ? offset0 : offset1);
    return;
}

#endif // TILEOP_TILE_OPERATOR_EXTRACT_L12L0__H