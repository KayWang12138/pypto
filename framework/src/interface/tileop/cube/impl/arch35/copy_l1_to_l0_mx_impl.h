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
 * \file copy_l1_to_l0_mx.h.h
 * \brief MX矩阵乘Scale数据L1->L0提取内部实现（仅A5架构）
 */

#ifndef TILEOP_TILE_OPERATOR_ARCH35_EXTRACT_L12L0_MX__H
#define TILEOP_TILE_OPERATOR_ARCH35_EXTRACT_L12L0_MX__H

#include "../utools.h"

// Copy data from L1 to L0A_MX scale or L0B_MX scale
template <typename Coord, typename T, typename U>
INLINE void TExtractMXImpl(T& dst, U& src, const Coord& coord)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert((T::FORMAT == Hardware::L0A_MX || T::FORMAT == Hardware::L0B_MX) && U::FORMAT == Hardware::L1);
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM3, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM3, 0>(coord);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM3, typename U::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto staticL0H = Std::tuple_element<shapeSize - SHAPE_DIM3, typename T::TileShape>::type::value;
    constexpr auto staticL0W = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    using tileL1Tensor = std::conditional_t<
        T::FORMAT == Hardware::L0A_MX,
        pto::Tile<
            pto::TileType::Mat, typename U::Type, staticL1H, staticL1W * SHAPE_DIM2, pto::BLayout::RowMajor, -1, -1,
            pto::SLayout::RowMajor, pto::TileConfig::alignedSize>,
        pto::Tile<
            pto::TileType::Mat, typename U::Type, staticL1H * SHAPE_DIM2, staticL1W, pto::BLayout::ColMajor, -1, -1,
            pto::SLayout::ColMajor, pto::TileConfig::alignedSize>>;
    using tileL0MXTensor = std::conditional_t<
        T::FORMAT == Hardware::L0A_MX,
        pto::TileLeftScaleCompact<typename T::Type, staticL0H, staticL0W * SHAPE_DIM2, -1, -1>,
        pto::TileRightScaleCompact<typename T::Type, staticL0H * SHAPE_DIM2, staticL0W, -1, -1>>;
    if constexpr (T::FORMAT == Hardware::L0A_MX) {
        tileL1Tensor l1Tile(srcShape0, srcShape1 * SHAPE_DIM2);
        tileL0MXTensor l0MXTile(dstShape0, dstShape1 * SHAPE_DIM2);
        pto::TASSIGN(l1Tile, (uint64_t)src.GetAddr());
        pto::TASSIGN(l0MXTile, (uint64_t)dst.GetAddr());
        pto::TEXTRACT(l0MXTile, l1Tile, offset0, offset1 * SHAPE_DIM2);
    } else {
        tileL1Tensor l1Tile(srcShape0 * SHAPE_DIM2, srcShape1);
        tileL0MXTensor l0MXTile(dstShape0 * SHAPE_DIM2, dstShape1);
        pto::TASSIGN(l1Tile, (uint64_t)src.GetAddr());
        pto::TASSIGN(l0MXTile, (uint64_t)dst.GetAddr());
        pto::TEXTRACT(l0MXTile, l1Tile, offset0 * SHAPE_DIM2, offset1);
    }
}

#endif // TILEOP_TILE_OPERATOR_ARCH35_EXTRACT_L12L0_MX__H