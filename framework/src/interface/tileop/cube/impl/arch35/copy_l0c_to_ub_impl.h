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
 * \file copy_l0c_to_ub_impl.h
 * \brief L0C->UB数据搬运实现（仅A5架构）
 */

#ifndef TILEOP_TILE_OPERATOR_EXTRACT_L0C2UB__H
#define TILEOP_TILE_OPERATOR_EXTRACT_L0C2UB__H

#include "../cube_tools.h"

template <CopyOutMode mode, typename Coord, typename T, typename U>
INLINE void TExtractL0CToUBStatic(T& dst, U& src, const Coord& coord, int16_t subblockId)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t>&>(coord).GetValue();
    constexpr int64_t staticUBH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t staticUBW = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    constexpr int64_t staticL0CH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t staticL0CW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    constexpr int64_t staticSrcShape0 = TileOp::GetTensorShapeDim<U, shapeSize - SHAPE_DIM2>();
    constexpr int64_t staticSrcShape1 = TileOp::GetTensorShapeDim<U, shapeSize - 1>();
    constexpr int64_t staticDstShape0 = TileOp::GetTensorShapeDim<T, shapeSize - SHAPE_DIM2>();
    constexpr int64_t staticDstShape1 = TileOp::GetTensorShapeDim<T, shapeSize - 1>();
    int64_t l0cOffset = CalNZOffset(staticSrcShape0, staticSrcShape1, offset0, offset1, c0Size);

    using tileUBTensor = pto::Tile<
        pto::TileType::Vec, typename T::Type, staticUBH, staticUBW,
        mode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, staticDstShape0, staticDstShape1,
        mode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;
    using tileL0CTensor = pto::TileAcc<typename U::Type, staticL0CH, staticL0CW, staticSrcShape0, staticSrcShape1>;

    tileUBTensor UBTile;
    tileL0CTensor l0cTile;
    pto::TASSIGN(UBTile, (uint64_t)dst.GetAddr());
    pto::TASSIGN(l0cTile, (uint64_t)src.GetAddr() + l0cOffset);
    if (subblockId == 0) {
        pto::TMOV<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec0>(UBTile, l0cTile);
    } else {
        pto::TMOV<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec1>(UBTile, l0cTile);
    }
}

template <CopyOutMode mode, typename Coord, typename T, typename U>
INLINE void TExtractL0CToUBDynamic(T& dst, U& src, const Coord& coord, int16_t subblockId)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t>&>(coord).GetValue();
    constexpr int64_t staticUBH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t staticUBW = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    constexpr int64_t staticL0CH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t staticL0CW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t l0cOffset = CalNZOffset(srcShape0, srcShape1, offset0, offset1, c0Size);

    using tileUBTensor = pto::Tile<
        pto::TileType::Vec, typename T::Type, staticUBH, staticUBW,
        mode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, -1, -1,
        mode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;
    using tileL0CTensor = pto::TileAcc<typename U::Type, staticL0CH, staticL0CW, -1, -1>;

    tileUBTensor UBTile(dstShape0, dstShape1);
    tileL0CTensor l0cTile(srcShape0, srcShape1);
    pto::TASSIGN(UBTile, (uint64_t)dst.GetAddr());
    pto::TASSIGN(l0cTile, (uint64_t)src.GetAddr() + l0cOffset);
    if (subblockId == 0) {
        pto::TMOV<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec0>(UBTile, l0cTile);
    } else {
        pto::TMOV<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec1>(UBTile, l0cTile);
    }
}

#endif // TILEOP_TILE_OPERATOR_EXTRACT_L0C2UB__H