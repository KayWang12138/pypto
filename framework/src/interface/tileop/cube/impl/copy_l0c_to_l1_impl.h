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
 * \file l0c_to_l1.h
 * \brief 实现L0C->L1大搬小及小搬大（A3/A5架构）
 */

#ifndef TILEOP_TILE_OPERATOR_EXTRACT_L0C2L1__H
#define TILEOP_TILE_OPERATOR_EXTRACT_L0C2L1__H

#include "utools.h"

// create Scale Tile Data
template <typename V>
INLINE auto CreateScaleTileData(V& fixbuf)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename V::Shape>::value;
    constexpr int64_t scaleTileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename V::TileShape>::type::value;
    constexpr int64_t scaleTileW = Std::tuple_element<shapeSize - 1, typename V::TileShape>::type::value;
    int64_t scaleShape0 = GetShape<0>(fixbuf);
    int64_t scaleShape1 = GetShape<1>(fixbuf);
    using scaleTileData =
        pto::Tile<pto::TileType::Scaling, uint64_t, scaleTileH, scaleTileW, pto::BLayout::RowMajor, -1, -1>;
    return scaleTileData(scaleShape0, scaleShape1);
}

// Copy data from L0C to L1(Extract)
template <typename config, typename l1Data, typename l0cData, typename V>
INLINE void TExtractL0CToL1(
    l1Data& dstL1, l0cData& srcL0C, V& fixbuf, uint16_t l0cOffset0, uint16_t l0cOffset1, uint64_t scaleValue = 0)
{
    if constexpr (
        std::is_same<typename l0cData::DType, int32_t>::value && std::is_same<typename l1Data::DType, half>::value) {
        if (scaleValue != 0) {
            constexpr pto::ReluPreMode relu_mode =
                (config::kReluMode == 0) ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu;
            pto::TEXTRACT<l1Data, l0cData, relu_mode>(dstL1, srcL0C, scaleValue, l0cOffset0, l0cOffset1);
        } else {
            auto scaleData = CreateScaleTileData(fixbuf);
            pto::TASSIGN(scaleData, (uint64_t)fixbuf.GetAddr());
            pto::TEXTRACT_FP<
                l1Data, l0cData, decltype(scaleData),
                config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstL1, srcL0C, scaleData, l0cOffset0, l0cOffset1);
        }
    } else {
        pto::TEXTRACT<
            l1Data, l0cData, config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
            dstL1, srcL0C, l0cOffset0, l0cOffset1);
    }
}

// Copy data from L0C to L1(Insert)
template <typename config, typename l1Data, typename l0cData, typename V>
INLINE void TInsertL0CToL1(
    l1Data& dstL1, l0cData& srcL0C, V& fixbuf, uint16_t l1Offset0, uint16_t l1Offset1, uint64_t scaleValue = 0)
{
    if constexpr (
        std::is_same<typename l0cData::DType, int32_t>::value && std::is_same<typename l1Data::DType, half>::value) {
        if (scaleValue != 0) {
            constexpr pto::ReluPreMode relu_mode =
                (config::kReluMode == 0) ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu;
            pto::TINSERT<l1Data, l0cData, relu_mode>(dstL1, srcL0C, scaleValue, l1Offset0, l1Offset1);
        } else {
            auto scaleData = CreateScaleTileData(fixbuf);
            pto::TASSIGN(scaleData, (uint64_t)fixbuf.GetAddr());
            pto::TINSERT_FP<
                l1Data, l0cData, decltype(scaleData),
                config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstL1, srcL0C, scaleData, l1Offset0, l1Offset1);
        }
    } else {
        pto::TINSERT<l1Data, l0cData, config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
            dstL1, srcL0C, l1Offset0, l1Offset1);
    }
}

// Copy data from L0C to L1 with quantization ability
template <typename config, typename Coord, typename T, typename U, typename V>
INLINE void TExtractL0C2L1Impl(T& dst, U& src, V& fixbuf, const Coord& l1Coord, const Coord& l0cCoord, uint64_t scaleValue = 0)
{
    constexpr int64_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    uint16_t l1Offset0 = l1Coord.GetValue();
    uint16_t l1Offset1 = static_cast<const Std::tuple<size_t>&>(l1Coord).GetValue();
    uint16_t l0cOffset0 = l0cCoord.GetValue();
    uint16_t l0cOffset1 = static_cast<const Std::tuple<size_t>&>(l0cCoord).GetValue();
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename T::Type);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    constexpr int64_t tileL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t tileL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    constexpr int64_t tileL0CH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t tileL0CW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    using l1TileData = pto::Tile<
        pto::TileType::Mat, typename T::Type, tileL1H, tileL1W,
        config::kMode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, -1, -1,
        config::kMode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;
    using l0cTileData = pto::Tile<
        pto::TileType::Acc, typename U::Type, tileL0CH, tileL0CW, pto::BLayout::ColMajor, -1, -1,
        pto::SLayout::RowMajor>;
    l1TileData dstL1(dstShape0, dstShape1);
    l0cTileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    if (dstShape0 < srcShape0 || dstShape1 < srcShape1) {
        TExtractL0CToL1<config, l1TileData, l0cTileData, V>(dstL1, srcL0C, fixbuf, l0cOffset0, l0cOffset1, scaleValue);
    } else {
        TInsertL0CToL1<config, l1TileData, l0cTileData, V>(dstL1, srcL0C, fixbuf, l1Offset0, l1Offset1, scaleValue);
    }
    return;
}

#endif // TILEOP_TILE_OPERATOR_EXTRACT_L0C2L1__H