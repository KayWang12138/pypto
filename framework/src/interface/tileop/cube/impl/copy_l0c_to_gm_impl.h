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
 * \file copy_l0c_to_gm_impl.h
 * \brief L0C->DDR数据搬运实现（A3/A5架构）
 */

#ifndef TILEOP_TILE_OPERATOR_TSTORE__H
#define TILEOP_TILE_OPERATOR_TSTORE__H

#include "utools.h"

template <typename config, typename globalData, typename tileData, typename V>
INLINE void TStoreExecute(globalData dstGlobal, tileData srcL0C, V& fixbuf, uint64_t scaleValue)
{
    if constexpr (
        std::is_same<typename tileData::DType, int32_t>::value &&
        std::is_same<typename globalData::DType, __gm__ half>::value) {
        if (scaleValue != 0) {
            pto::TSTORE<
                tileData, globalData, config::kIsAcc ? pto::AtomicType::AtomicAdd : pto::AtomicType::AtomicNone,
                config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstGlobal, srcL0C, scaleValue);
        } else {
            constexpr auto shapeSize = Std::tuple_size<typename V::Shape>::value;
            constexpr auto tileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename V::TileShape>::type::value;
            constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename V::TileShape>::type::value;
            using fpTileData =
                pto::Tile<pto::TileType::Scaling, uint64_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
            int64_t fixShape0 = GetShape<0>(fixbuf);
            int64_t fixShape1 = GetShape<1>(fixbuf);
            if (fixShape0 == 0 || fixShape1 == 0) {
                return;
            }
            fpTileData fpData(fixShape0, fixShape1);
            pto::TASSIGN(fpData, (uint64_t)fixbuf.GetAddr());
            pto::TSTORE_FP<
                tileData, globalData, fpTileData,
                config::kIsAcc ? pto::AtomicType::AtomicAdd : pto::AtomicType::AtomicNone,
                config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstGlobal, srcL0C, fpData);
        }
    } else {
        pto::TSTORE<
            tileData, globalData, config::kIsAcc ? pto::AtomicType::AtomicAdd : pto::AtomicType::AtomicNone,
            config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(dstGlobal, srcL0C);
    }
}

// Copy data from L0C to DDR with NZ -> ND format
template <typename config, typename T, typename U, typename V>
INLINE void TStoreNZ2ND(
    T& dst, U& src, V& fixbuf, const int64_t& offset0, const int64_t& offset1, uint64_t scaleValue = 0)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t dstStride0 = GetStride<0>(dst);
    int64_t dstStride1 = GetStride<1>(dst);

    constexpr auto tileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;

    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    int64_t gmOffset = offset1 + offset0 * dstShape1;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    using tileData = pto::Tile<
        pto::TileType::Acc, typename U::Type, tileH, tileW, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor,
        pto::TileConfig::fractalCSize, pto::PadValue::Null, pto::CompactMode::Normal>;
    globalData dstGlobal(
        (__gm__ typename T::Type*)(dst.GetAddr() + gmOffset), pto::Shape<1, 1, 1, -1, -1>(srcShape0, srcShape1),
        pto::Stride<1, 1, 1, -1, -1>(dstStride0, dstStride1));
    tileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    TStoreExecute<config, globalData, tileData>(dstGlobal, srcL0C, fixbuf, scaleValue);
    return;
}

// Copy data from L0C to DDR with NZ -> NZ format
template <typename config, typename T, typename U, typename V>
INLINE void TStoreNZ2NZ(
    T& dst, U& src, V& fixbuf, const int64_t& offset0, const int64_t& offset1, const int64_t& curH, const int64_t& curW,
    uint64_t scaleValue = 0)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t c0Size =
        std::is_same<typename U::Type, int32_t>::value ? BLOCK_CUBE_M_N : BLOCK_ALIGN_BYTE / sizeof(typename T::Type);
    int64_t dstShape0 = curH;
    int64_t dstShape1 = curW;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);

    constexpr auto tileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;

    int64_t gmOffset = CalNZOffset(dstShape0, dstShape1, offset0, offset1, c0Size);
    using shapeDim2 = pto::Shape<1, -1, -1, BLOCK_CUBE_M_N, c0Size>;
    using strideDim2 = pto::Stride<-1, -1, -1, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim2, strideDim2, pto::Layout::NZ>;
    globalData dstGlobal(
        (__gm__ typename T::Type*)(dst.GetAddr() + gmOffset), shapeDim2(dstShape1 / c0Size, dstShape0 / BLOCK_CUBE_M_N),
        strideDim2(dstShape0 * dstShape1, dstShape0 * c0Size, BLOCK_CUBE_M_N * c0Size));
    using tileData = pto::Tile<
        pto::TileType::Acc, typename U::Type, tileH, tileW, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor,
        pto::TileConfig::fractalCSize, pto::PadValue::Null, pto::CompactMode::Normal>;
    tileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    TStoreExecute<config, globalData, tileData>(dstGlobal, srcL0C, fixbuf, scaleValue);
    return;
}

// L1 spill
// When L1 space is insufficient, spill to GM. (Supported on A2/A3 only.)
template <typename config, typename Coord, typename T, typename U>
INLINE void TStoreL1SpillImpl(T& dst, U& src, const Coord& coord)
{
    constexpr auto shapeSize = Std::tuple_size<typename U::Shape>::value;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t dstStride0 = GetStride<0>(dst);
    int64_t dstStride1 = GetStride<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    if constexpr (config::kMode == CopyOutMode::ND2ND) {
        using globalData = pto::GlobalTensor<typename T::Type, shapeDim2, strideDim2, pto::Layout::ND>;
        using tileData =
            pto::Tile<pto::TileType::Mat, typename U::Type, staticL1H, staticL1W, pto::BLayout::RowMajor, -1, -1>;
        globalData dstGlobal(
            (__gm__ typename T::Type*)(dst.GetAddr()), shapeDim2(srcShape0, srcShape1),
            strideDim2(dstStride0, dstStride1));
        tileData srcL1(srcShape0, srcShape1);
        pto::TASSIGN(srcL1, (uint64_t)src.GetAddr());
        pto::TSTORE<tileData, globalData>(dstGlobal, srcL1);
    }
    return;
}

#endif // TILEOP_TILE_OPERATOR_TSTORE__H