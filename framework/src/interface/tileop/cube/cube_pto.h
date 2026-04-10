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
 * \file cube_pto.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_CUBE_PTO__H
#define TILEOP_TILE_OPERATOR_CUBE_PTO__H


#include "impl/copy_gm_to_l1_impl.h"
#include "impl/copy_l0c_to_gm_impl.h"
#include "impl/copy_l1_to_bt_fb_impl.h"
#include "impl/load_l1_to_l0_impl.h"
#include "impl/cube_tools.h"
#include "impl/gather_in_l1_impl.h"
#include "impl/copy_l0c_to_l1_impl.h"
#include "impl/mmad_impl.h"

#if defined PTO_NPU_ARCH_A5
#include "impl/arch35/copy_gm_to_l1_mx_impl.h"
#include "impl/arch35/copy_l0c_to_ub_impl.h"
#include "impl/arch35/copy_l1_to_l0_mx_impl.h"
#include "impl/arch35/mmad_mx_impl.h"
#endif

// Copy data from DDR to L1
template <CopyInMode copyMode, PaddingMode padMode, typename Coord, typename T, typename U>
TILEOP void TLoad(T& dst, U& src, const Coord& coord, const int64_t& curH, const int64_t& curW)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t>&>(coord).GetValue();

    static_assert(
        T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::GM,
        "[TLoad Error]: Dst format shoulde be L1 and Src format shoulde be GM");
    if constexpr (copyMode == CopyInMode::ND2NZ) {
        TLoadND2NZ<padMode>(dst, src, offset0, offset1);
    } else if constexpr (copyMode == CopyInMode::NZ2NZ) {
        TLoadNZ2NZ<padMode>(dst, src, offset0, offset1, curH, curW);
    } else if constexpr (copyMode == CopyInMode::ND2ND) {
        TLoadND2ND(dst, src, offset0, offset1);
    }
    return;
}

#if defined PTO_NPU_ARCH_A5
// Copy Scale A data from DDR to L1 for MX matmul
template <CopyInMode mode, typename Coord, typename T, typename U>
TILEOP void TLoadAMX(T& dst, U& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(
        shapeSize == SHAPE_DIM3 && Std::tuple_size<Coord>::value == SHAPE_DIM3,
        "[TLoadAMX Error]: MXMatmul A Scale Shape Size should be 3 Dim");
    static_assert(
        T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::GM,
        "[TLoadAMX Error]: Dst format should be L1 and Src format shoulde be GM");

    if constexpr (TileOp::IsConstContinuousMX<T>()) {
        TLoadAMXStatic<mode, Coord, T, U>(dst, src, coord);
    } else {
        TLoadAMXDynamic<mode, Coord, T, U>(dst, src, coord);
    }
}
#endif

#if defined PTO_NPU_ARCH_A5
// Copy Scale B data from DDR to L1 for MX matmul
template <CopyInMode mode, typename Coord, typename T, typename U>
TILEOP void TLoadBMX(T& dst, U& src, const Coord& coord)
{

    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(
        shapeSize == SHAPE_DIM3 && Std::tuple_size<Coord>::value == SHAPE_DIM3,
        "[TLoadBMX Error]: MXMatmul B Scale Shape Size should be 3 Dim");
    static_assert(
        T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::GM,
        "[TLoadBMX Error]: Dst format shoulde be L1 and Src format shoulde be GM");

    if constexpr (TileOp::IsConstContinuousMX<T>()) {
        TLoadBMXStatic<mode, Coord, T, U>(dst, src, coord);
    } else {
        TLoadBMXDynamic<mode, Coord, T, U>(dst, src, coord);
    }
}
#endif

#if defined PTO_NPU_ARCH_A5
// Copy data from UB to UB with ND -> NZ format
template <typename T, typename U>
TILEOP void TMoveND2NZ(T& dst, U& src)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(T::FORMAT == Hardware::UB && U::FORMAT == Hardware::UB);
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
#endif

#if defined PTO_NPU_ARCH_A5
// Copy data from UB to L1 with NZ -> NZ format
template <typename Coord, typename T, typename U>
TILEOP void TExtract(T& dst, U& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr bool isB4 = CheckIsB4<T>();
    constexpr int64_t c0Size = isB4 ? FP4_BLOCK_ALIGN_BYTE : BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::UB);
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
}
#endif

// Copy data from L0C to L1 with quantization ability
template <typename config, typename Coord, typename T, typename U, typename V>
TILEOP void TExtract(T& dst, U& src, V& fixbuf, const Coord& l1Coord, const Coord& l0cCoord, uint64_t scaleValue = 0)
{
    if (!CheckShapeValid(dst, src) || !CheckShapeValid(dst, fixbuf)) {
        return;
    }
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(U::FORMAT == Hardware::L0C && T::FORMAT == Hardware::L1);
    uint16_t l1Offset0 = l1Coord.GetValue();
    uint16_t l1Offset1 = static_cast<const Std::tuple<size_t>&>(l1Coord).GetValue();
    uint16_t l0cOffset0 = l0cCoord.GetValue();
    uint16_t l0cOffset1 = static_cast<const Std::tuple<size_t>&>(l0cCoord).GetValue();
    if constexpr (TileOp::IsConstContinous<T, U>()) {
        TExtractL0CToL1Static<config, T, U, V>(
            dst, src, fixbuf, l1Offset0, l1Offset1, l0cOffset0, l0cOffset1, scaleValue);
    } else {
        TExtractL0CToL1Dynamic<config, T, U, V>(
            dst, src, fixbuf, l1Offset0, l1Offset1, l0cOffset0, l0cOffset1, scaleValue);
    }
    return;
}

// Copy data from L1 to L0A/L0B
template <bool isTrans, typename Coord, typename T, typename U>
TILEOP void TExtract(T& dst, U& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t>&>(coord).GetValue();
    if constexpr ((T::FORMAT == Hardware::L0A || T::FORMAT == Hardware::L0B) && U::FORMAT == Hardware::L1) {
        TExtractL1ToL0<isTrans>(dst, src, offset0, offset1);
    }
    if constexpr ((T::FORMAT == Hardware::BIAS || T::FORMAT == Hardware::FIXBUF) && U::FORMAT == Hardware::L1) {
        TExtractL1ToBiasTableOrFixBuffer<isTrans>(dst, src);
    }
    return;
}

#if defined PTO_NPU_ARCH_A5
// Copy data from L1 to L0A_MX scale or L0B_MX scale
template <typename Coord, typename T, typename U>
TILEOP void TExtractMX(T& dst, U& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(
        shapeSize == SHAPE_DIM3 && Std::tuple_size<Coord>::value == SHAPE_DIM3,
        "[TExtractMX Error]: L0A_MX scale or L0B_MX scale Shape Size should be 3 Dim");
    static_assert((T::FORMAT == Hardware::L0A_MX || T::FORMAT == Hardware::L0B_MX) && U::FORMAT == Hardware::L1);

    if constexpr (TileOp::IsConstContinuousMX<T, U>()) {
        TExtractMXStatic<T, U, Coord>(dst, src, coord);
    } else {
        TExtractMXDynamic<T, U, Coord>(dst, src, coord);
    }
}
#endif

#if defined PTO_NPU_ARCH_A5
// Copy data from L0C to UB
template <CopyOutMode mode, typename Coord, typename T, typename U>
TILEOP void TExtract(T& dst, U& src, const Coord& coord, int16_t subblockId)
{
    if (!CheckShapeValid(dst, src)){
        return;
    }
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");

    if constexpr (T::FORMAT == Hardware::UB && U::FORMAT == Hardware::L0C) {
        if constexpr (TileOp::IsConstContinous<T, U>()) {
            TExtractL0CToUBStatic<mode, Coord, T, U>(dst, src, coord, subblockId);
        } else {
            TExtractL0CToUBDynamic<mode, Coord, T, U>(dst, src, coord, subblockId);
        }
    }
}
#endif

template <bool isZeroC, TransMode transMode, typename T, typename U, typename V>
TILEOP void TMatmul(T& c, U& a, V& b)
{
    if constexpr (TileOp::IsConstContinous<T, U, V>()) {
        TMatmulStatic<isZeroC, transMode, false>(c, a, b);
    } else {
        TMatmulDynamic<isZeroC, transMode, false>(c, a, b);
    }
}

template <TransMode transMode, typename T0, typename T1, typename T2, typename T3>
TILEOP void TMatmul(T0& c, T1& a, T2& b, T3 bias)
{
    if constexpr (TileOp::IsConstContinous<T0, T1, T2, T3>()) {
        TMatmulStatic<false, transMode, true>(c, a, b, bias);
    } else {
        TMatmulDynamic<false, transMode, true>(c, a, b, bias);
    }
}

#if defined PTO_NPU_ARCH_A5
template <bool isZeroC, typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void MatmulMX(T0& c, T1& a, T2& aScale, T3& b, T4& bScale)
{
    if constexpr (TileOp::IsConstContinous<T0, T1, T2, T3, T4>()) {
        MatmulMXStatic<isZeroC, false>(c, a, aScale, b, bScale);
    } else {
        MatmulMXDynamic<isZeroC, false>(c, a, aScale, b, bScale);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
TILEOP void MatmulMX(T0& c, T1& a, T2& aScale, T3& b, T4& bScale, T5& bias)
{
    if constexpr (TileOp::IsConstContinous<T0, T1, T2, T3, T4, T5>()) {
        MatmulMXStatic<false, true>(c, a, aScale, b, bScale, bias);
    } else {
        MatmulMXDynamic<false, true>(c, a, aScale, b, bScale, bias);
    }
}
#endif

// Copy data from L0C to DDR with quantization ability
template <typename config, typename Coord, typename T, typename U, typename V>
TILEOP void TStore(
    T& dst, U& src, V& fixbuf, const Coord& coord, const int64_t& curH, const int64_t& curW, uint64_t scaleValue = 0)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t>&>(coord).GetValue();
    if constexpr (U::FORMAT == Hardware::L0C && T::FORMAT == Hardware::GM) {
        if constexpr (config::kMode == CopyOutMode::NZ2ND) {
            TStoreNZ2ND<config>(dst, src, fixbuf, offset0, offset1, scaleValue);
        } else {
            TStoreNZ2NZ<config>(dst, src, fixbuf, offset0, offset1, curH, curW, scaleValue);
        }
    }
}

// L1 spill
// When L1 space is insufficient, spill to GM. (Supported on A2/A3 only.)
template <typename config, typename Coord, typename T, typename U>
TILEOP void TStore(T& dst, U& src, const Coord& coord)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename U::Shape>::value;
    constexpr int64_t staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t staticL1W = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t dstStride0 = GetStride<0>(dst);
    int64_t dstStride1 = GetStride<1>(dst);
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    if constexpr (config::kMode == CopyOutMode::ND2ND) {
        if constexpr (TileOp::IsConstContinous<U>()) {
            constexpr int64_t staticSrcShape0 = TileOp::GetTensorShapeDim<U, shapeSize - SHAPE_DIM2>();
            constexpr int64_t staticSrcShape1 = TileOp::GetTensorShapeDim<U, shapeSize - 1>();
            using globalData = pto::GlobalTensor<typename T::Type, shapeDim2, strideDim2, pto::Layout::ND>;
            using tileData = pto::Tile<
                pto::TileType::Mat, typename U::Type, staticL1H, staticL1W, pto::BLayout::RowMajor, staticSrcShape0,
                staticSrcShape1>;
            globalData dstGlobal(
                (__gm__ typename T::Type*)(dst.GetAddr()), shapeDim2(staticSrcShape0, staticSrcShape1),
                strideDim2(dstStride0, dstStride1));
            tileData srcL1;
            pto::TASSIGN(srcL1, (uint64_t)src.GetAddr());
            pto::TSTORE<tileData, globalData>(dstGlobal, srcL1);
        } else {
            int64_t srcShape0 = GetShape<0>(src);
            int64_t srcShape1 = GetShape<1>(src);
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
    }
}

template <
    int64_t blockSize, typename DstT, typename SrcT, typename BlockT, typename OffsetT, typename SrcCoord,
    typename OffsetCoord, typename BlockCoord>
TILEOP void TGatherInL1(
    DstT dst, SrcT src, BlockT block, OffsetT offset, SrcCoord srcCoord, OffsetCoord offsetCoord, BlockCoord blockCoord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint16_t shapeSize = Std::tuple_size<typename DstT::Shape>::value;

    int64_t srcShape1 = GetShape<1>(src);
    int64_t srcStride0 = GetStride<0>(src);
    int64_t srcStride1 = GetStride<1>(src);
    int64_t blockShape1 = GetShape<1>(block);
    int64_t offsetShape1 = GetShape<1>(offset);
    uint64_t srcColumnStartOffset = srcCoord.GetValue();
    uint64_t offsetsRowStartOffset = offsetCoord.GetValue();
    uint64_t offsetsColumnStartOffset = static_cast<const Std::tuple<size_t>&>(offsetCoord).GetValue();
    uint64_t offsetsStartOffset = offsetsRowStartOffset * offsetShape1 + offsetsColumnStartOffset;
    uint64_t GMBlockTableOffset0 = blockCoord.GetValue();
    uint64_t GMBlockTableOffset1 = static_cast<const Std::tuple<size_t>&>(blockCoord).GetValue();
    uint64_t GMBlockTableOffset = GMBlockTableOffset0 * blockShape1 + GMBlockTableOffset1;

    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    using globalData = pto::GlobalTensor<typename SrcT::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    if constexpr (TileOp::IsConstContinous<DstT>()) {
        TGatherInL1Static<blockSize, DstT, SrcT, BlockT, OffsetT, globalData>(
            dst, src, block, offset, srcShape1, srcStride0, srcStride1, srcColumnStartOffset, offsetsStartOffset,
            GMBlockTableOffset);
    } else {
        TGatherInL1Dynamic<blockSize, DstT, SrcT, BlockT, OffsetT, globalData>(
            dst, src, block, offset, srcShape1, srcStride0, srcStride1, srcColumnStartOffset, offsetsStartOffset,
            GMBlockTableOffset);
    }
}

#endif // TILEOP_TILE_OPERATOR_CUBE_PTO__H