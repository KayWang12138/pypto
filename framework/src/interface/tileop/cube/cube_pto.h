/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cube_pto.h
 * \brief TileOp Interface Definition
 */

#ifndef TILEOP_TILE_OPERATOR_CUBE_PTO__H
#define TILEOP_TILE_OPERATOR_CUBE_PTO__H

// Common Operator Definitions (Shared by Atlas A3, Ascend 950PR/Ascend 950DT)
#include "impl/copy_gm_to_l1_impl.h"
#include "impl/copy_l0c_to_gm_impl.h"
#include "impl/copy_l0c_to_l1_impl.h"
#include "impl/copy_l1_to_bt_fb_impl.h"
#include "impl/copy_l1_to_l0_impl.h"
#include "impl/cube_utils.h"
#include "impl/gather_in_l1_impl.h"
#include "impl/mmad_impl.h"

// Operator Header File for Ascend 950PR/Ascend 950DT Architectures, Enabled Only When PTO_NPU_ARCH_A5 Macro is Defined.
#if defined PTO_NPU_ARCH_A5
#include "impl/arch35/copy_gm_to_l1_mx_impl.h"
#include "impl/arch35/copy_l0c_to_ub_impl.h"
#include "impl/arch35/copy_l1_to_l0_mx_impl.h"
#include "impl/arch35/copy_ub_to_l1_impl.h"
#include "impl/arch35/copy_ub_to_ub_impl.h"
#include "impl/arch35/mmad_mx_impl.h"
#endif

// TileOp Definitions for Matrix Multiplication & Data Movement on Ascend 950PR/Ascend 950DT Architectures
#if defined PTO_NPU_ARCH_A5
// Copy Scale A data from DDR to L1 for MX matmul
template <CopyInMode mode, typename Coord, typename TileData, typename GlobalData>
TILEOP void TLoadAMX(TileData& dst, GlobalData& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    static_assert(
        shapeSize == SHAPE_DIM3 && Std::tuple_size<Coord>::value == SHAPE_DIM3,
        "[TLoadAMX Error]: MXMatmul A Scale Shape Size should be 3 Dim");
    static_assert(
        TileData::FORMAT == Hardware::L1 && GlobalData::FORMAT == Hardware::GM,
        "[TLoadAMX Error]: Dst format should be L1 and Src format should be GM");
    TLoadAMXImpl<mode, Coord, TileData, GlobalData>(dst, src, coord);
}

// Copy Scale B data from DDR to L1 for MX matmul
template <CopyInMode mode, typename Coord, typename TileData, typename GlobalData>
TILEOP void TLoadBMX(TileData& dst, GlobalData& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    static_assert(
        shapeSize == SHAPE_DIM3 && Std::tuple_size<Coord>::value == SHAPE_DIM3,
        "[TLoadBMX Error]: MXMatmul B Scale Shape Size should be 3 Dim");
    static_assert(
        TileData::FORMAT == Hardware::L1 && GlobalData::FORMAT == Hardware::GM,
        "[TLoadBMX Error]: Dst format should be L1 and Src format should be GM");
    TLoadBMXImpl<mode, Coord, TileData, GlobalData>(dst, src, coord);
}

// Copy data from UB to UB with ND -> NZ format
template <typename DstTileData, typename SrcTileData>
TILEOP void TMoveND2NZ(DstTileData& dst, SrcTileData& src)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(DstTileData::FORMAT == Hardware::UB && SrcTileData::FORMAT == Hardware::UB);
    TMoveND2NZImpl<DstTileData, SrcTileData>(dst, src);
}

// Copy data from UB to L1 with NZ -> NZ format
template <typename Coord, typename DstTileData, typename SrcTileData>
TILEOP void TExtract(DstTileData& dst, SrcTileData& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(DstTileData::FORMAT == Hardware::L1 && SrcTileData::FORMAT == Hardware::UB);
    TExtractUB2L1Impl<Coord, DstTileData, SrcTileData>(dst, src, coord);
}

// Copy data from L1 to L0A_MX scale or L0B_MX scale
template <typename Coord, typename DstTileData, typename SrcTileData>
TILEOP void TExtractMX(DstTileData& dst, SrcTileData& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(
        shapeSize == SHAPE_DIM3 && Std::tuple_size<Coord>::value == SHAPE_DIM3,
        "[TExtractMX Error]: L0A_MX scale or L0B_MX scale Shape Size should be 3 Dim");
    static_assert(
        (DstTileData::FORMAT == Hardware::L0A_MX || DstTileData::FORMAT == Hardware::L0B_MX) &&
        SrcTileData::FORMAT == Hardware::L1);
    TExtractMXImpl<Coord, DstTileData, SrcTileData>(dst, src, coord);
}

// Copy data from L0C to UB
template <CopyOutMode mode, typename Coord, typename DstTileData, typename SrcTileData>
TILEOP void TExtract(DstTileData& dst, SrcTileData& src, const Coord& coord, int16_t subblockId)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    TExtractL0C2UBImpl<mode, Coord, DstTileData, SrcTileData>(dst, src, coord, subblockId);
}

template <
    bool isZeroC, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
    typename TileRightScale>
TILEOP void MatmulMX(TileRes& c, TileLeft& a, TileLeftScale& aScale, TileRight& b, TileRightScale& bScale)
{
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileRes::Shape>::value;
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeAScale = Std::tuple_size<typename TileLeftScale::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeBScale = Std::tuple_size<typename TileRightScale::Shape>::value;
    static_assert(
        shapeSizeC == SHAPE_DIM2 && shapeSizeA == SHAPE_DIM2 && shapeSizeAScale == SHAPE_DIM3 &&
            shapeSizeB == SHAPE_DIM2 && shapeSizeBScale == SHAPE_DIM3,
        "[MatmulMX ERROR]: Tensor Shape dim size should be 2 and Scale Shape dim size should be 3");
    MatmulMXImpl<isZeroC>(c, a, aScale, b, bScale);
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename TileBias>
TILEOP void MatmulMX(
    TileRes& c, TileLeft& a, TileLeftScale& aScale, TileRight& b, TileRightScale& bScale, TileBias& bias)
{
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileRes::Shape>::value;
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeAScale = Std::tuple_size<typename TileLeftScale::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeBScale = Std::tuple_size<typename TileRightScale::Shape>::value;
    static_assert(
        shapeSizeC == SHAPE_DIM2 && shapeSizeA == SHAPE_DIM2 && shapeSizeAScale == SHAPE_DIM3 &&
            shapeSizeB == SHAPE_DIM2 && shapeSizeBScale == SHAPE_DIM3,
        "[MatmulMX ERROR]: Shape dim size should be 2 and Scale Shape dim size should be 3");
    MatmulMXImpl(c, a, aScale, b, bScale, bias);
}
// End of TileOp Interface Definitions for Ascend 950PR/Ascend 950DT Architecture
#endif

// Common Operator TileOp Interface Definitions

// Copy data from DDR to L1 with ND -> NZ format
template <PaddingMode padMode, typename TileData, typename GlobalData>
INLINE void TLoadND2NZ(TileData& dst, GlobalData& src, const int64_t& offset0, const int64_t& offset1)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t srcStride0 = GetStride<0>(src);
    int64_t srcStride1 = GetStride<1>(src);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileData::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename TileData::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    using globalData = pto::GlobalTensor<typename GlobalData::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    using tileData = pto::Tile<
        pto::TileType::Mat, typename TileData::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, -1, -1,
        pto::SLayout::RowMajor, pto::TileConfig::fractalABSize, pto::PadValue::Null,
        pto::CompactMode::RowAlignedPadding>;
    int64_t gmOffset = offset1 + offset0 * srcShape1;
    // FP4数据类型数据宽度减半，地址偏移需要右移1位
    if constexpr (CheckIsB4<TileData>()) {
        gmOffset = gmOffset >> 1;
    }
    globalData src0Global(
        (__gm__ typename GlobalData::Type*)(src.GetAddr() + gmOffset), shapeDim2(dstShape0, dstShape1),
        strideDim2(srcStride0, srcStride1));
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, static_cast<uint64_t>(dst.GetAddr()));
    pto::TLOAD(dstL1, src0Global);
    // L1数据为NZ时，外轴非16元素对齐需要pad到16对齐
    if ((dstShape0 & 0xF) != 0) {
        pto::TFILLPAD(dstL1, dstL1);
    }
}

// Copy data from DDR to L1 with NZ -> NZ format
template <PaddingMode padMode, typename T, typename U>
INLINE void TLoadNZ2NZ(
    T& dst, U& src, const int64_t& dstOffset0, const int64_t& dstOffset1, const int64_t& srcOffset0,
    const int64_t& srcOffset1, const int64_t& curH, const int64_t& curW)
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
    int64_t gmOffset = CalNZOffset(srcShape0, srcShape1, srcOffset0, srcOffset1, c0Size);
    int64_t l1Offset = CalNZOffset(dstShape0, dstShape1, dstOffset0, dstOffset1, c0Size);
    if constexpr (isB4) {
        gmOffset = gmOffset >> 1;
    }
    globalData src0Global(
        (__gm__ typename U::Type*)(src.GetAddr() + gmOffset), shapeDim2(dstShape1 / c0Size, dstShape0 / BLOCK_CUBE_M_N),
        strideDim2(srcShape0 * srcShape1, srcShape0 * c0Size, BLOCK_CUBE_M_N * c0Size));
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)((typename T::Type*)dst.GetAddr() + l1Offset));
    pto::TLOAD(dstL1, src0Global);
    if constexpr (padMode != PaddingMode::NO_PADDING) {
        pto::TFILLPAD(dstL1, dstL1);
    }
    return;
}

// Copy data from DDR to L1
template <CopyInMode copyMode, PaddingMode padMode, typename Coord, typename T, typename U>
TILEOP void TLoad(
    T& dst, U& src, const Coord& dstCoord, const Coord& srcCoord, const int64_t& curH, const int64_t& curW)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    int64_t dstOffset0 = TileOp::GetTupleElement<Coord, DIM_1ST, shapeSize, 0>(dstCoord);
    int64_t dstOffset1 = TileOp::GetTupleElement<Coord, DIM_2ND, shapeSize, 0>(dstCoord);
    int64_t srcOffset0 = TileOp::GetTupleElement<Coord, DIM_1ST, shapeSize, 0>(srcCoord);
    int64_t srcOffset1 = TileOp::GetTupleElement<Coord, DIM_2ND, shapeSize, 0>(srcCoord);
    static_assert(
        T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::GM,
        "[TLoad Error]: Dst format shoulde be L1 and Src format shoulde be GM");
    if constexpr (copyMode == CopyInMode::ND2NZ) {
        TLoadND2NZ<padMode>(dst, src, srcOffset0, srcOffset1);
    } else if constexpr (copyMode == CopyInMode::NZ2NZ) {
        TLoadNZ2NZ<padMode>(dst, src, dstOffset0, dstOffset1, srcOffset0, srcOffset1, curH, curW);
    } else if constexpr (copyMode == CopyInMode::ND2ND) {
        TLoadND2ND(dst, src, srcOffset0, srcOffset1);
    }
    return;
}


// Copy data from L0C to L1 with quantization ability
template <typename config, typename Coord, typename DstTileData, typename SrcTileData, typename FpTileData>
TILEOP void TExtract(
    DstTileData& dst, SrcTileData& src, FpTileData& fixbuf, const Coord& l1Coord, const Coord& l0cCoord,
    uint64_t scaleValue = 0)
{
    if (!CheckShapeValid(dst, src) || !CheckShapeValid(dst, fixbuf)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(SrcTileData::FORMAT == Hardware::L0C && DstTileData::FORMAT == Hardware::L1);
    TExtractL0C2L1Impl<config>(dst, src, fixbuf, l1Coord, l0cCoord, scaleValue);
}

// Copy data from L1 to L0A/L0B
template <bool isTrans, typename Coord, typename DstTileData, typename SrcTileData>
TILEOP void TExtract(DstTileData& dst, SrcTileData& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(coord);
    if constexpr (
        (DstTileData::FORMAT == Hardware::L0A || DstTileData::FORMAT == Hardware::L0B) &&
        SrcTileData::FORMAT == Hardware::L1) {
        TExtractL1ToL0Impl<isTrans>(dst, src, offset0, offset1);
    }
    if constexpr (
        (DstTileData::FORMAT == Hardware::BIAS || DstTileData::FORMAT == Hardware::FIXBUF) &&
        SrcTileData::FORMAT == Hardware::L1) {
        TExtractL1ToBTOrFBImpl<isTrans>(dst, src);
    }
}

template <bool isZeroC, TransMode transMode, typename TileAcc, typename TileLeft, typename TileRight>
TILEOP void TMatmul(TileAcc& c, TileLeft& a, TileRight& b)
{
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    static_assert(
        shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
        "[Matmul ERROR]: Shape dim size should be 2");
    TMatmulImpl<isZeroC, transMode>(c, a, b);
}

template <TransMode transMode, typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
TILEOP void TMatmul(TileAcc& c, TileLeft& a, TileRight& b, TileBias& bias)
{
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    static_assert(
        shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
        "[Matmul ERROR]: Shape dim size should be 2");
    TMatmulImpl<transMode>(c, a, b, bias);
}

// Copy data from L0C to DDR with quantization ability
template <typename config, typename Coord, typename GlobalData, typename TileData, typename FpTileData>
TILEOP void TStore(
    GlobalData& dst, TileData& src, FpTileData& fixbuf, const Coord& coord, const int64_t& curH, const int64_t& curW,
    uint64_t scaleValue = 0)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint64_t shapeSize = Std::tuple_size<typename GlobalData::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(coord);
    if constexpr (TileData::FORMAT == Hardware::L0C && GlobalData::FORMAT == Hardware::GM) {
        if constexpr (config::kMode == CopyOutMode::NZ2ND) {
            TStoreNZ2ND<config>(dst, src, fixbuf, offset0, offset1, scaleValue);
        } else {
            TStoreNZ2NZ<config>(dst, src, fixbuf, offset0, offset1, curH, curW, scaleValue);
        }
    }
}

// L1 spill(Only used in deepseek model)
// When L1 space is insufficient, spill to GM. (Supported on A2/A3 only.)
template <typename config, typename Coord, typename GlobalData, typename TileData>
TILEOP void TStore(GlobalData& dst, TileData& src, const Coord& coord)
{
    TStoreL1SpillImpl<config>(dst, src, coord);
}

template <
    int64_t blockSize, typename TileData, typename GlobalData, typename BlockT, typename OffsetT, typename SrcCoord,
    typename OffsetCoord, typename BlockCoord>
TILEOP void TGatherInL1(
    TileData dst, GlobalData src, BlockT block, OffsetT offset, SrcCoord srcCoord, OffsetCoord offsetCoord,
    BlockCoord blockCoord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    TGatherInL1Impl<blockSize>(dst, src, block, offset, srcCoord, offsetCoord, blockCoord);
}

#endif // TILEOP_TILE_OPERATOR_CUBE_PTO__H