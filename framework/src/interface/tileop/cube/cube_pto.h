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

#include "impl/utools.h"
#include "impl/copy_gm_to_l1_impl.h"
#include "impl/copy_l0c_to_gm_impl.h"
#include "impl/copy_l1_to_bt_fb_impl.h"
#include "impl/copy_l1_to_l0_impl.h"
#include "impl/gather_in_l1_impl.h"
#include "impl/copy_l0c_to_l1_impl.h"
#include "impl/mmad_impl.h"

#if defined PTO_NPU_ARCH_A5
#include "impl/arch35/copy_gm_to_l1_mx_impl.h"
#include "impl/arch35/copy_l0c_to_ub_impl.h"
#include "impl/arch35/copy_l1_to_l0_mx_impl.h"
#include "impl/arch35/mmad_mx_impl.h"
#include "impl/arch35/copy_ub_to_l1_impl.h"
#include "impl/arch35/copy_ub_to_ub_impl.h"
#endif

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
        "[TLoadAMX Error]: Dst format should be L1 and Src format should be GM");
    TLoadAMXImpl<mode, Coord, T, U>(dst, src, coord);
}

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
        "[TLoadBMX Error]: Dst format should be L1 and Src format should be GM");
    TLoadBMXImpl<mode, Coord, T, U>(dst, src, coord);
}

// Copy data from UB to UB with ND -> NZ format
template <typename T, typename U>
TILEOP void TMoveND2NZ(T& dst, U& src)
{
    constexpr int64_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(T::FORMAT == Hardware::UB && U::FORMAT == Hardware::UB);
    TMoveND2NZImpl(dst, src);
}

// Copy data from UB to L1 with NZ -> NZ format
template <typename Coord, typename T, typename U>
TILEOP void TExtract(T& dst, U& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::UB);
    TExtractUB2L1Impl(dst, src, coord);
}

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
    TExtractMXImpl<Coord, T, U>(dst, src, coord);
}

// Copy data from L0C to UB
template <CopyOutMode mode, typename Coord, typename T, typename U>
TILEOP void TExtract(T& dst, U& src, const Coord& coord, int16_t subblockId)
{
    if (!CheckShapeValid(dst, src)){
        return;
    }
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");

    TExtractL0C2UBImpl<mode, Coord, T, U>(dst, src, coord, subblockId);
}

template <bool isZeroC, typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void MatmulMX(T0& c, T1& a, T2& aScale, T3& b, T4& bScale)
{
    constexpr auto shapeSizeC = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto shapeSizeA = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto shapeSizeAScale = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto shapeSizeB = Std::tuple_size<typename T3::Shape>::value;
    constexpr auto shapeSizeBScale = Std::tuple_size<typename T4::Shape>::value;
    static_assert(
        shapeSizeC == SHAPE_DIM2 && shapeSizeA == SHAPE_DIM2 && shapeSizeAScale == SHAPE_DIM3 &&
            shapeSizeB == SHAPE_DIM2 && shapeSizeBScale == SHAPE_DIM3,
        "[MatmulMX ERROR]: Tensor Shape dim size should be 2 and Scale Shape dim size should be 3");
    MatmulMXImpl<isZeroC>(c, a, aScale, b, bScale);
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
TILEOP void MatmulMX(T0& c, T1& a, T2& aScale, T3& b, T4& bScale, T5& bias)
{
    constexpr auto shapeSizeC = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto shapeSizeA = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto shapeSizeAScale = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto shapeSizeB = Std::tuple_size<typename T3::Shape>::value;
    constexpr auto shapeSizeBScale = Std::tuple_size<typename T4::Shape>::value;
    static_assert(
        shapeSizeC == SHAPE_DIM2 && shapeSizeA == SHAPE_DIM2 && shapeSizeAScale == SHAPE_DIM3 &&
            shapeSizeB == SHAPE_DIM2 && shapeSizeBScale == SHAPE_DIM3,
        "[MatmulMX ERROR]: Shape dim size should be 2 and Scale Shape dim size should be 3");
    MatmulMXImpl(c, a, aScale, b, bScale, bias);
}
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
        "[TLoad Error]: Dst format should be L1 and Src format should be GM");
    if constexpr (copyMode == CopyInMode::ND2NZ) {
        TLoadND2NZ<padMode>(dst, src, offset0, offset1);
    } else if constexpr (copyMode == CopyInMode::NZ2NZ) {
        TLoadNZ2NZ<padMode>(dst, src, offset0, offset1, curH, curW);
    } else if constexpr (copyMode == CopyInMode::ND2ND) {
        TLoadND2ND(dst, src, offset0, offset1);
    }
    return;
}

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
    TExtractL0C2L1Impl<config>(dst, src, fixbuf, l1Coord, l0cCoord, scaleValue);
}

// Copy data from L1 to L0A/L0B
template <bool isTrans, typename Coord, typename T, typename U>
TILEOP void TExtract(T& dst, U& src, const Coord& coord)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t>&>(coord).GetValue();
    if constexpr ((T::FORMAT == Hardware::L0A || T::FORMAT == Hardware::L0B) && U::FORMAT == Hardware::L1) {
        TExtractL1ToL0Impl<isTrans>(dst, src, offset0, offset1);
    }
    if constexpr ((T::FORMAT == Hardware::BIAS || T::FORMAT == Hardware::FIXBUF) && U::FORMAT == Hardware::L1) {
        TExtractL1ToBTOrFBImpl<isTrans>(dst, src);
    }
    return;
}

template <bool isZeroC, TransMode transMode, typename T, typename U, typename V>
TILEOP void TMatmul(T& c, U& a, V& b)
{
    constexpr auto shapeSizeA = Std::tuple_size<typename U::Shape>::value;
    constexpr auto shapeSizeB = Std::tuple_size<typename V::Shape>::value;
    constexpr auto shapeSizeC = Std::tuple_size<typename T::Shape>::value;
    static_assert(
        shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
        "[Matmul ERROR]: Shape dim size should be 2");
    TMatmulImpl<isZeroC, transMode>(c, a, b);
}

template <TransMode transMode, typename T0, typename T1, typename T2, typename T3>
TILEOP void TMatmul(T0& c, T1& a, T2& b, T3 bias)
{
    constexpr auto shapeSizeA = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto shapeSizeB = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto shapeSizeC = Std::tuple_size<typename T0::Shape>::value;
    static_assert(
        shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
        "[Matmul ERROR]: Shape dim size should be 2");
    TMatmulImpl<transMode>(c, a, b, bias);
}

// Copy data from L0C to DDR with quantization ability
template <typename config, typename Coord, typename T, typename U, typename V>
TILEOP void TStore(
    T& dst, U& src, V& fixbuf, const Coord& coord, const int64_t& curH, const int64_t& curW, uint64_t scaleValue = 0)
{
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
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
    TStoreL1SpillImpl<config>(dst, src, coord);
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
    TGatherInL1Impl<blockSize>(dst, src, block, offset, srcCoord, offsetCoord, blockCoord);
}

#endif // TILEOP_TILE_OPERATOR_CUBE_PTO__H