/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file logicaland.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_LOGICALAND__H
#define TILEOP_TILE_OPERATOR_LOGICALAND__H
#include <type_traits>

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename Type>
using select_type = std::conditional_t<std::is_same_v<Type, half> || (sizeof(Type) == 1), half, float>;

template <typename T, typename U>
using select_final_type = std::conditional_t<std::is_same_v<select_type<T>, select_type<U>>, select_type<T>, float>;

template <typename T, typename U>
TILEOP T &StandardizedSrcTile([[maybe_unused]] T &dst, U &src) {
    if constexpr (std::is_same_v<typename T::DType, typename U::DType>) {
        return src;
    } else {
        pto::TCVT(dst, src, pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        return dst;
    }
}

template <typename T, typename U, typename R>
TILEOP void CalculateLogicalTile(T &dst, U &src, U &zeros, U &ones, R &res) {
    pto::TCMP(res, src, zeros, pto::CmpMode::EQ);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSEL(dst, res, zeros, ones);
}

template <typename T, typename U, typename R>
TILEOP void CalculateSrcLogicalTile(T &dst, U &src, T &zeros, T &ones, R &res) {
    auto tile = StandardizedTile(dst, src);
    CalculateLogicalTile(dst, tile, zeros, ones, res);
}

template <typename T, typename U, typename TMP>
TILEOP void SelectLogicalResult(T &dst, U &src1, U &src2, TMP &tmp) {
    pto::TMIN(src1, src1, src2);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    if constexpr (std::is_same_v<typename TMP::DType, typename U::DType>) {
        pto::TCVT(dst, src1, pto::RoundMode::CAST_NONE);
    } else {
        pto::TCVT(tmp, src1, pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        pto::TCVT(dst, tmp, pto::RoundMode::CAST_NONE);
    }
}

template <typename T, typename U1, typename U2, typename L, typename Res, typename Cvt>
TILEOP void LogicalAndImpl(
    T &dst, U1 &src1, U2 &src2, L &tmp1, L &tmp2, L &ones, L &zeros, Res &res1, Res &res2, Cvt &cvt) {
    CalculateSrcLogicalTile(tmp1, src1, zeros, ones, res1);
    CalculateSrcLogicalTile(tmp2, src2, zeros, ones, res2);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    SelectLogicalResult(dst, tmp1, tmp2, cvt);
}

TILEOP uint64_t GenTmpTileAddr(uint64_t addr, uint64_t offset) {
    constexpr uint32_t ALIGN_SIZE = 32;
    uintptr_t start = reinterpret_cast<uintptr_t>(addr + offset);
    return (start + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TLogicalAnd(T0 dst, T1 src1, T2 src2, T3 tmp) {
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    using DType = select_final_type<typename T1::Type, typename T2::Type>;
    constexpr auto COUNT_MAX = 2048;
    constexpr auto BITS_PER_BYTE = 8;
    constexpr auto CMP_BYTE_SIZE = (COUNT_MAX + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
    constexpr auto TMP_OFFSET = sizeof(DType) * COUNT_MAX;
    constexpr auto CMP_SHAPE = COUNT_MAX / BITS_PER_BYTE;

    using DstTileTensor = TileTensor<typename T0::Type, LocalLayout2Dim<1, COUNT_MAX>, Hardware::UB>;
    using Src1TileTensor = TileTensor<typename T1::Type, LocalLayout2Dim<1, COUNT_MAX>, Hardware::UB>;
    using Src2TileTensor = TileTensor<typename T2::Type, LocalLayout2Dim<1, COUNT_MAX>, Hardware::UB>;
    using TmpTileTensor = TileTensor<DType, LocalLayout2Dim<1, COUNT_MAX>, Hardware::UB>;
    using CvtTileTensor = TileTensor<half, LocalLayout2Dim<1, COUNT_MAX>, Hardware::UB>;
    using CmpBitsTileTensor = TileTensor<uint8_t, StaticLayout2Dim<1, CMP_SHAPE, 1, CMP_SHAPE>, Hardware::UB>;

    auto numLoop = dstShape4 / COUNT_MAX;
    auto remainAfterLoop = dstShape4 % COUNT_MAX;

    auto w = numLoop > 0 ? COUNT_MAX : remainAfterLoop;
    auto dstTile = PtoTile<DstTileTensor>(1, w);
    auto src1Tile = PtoTile<Src1TileTensor>(1, w);
    auto src2Tile = PtoTile<Src2TileTensor>(1, w);
    auto tmp1Tile = PtoTile<TmpTileTensor>(1, w);
    auto tmp2Tile = PtoTile<TmpTileTensor>(1, w);
    auto oneTile = PtoTile<TmpTileTensor>(1, w);
    auto zeroTile = PtoTile<TmpTileTensor>(1, w);
    auto cvtTile = PtoTile<CvtTileTensor>(1, w);
    auto cmp1Tile = PtoTile<CmpBitsTileTensor>();
    auto cmp2Tile = PtoTile<CmpBitsTileTensor>();

    auto cmp1Addr = (uint64_t)tmp.GetAddr();
    auto cmp2Addr = GenTmpTileAddr(cmp1Addr, CMP_BYTE_SIZE);
    auto zeroAddr = GenTmpTileAddr(cmp2Addr, CMP_BYTE_SIZE);
    auto oneAddr = GenTmpTileAddr(zeroAddr, TMP_OFFSET);
    auto tmp1Addr = GenTmpTileAddr(oneAddr, TMP_OFFSET);
    auto tmp2Addr = GenTmpTileAddr(tmp1Addr, TMP_OFFSET);
    auto cvtAddr = GenTmpTileAddr(tmp2Addr, TMP_OFFSET);

    cmp1Tile.Assign(cmp1Addr);
    cmp2Tile.Assign(cmp2Addr);
    zeroTile.Assign(zeroAddr);
    oneTile.Assign(oneAddr);
    tmp1Tile.Assign(tmp1Addr);
    tmp2Tile.Assign(tmp2Addr);
    cvtTile.Assign(cvtAddr);

    pto::TEXPANDS(oneTile.Data(), 1.0);
    pto::TEXPANDS(zeroTile.Data(), 0.0);

    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto src1TypeSize = sizeof(typename T1::Type);
    constexpr auto src2TypeSize = sizeof(typename T2::Type);
    for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                for (size_t n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    auto tileOffsets = TileOffset4Dim(n0Index, n1Index, n2Index, n3Index);
                    auto dstOffset = GenTileOffset(dst, tileOffsets);
                    auto src1Offset = GenTileOffset(src1, tileOffsets);
                    auto src2Offset = GenTileOffset(src2, tileOffsets);
                    for (int j = 0; j < numLoop; j++) {
                        dstTile.Assign(dst.GetAddr() + (dstOffset + j * COUNT_MAX) * dstTypeSize);
                        src1Tile.Assign(src1.GetAddr() + (src1Offset + j * COUNT_MAX) * src1TypeSize);
                        src2Tile.Assign(src2.GetAddr(), (src2Offset + j * COUNT_MAX) * src2TypeSize);
                        LogicalAndImpl(dstTile.Data(), src1Tile.Data(), src2Tile.Data(), tmp1Tile.Data(),
                            tmp2Tile.Data(), oneTile.Data(), zeroTile.Data(), cmp1Tile.Data(), cmp2Tile.Data(),
                            cvtTile.Data());
                    }
                    if (remainAfterLoop > 0) {
                        if (numLoop > 0) {
                            w = remainAfterLoop;
                            dstTile = PtoTile<DstTileTensor>(1, w);
                            src1Tile = PtoTile<Src1TileTensor>(1, w);
                            src2Tile = PtoTile<Src2TileTensor>(1, w);
                            oneTile = PtoTile<TmpTileTensor>(1, w);
                            zeroTile = PtoTile<TmpTileTensor>(1, w);
                            tmp1Tile = PtoTile<TmpTileTensor>(1, w);
                            tmp2Tile = PtoTile<TmpTileTensor>(1, w);
                            cvtTile = PtoTile<CvtTileTensor>(1, w);

                            zeroTile.Assign(zeroAddr);
                            oneTile.Assign(oneAddr);
                            tmp1Tile.Assign(tmp1Addr);
                            tmp2Tile.Assign(tmp2Addr);
                            cvtTile.Assign(cvtAddr);
                            pto::TEXPANDS(oneTile.Data(), 1.0);
                            pto::TEXPANDS(zeroTile.Data(), 0.0);
                        }
                        dstTile.Assign(dst.GetAddr() + (dstOffset + numLoop * COUNT_MAX) * dstTypeSize);
                        src1Tile.Assign(src1.GetAddr() + (src1Offset + numLoop * COUNT_MAX) * src1TypeSize);
                        src2Tile.Assign(src2.GetAddr(), (src2Offset + numLoop * COUNT_MAX) * src2TypeSize);
                        LogicalAndImpl(dstTile.Data(), src1Tile.Data(), src2Tile.Data(), tmp1Tile.Data(),
                            tmp2Tile.Data(), oneTile.Data(), zeroTile.Data(), cmp1Tile.Data(), cmp2Tile.Data(),
                            cvtTile.Data());
                    }
                }
            }
        }
    }
}
#endif // TILEOP_TILE_OPERATOR_LOGICALAND__H
