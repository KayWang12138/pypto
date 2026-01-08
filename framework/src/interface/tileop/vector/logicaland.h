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

template <typename T>
using select_type =
    std::conditional_t<std::is_same_v<typename T::Type, half> || (sizeof(typename T::Type) == 1), half, float>;

template <typename T, typename U>
using need_cvt = !std::is_same<select_type<T>, select_type<U>>::value;

template <typename T, typename U>
using select_final_type = std::conditional_t<need_cvt<T, U>, float, half>;

template <typename T, typename U>
TILEOP T &StandardizedTile([[maybe_unused]] T &dst, U &src) {
    if constexpr (std::is_same_v<T::Type, U::Type>) {
        return src;
    } else {
        pto::TCVT(dst, src, pto::RoundMode::CAST_NONE);
    }
}

template <typename T, typename U, typename Z, typename O, typename R>
TILEOP void GenLogicalTile(T &dst, U &src, Z &zeros, O &ones, R &res) {
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TCMP(res, src, zeros, pto::CmpMode::EQ);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSEL(dst, res, ones, zeros);
}

template <typename T, typename U1, typename U2>
TILEOP void SelectLogicalResult(T &dst, U1 &src1, U2 &src2) {
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMIN(src1, src1, src2);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TCVT(dst, src1, pto::RoundMode::CAST_NONE);
}

template <typename T, typename U1, typename U2, typename L1, typename L2, typename Ones, typename Zeros, typename Res>
TILEOP void LogicalAndImpl(T &dst, U1 &src1, U2 &src2, L1 &tmp1, L2 &tmp2, Ones &ones, Zeros &zeros, Res &res) {
    auto input1 = StandardizedTile(tmp1, src1);
    GenLogicalTile(input1, input1, ones, zeros, res);
    auto input2 = StandardizedTile(tmp2, src2);
    GenLogicalTile(input2, input2, ones, zeros, res);
    SelectLogicalResult(dst, input1, input2);
}

template <typename T, size_t offset = sizeof(typename T::Type) * COUNT_MAX>
TILEOP uint64_t GenTileAddr(const uint64_t addr) {
    constexpr uint32_t ALIGN_SIZE = 32;
    uintptr_t start = reinterpret_cast<uintptr_t>(addr + offset);
    return (start + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TLogicalAnd(T0 dst, T1 src1, T2 src2, T3 tmp) {
    constexpr auto COUNT_MAX = 2048;
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    auto numLoop = dstShape4 / COUNT_MAX;
    auto remainAfterLoop = dstShape4 % COUNT_MAX;

    using Dtype = select_final_type<T1, T2>;
    using DstTileTensor = TileTensor<typename T0::Type, LocalLayout5Dim<1, 1, 1, 1, COUNT_MAX>, Hardware::UB>;
    using Src1TileTensor = TileTensor<typename T1::Type, LocalLayout5Dim<1, 1, 1, 1, COUNT_MAX>, Hardware::UB>;
    using Src2TileTensor = TileTensor<typename T2::Type, LocalLayout5Dim<1, 1, 1, 1, COUNT_MAX>, Hardware::UB>;
    using StaticTileTensor =
        TileTensor<uint8_t, StaticLayout5Dim<1, 1, 1, 1, COUNT_MAX, 1, 1, 1, 1, COUNT_MAX>, Hardware::UB>;
    using TmpTileTensor = TileTensor<Dtype, LocalLayout5Dim<1, 1, 1, 1, COUNT_MAX>, Hardware::UB>;
    using ResTileTensor =
        TileTensor<uint8_t, StaticLayout5Dim<1, 1, 1, 1, COUNT_MAX / 8, 1, 1, 1, 1, COUNT_MAX / 8>, Hardware::UB>;

    auto h = 1;
    auto w = numLoop > 0 ? COUNT_MAX : remainAfterLoop;
    auto dstTile = PtoTile<DstTileTensor>(h, w);
    auto src1Tile = PtoTile<Src1TileTensor>(h, w);
    auto src2Tile = PtoTile<Src2TileTensor>(h, w);
    auto oneTile = PtoTile<StaticTileTensor>(h, w);
    auto zeroTile = PtoTile<StaticTileTensor>(h, w);
    auto tmp1Tile = PtoTile<TmpTileTensor>(h, w);
    auto tmp2Tile = PtoTile<TmpTileTensor>(h, w);
    auto resTile = PtoTile<ResTileTensor>(h, w);

    resTile.Assign((uint64_t)tmp.GetAddr());

    constexpr auto BITS_PER_BYTES = 8;
    constexpr auto vcmpBitSize = (COUNT_MAX + BITS_PER_BYTES - 1) / BITS_PER_BYTES;

    auto zeroAddr = GenTileAddr<StaticTileTensor, vcmpBitSize>((uint64_t)tmp.GetAddr());
    zeroTile.Assign((uint64_t)zeroAddr);

    auto oneAddr = GenTileAddr<StaticTileTensor>((uint64_t)zeroAddr);
    oneTile.Assign((uint64_t)oneAddr);

    auto tmp1Addr = GenTileAddr<TmpTileTensor>((uint64_t)oneAddr);
    tmp1Tile.Assign((uint64_t)tmp1Addr);

    auto tmp2Addr = GenTileAddr<TmpTileTensor>((uint64_t)tmp1Addr);
    tmp2Tile.Assign((uint64_t)tmp2Addr);

    pto::TEXPANDS(oneTile.Data(), 1.0);
    pto::TEXPANDS(zeroTile.Data(), 0.0);

    for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                for (size_t n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    auto tileOffsets = TileOffset(n0Index, n1Index, n2Index, n3Index);
                    for (int j = 0; j < numLoop; j++) {
                        dstTile.Assign(dst, tileOffsets, j * COUNT_MAX);
                        src1Tile.Assign(src1, tileOffsets, j * COUNT_MAX);
                        src2Tile.Assign(src2, tileOffsets, j * COUNT_MAX);
                        LogicalAndImpl(dstTile.Data(), src1Tile.Data(), src2Tile.Data(), tmp1Tile.Data(),
                            tmp2Tile.Data(), oneTile.Data(), zeroTile.Data(), resTile.Data());
                    }
                    if (remainAfterLoop > 0) {
                        w = remainAfterLoop;
                        dstTile = PtoTile<DstTileTensor>(h, w);
                        src1Tile = PtoTile<Src1TileTensor>(h, w);
                        src2Tile = PtoTile<Src2TileTensor>(h, w);
                        oneTile = PtoTile<StaticTileTensor>(h, w);
                        zeroTile = PtoTile<StaticTileTensor>(h, w);
                        tmp1Tile = PtoTile<TmpTileTensor>(h, w);
                        tmp2Tile = PtoTile<TmpTileTensor>(h, w);
                        resTile = PtoTile<ResTileTensor>(h, w);
                        dstTile.Assign(dst, tileOffsets, numLoop * COUNT_MAX);
                        src1Tile.Assign(src1, tileOffsets, numLoop * COUNT_MAX);
                        src2Tile.Assign(src2, tileOffsets, numLoop * COUNT_MAX);
                        resTile.Assign((uint64_t)tmp.GetAddr());
                        zeroTile.Assign((uint64_t)zeroAddr);
                        oneTile.Assign((uint64_t)oneAddr);
                        tmp1Tile.Assign((uint64_t)tmp1Addr);
                        tmp2Tile.Assign((uint64_t)tmp2Addr);
                        pto::TEXPANDS(oneTile.Data(), 1.0);
                        pto::TEXPANDS(zeroTile.Data(), 0.0);
                        LogicalAndImpl(dstTile.Data(), src1Tile.Data(), src2Tile.Data(), tmp1Tile.Data(),
                            tmp2Tile.Data(), oneTile.Data(), zeroTile.Data(), resTile.Data());
                    }
                }
            }
        }
    }
}
#endif // TILEOP_TILE_OPERATOR_LOGICALAND__H
