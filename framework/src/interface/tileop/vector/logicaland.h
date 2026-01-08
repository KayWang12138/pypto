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
TILEOP T &StandardizedTile([[maybe_unused]] T &dst, U &src) {
    if constexpr (std::is_same_v<typename T::DType, typename U::DType>) {
        return src;
    } else {
        pto::TCVT(dst, src, pto::RoundMode::CAST_NONE);
        return dst;
    }
}

template <typename T, typename U, typename R>
TILEOP void GenLogicalTile(T &dst, U &src, U &zeros, U &ones, R &res) {
    pto::TCMP(res, src, zeros, pto::CmpMode::EQ);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSEL(dst, res, zeros, ones);
}

template <typename T, typename U>
TILEOP void SelectLogicalResult(T &dst, U &src1, U &src2) {
    pto::TMIN(src1, src1, src2);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TCVT(dst, src1, pto::RoundMode::CAST_NONE);
}

template <typename T, typename U1, typename U2, typename L, typename Res>
TILEOP void LogicalAndImpl(T &dst, U1 &src1, U2 &src2, L &tmp1, L &tmp2, L &ones, L &zeros, Res &res1, Res &res2) {
    auto input1 = StandardizedTile(tmp1, src1);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    GenLogicalTile(tmp1, input1, zeros, ones, res1);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    auto input2 = StandardizedTile(tmp2, src2);
    GenLogicalTile(tmp2, input2, zeros, ones, res2);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    SelectLogicalResult(dst, tmp1, tmp2);
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

    using DstTileTensor = TileTensor<typename T0::Type, LocalLayout5Dim<1, COUNT_MAX>, Hardware::UB>;
    using Src1TileTensor = TileTensor<typename T1::Type, LocalLayout5Dim<1, COUNT_MAX>, Hardware::UB>;
    using Src2TileTensor = TileTensor<typename T2::Type, LocalLayout5Dim<1, COUNT_MAX>, Hardware::UB>;
    using TmpTileTensor = TileTensor<DType, LocalLayout5Dim<1, COUNT_MAX>, Hardware::UB>;
    using CmpBitsTileTensor = TileTensor<uint8_t, StaticLayout5Dim<1, CMP_SHAPE, 1, CMP_SHAPE>, Hardware::UB>;

    auto numLoop = dstShape4 / COUNT_MAX;
    auto remainAfterLoop = dstShape4 % COUNT_MAX;

    auto h = 1;
    auto w = numLoop > 0 ? COUNT_MAX : remainAfterLoop;
    auto dstTile = PtoTile<DstTileTensor>(h, w);
    auto src1Tile = PtoTile<Src1TileTensor>(h, w);
    auto src2Tile = PtoTile<Src2TileTensor>(h, w);
    auto tmp1Tile = PtoTile<TmpTileTensor>(h, w);
    auto tmp2Tile = PtoTile<TmpTileTensor>(h, w);
    auto oneTile = PtoTile<TmpTileTensor>(h, w);
    auto zeroTile = PtoTile<TmpTileTensor>(h, w);
    auto cmp1Tile = PtoTile<CmpBitsTileTensor>();
    auto cmp2Tile = PtoTile<CmpBitsTileTensor>();

    auto cmp1Addr = (uint64_t)tmp.GetAddr();
    auto cmp2Addr = GenTmpTileAddr(cmp1Addr, CMP_BYTE_SIZE);
    auto zeroAddr = GenTmpTileAddr(cmp2Addr, CMP_BYTE_SIZE);
    auto oneAddr = GenTmpTileAddr(zeroAddr, TMP_OFFSET);
    auto tmp1Addr = GenTmpTileAddr(oneAddr, TMP_OFFSET);
    auto tmp2Addr = GenTmpTileAddr(tmp1Addr, TMP_OFFSET);

    cmp1Tile.Assign(cmp1Addr);
    cmp2Tile.Assign(cmp2Addr);
    zeroTile.Assign(zeroAddr);
    oneTile.Assign(oneAddr);
    tmp1Tile.Assign(tmp1Addr);
    tmp2Tile.Assign(tmp2Addr);

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
                            tmp2Tile.Data(), oneTile.Data(), zeroTile.Data(), cmp1Tile.Data(), cmp2Tile.Data());
                    }
                    if (remainAfterLoop > 0) {
                        if (numLoop > 0) {
                            w = remainAfterLoop;
                            dstTile = PtoTile<DstTileTensor>(h, w);
                            src1Tile = PtoTile<Src1TileTensor>(h, w);
                            src2Tile = PtoTile<Src2TileTensor>(h, w);
                            oneTile = PtoTile<TmpTileTensor>(h, w);
                            zeroTile = PtoTile<TmpTileTensor>(h, w);
                            tmp1Tile = PtoTile<TmpTileTensor>(h, w);
                            tmp2Tile = PtoTile<TmpTileTensor>(h, w);
                            dstTile.Assign(dst.GetAddr() + (dstOffset + numLoop * COUNT_MAX) * dstTypeSize);
                            src1Tile.Assign(src1.GetAddr() + (src1Offset + numLoop * COUNT_MAX) * src1TypeSize);
                            src2Tile.Assign(src2.GetAddr(), (src2Offset + numLoop * COUNT_MAX) * src2TypeSize);
                            zeroTile.Assign((uint64_t)zeroAddr);
                            oneTile.Assign((uint64_t)oneAddr);
                            tmp1Tile.Assign((uint64_t)tmp1Addr);
                            tmp2Tile.Assign((uint64_t)tmp2Addr);
                            pto::TEXPANDS(oneTile.Data(), 1.0);
                            pto::TEXPANDS(zeroTile.Data(), 0.0);
                        }
                        LogicalAndImpl(dstTile.Data(), src1Tile.Data(), src2Tile.Data(), tmp1Tile.Data(),
                            tmp2Tile.Data(), oneTile.Data(), zeroTile.Data(), cmp1Tile.Data(), cmp2Tile.Data());
                    }
                }
            }
        }
    }
}
#endif // TILEOP_TILE_OPERATOR_LOGICALAND__H
