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
 * \file signbit.h
 * \brief signbit operator: identify sign bit, negative returns true, positive returns false
 */

#ifndef TILEOP_TILE_OPERATOR_SIGNBIT__H
#define TILEOP_TILE_OPERATOR_SIGNBIT__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <type_traits>

template <typename DstTile, typename SrcUint32Tile, typename TempTile>
TILEOP void SignbitImpl32Bit(DstTile dstTile, SrcUint32Tile srcUint32Tile, TempTile tempTile)
{
    constexpr auto shiftAmount = 31;
    pto::TSHRS(srcUint32Tile, srcUint32Tile, static_cast<uint32_t>(shiftAmount));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TCVT(tempTile, srcUint32Tile, pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TCVT(dstTile, tempTile, pto::RoundMode::CAST_NONE);
}

template <typename DstTile, typename SrcUint16Tile>
TILEOP void SignbitImpl16Bit(DstTile dstTile, SrcUint16Tile srcUint16Tile)
{
    constexpr auto shiftAmount = 15;
    pto::TSHRS(srcUint16Tile, srcUint16Tile, static_cast<uint16_t>(shiftAmount));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TCVT(dstTile, srcUint16Tile, pto::RoundMode::CAST_NONE);
}

template <typename DstTile, typename SrcUint8Tile, typename TempTile>
TILEOP void SignbitImpl8Bit(DstTile dstTile, SrcUint8Tile srcUint8Tile, TempTile tempTile)
{
    constexpr auto shiftAmount = 7;
    pto::TCVT(tempTile, srcUint8Tile, pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSHRS(tempTile, tempTile, static_cast<uint16_t>(shiftAmount));
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TCVT(dstTile, tempTile, pto::RoundMode::CAST_NONE);
}

template <typename T, typename DstTile, typename SrcTile, typename SrcUint32Tile, typename SrcUint16Tile, typename SrcUint8Tile, typename TempTile>
TILEOP void SignbitImpl(DstTile dstTile, SrcTile srcTile, SrcUint32Tile srcUint32Tile, SrcUint16Tile srcUint16Tile, SrcUint8Tile srcUint8Tile, TempTile tempTile)
{
    if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value ||
                  std::is_same<T, uint32_t>::value) {
        pto::TASSIGN(srcUint32Tile, (uint64_t)srcTile.GetAddr());
        SignbitImpl32Bit<DstTile, SrcUint32Tile, TempTile>(dstTile, srcUint32Tile, tempTile);
    } else if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value ||
                         std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value) {
        pto::TASSIGN(srcUint16Tile, (uint64_t)srcTile.GetAddr());
        SignbitImpl16Bit<DstTile, SrcUint16Tile>(dstTile, srcUint16Tile);
    } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, bool>::value ||
                         std::is_same<T, uint8_t>::value) {
        pto::TASSIGN(srcUint8Tile, (uint64_t)srcTile.GetAddr());
        SignbitImpl8Bit<DstTile, SrcUint8Tile, TempTile>(dstTile, srcUint8Tile, tempTile);
    }
    return;
}

#define OP_TILE_OP_SIGNBIT TSignbit
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1, typename T3>
TILEOP void TSignbit(T0 dst, T1 src, T3 tmp) {
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);

    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    auto srcShape0 = srcLayout.template GetShapeDim<0, expectSize>();
    auto srcShape1 = srcLayout.template GetShapeDim<1, expectSize>();
    auto srcShape2 = srcLayout.template GetShapeDim<2, expectSize>();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();

    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();

    constexpr auto ALIGN32HALF = 16;
    constexpr auto tmpTileW = (srcTileW + ALIGN32HALF - 1) / ALIGN32HALF * ALIGN32HALF;

    using DstTile =
        pto::Tile<pto::TileType::Vec, uint8_t, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcTile =
        pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcUint32Tile =
        pto::Tile<pto::TileType::Vec, uint32_t, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcUint16Tile =
        pto::Tile<pto::TileType::Vec, uint16_t, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcUint8Tile =
        pto::Tile<pto::TileType::Vec, uint8_t, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
    using TempTile =
        pto::Tile<pto::TileType::Vec, uint16_t, srcTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
    DstTile dstTile(dstShape3, dstShape4);
    SrcTile srcTile(srcShape3, srcShape4);
    SrcUint32Tile srcUint32Tile(srcShape3, srcShape4);
    SrcUint16Tile srcUint16Tile(srcShape3, srcShape4);
    SrcUint8Tile srcUint8Tile(srcShape3, srcShape4);
    TempTile tempTile(srcShape3, srcShape4);
    pto::TASSIGN(tempTile, (uint64_t)(tmp.GetAddr()));

    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                SignbitImpl<typename T1::Type, DstTile, SrcTile, SrcUint32Tile, SrcUint16Tile, SrcUint8Tile, TempTile>(dstTile, srcTile, srcUint32Tile, srcUint16Tile, srcUint8Tile, tempTile);
            }
        }
    }
}

#endif
