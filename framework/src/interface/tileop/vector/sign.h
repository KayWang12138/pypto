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
 * \file sign.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_SIGN__H
#define TILEOP_TILE_OPERATOR_SIGN__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <type_traits>

template <typename T, typename DstTile, typename SrcTile>
TILEOP void SignInt(DstTile dstTile, SrcTile srcTile)
{
    pto::MINS(srcTile, srcTile, (T)1);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::MAXS(dstTile, srcTile, (T)-1);
}

template <typename T, typename DstTile, typename SrcTile>
TILEOP void SignHalf(DstTile dstTile, SrcTile srcTile)
{
    pto::MINS(srcTile, srcTile, (T)5.960464e-08f);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::MAXS(srcTile, srcTile, (T)-5.960464e-08f);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::MULS(srcTile, srcTile, (T)4.096000e+03f);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::MULS(dstTile, srcTile, (T)4.096000e+03f);
}

template <typename T, typename DstTile, typename SrcTile>
TILEOP void SignFloat(DstTile dstTile, SrcTile srcTile)
{
    pto::MINS(srcTile, srcTile, (T)1.1754943508222875e-38f);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::MAXS(srcTile, srcTile, (T)-1.1754943508222875e-38f);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::MULS(srcTile, srcTile, (T)4.6116860184273879e+18f);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::MULS(srcTile, srcTile, (T)4.6116860184273879e+18f);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::MULS(dstTile, srcTile, (T)4.0000000000000000e+00f);
}

template <typename T, typename DstTile, typename SrcTile>
TILEOP void SignImpl(DstTile dstTile, SrcTile srcTile)
{
    if constexpr (std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value) {
        SignInt<DstTile, SrcTile>(dstTile, srcTile);
    } else if (std::is_same<T, half>::value) {
        SignHalf<DstTile, SrcTile>(dstTile, srcTile);
    } else if (std::is_same<T, float>::value || std::is_same<T, bfloat16_t>::value) {
        SignFloat<T, DstTile, SrcTile>(dstTile, srcTile);
    }
    return;
}

#define OP_TILE_OP_SIGN TSign
template <typename T0, typename T1>
TILEOP void TSign(T0 dst, T1 src) {
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

    using DstTileDefine =
        pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcTileDefine =
        pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
    DstTileDefine dstTile(dstShape3, dstShape4);
    SrcTileDefine srcTile(srcShape3, srcShape4);

    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                SignImpl<typename T0::Type, DstTileDefine, SrcTileDefine>(dstTile, srcTile);
            }
        }
    }
}

#endif