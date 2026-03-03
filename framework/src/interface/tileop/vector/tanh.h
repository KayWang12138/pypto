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
 * \file tanh.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_TANH__H
#define TILEOP_TILE_OPERATOR_TANH__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <type_traits>

constexpr float EPSILON = 1.1754943508222875e-38f;

TILEOP void SyncPipeBarrier()
{
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
}

template <typename LastUse, typename T, typename DstTile, typename SrcTile, typename TmpTile>
TILEOP void TanhFP32(DstTile dstTile, SrcTile srcTile, TmpTile tmpTile)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;

    pto::TABS(tmpTile, srcTile);
    SyncPipeBarrier();

    pto::TMULS(dstTile, tmpTile, static_cast<T>(-2.0f));
    SyncPipeBarrier();

    pto::TEXP(dstTile, dstTile);
    SyncPipeBarrier();

    pto::TMULS(tmpTile, dstTile, static_cast<T>(-1.0f));
    SyncPipeBarrier();

    pto::TADDS(tmpTile, tmpTile, static_cast<T>(1.0f));
    SyncPipeBarrier();

    pto::TMUL(tmpTile, srcTile, tmpTile);
    SyncPipeBarrier();

    pto::TADDS(dstTile, dstTile, static_cast<T>(1.0f));
    SyncPipeBarrier();

    pto::TABS(srcTile, srcTile);
    SyncPipeBarrier();

    pto::TADDS(srcTile, srcTile, static_cast<T>(EPSILON));
    SyncPipeBarrier();

    pto::TMUL(dstTile, dstTile, srcTile);
    SyncPipeBarrier();

    pto::TDIV(dstTile, tmpTile, dstTile);
}

template <typename LastUse, typename T, typename DstTile, typename SrcTile, typename TmpTile1, typename TmpTile2, typename TmpTile3>
TILEOP void TanhCast(DstTile dstTile, SrcTile srcTile, TmpTile1 tmpTile1, TmpTile2 tmpTile2, TmpTile3 tmpTile3)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;

    pto::TCVT(tmpTile1, srcTile, pto::RoundMode::CAST_NONE);
    SyncPipeBarrier();

    pto::TABS(tmpTile2, tmpTile1);
    SyncPipeBarrier();

    pto::TMULS(tmpTile3, tmpTile2, static_cast<float>(-2.0f));
    SyncPipeBarrier();

    pto::TEXP(tmpTile3, tmpTile3);
    SyncPipeBarrier();

    pto::TMULS(tmpTile2, tmpTile3, static_cast<float>(-1.0f));
    SyncPipeBarrier();

    pto::TADDS(tmpTile2, tmpTile2, static_cast<float>(1.0f));
    SyncPipeBarrier();

    pto::TMUL(tmpTile2, tmpTile1, tmpTile2);
    SyncPipeBarrier();

    pto::TADDS(tmpTile3, tmpTile3, static_cast<float>(1.0f));
    SyncPipeBarrier();

    pto::TABS(tmpTile1, tmpTile1);
    SyncPipeBarrier();

    pto::TADDS(tmpTile1, tmpTile1, static_cast<float>(EPSILON));
    SyncPipeBarrier();

    pto::TMUL(tmpTile3, tmpTile3, tmpTile1);
    SyncPipeBarrier();

    pto::TDIV(tmpTile2, tmpTile2, tmpTile3);
    SyncPipeBarrier();

    pto::TCVT(dstTile, tmpTile2, pto::RoundMode::CAST_NONE);
}

#define OP_TILE_OP_TANH Ttanh
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1, typename T3>
TILEOP void TTanh(T0 dst, T1 src, T3 tmp) {
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

    using DstTile =
        pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcTile =
        pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;

    DstTile dstTile(dstShape3, dstShape4);
    SrcTile srcTile(srcShape3, srcShape4);

    if constexpr (std::is_same<typename T0::Type, float>::value) {
        constexpr auto ALIGN32FP32 = 8;
        constexpr auto tmpTileW = (srcTileW + ALIGN32FP32 - 1) / ALIGN32FP32 * ALIGN32FP32;
        using TmpTile =
            pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
        TmpTile tmpTile(srcShape3, srcShape4);
        pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    TanhFP32<LastUse, typename T0::Type, DstTile, SrcTile, TmpTile>(dstTile, srcTile, tmpTile);
                }
            }
        }
    } else if constexpr (std::is_same<typename T0::Type, half>::value || std::is_same<typename T0::Type, bfloat16_t>::value) {
        constexpr auto ALIGN32FP32 = 8;
        constexpr auto tmpTileW = (srcTileW + ALIGN32FP32 - 1) / ALIGN32FP32 * ALIGN32FP32;
        using TmpTile =
            pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
        auto tmpOffset = srcTileH * tmpTileW;
        TmpTile tmpTile1(srcShape3, srcShape4);
        TmpTile tmpTile2(srcShape3, srcShape4);
        TmpTile tmpTile3(srcShape3, srcShape4);
        pto::TASSIGN(tmpTile1, (uint64_t)(tmp.GetAddr()));
        pto::TASSIGN(tmpTile2, (uint64_t)(tmp.GetAddr() + tmpOffset * sizeof(float)));
        pto::TASSIGN(tmpTile3, (uint64_t)(tmp.GetAddr() + 2 * tmpOffset * sizeof(float)));

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    TanhCast<LastUse, typename T0::Type, DstTile, SrcTile, TmpTile, TmpTile, TmpTile>(dstTile, srcTile, tmpTile1, tmpTile2, tmpTile3);
                }
            }
        }
    }
}

#endif
