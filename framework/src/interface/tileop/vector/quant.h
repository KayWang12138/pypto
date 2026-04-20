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
 * \file quant.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_QUANT__H
#define TILEOP_TILE_OPERATOR_QUANT__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_QUANT_MX TQuantMX
constexpr int kDequantScaleRoundingModeRoundUp = 0;
constexpr int kDequantScaleRoundingModeRoundDown = 1;
constexpr int kQuantMXPerformanceModeOn = 1;

template <
    int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, typename T0, typename T1,
    typename T2, typename T3, typename T4>
TILEOP void TQuantMXGeneral(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    (void)AXIS;
    constexpr int kMxQuantGroupSize = 32;
    const auto dstLayout = dst.GetLayout();
    const auto expLayout = exp.GetLayout();
    const auto maxLayout = maxScratch.GetLayout();
    const auto scalingLayout = scalingScratch.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto expStride0 = expLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto expStride1 = expLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto expStride2 = expLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    constexpr auto expTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto expTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    using ExpByteTile = pto::Tile<pto::TileType::Vec, uint8_t, expTileH, expTileW, pto::BLayout::RowMajor, -1, -1>;

    auto dstTile = PtoTile<T0>(dst);
    auto maxTile = PtoTile<T2>(maxScratch);
    auto scalingTile = PtoTile<T3>(scalingScratch);
    auto srcTile = PtoTile<T4>(src);
    using SrcTileType = typename decltype(srcTile)::Type;
    using SrcPadTileType = pto::Tile<
        SrcTileType::Loc, typename SrcTileType::DType, SrcTileType::Rows, SrcTileType::Cols, SrcTileType::BFractal,
        SrcTileType::ValidRow, SrcTileType::ValidCol, SrcTileType::SFractal, SrcTileType::SFractalSize,
        pto::PadValue::Zero, SrcTileType::Compact>;
    ExpByteTile expByteTile(
        expLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(), expLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>());

    (void)maxLayout;
    (void)scalingLayout;
    (void)srcLayout;
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto expTileOffset = n0Index * expStride0 + n1Index * expStride1 + n2Index * expStride2;
                auto srcTileAddr =
                    (uint64_t)(src.GetAddr() + GenTileOffset(src, tileOffsets) * sizeof(typename T4::Type));
                dstTile.Assign(dst, tileOffsets);
                maxTile.Assign(maxScratch, tileOffsets);
                scalingTile.Assign(scalingScratch, tileOffsets);
                srcTile.Assign(srcTileAddr);
                pto::TASSIGN(expByteTile, (uint64_t)(exp.GetAddr() + expTileOffset * sizeof(typename T1::Type)));
                if (srcTile.Data().GetValidCol() % kMxQuantGroupSize != 0) {
                    if constexpr (T4::IsStaticLayout()) {
                        SrcPadTileType srcPadTile;
                        pto::TASSIGN(srcPadTile, srcTileAddr);
                        pto::TFILLPAD_INPLACE(srcPadTile, srcTile.Data());
                    } else {
                        SrcPadTileType srcPadTile(srcTile.Data().GetValidRow(), srcTile.Data().GetValidCol());
                        pto::TASSIGN(srcPadTile, srcTileAddr);
                        pto::TFILLPAD_INPLACE(srcPadTile, srcTile.Data());
                    }
                }
                if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown) {
                    pto::TQUANT<pto::QuantType::MXFP8>(
                        dstTile.Data(), srcTile.Data(), &expByteTile, &maxTile.Data(), &scalingTile.Data());
                } else {
                    static_assert(
                        DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown,
                        "TQuantMX only supports ROUND_DOWN (OCP standard) mode currently.");
                }
            }
        }
    }
}

template <
    int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, typename T0, typename T1,
    typename T2, typename T3, typename T4>
TILEOP void TQuantMXPerformance(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    (void)AXIS;
    constexpr int kMxQuantGroupSize = 32;
    const auto dstLayout = dst.GetLayout();
    const auto expLayout = exp.GetLayout();
    const auto maxLayout = maxScratch.GetLayout();
    const auto scalingLayout = scalingScratch.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto expStride0 = expLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto expStride1 = expLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto expStride2 = expLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    constexpr auto expTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto expTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    using ExpByteTile = pto::Tile<pto::TileType::Vec, uint8_t, expTileH, expTileW, pto::BLayout::RowMajor, -1, -1>;

    auto dstTile = PtoTile<T0>(dst);
    auto maxTile = PtoTile<T2>(maxScratch);
    auto scalingTile = PtoTile<T3>(scalingScratch);
    auto srcTile = PtoTile<T4>(src);
    using SrcTileType = typename decltype(srcTile)::Type;
    using SrcPadTileType = pto::Tile<
        SrcTileType::Loc, typename SrcTileType::DType, SrcTileType::Rows, SrcTileType::Cols, SrcTileType::BFractal,
        SrcTileType::ValidRow, SrcTileType::ValidCol, SrcTileType::SFractal, SrcTileType::SFractalSize,
        pto::PadValue::Zero, SrcTileType::Compact>;
    ExpByteTile expByteTile(
        expLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(), expLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>());

    (void)maxLayout;
    (void)scalingLayout;
    (void)srcLayout;
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto expTileOffset = n0Index * expStride0 + n1Index * expStride1 + n2Index * expStride2;
                auto srcTileAddr =
                    (uint64_t)(src.GetAddr() + GenTileOffset(src, tileOffsets) * sizeof(typename T4::Type));
                dstTile.Assign(dst, tileOffsets);
                maxTile.Assign(maxScratch, tileOffsets);
                scalingTile.Assign(scalingScratch, tileOffsets);
                srcTile.Assign(srcTileAddr);
                pto::TASSIGN(expByteTile, (uint64_t)(exp.GetAddr() + expTileOffset * sizeof(typename T1::Type)));
                if (srcTile.Data().GetValidCol() % kMxQuantGroupSize != 0) {
                    if constexpr (T4::IsStaticLayout()) {
                        SrcPadTileType srcPadTile;
                        pto::TASSIGN(srcPadTile, srcTileAddr);
                        pto::TFILLPAD_INPLACE(srcPadTile, srcTile.Data());
                    } else {
                        SrcPadTileType srcPadTile(srcTile.Data().GetValidRow(), srcTile.Data().GetValidCol());
                        pto::TASSIGN(srcPadTile, srcTileAddr);
                        pto::TFILLPAD_INPLACE(srcPadTile, srcTile.Data());
                    }
                }
                if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown) {
                    pto::TQUANT<pto::QuantType::MXFP8>(
                        dstTile.Data(), srcTile.Data(), &expByteTile, &maxTile.Data(), &scalingTile.Data());
                } else {
                    static_assert(
                        DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown,
                        "TQuantMX only supports ROUND_DOWN (OCP standard) mode currently.");
                }
            }
        }
    }
}

template <
    int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, int PERFORMANCE_MODE = 0,
    typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void TQuantMX(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    if constexpr (PERFORMANCE_MODE == kQuantMXPerformanceModeOn) {
        TQuantMXPerformance<DEQUANT_SCALE_ROUNDING_MODE, AXIS>(dst, exp, maxScratch, scalingScratch, src);
    } else {
        TQuantMXGeneral<DEQUANT_SCALE_ROUNDING_MODE, AXIS>(dst, exp, maxScratch, scalingScratch, src);
    }
}

#endif
