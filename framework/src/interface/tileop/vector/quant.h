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

#include <pto/npu/a5/TQuant.hpp>

#include "pto_tile.h"

#define OP_TILE_OP_QUANT_MX TQuantMX
template <typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void TQuantMX(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
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
                dstTile.Assign(dst, tileOffsets);
                maxTile.Assign(maxScratch, tileOffsets);
                scalingTile.Assign(scalingScratch, tileOffsets);
                srcTile.Assign(src, tileOffsets);
                pto::TASSIGN(expByteTile, (uint64_t)(exp.GetAddr() + expTileOffset * sizeof(typename T1::Type)));
                pto::TQUANT_IMPL<pto::QuantType::MXFP8>(
                    dstTile.Data(), srcTile.Data(), &expByteTile, &maxTile.Data(), &scalingTile.Data());
            }
        }
    }
}

#endif
