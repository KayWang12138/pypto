/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file welford_var.h
 * \brief Welford online variance TileOp for PyPTO
 */
#ifndef TILEOP_TILE_OPERATOR_WELFORD_VAR__H
#define TILEOP_TILE_OPERATOR_WELFORD_VAR__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void WelfordVarSingleTileCompute(T0 dstSumX, T1 dstSumX2, T2 src, T3 tmp)
{
    constexpr auto srcShapeSize = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto tmpShapeSize = Std::tuple_size<typename T3::Shape>::value;
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T3, 3, 5>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T3, 4, 5>();
    using TmpTileDefine = pto::Tile<
        pto::TileType::Vec, typename T3::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor, tmpTileH, tmpTileW>;
    TmpTileDefine tmpTile;

    constexpr size_t expectSize = 5;
    const auto dstLayout = dstSumX.GetLayout();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();

    const auto srcLayout = src.GetLayout();
    auto srcShape0 = srcLayout.template GetShapeDim<0, expectSize>();
    auto srcShape1 = srcLayout.template GetShapeDim<1, expectSize>();
    auto srcShape2 = srcLayout.template GetShapeDim<2, expectSize>();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    if (srcShape0 == 0 || srcShape1 == 0 || srcShape2 == 0 || srcShape3 == 0 || srcShape4 == 0) {
        return;
    }
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T2, 3, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();
    constexpr auto srcTypeSize = sizeof(typename T2::Type);

    for (LoopVar n0Index = 0; n0Index < srcShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < srcShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < srcShape2; ++n2Index) {
                using DstTileDefine = typename std::conditional<
                    (dstTileW == 1),
                    pto::Tile<
                        pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::ColMajor, -1, -1>,
                    pto::Tile<
                        pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1,
                        -1>>::type;
                using SrcTileDefine = pto::Tile<
                    pto::TileType::Vec, typename T2::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;

                DstTileDefine dstSumXTile(dstShape3, dstShape4);
                DstTileDefine dstSumX2Tile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);

                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;

                pto::TASSIGN(dstSumXTile, (uint64_t)(dstSumX.GetAddr() + dstOffset * srcTypeSize));
                pto::TASSIGN(dstSumX2Tile, (uint64_t)(dstSumX2.GetAddr() + dstOffset * srcTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));

                if (srcShape3 == 0 || srcShape4 == 0) {
                    return;
                }

                pto::TROWSUM(dstSumXTile, srcTile, tmpTile);
                pto::TMUL(dstSumX2Tile, srcTile, srcTile);
                SyncV();
                pto::TROWSUM(dstSumX2Tile, dstSumX2Tile, tmpTile);
            }
        }
    }
}

#define OP_TILE_OP_WELFORDVAR_TWOTILE TWelfordVarTwoTile
template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
TILEOP void TWelfordVarTwoTile(T0 dstMean, T1 dstM2, T2 src0Mean, T3 src0M2, T4 src1Mean, T5 src1M2)
{
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr size_t expectSize = 5;

    const auto dstLayout = dstMean.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();
    constexpr auto typeSize = sizeof(typename T0::Type);

    using TileDefine =
        pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;

    for (LoopVar n0 = 0; n0 < dstShape0; ++n0) {
        for (LoopVar n1 = 0; n1 < dstShape1; ++n1) {
            for (LoopVar n2 = 0; n2 < dstShape2; ++n2) {
                auto offset = n0 * dstStride0 + n1 * dstStride1 + n2 * dstStride2;
                TileDefine dstMeanTile(dstShape3, dstShape4);
                TileDefine dstM2Tile(dstShape3, dstShape4);
                TileDefine src0MeanTile(dstShape3, dstShape4);
                TileDefine src0M2Tile(dstShape3, dstShape4);
                TileDefine src1MeanTile(dstShape3, dstShape4);
                TileDefine src1M2Tile(dstShape3, dstShape4);

                pto::TASSIGN(dstMeanTile, (uint64_t)(dstMean.GetAddr() + offset * typeSize));
                pto::TASSIGN(dstM2Tile, (uint64_t)(dstM2.GetAddr() + offset * typeSize));
                pto::TASSIGN(src0MeanTile, (uint64_t)(src0Mean.GetAddr() + offset * typeSize));
                pto::TASSIGN(src0M2Tile, (uint64_t)(src0M2.GetAddr() + offset * typeSize));
                pto::TASSIGN(src1MeanTile, (uint64_t)(src1Mean.GetAddr() + offset * typeSize));
                pto::TASSIGN(src1M2Tile, (uint64_t)(src1M2.GetAddr() + offset * typeSize));

                // delta = src1Mean - src0Mean
                TileDefine deltaTile(dstShape3, dstShape4);
                pto::TSUB(deltaTile, src1MeanTile, src0MeanTile);

                // newM2 = src0M2 + src1M2 + delta * delta
                TileDefine deltaSqTile(dstShape3, dstShape4);
                pto::TMUL(deltaSqTile, deltaTile, deltaTile);
                pto::TADD(dstM2Tile, src0M2Tile, src1M2Tile);
                pto::TADD(dstM2Tile, dstM2Tile, deltaSqTile);

                // newMean = src0Mean + delta (simplified pairwise)
                pto::TADD(dstMeanTile, src0MeanTile, deltaTile);
            }
        }
    }
}

#define OP_TILE_OP_WELFORDVAR_SINGLE TWelfordVarSingle
template <typename LastUse = LastUse3Dim<0, 0, 0>, typename T0, typename T1, typename T2, typename T3>
TILEOP void TWelfordVarSingle(T0 dstSumX, T1 dstSumX2, T2 src, T3 tmp)
{
    WelfordVarSingleTileCompute(dstSumX, dstSumX2, src, tmp);
}

#endif
