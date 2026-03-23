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
 * \file expand.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_EXPAND__H
#define TILEOP_TILE_OPERATOR_EXPAND__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_EXPAND TExpand

template <unsigned Head, unsigned... Tail>
struct FirstAxis {
    static constexpr unsigned value = Head;
};

template <typename LastUse = LastUse2Dim<0, 0>, unsigned... Axes, typename T0, typename T1>
TILEOP void TExpand(T0 dst, T1 src) {
    static_assert(sizeof...(Axes) >= 1, "At lease one expand axis required");
    constexpr unsigned axis = FirstAxis<Axes...>::value;
    const auto dstLayout = dst.GetLayout();
    
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    const auto srcLayout = src.GetLayout();
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    using SrcDtype = std::conditional_t<std::is_same_v<typename T1::Type, bool>, uint8_t, typename T1::Type>;
    using DstDtype = std::conditional_t<std::is_same_v<typename T0::Type, bool>, uint8_t, typename T0::Type>;

    constexpr auto typeSize = sizeof(DstDtype);

    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();

    if constexpr (axis == 3) {
        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    using dstTileDefine =
                        pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    using srcTileDefine =
                        pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
                    dstTileDefine dstTile(dstShape3, dstShape4);
                    srcTileDefine srcTile(srcShape3, srcShape4);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * typeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * typeSize));
                    PTO_WITH_LAST_USE(pto::TROWEXPAND(dstTile, srcTile), n1, n2);
                }
            }
        }
    } else if constexpr (axis == 2) {
        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    using dstTileDefine =
                        pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    using srcTileDefine =
                        pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
                    dstTileDefine dstTile(dstShape3, dstShape4);
                    srcTileDefine srcTile(srcShape3, srcShape4);
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * typeSize));
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * typeSize));
                    PTO_WITH_LAST_USE(pto::TCOLEXPAND(dstTile, srcTile), n1, n2);
                }
            }
        }
    } else if constexpr (axis == 1) {
        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1;    
                constexpr size_t minTileH = dstTileH > srcTileH ? srcTileH : dstTileH;
                using dstTileDefine =
                    pto::Tile<pto::TileType::Vec, DstDtype, minTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                using srcTileDefine =
                    pto::Tile<pto::TileType::Vec, SrcDtype, minTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
                dstTileDefine dstTile(dstShape3, dstShape4);
                srcTileDefine srcTile(srcShape3, srcShape4);
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * typeSize));
                for (LoopVar i = 0; i < dstShape2; i++) {
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + (dstOffset + i * dstTileH * dstTileW) * typeSize));
                    pto::TMOV(dstTile, srcTile);
                }
            }
        }
    } else if constexpr (axis == 0) {
        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            auto dstOffset = n0Index * dstStride0;
            auto srcOffset = n0Index * srcStride0;    
            constexpr size_t minTileH = dstTileH > srcTileH ? srcTileH : dstTileH;
            using dstTileDefine =
                pto::Tile<pto::TileType::Vec, DstDtype, minTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
            using srcTileDefine =
                pto::Tile<pto::TileType::Vec, SrcDtype, minTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
            dstTileDefine dstTile(dstShape3, dstShape4);
            srcTileDefine srcTile(srcShape3, srcShape4);

            constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
            dstShape2 = shapeSize > 2 ? TileOp::GetTensorTileShapeDim<T0, shapeSize - 3>() : dstShape2;
            for (LoopVar i = 0; i < dstShape1; ++i) {
                for (LoopVar j = 0; j < dstShape2; j++) {
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + (srcOffset + j * srcTileH * srcTileW) * typeSize));
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + (dstOffset + i * dstShape2 * dstTileH * dstTileW
                                                                        + j * dstTileH * dstTileW) * typeSize));
                    pto::TMOV(dstTile, srcTile);
                }
            }
        }
    }
}
#endif // TILEOP_TILE_OPERATOR_VEC_EXPAND__H