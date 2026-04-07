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

enum class ExpandTile : uint8_t {
    NONE,
    H,
    W,
    HW
};

template <unsigned... Axes>
TILEOP constexpr ExpandTile GetExpandTile()
{
    constexpr auto expandAxesNum = sizeof...(Axes);
    constexpr unsigned axesList[] = { Axes... };
    bool hasH = false;
    bool hasW = false;
    for (size_t i = 0; i < expandAxesNum; i++) {
        if (axesList[i] == 3) {
            hasH = true;
        }
        if (axesList[i] == 4) {
            hasW = true;
        }
    }
    
    if (hasH && hasW) {
        return ExpandTile::HW;
    } else if (hasH) {
        return ExpandTile::H;
    } else if (hasW) {
        return ExpandTile::W;
    }
    return ExpandTile::NONE;
}

#define OP_TILE_OP_EXPAND TExpand
template <typename LastUse = LastUse2Dim<0, 0>, ExpandTile expandTile, typename TileDst, typename TileSrc>
TILEOP void ExpandImpl(TileDst& dst, TileSrc& src)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    if constexpr (expandTile == ExpandTile::H) {
        PTO_WITH_LAST_USE(pto::TCOLEXPAND(dstTile, srcTile), n1, n2);
    } else if constexpr (expandTile == ExpandTile::W) {
        PTO_WITH_LAST_USE(pto::TROWEXPAND(dstTile, srcTile), n1, n2);
    } else if constexpr (expandTile == ExpandTile::HW) {
        // TODO
    } else {
        PTO_WITH_LAST_USE(pto::TMOV(dstTile, srcTile), n1, n2);
    }
}

template <typename LastUse = LastUse2Dim<0, 0>, unsigned... Axes, typename T0, typename T1>
TILEOP void TExpand(T0 dst, T1 src)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    const auto srcLayout = src.GetLayout();
    auto srcShape3 = srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto srcShape4 = srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    constexpr ExpandTile expandTile = GetExpandTile<Axes...>();

    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    using DstTileInfo = TensorTileInfo<T0>;
    using SrcTileInfo = TensorTileInfo<T1>;
    using DstDtype = std::conditional_t<std::is_same_v<typename T0::Type, bool>, uint8_t, typename T0::Type>;
    using SrcDtype = std::conditional_t<std::is_same_v<typename T1::Type, bool>, uint8_t, typename T1::Type>;
    constexpr auto typeSize = sizeof(DstDtype);

    if constexpr (expandTile == ExpandTile::NONE) {
        constexpr size_t minTileH = DstTileInfo::tileH < SrcTileInfo::tileH ? DstTileInfo::tileH : SrcTileInfo::tileH;
        using dstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, minTileH, DstTileInfo::tileW, pto::BLayout::RowMajor, -1, -1>;
        using srcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, minTileH, SrcTileInfo::tileW, pto::BLayout::RowMajor, -1, -1>;
        dstTileDefine dstTile(dstShape3, dstShape4);
        srcTileDefine srcTile(srcShape3, srcShape4);

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    auto dstOffset = GenTileOffset(dst, TileOffset(n0Index, n1Index, n2Index));
                    auto srcOffset = GenTileOffset(src, TileOffset(SrcTileInfo::tile0 == 1 ? 0 : n0Index, SrcTileInfo::tile1 == 1 ? 0 : n1Index,
                                                                   SrcTileInfo::tile2 == 1 ? 0 : n2Index));
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * typeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * typeSize));
                    ExpandImpl<LastUse, expandTile>(dstTile, srcTile);
                }
            }
        }
    } else {
        auto dstTile = PtoTile<T0>(dst);
        auto srcTile = PtoTile<T1>(src);
        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    auto dstOffset = TileOffset(n0Index, n1Index, n2Index);
                    auto srcOffset = TileOffset(SrcTileInfo::tile0 == 1 ? 0 : n0Index, SrcTileInfo::tile1 == 1 ? 0 : n1Index,
                                                SrcTileInfo::tile2 == 1 ? 0 : n2Index);
                    dstTile.Assign(dst, dstOffset);
                    srcTile.Assign(src, srcOffset);
                    ExpandImpl<LastUse, expandTile>(dstTile.Data(), srcTile.Data());
                }
            }
        }
    }
}
#endif // TILEOP_TILE_OPERATOR_VEC_EXPAND__H
