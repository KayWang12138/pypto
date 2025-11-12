/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_REDUCE__H
#define TILEOP_TILE_OPERATOR_REDUCE__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename T0, typename T1, typename T2>
TILEOP void TRowSumSingle(T0 dst, T1 src, T2 tmp) {
    using ShapeValueType = typename Std::tuple_element<0, typename T1::Shape>::type;
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto tmpShapeSize = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto tmpTileH = Std::tuple_element<tmpShapeSize - 2, typename T2::TileShape>::type::value;
    constexpr auto tmpTileW = Std::tuple_element<tmpShapeSize - 1, typename T2::TileShape>::type::value;
    using TmpTileDefine =
            pto::Tile<pto::Location::Vec, typename T2::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor, tmpTileH, tmpTileW>;
    TmpTileDefine tmpTile;
    if constexpr (TileOp::IsConstContinous<T0, T1>() == true) {
        constexpr auto srcTileH = TileOp::GetOutterAxisMergeResult<srcShapeSize, typename T1::TileShape>();
        constexpr auto srcTileW = Std::tuple_element<srcShapeSize - 1, typename T1::TileShape>::type::value;
        constexpr auto dstTileH = TileOp::GetOutterAxisMergeResult<dstShapeSize, typename T0::TileShape>();
        constexpr auto dstTileW = Std::tuple_element<dstShapeSize - 1, typename T0::TileShape>::type::value;
        using SrcTileDefine =
            pto::Tile<pto::Location::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, srcTileH, srcTileW>;
        using DstTileDefine =
            pto::Tile<pto::Location::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, dstTileH, dstTileW>;
        SrcTileDefine srcTile;
        DstTileDefine dstTile;
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        pto::TASSIGN(tmpTile, (uint64_t)tmp.GetAddr());
        pto::TROWSUM(dstTile, srcTile, tmpTile);
        return;
    }
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    constexpr auto dstTileH = Std::tuple_element<dstShapeSize - 2, typename T0::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<dstShapeSize - 1, typename T0::TileShape>::type::value;

    const auto srcLayout = src.GetLayout();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    constexpr auto srcTileH = Std::tuple_element<srcShapeSize - 2, typename T1::TileShape>::type::value;
    constexpr auto srcTileW = Std::tuple_element<srcShapeSize - 1, typename T1::TileShape>::type::value;
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                using DstTileDefine =
                    pto::Tile<pto::Location::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                using SrcTileDefine =
                    pto::Tile<pto::Location::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * srcTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                pto::TROWSUM(dstTile, srcTile, tmpTile);
            }
        }
    }
}
#endif