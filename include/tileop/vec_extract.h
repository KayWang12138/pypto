/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vec_binary.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_EXTRACT__H
#define TILEOP_TILE_OPERATOR_EXTRACT__H
#include "a2a3/utils/layout.h"
#include "a2a3/utils/tile_tensor.h"

template <int k, int extractMode, int isLargest, typename T0, typename T1>
TILEOP void TExtract(T0 dst, T1 src) {
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto dstTileH = Std::tuple_element<shapeSize - 2, typename T0::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<shapeSize - 1, typename T0::TileShape>::type::value;
    constexpr auto srcTileH = Std::tuple_element<shapeSize - 2, typename T1::TileShape>::type::value;
    constexpr auto srcTileW = Std::tuple_element<shapeSize - 1, typename T1::TileShape>::type::value;
    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }
    for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                using DstTileDefine = pto::Tile<pto::Location::Vec, typename T0::Type, dstTileH, dstTileW,
                                                pto::BLayout::RowMajor, -1, -1>;
                using SrcTileDefine = pto::Tile<pto::Location::Vec, typename T1::Type, srcTileH, srcTileW,
                                                pto::BLayout::RowMajor, -1, -1>;
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset));
                constexpr auto pattern = (extractMode == 0) ? pto::MaskPattern::P0101 : pto::MaskPattern::P1010;
                pto::TGATHER<DstTileDefine, SrcTileDefine, pattern>(dstTile, srcTile);

                if constexpr (extractMode == 0 && isLargest == 0) {
                    pipe_barrier(PIPE_V);
                    set_mask_count();
                    set_vector_mask(0, dstShape3 * dstTileW);
                    vadds((__ubuf__ int32_t *)((uint64_t)(dst.GetAddr() + dstOffset)),
                          (__ubuf__ int32_t *)((uint64_t)(dst.GetAddr() + dstOffset)),
                          0x80000000, 1, 1, 1, 8, 8);
                    set_mask_norm();
                    set_vector_mask(-1, -1);
                    pipe_barrier(PIPE_V);
                }
            }
        }
    }
}

#endif