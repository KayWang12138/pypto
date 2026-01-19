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
 * \file triul.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_GATHER__H
#define TILEOP_TILE_OPERATOR_GATHER__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_TRILU TTriLU
template <int diagonal, bool isUpper, typename DstTensor, typename SrcTensor>
TILEOP void TTriLU(DstTensor dst, SrcTensor src) {
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<DstTensor>(dst); // 内部自动算tileH,validH等
    auto srcTile = PtoTile<SrcTensor>(src);
    using TypeDefine = pto::Tile<pto::TileType::Vec, typename DstTensor::Type, dstTile.tileH, dstTile.tileW,
        pto::BLayout::RowMajor, dstTile.validH, dstTile.validW>;
    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                srcTile.Assign(src, tileOffsets);
                if ((diagonal >= dstTile.validW - 1 && !isUpper) || (diagonal <= -dstTile.validH + 1 && isUpper)) {
                    pto::TMOV(dstTile, srcTile);
                } else if (diagonal < dstTile.validW - 1 && diagonal > -dstTile.validH + 1) {
                    pto::TTRI<TypeDefine, isUpper, diagonal>(dstTile);
                    pto::MUL(dstTile, dstTile, srcTile);
                } else {
                    pto::EXPANDS(dstTile, 0);
                }
            }
        }
    }
}

#endif