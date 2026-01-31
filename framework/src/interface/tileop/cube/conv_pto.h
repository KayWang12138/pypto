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
 * \file conv_pto.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_PTO__H
#define TILEOP_TILE_OPERATOR_CONV_PTO__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

namespace TileOp {

template <typename T, typename U>
TILEOP void TLoad3D(T &dst, U &src, const int64_t &mPos, const int64_t &kPos, 
                    uint8_t padList[4], uint16_t fmapH, uint16_t fmapW, uint16_t filterH, uint16_t filterW,
                    uint8_t dilationH, uint8_t dilationW, uint8_t strideH, uint8_t strideW, uint16_t channelSize,
                    T padValue, bool transpose, bool smallChannel, int capacity, int n, int c1, int c0) {
    // 构造 convCfg
    pto::Img2colTileConfig<T> convCfg;
    convCfg.fmapH = fmapH;
    convCfg.fmapW = fmapW;
    convCfg.padList = padList;
    convCfg.filterH = filterH;
    convCfg.filterW = filterW;
    convCfg.dilationH = dilationH;
    convCfg.dilationW = dilationW;
    convCfg.strideH = strideH;
    convCfg.strideW = strideW;
    convCfg.channelSize = channelSize;
    convCfg.padValue = padValue;
    convCfg.transpose = transpose;
    convCfg.smallChannel = smallChannel;
    
    // 构造 ConvTileData
    int bufferSize = capacity;
    using srcTensor = pto::ConvTile<pto::TileType::Mat, U, bufferSize, Layout::NC1HWC0, ConvTileShape<n, c1, fmapH, fmapW, c0>>;
    srcTensor l1;

    // 构造 TileData
    auto m = fmapH * fmapW;
    auto k = c1 * filterH * filterW * c0;
    using dstTensor = pto::Tile<pto::TileType::Left, T, m, k, pto::BLayout::RowMajor, -1, -1, pto::SLayout::RowMajor>;
    dstTensor l0;

    // 地址赋值
    pto::TASSIGN(l1, (uint64_t)src.GetAddr());
    pto::TASSIGN(l0, (uint64_t)dst.GetAddr());
    // 指令调用
    pto::TSETFMATRIX(&convCfg);
    pto::TIMG2COL<dstTensor, srcTensor, SetFmatrixMode::FMATRIX_A_MANUAL, U>(dst, src, *mPos, *kPos, &convcfg);
}

} // namespace TileOp
#endif // TILEOP_TILE_OPERATOR_CONV_PTO__H