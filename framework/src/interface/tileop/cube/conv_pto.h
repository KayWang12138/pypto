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
TILEOP void TLoad3D(T &dst, U &src, const int64_t &mPos, const int64_t &kPos, int n, int c1, uint16_t fmapH, uint16_t fmapW, int c0, 
                    uint8_t padLeft, uint8_t padRight, uint8_t padTop, uint8_t padBottom, uint16_t filterH, uint16_t filterW, 
                    uint8_t dilationH, uint8_t dilationW, uint8_t strideH, uint8_t strideW, T padValue, int mL0, int kL0) {
    // 构造 convCfg
    pto::Img2colTileConfig<T> convCfg;
    convCfg.fmapH = fmapH;
    convCfg.fmapW = fmapW;
    convCfg.filterH = filterH;
    convCfg.filterW = filterW;
    convCfg.dilationH = dilationH;
    convCfg.dilationW = dilationW;
    convCfg.strideH = strideH; 
    convCfg.strideW = strideW;
    convCfg.channelSize = c1 * c0;
    convCfg.padValue = padValue;
    convCfg.padList[0] = padLeft;
    convCfg.padList[1] = padRight;
    convCfg.padList[2] = padTop;
    convCfg.padList[3] = padBottom;
    
    // 构造 ConvTileData
    int bufferSize = n * c1 * fmapH * fmapW * c0 * sizeof(U);
    using srcTensor = pto::ConvTile<pto::TileType::Mat, U, bufferSize, Layout::NC1HWC0, pto::ConvTileShape<n, c1, fmapH, fmapW, c0>>;
    srcTensor l1;

    // 构造 TileData
    using dstTensor = pto::TileLeft<T, mL0, kL0, mL0, kL0>
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