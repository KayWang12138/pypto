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
#include <limits.h>
#include "utils/layout.h"
#include "utils/tile_tensor.h"

namespace TileOp {

template <typename T, typename U>
TILEOP void TLoad3D(T &dst, U &src, const uint16_t &mPos, const uint16_t &kPos, int n, int c1, int h, int w, int c0, 
                    uint8_t padLeft, uint8_t padRight, uint8_t padTop, uint8_t padBottom, T padValue, uint16_t filterH, uint16_t filterW, 
                    uint8_t dilationH, uint8_t dilationW, uint8_t strideH, uint8_t strideW, int mL0, int kL0) {

    pto::Img2colTileConfig<T> convCfg;
    convCfg.fmapH = static_cast<uint16_t>(h);
    convCfg.fmapW = static_cast<uint16_t>(w);
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

    size_t tempBufferSize = static_cast<size_t>(n) * 
                        static_cast<size_t>(c1) * 
                        static_cast<size_t>(h) * 
                        static_cast<size_t>(w) * 
                        static_cast<size_t>(c0) * 
                        sizeof(U);
    static_assert(tempBufferSize > static_cast<size_t>(INT_MAX), "[TLoad3D ERROR]: fmap shape in L1 exceed INT_MAX");
    int bufferSize = static_cast<int>(tempBufferSize);
    using srcTensor = pto::ConvTile<pto::TileType::Mat, U, bufferSize, Layout::NC1HWC0, pto::ConvTileShape<n, c1, h, w, c0>>;
    srcTensor l1;

    using dstTensor = pto::TileLeft<T, mL0, kL0>
    dstTensor l0;

    pto::TASSIGN(l1, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(l0, static_cast<uint64_t>(dst.GetAddr()));
    pto::TSETFMATRIX(&convCfg);
    pto::TIMG2COL<dstTensor, srcTensor, SetFmatrixMode::FMATRIX_A_MANUAL, U>(dst, src, *mPos, *kPos, &convcfg);
}

template <typename T, typename U>
TILEOP void TLoad2D(T &dst, U &src, const uint16_t &indexRow, const uint16_t &indexCol, int c1hw, int n1, int n0, int c0, int kL0, int nL0) {
    size_t tempBufferSize = static_cast<size_t>(c1hw) * 
                            static_cast<size_t>(n1) * 
                            static_cast<size_t>(n0) * 
                            static_cast<size_t>(c0) * 
                            sizeof(U);
    static_assert(tempBufferSize > static_cast<size_t>(INT_MAX), "[TLoad2D ERROR]: weight shape in L1 exceed INT_MAX");
    int bufferSize = static_cast<int>(tempBufferSize);
    using srcTensor = pto::ConvTile<pto::TileType::Mat, U, bufferSize, Layout::FRACTAL_Z, pto::ConvTileShape<c1hw, n1, n0, c0>>;
    srcTensor l1;

    using dstTensor = pto::TileRight<T, kL0, nL0>;
    dstTensor l0;

    pto::TASSIGN(l1, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(l0, static_cast<uint64_t>(dst.GetAddr()));
    pto::TEXTRACT<dstTensor, srcTensor>(dst, src, *indexRow, *indexCol);
}

} // namespace TileOp
#endif // TILEOP_TILE_OPERATOR_CONV_PTO__H