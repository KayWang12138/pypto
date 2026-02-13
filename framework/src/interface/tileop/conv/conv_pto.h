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

// Copy data from DDR to L1
template <CopyInMode mode, typename Coord, typename T, typename U>
TILEOP void TLoadConv(T &dst, U &src, const Coord &coord, const int64_t &curH, const int64_t &curW) {
    return;
}

template <typename T, typename U>
TILEOP void TLoad3D(T &dst, U &src, const int64_t &mPos, const int64_t &kPos, 
                    const int64_t &padLeft, const int64_t &padRight, const int64_t &padTop, const int64_t &padBottom, const int64_t &padValue, 
                    const int64_t &filterH, const int64_t &filterW, const int64_t &dilationH, const int64_t &dilationW, 
                    const int64_t &strideH, const int64_t &strideW) {
    constexpr auto staticN = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto staticC1 = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    constexpr auto staticH = Std::tuple_element<CONV_IDX_2, typename U::TileShape>::type::value;
    constexpr auto staticW = Std::tuple_element<CONV_IDX_3, typename U::TileShape>::type::value;
    constexpr auto staticC0 = Std::tuple_element<CONV_IDX_4, typename U::TileShape>::type::value;
    constexpr auto bufferSize = staticN * staticC1 * staticH * staticW * staticC0;
    int n = GetConvShape<CONV_IDX_0>(src);
    int c1 = GetConvShape<CONV_IDX_1>(src);
    int h = GetConvShape<CONV_IDX_2>(src);
    int w = GetConvShape<CONV_IDX_3>(src);
    int c0 = GetConvShape<CONV_IDX_4>(src);
    using srcTensor = pto::ConvTile<pto::TileType::Mat, typename U::Type, bufferSize, pto::Layout::NC1HWC0, pto::ConvTileShape<-1, -1, -1, -1, -1>>;
    srcTensor l1(n, c1, h, w, c0);
    l1.SetFmapH(static_cast<uint16_t>(h));
    l1.SetFmapW(static_cast<uint16_t>(w));
    uint8_t values[4] = {static_cast<uint8_t>(padLeft), static_cast<uint8_t>(padRight), static_cast<uint8_t>(padTop), static_cast<uint8_t>(padBottom)};
    l1.SetPadListArray(values);
    l1.SetFilterH(filterH);
    l1.SetFilterW(filterW);
    l1.SetDilationH(dilationH);
    l1.SetDilationW(dilationW);
    l1.SetStrideH(strideH);
    l1.SetStrideW(strideW);
    l1.SetPadValue(padValue);
    l1.SetChannelSize(c1 * c0);

    constexpr auto staticML0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto staticKL0 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    int64_t mL0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t kL0 = GetConvShape<CONV_IDX_1>(dst);
    using dstTensor = pto::TileLeft<typename T::Type, staticML0, staticKL0, -1, -1>;
    dstTensor l0(mL0, kL0);

    pto::TASSIGN(l1, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(l0, static_cast<uint64_t>(dst.GetAddr()));
    pto::TSETFMATRIX(l1);
    pto::TIMG2COL(l0, l1, mPos, kPos);
}

template <typename T, typename U>
TILEOP void TLoad2D(T &dst, U &src, const int64_t &indexRow, const int64_t &indexCol) {
    constexpr auto staticC1HW = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto staticN1 = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    constexpr auto staticN0 = Std::tuple_element<CONV_IDX_2, typename U::TileShape>::type::value;
    constexpr auto staticC0 = Std::tuple_element<CONV_IDX_3, typename U::TileShape>::type::value;
    constexpr auto bufferSize = staticC1HW * staticN1 * staticN0 * staticC0;
    int c1hw = GetConvShape<CONV_IDX_0>(src);
    int n1 = GetConvShape<CONV_IDX_1>(src);
    int n0 = GetConvShape<CONV_IDX_2>(src);
    int c0 = GetConvShape<CONV_IDX_3>(src);
    using srcTensor = pto::ConvTile<pto::TileType::Mat, typename U::Type, bufferSize, pto::Layout::FRACTAL_Z, pto::ConvTileShape<-1, -1, -1, -1>>;
    srcTensor l1(c1hw, n1, n0, c0);

    constexpr auto staticKL0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto staticNL0 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    int64_t kL0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t nL0 = GetConvShape<CONV_IDX_1>(dst);
    using dstTensor = pto::TileRight<typename T::Type, staticKL0, staticNL0, -1, -1>;
    dstTensor l0(kL0, nL0);

    pto::TASSIGN(l1, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(l0, static_cast<uint64_t>(dst.GetAddr()));
    pto::TEXTRACT<dstTensor, srcTensor>(l0, l1, indexRow, indexCol);
}
} // namespace TileOp
#endif // TILEOP_TILE_OPERATOR_CONV_PTO__H