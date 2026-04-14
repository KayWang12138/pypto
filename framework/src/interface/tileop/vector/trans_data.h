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
 * \file trans_data.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_TRANS_DATA__H
#define TILEOP_TILE_OPERATOR_TRANS_DATA__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_TRANSDATA TTransData
template <int N, int C, int H, int W, typename DST, typename TYPEC, typename TMPDST, typename TMP, typename INPUT>
__aicore__ inline void TTransData(
    DST dst, TYPEC coordinate, TMPDST tmpDstTensor, TMP tmpTensor, INPUT input, int n, int c, int h, int w)
{
    constexpr size_t expectSize = 5;
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto tmpDstTensorN = Std::tuple_element<DIM_1ST, typename TMPDST::TileShape>::type::value;
    constexpr auto tmpDstTensorC1 = Std::tuple_element<DIM_2ND, typename TMPDST::TileShape>::type::value;
    constexpr auto tmpDstTensorH = Std::tuple_element<DIM_3RD, typename TMPDST::TileShape>::type::value;
    constexpr auto tmpDstTensorW = Std::tuple_element<DIM_4TH, typename TMPDST::TileShape>::type::value;
    constexpr auto C0 = Std::tuple_element<DIM_5TH, typename TMPDST::TileShape>::type::value;

    constexpr int elementSize = tmpDstTensorN * tmpDstTensorC1 * tmpDstTensorH * tmpDstTensorW * C0;
    constexpr int bufferSize = elementSize * sizeof(typename TMPDST::Type);

    using inputTileData = pto::ConvTile<
        pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCHW,
        pto::ConvTileShape<tmpDstTensorN, tmpDstTensorC1 * C0, tmpDstTensorH, tmpDstTensorW>>;
    using tmpDstTileData = pto::ConvTile<
        pto::TileType::Vec, typename TMPDST::Type, bufferSize, pto::Layout::NC1HWC0,
        pto::ConvTileShape<tmpDstTensorN, tmpDstTensorC1, tmpDstTensorH, tmpDstTensorW, C0>>;
    using tmpTileData = pto::Tile<
        pto::TileType::Vec, typename TMP::Type, tmpDstTensorH * tmpDstTensorW, C0, pto::BLayout::RowMajor,
        tmpDstTensorH * tmpDstTensorW, C0>;
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpTile;
    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDst, (uint64_t)tmpDstTensor.GetAddr());
    pto::TASSIGN(tmpTile, (uint64_t)tmpTensor.GetAddr());

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    pto::TTRANS(convTmpDst, convInput, tmpTile);

    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    const auto inputLayout = input.GetLayout();
    const auto tmpDstTensorLayout = tmpDstTensor.GetLayout();
    auto n0tmpDstStride = tmpDstTensorLayout.template GetStrideDim<DIM_1ST, expectSize>();
    auto n1tmpDstStride = tmpDstTensorLayout.template GetStrideDim<DIM_2ND, expectSize>();
    auto n2tmpDstStride = tmpDstTensorLayout.template GetStrideDim<DIM_3RD, expectSize>();
    auto n3tmpDstStride = tmpDstTensorLayout.template GetStrideDim<DIM_4TH, expectSize>();
    auto n4tmpDstStride = tmpDstTensorLayout.template GetStrideDim<DIM_5TH, expectSize>();

    constexpr auto dstStride0 = C * H * W;
    constexpr auto dstStride1 = H * W * C0;
    constexpr auto dstStride2 = W * C0;
    constexpr auto dstStride3 = C0;

    const auto gmLayout = dst.GetLayout();
    auto tmpDstAddr = (__ubuf__ typename TMPDST::Type*)((uint64_t)(tmpDstTensor.GetAddr()));
    auto DstAddr = (__gm__ typename DST::Type*)((uint64_t)(dst.GetAddr()));
    size_t gmOffset = static_cast<size_t>(gmLayout.template GetGmOffset<TYPEC, 5>(coordinate));
    auto inputN = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputC = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto inputC1 = inputC / C0;
    using TileDefine =
        pto::Tile<pto::TileType::Vec, typename INPUT::Type, tmpDstTensorW, C0, pto::BLayout::RowMajor, -1, -1>;
    using GlobalData =
        pto::GlobalTensor<typename DST::Type, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    TileDefine tmpDstTile(inputW, C0);
    for (LoopVar i = 0; i < inputN; i++) {
        for (LoopVar j = 0; j < inputC1; j++) {
            for (LoopVar k = 0; k < inputH; k++) {
                uint64_t tmpDstStride = i * n0tmpDstStride + j * n1tmpDstStride + k * n2tmpDstStride;
                uint64_t DstStride =
                    (n + i) * dstStride0 + (c / C0 + j) * dstStride1 + (h + k) * dstStride2 + w * dstStride3;
                pto::TASSIGN(tmpDstTile, (uint64_t)(tmpDstAddr + tmpDstStride));
                GlobalData globalData(
                    DstAddr + gmOffset + DstStride, pto::Shape(1, 1, 1, inputW, C0), pto::Stride(1, 1, 1, C0, 1));
                pto::TSTORE(globalData, tmpDstTile);
            }
        }
    }
}

#endif
