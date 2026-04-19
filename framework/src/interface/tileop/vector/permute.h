/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file permute.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_PERMUTE__H
#define TILEOP_TILE_OPERATOR_PERMUTE__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <cstdint>

namespace permute_detail {

template <int tileH, int tileW, typename dstType, typename ubType>
TILEOP void LoadVecTile(
    __ubuf__ dstType* dstPtr, __gm__ ubType* srcAddr, int64_t gmOff, int64_t innerDim, int64_t srcStride4)
{
    using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
    using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = pto::GlobalTensor<ubType, ShapeDim5, StrideDim5>;
    using TileDefine = pto::Tile<pto::TileType::Vec, dstType, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    TileDefine dstTile(1, innerDim);
    GlobalData srcGlobal(srcAddr + gmOff, pto::Shape(1, 1, 1, 1, innerDim), pto::Stride(0, 0, 0, 0, srcStride4));
    pto::TASSIGN(dstTile, (uint64_t)dstPtr);
    pto::TLOAD(dstTile, srcGlobal);
}

template <int tileH, int tileW, typename dstType, typename ubType>
TILEOP void StoreVecTile(
    __gm__ dstType* dstAddr, __ubuf__ ubType* srcPtr, int64_t gmOff, int64_t innerDim, int64_t dstStride4)
{
    using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
    using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = pto::GlobalTensor<dstType, ShapeDim5, StrideDim5>;
    using TileDefine = pto::Tile<pto::TileType::Vec, ubType, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    TileDefine srcTile(1, innerDim);
    GlobalData dstGlobal(dstAddr + gmOff, pto::Shape(1, 1, 1, 1, innerDim), pto::Stride(0, 0, 0, 0, dstStride4));
    pto::TASSIGN(srcTile, (uint64_t)srcPtr);
    pto::TSTORE(dstGlobal, srcTile);
}

template <
    int pad, int axis0, int axis1, int tileH, int tileW, typename dstType, typename ubType, typename SrcLayoutT,
    typename DstLayoutT>
TILEOP void PermuteDim2(
    __ubuf__ dstType* dstAddr, __gm__ ubType* srcAddr, const SrcLayoutT& srcLayout, const DstLayoutT& dstLayout,
    int64_t srcStride0, int64_t srcStride1, int64_t srcStride4)
{
    auto d0 = dstLayout.template GetShapeDim<0, 2>();
    auto d1 = dstLayout.template GetShapeDim<1, 2>();
    auto ds0 = dstLayout.template GetStrideDim<0, 2>();
    for (LoopVar i0 = 0; i0 < d0; ++i0) {
        int64_t sc[5] = {0, 0, 0, 0, 0};
        sc[pad + axis0] = static_cast<int64_t>(i0);
        sc[pad + axis1] = 0;
        auto gmOff = sc[0] * srcStride0 + sc[1] * srcStride1 + sc[2] * 0 + sc[3] * 0 + sc[4] * srcStride4;
        LoadVecTile<tileH, tileW>(dstAddr + i0 * ds0, srcAddr, gmOff, d1, srcStride4);
    }
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
}

template <
    int pad, int axis0, int axis1, int axis2, int tileH, int tileW, typename dstType, typename ubType,
    typename SrcLayoutT, typename DstLayoutT>
TILEOP void PermuteDim3(
    __ubuf__ dstType* dstAddr, __gm__ ubType* srcAddr, const SrcLayoutT& srcLayout, const DstLayoutT& dstLayout,
    int64_t srcStride0, int64_t srcStride1, int64_t srcStride2, int64_t srcStride3, int64_t srcStride4)
{
    auto d0 = dstLayout.template GetShapeDim<0, 3>();
    auto d1 = dstLayout.template GetShapeDim<1, 3>();
    auto d2 = dstLayout.template GetShapeDim<2, 3>();
    auto ds0 = dstLayout.template GetStrideDim<0, 3>();
    auto ds1 = dstLayout.template GetStrideDim<1, 3>();
    for (LoopVar i0 = 0; i0 < d0; ++i0) {
        __ubuf__ dstType* dst1 = dstAddr + i0 * ds0;
        for (LoopVar i1 = 0; i1 < d1; ++i1) {
            int64_t sc[5] = {0, 0, 0, 0, 0};
            sc[pad + axis0] = static_cast<int64_t>(i0);
            sc[pad + axis1] = static_cast<int64_t>(i1);
            sc[pad + axis2] = 0;
            auto gmOff =
                sc[0] * srcStride0 + sc[1] * srcStride1 + sc[2] * srcStride2 + sc[3] * srcStride3 + sc[4] * srcStride4;
            LoadVecTile<tileH, tileW>(dst1 + i1 * ds1, srcAddr, gmOff, d2, srcStride4);
        }
    }
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
}

template <
    int pad, int axis0, int axis1, int axis2, int axis3, int tileH, int tileW, typename dstType, typename ubType,
    typename SrcLayoutT, typename DstLayoutT>
TILEOP void PermuteDim4(
    __ubuf__ dstType* dstAddr, __gm__ ubType* srcAddr, const SrcLayoutT& srcLayout, const DstLayoutT& dstLayout,
    int64_t srcStride0, int64_t srcStride1, int64_t srcStride2, int64_t srcStride3, int64_t srcStride4)
{
    auto d0 = dstLayout.template GetShapeDim<0, 4>();
    auto d1 = dstLayout.template GetShapeDim<1, 4>();
    auto d2 = dstLayout.template GetShapeDim<2, 4>();
    auto d3 = dstLayout.template GetShapeDim<3, 4>();
    auto ds0 = dstLayout.template GetStrideDim<0, 4>();
    auto ds1 = dstLayout.template GetStrideDim<1, 4>();
    auto ds2 = dstLayout.template GetStrideDim<2, 4>();
    for (LoopVar i0 = 0; i0 < d0; ++i0) {
        __ubuf__ dstType* dst1 = dstAddr + i0 * ds0;
        for (LoopVar i1 = 0; i1 < d1; ++i1) {
            __ubuf__ dstType* dst2 = dst1 + i1 * ds1;
            for (LoopVar i2 = 0; i2 < d2; ++i2) {
                int64_t sc[5] = {0, 0, 0, 0, 0};
                sc[pad + axis0] = static_cast<int64_t>(i0);
                sc[pad + axis1] = static_cast<int64_t>(i1);
                sc[pad + axis2] = static_cast<int64_t>(i2);
                sc[pad + axis3] = 0;
                auto gmOff = sc[0] * srcStride0 + sc[1] * srcStride1 + sc[2] * srcStride2 + sc[3] * srcStride3 +
                             sc[4] * srcStride4;
                LoadVecTile<tileH, tileW>(dst2 + i2 * ds2, srcAddr, gmOff, d3, srcStride4);
            }
        }
    }
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
}

template <
    int pad, int axis0, int axis1, int axis2, int axis3, int axis4, int tileH, int tileW, typename dstType,
    typename ubType, typename SrcLayoutT, typename DstLayoutT>
TILEOP void PermuteDim5(
    __ubuf__ dstType* dstAddr, __gm__ ubType* srcAddr, const SrcLayoutT& srcLayout, const DstLayoutT& dstLayout,
    int64_t srcStride0, int64_t srcStride1, int64_t srcStride2, int64_t srcStride3, int64_t srcStride4)
{
    auto d0 = dstLayout.template GetShapeDim<0, 5>();
    auto d1 = dstLayout.template GetShapeDim<1, 5>();
    auto d2 = dstLayout.template GetShapeDim<2, 5>();
    auto d3 = dstLayout.template GetShapeDim<3, 5>();
    auto d4 = dstLayout.template GetShapeDim<4, 5>();
    auto ds0 = dstLayout.template GetStrideDim<0, 5>();
    auto ds1 = dstLayout.template GetStrideDim<1, 5>();
    auto ds2 = dstLayout.template GetStrideDim<2, 5>();
    auto ds3 = dstLayout.template GetStrideDim<3, 5>();
    for (LoopVar i0 = 0; i0 < d0; ++i0) {
        __ubuf__ dstType* dst1 = dstAddr + i0 * ds0;
        for (LoopVar i1 = 0; i1 < d1; ++i1) {
            __ubuf__ dstType* dst2 = dst1 + i1 * ds1;
            for (LoopVar i2 = 0; i2 < d2; ++i2) {
                __ubuf__ dstType* dst3 = dst2 + i2 * ds2;
                for (LoopVar i3 = 0; i3 < d3; ++i3) {
                    int64_t sc[5] = {0, 0, 0, 0, 0};
                    sc[pad + axis0] = static_cast<int64_t>(i0);
                    sc[pad + axis1] = static_cast<int64_t>(i1);
                    sc[pad + axis2] = static_cast<int64_t>(i2);
                    sc[pad + axis3] = static_cast<int64_t>(i3);
                    sc[pad + axis4] = 0;
                    auto gmOff = sc[0] * srcStride0 + sc[1] * srcStride1 + sc[2] * srcStride2 + sc[3] * srcStride3 +
                                 sc[4] * srcStride4;
                    LoadVecTile<tileH, tileW>(dst3 + i3 * ds3, srcAddr, gmOff, d4, srcStride4);
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
}

template <
    int pad, int axis0, int axis1, int tileH, int tileW, typename dstType, typename ubType, typename SrcLayoutT,
    typename DstLayoutT>
TILEOP void PermuteMoveOutDim2(
    __gm__ dstType* dstAddr, __ubuf__ ubType* srcAddr, const SrcLayoutT& srcLayout, const DstLayoutT& dstLayout,
    int64_t srcStride0, int64_t srcStride1, int64_t dstStride0, int64_t dstStride1, int64_t dstStride4)
{
    auto d0 = dstLayout.template GetShapeDim<0, 2>();
    auto d1 = dstLayout.template GetShapeDim<1, 2>();
    for (LoopVar i0 = 0; i0 < d0; ++i0) {
        __ubuf__ ubType* src1 = srcAddr + i0 * srcStride0;
        int64_t dc[5] = {0, 0, 0, 0, 0};
        dc[pad + axis0] = static_cast<int64_t>(i0);
        dc[pad + axis1] = 0;
        auto gmOff = dc[0] * dstStride0 + dc[1] * dstStride1 + dc[2] * 0 + dc[3] * 0 + dc[4] * dstStride4;
        StoreVecTile<tileH, tileW>(dstAddr, src1, gmOff, d1, dstStride4);
    }
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
}

template <
    int pad, int axis0, int axis1, int axis2, int tileH, int tileW, typename dstType, typename ubType,
    typename SrcLayoutT, typename DstLayoutT>
TILEOP void PermuteMoveOutDim3(
    __gm__ dstType* dstAddr, __ubuf__ ubType* srcAddr, const SrcLayoutT& srcLayout, const DstLayoutT& dstLayout,
    int64_t srcStride0, int64_t srcStride1, int64_t srcStride2, int64_t dstStride0, int64_t dstStride1,
    int64_t dstStride2, int64_t dstStride3, int64_t dstStride4)
{
    auto d0 = dstLayout.template GetShapeDim<0, 3>();
    auto d1 = dstLayout.template GetShapeDim<1, 3>();
    auto d2 = dstLayout.template GetShapeDim<2, 3>();
    for (LoopVar i0 = 0; i0 < d0; ++i0) {
        __ubuf__ ubType* src1 = srcAddr + i0 * srcStride0;
        for (LoopVar i1 = 0; i1 < d1; ++i1) {
            __ubuf__ ubType* src2 = src1 + i1 * srcStride1;
            int64_t dc[5] = {0, 0, 0, 0, 0};
            dc[pad + axis0] = static_cast<int64_t>(i0);
            dc[pad + axis1] = static_cast<int64_t>(i1);
            dc[pad + axis2] = 0;
            auto gmOff =
                dc[0] * dstStride0 + dc[1] * dstStride1 + dc[2] * dstStride2 + dc[3] * dstStride3 + dc[4] * dstStride4;
            StoreVecTile<tileH, tileW>(dstAddr, src2, gmOff, d2, dstStride4);
        }
    }
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
}

template <
    int pad, int axis0, int axis1, int axis2, int axis3, int tileH, int tileW, typename dstType, typename ubType,
    typename SrcLayoutT, typename DstLayoutT>
TILEOP void PermuteMoveOutDim4(
    __gm__ dstType* dstAddr, __ubuf__ ubType* srcAddr, const SrcLayoutT& srcLayout, const DstLayoutT& dstLayout,
    int64_t srcStride0, int64_t srcStride1, int64_t srcStride2, int64_t srcStride3, int64_t dstStride0,
    int64_t dstStride1, int64_t dstStride2, int64_t dstStride3, int64_t dstStride4)
{
    auto d0 = dstLayout.template GetShapeDim<0, 4>();
    auto d1 = dstLayout.template GetShapeDim<1, 4>();
    auto d2 = dstLayout.template GetShapeDim<2, 4>();
    auto d3 = dstLayout.template GetShapeDim<3, 4>();
    for (LoopVar i0 = 0; i0 < d0; ++i0) {
        __ubuf__ ubType* src1 = srcAddr + i0 * srcStride0;
        for (LoopVar i1 = 0; i1 < d1; ++i1) {
            __ubuf__ ubType* src2 = src1 + i1 * srcStride1;
            for (LoopVar i2 = 0; i2 < d2; ++i2) {
                __ubuf__ ubType* src3 = src2 + i2 * srcStride2;
                int64_t dc[5] = {0, 0, 0, 0, 0};
                dc[pad + axis0] = static_cast<int64_t>(i0);
                dc[pad + axis1] = static_cast<int64_t>(i1);
                dc[pad + axis2] = static_cast<int64_t>(i2);
                dc[pad + axis3] = 0;
                auto gmOff = dc[0] * dstStride0 + dc[1] * dstStride1 + dc[2] * dstStride2 + dc[3] * dstStride3 +
                             dc[4] * dstStride4;
                StoreVecTile<tileH, tileW>(dstAddr, src3, gmOff, d3, dstStride4);
            }
        }
    }
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
}

template <
    int pad, int axis0, int axis1, int axis2, int axis3, int axis4, int tileH, int tileW, typename dstType,
    typename ubType, typename SrcLayoutT, typename DstLayoutT>
TILEOP void PermuteMoveOutDim5(
    __gm__ dstType* dstAddr, __ubuf__ ubType* srcAddr, const SrcLayoutT& srcLayout, const DstLayoutT& dstLayout,
    int64_t srcStride0, int64_t srcStride1, int64_t srcStride2, int64_t srcStride3, int64_t srcStride4,
    int64_t dstStride0, int64_t dstStride1, int64_t dstStride2, int64_t dstStride3, int64_t dstStride4)
{
    auto d0 = dstLayout.template GetShapeDim<0, 5>();
    auto d1 = dstLayout.template GetShapeDim<1, 5>();
    auto d2 = dstLayout.template GetShapeDim<2, 5>();
    auto d3 = dstLayout.template GetShapeDim<3, 5>();
    auto d4 = dstLayout.template GetShapeDim<4, 5>();
    for (LoopVar i0 = 0; i0 < d0; ++i0) {
        __ubuf__ ubType* src1 = srcAddr + i0 * srcStride0;
        for (LoopVar i1 = 0; i1 < d1; ++i1) {
            __ubuf__ ubType* src2 = src1 + i1 * srcStride1;
            for (LoopVar i2 = 0; i2 < d2; ++i2) {
                __ubuf__ ubType* src3 = src2 + i2 * srcStride2;
                for (LoopVar i3 = 0; i3 < d3; ++i3) {
                    int64_t dc[5] = {0, 0, 0, 0, 0};
                    dc[pad + axis0] = static_cast<int64_t>(i0);
                    dc[pad + axis1] = static_cast<int64_t>(i1);
                    dc[pad + axis2] = static_cast<int64_t>(i2);
                    dc[pad + axis3] = static_cast<int64_t>(i3);
                    dc[pad + axis4] = 0;
                    auto gmOff = dc[0] * dstStride0 + dc[1] * dstStride1 + dc[2] * dstStride2 + dc[3] * dstStride3 +
                                 dc[4] * dstStride4;
                    StoreVecTile<tileH, tileW>(dstAddr, src3 + i3 * srcStride3, gmOff, d4, dstStride4);
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
}

template <typename dstType, typename ubType>
TILEOP void CopyElement(
    __ubuf__ dstType* dstAddr, __gm__ ubType* srcAddr, int64_t dstOff, const int64_t* sc, int64_t ss0, int64_t ss1,
    int64_t ss2, int64_t ss3, int64_t ss4)
{
    auto gmOff = sc[0] * ss0 + sc[1] * ss1 + sc[2] * ss2 + sc[3] * ss3 + sc[4] * ss4;
    dstAddr[dstOff] = srcAddr[gmOff];
}

template <int axis0, int axis1, typename dstType, typename ubType, typename SrcLayoutT, typename DstLayoutT>
TILEOP void PermuteEleDim2(
    __ubuf__ dstType* dstAddr, __gm__ ubType* srcAddr, const SrcLayoutT& srcLayout, const DstLayoutT& dstLayout)
{
    auto ss0 = srcLayout.template GetStrideDim<0, 2>();
    auto ss1 = srcLayout.template GetStrideDim<1, 2>();
    auto d0 = dstLayout.template GetShapeDim<0, 2>();
    auto d1 = dstLayout.template GetShapeDim<1, 2>();
    auto ds0 = dstLayout.template GetStrideDim<0, 2>();
    auto ds1 = dstLayout.template GetStrideDim<1, 2>();
    for (LoopVar i0 = 0; i0 < d0; ++i0) {
        for (LoopVar i1 = 0; i1 < d1; ++i1) {
            int64_t sc[2] = {0, 0};
            sc[axis0] = static_cast<int64_t>(i0);
            sc[axis1] = static_cast<int64_t>(i1);
            auto gmOff = sc[0] * ss0 + sc[1] * ss1;
            dstAddr[i0 * ds0 + i1 * ds1] = srcAddr[gmOff];
        }
    }
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
}

template <int axis0, int axis1, int axis2, typename dstType, typename ubType, typename SrcLayoutT, typename DstLayoutT>
TILEOP void PermuteEleDim3(
    __ubuf__ dstType* dstAddr, __gm__ ubType* srcAddr, const SrcLayoutT& srcLayout, const DstLayoutT& dstLayout)
{
    auto ss0 = srcLayout.template GetStrideDim<0, 3>();
    auto ss1 = srcLayout.template GetStrideDim<1, 3>();
    auto ss2 = srcLayout.template GetStrideDim<2, 3>();
    auto d0 = dstLayout.template GetShapeDim<0, 3>();
    auto d1 = dstLayout.template GetShapeDim<1, 3>();
    auto d2 = dstLayout.template GetShapeDim<2, 3>();
    auto ds0 = dstLayout.template GetStrideDim<0, 3>();
    auto ds1 = dstLayout.template GetStrideDim<1, 3>();
    auto ds2 = dstLayout.template GetStrideDim<2, 3>();
    for (LoopVar i0 = 0; i0 < d0; ++i0) {
        for (LoopVar i1 = 0; i1 < d1; ++i1) {
            for (LoopVar i2 = 0; i2 < d2; ++i2) {
                int64_t sc[3] = {0, 0, 0};
                sc[axis0] = static_cast<int64_t>(i0);
                sc[axis1] = static_cast<int64_t>(i1);
                sc[axis2] = static_cast<int64_t>(i2);
                auto gmOff = sc[0] * ss0 + sc[1] * ss1 + sc[2] * ss2;
                dstAddr[i0 * ds0 + i1 * ds1 + i2 * ds2] = srcAddr[gmOff];
            }
        }
    }
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
}

template <
    int axis0, int axis1, int axis2, int axis3, typename dstType, typename ubType, typename SrcLayoutT,
    typename DstLayoutT>
TILEOP void PermuteEleDim4(
    __ubuf__ dstType* dstAddr, __gm__ ubType* srcAddr, const SrcLayoutT& srcLayout, const DstLayoutT& dstLayout)
{
    auto ss0 = srcLayout.template GetStrideDim<0, 4>();
    auto ss1 = srcLayout.template GetStrideDim<1, 4>();
    auto ss2 = srcLayout.template GetStrideDim<2, 4>();
    auto ss3 = srcLayout.template GetStrideDim<3, 4>();
    auto d0 = dstLayout.template GetShapeDim<0, 4>();
    auto d1 = dstLayout.template GetShapeDim<1, 4>();
    auto d2 = dstLayout.template GetShapeDim<2, 4>();
    auto d3 = dstLayout.template GetShapeDim<3, 4>();
    auto ds0 = dstLayout.template GetStrideDim<0, 4>();
    auto ds1 = dstLayout.template GetStrideDim<1, 4>();
    auto ds2 = dstLayout.template GetStrideDim<2, 4>();
    auto ds3 = dstLayout.template GetStrideDim<3, 4>();
    for (LoopVar i0 = 0; i0 < d0; ++i0) {
        for (LoopVar i1 = 0; i1 < d1; ++i1) {
            for (LoopVar i2 = 0; i2 < d2; ++i2) {
                for (LoopVar i3 = 0; i3 < d3; ++i3) {
                    int64_t sc[4] = {0, 0, 0, 0};
                    sc[axis0] = static_cast<int64_t>(i0);
                    sc[axis1] = static_cast<int64_t>(i1);
                    sc[axis2] = static_cast<int64_t>(i2);
                    sc[axis3] = static_cast<int64_t>(i3);
                    auto gmOff = sc[0] * ss0 + sc[1] * ss1 + sc[2] * ss2 + sc[3] * ss3;
                    dstAddr[i0 * ds0 + i1 * ds1 + i2 * ds2 + i3 * ds3] = srcAddr[gmOff];
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
}

template <
    int axis0, int axis1, int axis2, int axis3, int axis4, typename dstType, typename ubType, typename SrcLayoutT,
    typename DstLayoutT>
TILEOP void PermuteEleDim5(
    __ubuf__ dstType* dstAddr, __gm__ ubType* srcAddr, const SrcLayoutT& srcLayout, const DstLayoutT& dstLayout)
{
    auto ss0 = srcLayout.template GetStrideDim<0, 5>();
    auto ss1 = srcLayout.template GetStrideDim<1, 5>();
    auto ss2 = srcLayout.template GetStrideDim<2, 5>();
    auto ss3 = srcLayout.template GetStrideDim<3, 5>();
    auto ss4 = srcLayout.template GetStrideDim<4, 5>();
    auto d0 = dstLayout.template GetShapeDim<0, 5>();
    auto d1 = dstLayout.template GetShapeDim<1, 5>();
    auto d2 = dstLayout.template GetShapeDim<2, 5>();
    auto d3 = dstLayout.template GetShapeDim<3, 5>();
    auto d4 = dstLayout.template GetShapeDim<4, 5>();
    auto ds0 = dstLayout.template GetStrideDim<0, 5>();
    auto ds1 = dstLayout.template GetStrideDim<1, 5>();
    auto ds2 = dstLayout.template GetStrideDim<2, 5>();
    auto ds3 = dstLayout.template GetStrideDim<3, 5>();
    for (LoopVar i0 = 0; i0 < d0; ++i0) {
        for (LoopVar i1 = 0; i1 < d1; ++i1) {
            for (LoopVar i2 = 0; i2 < d2; ++i2) {
                for (LoopVar i3 = 0; i3 < d3; ++i3) {
                    for (LoopVar i4 = 0; i4 < d4; ++i4) {
                        int64_t sc[5] = {0, 0, 0, 0, 0};
                        sc[axis0] = static_cast<int64_t>(i0);
                        sc[axis1] = static_cast<int64_t>(i1);
                        sc[axis2] = static_cast<int64_t>(i2);
                        sc[axis3] = static_cast<int64_t>(i3);
                        sc[axis4] = static_cast<int64_t>(i4);
                        auto gmOff = sc[0] * ss0 + sc[1] * ss1 + sc[2] * ss2 + sc[3] * ss3 + sc[4] * ss4;
                        dstAddr[i0 * ds0 + i1 * ds1 + i2 * ds2 + i3 * ds3 + i4] = srcAddr[gmOff];
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
}

} // namespace permute_detail

#define OP_TILE_OP_PERMUTE TPermute
template <int axis0, int axis1, int axis2, int axis3, int axis4, int dimCount, typename T0, typename T1, typename C0>
TILEOP void TPermute(T0 dst, T1 src, C0 srcCoordinate)
{
    constexpr size_t srcExpectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto srcOffset = srcLayout.template GetGmOffset<C0, srcExpectSize>(srcCoordinate);
    using ubType = typename T1::Type;
    using dstType = typename T0::Type;
    __gm__ ubType* srcAddr = (__gm__ ubType*)((uint64_t)(src.GetAddr()));
    srcAddr += srcOffset;
    __ubuf__ dstType* dstAddr = (__ubuf__ dstType*)((uint64_t)(dst.GetAddr()));

    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto tileH = Std::tuple_element<shapeSize - 2, typename T0::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename T0::TileShape>::type::value;
    auto srcStride0 = srcLayout.template GetStrideDim<0, srcExpectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, srcExpectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, srcExpectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, srcExpectSize>();
    auto srcStride4 = srcLayout.template GetStrideDim<4, srcExpectSize>();

    constexpr int pad = 5 - dimCount;
    if constexpr (dimCount == 2 && axis0 == 1 && axis1 == 0) {
        permute_detail::PermuteDim2<pad, axis0, axis1, tileH, tileW, dstType, ubType>(
            dstAddr, srcAddr, srcLayout, dstLayout, srcStride0, srcStride1, srcStride4);
    } else if constexpr (dimCount == 3) {
        permute_detail::PermuteDim3<pad, axis0, axis1, axis2, tileH, tileW, dstType, ubType>(
            dstAddr, srcAddr, srcLayout, dstLayout, srcStride0, srcStride1, srcStride2, srcStride3, srcStride4);
    } else if constexpr (dimCount == 4) {
        permute_detail::PermuteDim4<pad, axis0, axis1, axis2, axis3, tileH, tileW, dstType, ubType>(
            dstAddr, srcAddr, srcLayout, dstLayout, srcStride0, srcStride1, srcStride2, srcStride3, srcStride4);
    } else if constexpr (dimCount == 5) {
        permute_detail::PermuteDim5<pad, axis0, axis1, axis2, axis3, axis4, tileH, tileW, dstType, ubType>(
            dstAddr, srcAddr, srcLayout, dstLayout, srcStride0, srcStride1, srcStride2, srcStride3, srcStride4);
    }
}

#define OP_TILE_OP_PERMUTE_ELEMENT TPermuteElewise
template <int axis0, int axis1, int axis2, int axis3, int axis4, int dimCount, typename T0, typename T1, typename C0>
TILEOP void TPermuteElewise(T0 dst, T1 src, C0 srcCoordinate)
{
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    constexpr size_t N = Std::tuple_size<typename T1::Shape>::value;
    using ubType = typename T1::Type;
    using dstType = typename T0::Type;
    __gm__ ubType* srcAddr = (__gm__ ubType*)((uint64_t)(src.GetAddr()));
    __ubuf__ dstType* dstAddr = (__ubuf__ dstType*)((uint64_t)(dst.GetAddr()));

    if constexpr (dimCount == 2) {
        auto ss0 = srcLayout.template GetStrideDim<0, N>();
        auto ss1 = srcLayout.template GetStrideDim<1, N>();
        auto c0 = Std::get<0>(srcCoordinate);
        auto c1 = Std::get<1>(srcCoordinate);
        srcAddr += c0 * ss0 + c1 * ss1;
        permute_detail::PermuteEleDim2<axis0, axis1, dstType, ubType>(dstAddr, srcAddr, srcLayout, dstLayout);
    } else if constexpr (dimCount == 3) {
        auto ss0 = srcLayout.template GetStrideDim<0, N>();
        auto ss1 = srcLayout.template GetStrideDim<1, N>();
        auto ss2 = srcLayout.template GetStrideDim<2, N>();
        auto c0 = Std::get<0>(srcCoordinate);
        auto c1 = Std::get<1>(srcCoordinate);
        auto c2 = Std::get<2>(srcCoordinate);
        srcAddr += c0 * ss0 + c1 * ss1 + c2 * ss2;
        permute_detail::PermuteEleDim3<axis0, axis1, axis2, dstType, ubType>(dstAddr, srcAddr, srcLayout, dstLayout);
    } else if constexpr (dimCount == 4) {
        auto ss0 = srcLayout.template GetStrideDim<0, N>();
        auto ss1 = srcLayout.template GetStrideDim<1, N>();
        auto ss2 = srcLayout.template GetStrideDim<2, N>();
        auto ss3 = srcLayout.template GetStrideDim<3, N>();
        auto c0 = Std::get<0>(srcCoordinate);
        auto c1 = Std::get<1>(srcCoordinate);
        auto c2 = Std::get<2>(srcCoordinate);
        auto c3 = Std::get<3>(srcCoordinate);
        srcAddr += c0 * ss0 + c1 * ss1 + c2 * ss2 + c3 * ss3;
        permute_detail::PermuteEleDim4<axis0, axis1, axis2, axis3, dstType, ubType>(
            dstAddr, srcAddr, srcLayout, dstLayout);
    } else if constexpr (dimCount == 5) {
        auto ss0 = srcLayout.template GetStrideDim<0, N>();
        auto ss1 = srcLayout.template GetStrideDim<1, N>();
        auto ss2 = srcLayout.template GetStrideDim<2, N>();
        auto ss3 = srcLayout.template GetStrideDim<3, N>();
        auto ss4 = srcLayout.template GetStrideDim<4, N>();
        srcAddr += srcLayout.template GetGmOffset<C0, N>(srcCoordinate);
        permute_detail::PermuteEleDim5<axis0, axis1, axis2, axis3, axis4, dstType, ubType>(
            dstAddr, srcAddr, srcLayout, dstLayout);
    }
}

#define OP_TILE_OP_PERMUTE_MOVEOUT TPermuteMoveOut
template <int axis0, int axis1, int axis2, int axis3, int axis4, int dimCount, typename T0, typename T1, typename C0>
TILEOP void TPermuteMoveOut(T0 dst, T1 src, C0 dstCoordinate)
{
    constexpr size_t dstExpectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto dstOffset = dstLayout.template GetGmOffset<C0, dstExpectSize>(dstCoordinate);
    using ubType = typename T1::Type;
    using dstType = typename T0::Type;
    __gm__ dstType* dstAddr = (__gm__ dstType*)((uint64_t)(dst.GetAddr()));
    dstAddr += dstOffset;
    __ubuf__ ubType* srcAddr = (__ubuf__ ubType*)((uint64_t)(src.GetAddr()));

    constexpr auto shapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto tileH = Std::tuple_element<shapeSize - 2, typename T1::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename T1::TileShape>::type::value;

    auto srcStride0 = srcLayout.template GetStrideDim<0, shapeSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, shapeSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, shapeSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, shapeSize>();
    auto srcStride4 = srcLayout.template GetStrideDim<4, shapeSize>();

    auto dstStride0 = dstLayout.template GetStrideDim<0, dstExpectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, dstExpectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, dstExpectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, dstExpectSize>();
    auto dstStride4 = dstLayout.template GetStrideDim<4, dstExpectSize>();

    constexpr int pad = 5 - dimCount;
    if constexpr (dimCount == 2 && axis0 == 1 && axis1 == 0) {
        permute_detail::PermuteMoveOutDim2<pad, axis0, axis1, tileH, tileW, dstType, ubType>(
            dstAddr, srcAddr, srcLayout, dstLayout, srcStride0, srcStride1, dstStride0, dstStride1, dstStride4);
    } else if constexpr (dimCount == 3) {
        permute_detail::PermuteMoveOutDim3<pad, axis0, axis1, axis2, tileH, tileW, dstType, ubType>(
            dstAddr, srcAddr, srcLayout, dstLayout, srcStride0, srcStride1, srcStride2, dstStride0, dstStride1,
            dstStride2, dstStride3, dstStride4);
    } else if constexpr (dimCount == 4) {
        permute_detail::PermuteMoveOutDim4<pad, axis0, axis1, axis2, axis3, tileH, tileW, dstType, ubType>(
            dstAddr, srcAddr, srcLayout, dstLayout, srcStride0, srcStride1, srcStride2, srcStride3, dstStride0,
            dstStride1, dstStride2, dstStride3, dstStride4);
    } else if constexpr (dimCount == 5) {
        permute_detail::PermuteMoveOutDim5<pad, axis0, axis1, axis2, axis3, axis4, tileH, tileW, dstType, ubType>(
            dstAddr, srcAddr, srcLayout, dstLayout, srcStride0, srcStride1, srcStride2, srcStride3, srcStride4,
            dstStride0, dstStride1, dstStride2, dstStride3, dstStride4);
    }
}

#define OP_TILE_OP_TAIL_AXIS_PERMUTE TailAxisPermute
template <
    int s1_axis0, int s1_axis1, int s1_axis2, int s1_axis3, int s1_axis4, int s3_axis0, int s3_axis1, int s3_axis2,
    int s3_axis3, int s3_axis4, int dimCount, typename T0, typename T1, typename T2, typename T3, typename T4,
    typename C0, typename C1>
TILEOP void TailAxisPermute(T0 dst, T1 src, T2 ubBuf1, T3 ubBuf2, T4 ubBuf3, C0 srcCoordinate, C1 dstCoordinate)
{
    using gmType = typename T1::Type;
    using ubType = typename T2::Type;
    using dstType = typename T0::Type;
    constexpr size_t N = 5;

    const auto srcLayout = src.GetLayout();
    const auto dstLayout = dst.GetLayout();
    constexpr auto tileH = Std::tuple_element<3, typename T2::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<4, typename T2::TileShape>::type::value;

    constexpr auto ub2StaticH = tileW;
    constexpr auto ub2StaticW = (tileH * sizeof(ubType) + 32 - 1) / 32 * 32 / sizeof(ubType);
    constexpr auto ub1StaticH = ub2StaticW;
    constexpr auto ub1StaticW = tileW;
    constexpr unsigned yTileSize = (sizeof(ubType) == 1) ? 32 : 16;
    constexpr auto tmpStaticH = ub2StaticH;
    constexpr auto tmpStaticW = (ub2StaticW + yTileSize - 1) / yTileSize * yTileSize;

    using Ub1TileDefine = pto::Tile<pto::TileType::Vec, ubType, ub1StaticH, ub1StaticW, pto::BLayout::RowMajor, -1, -1>;
    using Ub2TileDefine = pto::Tile<pto::TileType::Vec, ubType, ub2StaticH, ub2StaticW, pto::BLayout::RowMajor, -1, -1>;
    using TmpTileDefine =
        pto::Tile<pto::TileType::Vec, ubType, tmpStaticH, tmpStaticW, pto::BLayout::RowMajor, tmpStaticH, tmpStaticW>;

    __gm__ gmType* srcAddr = (__gm__ gmType*)((uint64_t)(src.GetAddr()));
    __ubuf__ ubType* ubBuf1Addr = (__ubuf__ ubType*)((uint64_t)(ubBuf1.GetAddr()));
    __ubuf__ ubType* ubBuf2Addr = (__ubuf__ ubType*)((uint64_t)(ubBuf2.GetAddr()));
    __ubuf__ ubType* ubBuf3Addr = (__ubuf__ ubType*)((uint64_t)(ubBuf3.GetAddr()));
    __gm__ dstType* dstAddr = (__gm__ dstType*)((uint64_t)(dst.GetAddr()));

    auto srcOffset = srcLayout.template GetGmOffset<C0, N>(srcCoordinate);
    srcAddr += srcOffset;

    constexpr int pad = 5 - dimCount;

    auto ubShape0 = ubBuf1.GetLayout().template GetShapeDim<0, N>();
    auto ubShape1 = ubBuf1.GetLayout().template GetShapeDim<1, N>();
    auto ubShape2 = ubBuf1.GetLayout().template GetShapeDim<2, N>();
    auto ubShape3 = ubBuf1.GetLayout().template GetShapeDim<3, N>();
    auto ubShape4 = ubBuf1.GetLayout().template GetShapeDim<4, N>();

    Ub1TileDefine ub1Tile(ubShape3, ubShape4);
    Ub2TileDefine ub2Tile(ubShape4, ubShape3);
    TmpTileDefine tmpTile;
    pto::TASSIGN(tmpTile, (uint64_t)ubBuf3Addr);

    if constexpr (dimCount == 3) {
        permute_detail::PermuteDim3<pad, s1_axis0, s1_axis1, s1_axis2, tileH, tileW, ubType, ubType>(
            ubBuf1Addr, srcAddr, srcLayout, ubBuf1.GetLayout(), srcLayout.template GetStrideDim<0, N>(),
            srcLayout.template GetStrideDim<1, N>(), srcLayout.template GetStrideDim<2, N>(),
            srcLayout.template GetStrideDim<3, N>(), srcLayout.template GetStrideDim<4, N>());
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        for (LoopVar i0 = 0; i0 < ubShape0; ++i0) {
            for (LoopVar i1 = 0; i1 < ubShape1; ++i1) {
                for (LoopVar i2 = 0; i2 < ubShape2; ++i2) {
                    pto::TASSIGN(
                        ub1Tile, (uint64_t)(ubBuf1Addr + (i0 * ubBuf1.GetLayout().template GetStrideDim<0, N>() +
                                                          i1 * ubBuf1.GetLayout().template GetStrideDim<1, N>() +
                                                          i2 * ubBuf1.GetLayout().template GetStrideDim<2, N>())));
                    pto::TASSIGN(
                        ub2Tile, (uint64_t)(ubBuf2Addr + (i0 * ubBuf2.GetLayout().template GetStrideDim<0, N>() +
                                                          i1 * ubBuf2.GetLayout().template GetStrideDim<1, N>() +
                                                          i2 * ubBuf2.GetLayout().template GetStrideDim<2, N>())));
                    pto::TTRANS(ub2Tile, ub1Tile, tmpTile);
                }
            }
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        auto dstOffset = dstLayout.template GetGmOffset<C1, N>(dstCoordinate);
        permute_detail::PermuteMoveOutDim3<pad, s3_axis0, s3_axis1, s3_axis2, tileH, tileW, dstType, ubType>(
            dstAddr + dstOffset, ubBuf2Addr, ubBuf2.GetLayout(), dstLayout,
            ubBuf2.GetLayout().template GetStrideDim<0, N>(), ubBuf2.GetLayout().template GetStrideDim<1, N>(),
            ubBuf2.GetLayout().template GetStrideDim<2, N>(), dstLayout.template GetStrideDim<0, N>(),
            dstLayout.template GetStrideDim<1, N>(), dstLayout.template GetStrideDim<2, N>(),
            dstLayout.template GetStrideDim<3, N>(), dstLayout.template GetStrideDim<4, N>());
    } else if constexpr (dimCount == 4) {
        permute_detail::PermuteDim4<pad, s1_axis0, s1_axis1, s1_axis2, s1_axis3, tileH, tileW, ubType, ubType>(
            ubBuf1Addr, srcAddr, srcLayout, ubBuf1.GetLayout(), srcLayout.template GetStrideDim<0, N>(),
            srcLayout.template GetStrideDim<1, N>(), srcLayout.template GetStrideDim<2, N>(),
            srcLayout.template GetStrideDim<3, N>(), srcLayout.template GetStrideDim<4, N>());
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        for (LoopVar i0 = 0; i0 < ubShape0; ++i0) {
            for (LoopVar i1 = 0; i1 < ubShape1; ++i1) {
                for (LoopVar i2 = 0; i2 < ubShape2; ++i2) {
                    pto::TASSIGN(
                        ub1Tile, (uint64_t)(ubBuf1Addr + (i0 * ubBuf1.GetLayout().template GetStrideDim<0, N>() +
                                                          i1 * ubBuf1.GetLayout().template GetStrideDim<1, N>() +
                                                          i2 * ubBuf1.GetLayout().template GetStrideDim<2, N>())));
                    pto::TASSIGN(
                        ub2Tile, (uint64_t)(ubBuf2Addr + (i0 * ubBuf2.GetLayout().template GetStrideDim<0, N>() +
                                                          i1 * ubBuf2.GetLayout().template GetStrideDim<1, N>() +
                                                          i2 * ubBuf2.GetLayout().template GetStrideDim<2, N>())));
                    pto::TTRANS(ub2Tile, ub1Tile, tmpTile);
                }
            }
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        auto dstOffset = dstLayout.template GetGmOffset<C1, N>(dstCoordinate);
        permute_detail::PermuteMoveOutDim4<pad, s3_axis0, s3_axis1, s3_axis2, s3_axis3, tileH, tileW, dstType, ubType>(
            dstAddr + dstOffset, ubBuf2Addr, ubBuf2.GetLayout(), dstLayout,
            ubBuf2.GetLayout().template GetStrideDim<0, N>(), ubBuf2.GetLayout().template GetStrideDim<1, N>(),
            ubBuf2.GetLayout().template GetStrideDim<2, N>(), ubBuf2.GetLayout().template GetStrideDim<3, N>(),
            dstLayout.template GetStrideDim<0, N>(), dstLayout.template GetStrideDim<1, N>(),
            dstLayout.template GetStrideDim<2, N>(), dstLayout.template GetStrideDim<3, N>(),
            dstLayout.template GetStrideDim<4, N>());
    } else if constexpr (dimCount == 5) {
        permute_detail::PermuteDim5<
            pad, s1_axis0, s1_axis1, s1_axis2, s1_axis3, s1_axis4, tileH, tileW, ubType, ubType>(
            ubBuf1Addr, srcAddr, srcLayout, ubBuf1.GetLayout(), srcLayout.template GetStrideDim<0, N>(),
            srcLayout.template GetStrideDim<1, N>(), srcLayout.template GetStrideDim<2, N>(),
            srcLayout.template GetStrideDim<3, N>(), srcLayout.template GetStrideDim<4, N>());
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        for (LoopVar i0 = 0; i0 < ubShape0; ++i0) {
            for (LoopVar i1 = 0; i1 < ubShape1; ++i1) {
                for (LoopVar i2 = 0; i2 < ubShape2; ++i2) {
                    pto::TASSIGN(
                        ub1Tile, (uint64_t)(ubBuf1Addr + (i0 * ubBuf1.GetLayout().template GetStrideDim<0, N>() +
                                                          i1 * ubBuf1.GetLayout().template GetStrideDim<1, N>() +
                                                          i2 * ubBuf1.GetLayout().template GetStrideDim<2, N>())));
                    pto::TASSIGN(
                        ub2Tile, (uint64_t)(ubBuf2Addr + (i0 * ubBuf2.GetLayout().template GetStrideDim<0, N>() +
                                                          i1 * ubBuf2.GetLayout().template GetStrideDim<1, N>() +
                                                          i2 * ubBuf2.GetLayout().template GetStrideDim<2, N>())));
                    pto::TTRANS(ub2Tile, ub1Tile, tmpTile);
                }
            }
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        auto dstOffset = dstLayout.template GetGmOffset<C1, N>(dstCoordinate);
        permute_detail::PermuteMoveOutDim5<
            pad, s3_axis0, s3_axis1, s3_axis2, s3_axis3, s3_axis4, tileH, tileW, dstType, ubType>(
            dstAddr + dstOffset, ubBuf2Addr, ubBuf2.GetLayout(), dstLayout,
            ubBuf2.GetLayout().template GetStrideDim<0, N>(), ubBuf2.GetLayout().template GetStrideDim<1, N>(),
            ubBuf2.GetLayout().template GetStrideDim<2, N>(), ubBuf2.GetLayout().template GetStrideDim<3, N>(),
            ubBuf2.GetLayout().template GetStrideDim<4, N>(), dstLayout.template GetStrideDim<0, N>(),
            dstLayout.template GetStrideDim<1, N>(), dstLayout.template GetStrideDim<2, N>(),
            dstLayout.template GetStrideDim<3, N>(), dstLayout.template GetStrideDim<4, N>());
    }
}
#endif
