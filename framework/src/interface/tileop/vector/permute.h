/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FIT FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file permute.h
 * \brief Permute tileOp - GM->UB mode (reference: Tgather)
 */

#ifndef TILEOP_TILE_OPERATOR_PERMUTE__H
#define TILEOP_TILE_OPERATOR_PERMUTE__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#include <cstdint>

#define OP_TILE_OP_PERMUTE Tpermute

template <int axis0, int axis1, int axis2, int axis3, int axis4, int dimCount,
          typename T0, typename T1, typename C0>
TILEOP void Tpermute(T0 dst, T1 src, C0 srcCoordinate) {
    constexpr size_t dstExpectSize = 5;
    constexpr size_t srcExpectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto srcOffset = srcLayout.template GetGmOffset<C0, srcExpectSize>(srcCoordinate);
    using srcType = typename T1::Type;
    using dstType = typename T0::Type;
    __gm__ srcType *srcAddr = (__gm__ srcType *)((uint64_t)(src.GetAddr()));
    srcAddr += srcOffset;
    __ubuf__ dstType *dstAddr = (__ubuf__ dstType *)((uint64_t)(dst.GetAddr()));

    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto tileH = Std::tuple_element<shapeSize - 2, typename T0::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename T0::TileShape>::type::value;
    using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
    using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = pto::GlobalTensor<srcType, ShapeDim5, StrideDim5>;
    using TileDefine = pto::Tile<pto::TileType::Vec, dstType, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    auto srcStride0 = srcLayout.template GetStrideDim<0, srcExpectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, srcExpectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, srcExpectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, srcExpectSize>();
    auto srcStride4 = srcLayout.template GetStrideDim<4, srcExpectSize>();

    if constexpr (dimCount == 2 && axis0 == 1 && axis1 == 0) {
        auto d0 = dstLayout.template GetShapeDim<0, 2>();
        auto d1 = dstLayout.template GetShapeDim<1, 2>();
        auto ds0 = dstLayout.template GetStrideDim<0, 2>();
        for (LoopVar i0 = 0; i0 < d0; ++i0) {
            __ubuf__ dstType *dst0 = dstAddr + i0 * ds0;
            int64_t sc[5] = {0, 0, 0, 0, 0};
            sc[axis0] = static_cast<int64_t>(i0);
            sc[axis1] = 0;
            auto gmOff = sc[0] * srcStride0 + sc[1] * srcStride1 + sc[2] * srcStride2 + sc[3] * srcStride3 + sc[4] * srcStride4;
            TileDefine dstTile(1, d1);
            GlobalData srcGlobal(srcAddr + gmOff,
                pto::Shape(1, 1, 1, 1, d1), pto::Stride(0, 0, 0, 0, srcStride0));
            pto::TASSIGN(dstTile, (uint64_t)dst0);
            pto::TLOAD(dstTile, srcGlobal);
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    } else if constexpr (dimCount == 3) {
        auto d0 = dstLayout.template GetShapeDim<0, 3>();
        auto d1 = dstLayout.template GetShapeDim<1, 3>();
        auto d2 = dstLayout.template GetShapeDim<2, 3>();
        auto ds0 = dstLayout.template GetStrideDim<0, 3>();
        auto ds1 = dstLayout.template GetStrideDim<1, 3>();
        auto ds2 = dstLayout.template GetStrideDim<2, 3>();
        for (LoopVar i0 = 0; i0 < d0; ++i0) {
            __ubuf__ dstType *dst1 = dstAddr + i0 * ds0;
            for (LoopVar i1 = 0; i1 < d1; ++i1) {
                __ubuf__ dstType *dst2 = dst1 + i1 * ds1;
                int64_t sc[5] = {0, 0, 0, 0, 0};
                sc[axis0] = static_cast<int64_t>(i0);
                sc[axis1] = static_cast<int64_t>(i1);
                sc[axis2] = 0;
                auto gmOff = sc[0] * srcStride0 + sc[1] * srcStride1 + sc[2] * srcStride2 + sc[3] * srcStride3 + sc[4] * srcStride4;
                TileDefine dstTile(1, d2);
                GlobalData srcGlobal(srcAddr + gmOff,
                    pto::Shape(1, 1, 1, 1, d2), pto::Stride(0, 0, 0, 0, srcStride1));
                pto::TASSIGN(dstTile, (uint64_t)dst2);
                pto::TLOAD(dstTile, srcGlobal);
            }
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    } else if constexpr (dimCount == 4) {
        auto d0 = dstLayout.template GetShapeDim<0, 4>();
        auto d1 = dstLayout.template GetShapeDim<1, 4>();
        auto d2 = dstLayout.template GetShapeDim<2, 4>();
        auto d3 = dstLayout.template GetShapeDim<3, 4>();
        auto ds0 = dstLayout.template GetStrideDim<0, 4>();
        auto ds1 = dstLayout.template GetStrideDim<1, 4>();
        auto ds2 = dstLayout.template GetStrideDim<2, 4>();
        auto ds3 = dstLayout.template GetStrideDim<3, 4>();
        for (LoopVar i0 = 0; i0 < d0; ++i0) {
            __ubuf__ dstType *dst1 = dstAddr + i0 * ds0;
            for (LoopVar i1 = 0; i1 < d1; ++i1) {
                __ubuf__ dstType *dst2 = dst1 + i1 * ds1;
                for (LoopVar i2 = 0; i2 < d2; ++i2) {
                    __ubuf__ dstType *dst3 = dst2 + i2 * ds2;
                    int64_t sc[5] = {0, 0, 0, 0, 0};
                    sc[axis0] = static_cast<int64_t>(i0);
                    sc[axis1] = static_cast<int64_t>(i1);
                    sc[axis2] = static_cast<int64_t>(i2);
                    sc[axis3] = 0;
                    auto gmOff = sc[0] * srcStride0 + sc[1] * srcStride1 + sc[2] * srcStride2 + sc[3] * srcStride3 + sc[4] * srcStride4;
                    TileDefine dstTile(1, d3);
                    GlobalData srcGlobal(srcAddr + gmOff,
                        pto::Shape(1, 1, 1, 1, d3), pto::Stride(0, 0, 0, 0, srcStride2));
                    pto::TASSIGN(dstTile, (uint64_t)dst3);
                    pto::TLOAD(dstTile, srcGlobal);
                }
            }
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    } else if constexpr (dimCount == 5) {
        auto d0 = dstLayout.template GetShapeDim<0, 5>();
        auto d1 = dstLayout.template GetShapeDim<1, 5>();
        auto d2 = dstLayout.template GetShapeDim<2, 5>();
        auto d3 = dstLayout.template GetShapeDim<3, 5>();
        auto d4 = dstLayout.template GetShapeDim<4, 5>();
        auto ds0 = dstLayout.template GetStrideDim<0, 5>();
        auto ds1 = dstLayout.template GetStrideDim<1, 5>();
        auto ds2 = dstLayout.template GetStrideDim<2, 5>();
        auto ds3 = dstLayout.template GetStrideDim<3, 5>();
        auto ds4 = dstLayout.template GetStrideDim<4, 5>();
        for (LoopVar i0 = 0; i0 < d0; ++i0) {
            __ubuf__ dstType *dst1 = dstAddr + i0 * ds0;
            for (LoopVar i1 = 0; i1 < d1; ++i1) {
                __ubuf__ dstType *dst2 = dst1 + i1 * ds1;
                for (LoopVar i2 = 0; i2 < d2; ++i2) {
                    __ubuf__ dstType *dst3 = dst2 + i2 * ds2;
                    for (LoopVar i3 = 0; i3 < d3; ++i3) {
                        __ubuf__ dstType *dst4 = dst3 + i3 * ds3;
                        int64_t sc[5] = {0, 0, 0, 0, 0};
                        sc[axis0] = static_cast<int64_t>(i0);
                        sc[axis1] = static_cast<int64_t>(i1);
                        sc[axis2] = static_cast<int64_t>(i2);
                        sc[axis3] = static_cast<int64_t>(i3);
                        sc[axis4] = 0;
                        auto gmOff = sc[0] * srcStride0 + sc[1] * srcStride1 + sc[2] * srcStride2 + sc[3] * srcStride3 + sc[4] * srcStride4;
                        TileDefine dstTile(1, d4);
                        GlobalData srcGlobal(srcAddr + gmOff,
                            pto::Shape(1, 1, 1, 1, d4), pto::Stride(0, 0, 0, 0, srcStride3));
                        pto::TASSIGN(dstTile, (uint64_t)dst4);
                        pto::TLOAD(dstTile, srcGlobal);
                    }
                }
            }
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    }
}

#endif
