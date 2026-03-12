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
 * \file permute_element.h
 * \brief Element-wise permute tileOp - GM->UB mode for permutations involving the last axis
 */

#ifndef TILEOP_TILE_OPERATOR_PERMUTE_ELEMENT__H
#define TILEOP_TILE_OPERATOR_PERMUTE_ELEMENT__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#include <cstdint>

#define OP_TILE_OP_PERMUTE_ELEMENT TpermuteElement

template <int axis0, int axis1, int axis2, int axis3, int axis4, int dimCount, typename T0, typename T1, typename C0>
TILEOP void TpermuteElement(T0 dst, T1 src, C0 srcCoordinate) {
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    constexpr size_t N = Std::tuple_size<typename T1::Shape>::value;
    using srcType = typename T1::Type;
    using dstType = typename T0::Type;
    __gm__ srcType *srcAddr = (__gm__ srcType *)((uint64_t)(src.GetAddr()));
    __ubuf__ dstType *dstAddr = (__ubuf__ dstType *)((uint64_t)(dst.GetAddr()));

    constexpr int pad = 5 - dimCount;

    if constexpr (dimCount == 2) {
        auto ss0 = srcLayout.template GetStrideDim<pad + 0, N>();
        auto ss1 = srcLayout.template GetStrideDim<pad + 1, N>();
        auto c0 = Std::get<0>(srcCoordinate);
        auto c1 = Std::get<1>(srcCoordinate);
        auto srcBase = c0 * ss0 + c1 * ss1;
        srcAddr += srcBase;
        auto d0 = dstLayout.template GetShapeDim<0, 2>();
        auto d1 = dstLayout.template GetShapeDim<1, 2>();
        auto ds0 = dstLayout.template GetStrideDim<0, 2>();
        auto ds1 = dstLayout.template GetStrideDim<1, 2>();
        for (LoopVar i0 = 0; i0 < d0; ++i0) {
            for (LoopVar i1 = 0; i1 < d1; ++i1) {
                int64_t sc[5] = {0, 0, 0, 0, 0};
                sc[pad + axis0] = static_cast<int64_t>(i0);
                sc[pad + axis1] = static_cast<int64_t>(i1);
                auto gmOff = sc[0] * ss0 + sc[1] * ss1;
                dstAddr[i0 * ds0 + i1 * ds1] = srcAddr[gmOff];
            }
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    } else if constexpr (dimCount == 3) {
        auto ss0 = srcLayout.template GetStrideDim<pad + 0, N>();
        auto ss1 = srcLayout.template GetStrideDim<pad + 1, N>();
        auto ss2 = srcLayout.template GetStrideDim<pad + 2, N>();
        auto c0 = Std::get<0>(srcCoordinate);
        auto c1 = Std::get<1>(srcCoordinate);
        auto c2 = Std::get<2>(srcCoordinate);
        auto srcBase = c0 * ss0 + c1 * ss1 + c2 * ss2;
        srcAddr += srcBase;
        auto d0 = dstLayout.template GetShapeDim<0, 3>();
        auto d1 = dstLayout.template GetShapeDim<1, 3>();
        auto d2 = dstLayout.template GetShapeDim<2, 3>();
        auto ds0 = dstLayout.template GetStrideDim<0, 3>();
        auto ds1 = dstLayout.template GetStrideDim<1, 3>();
        auto ds2 = dstLayout.template GetStrideDim<2, 3>();
        for (LoopVar i0 = 0; i0 < d0; ++i0) {
            for (LoopVar i1 = 0; i1 < d1; ++i1) {
                for (LoopVar i2 = 0; i2 < d2; ++i2) {
                    int64_t sc[5] = {0, 0, 0, 0, 0};
                    sc[pad + axis0] = static_cast<int64_t>(i0);
                    sc[pad + axis1] = static_cast<int64_t>(i1);
                    sc[pad + axis2] = static_cast<int64_t>(i2);
                    auto gmOff = sc[0] * ss0 + sc[1] * ss1 + sc[2] * ss2;
                    dstAddr[i0 * ds0 + i1 * ds1 + i2 * ds2] = srcAddr[gmOff];
                }
            }
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    } else if constexpr (dimCount == 4) {
        auto ss0 = srcLayout.template GetStrideDim<pad + 0, N>();
        auto ss1 = srcLayout.template GetStrideDim<pad + 1, N>();
        auto ss2 = srcLayout.template GetStrideDim<pad + 2, N>();
        auto ss3 = srcLayout.template GetStrideDim<pad + 3, N>();
        auto c0 = Std::get<0>(srcCoordinate);
        auto c1 = Std::get<1>(srcCoordinate);
        auto c2 = Std::get<2>(srcCoordinate);
        auto c3 = Std::get<3>(srcCoordinate);
        auto srcBase = c0 * ss0 + c1 * ss1 + c2 * ss2 + c3 * ss3;
        srcAddr += srcBase;
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
                        int64_t sc[5] = {0, 0, 0, 0, 0};
                        sc[pad + axis0] = static_cast<int64_t>(i0);
                        sc[pad + axis1] = static_cast<int64_t>(i1);
                        sc[pad + axis2] = static_cast<int64_t>(i2);
                        sc[pad + axis3] = static_cast<int64_t>(i3);
                        auto gmOff = sc[0] * ss0 + sc[1] * ss1 + sc[2] * ss2 + sc[3] * ss3;
                        dstAddr[i0 * ds0 + i1 * ds1 + i2 * ds2 + i3 * ds3] = srcAddr[gmOff];
                    }
                }
            }
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    } else if constexpr (dimCount == 5) {
        auto ss0 = srcLayout.template GetStrideDim<0, N>();
        auto ss1 = srcLayout.template GetStrideDim<1, N>();
        auto ss2 = srcLayout.template GetStrideDim<2, N>();
        auto ss3 = srcLayout.template GetStrideDim<3, N>();
        auto ss4 = srcLayout.template GetStrideDim<4, N>();
        auto srcBase = srcLayout.template GetGmOffset<C0, N>(srcCoordinate);
        srcAddr += srcBase;
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
}

#endif
