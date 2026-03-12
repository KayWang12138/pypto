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

    auto srcStride0 = srcLayout.template GetStrideDim<0, srcExpectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, srcExpectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, srcExpectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, srcExpectSize>();
    auto srcStride4 = srcLayout.template GetStrideDim<4, srcExpectSize>();

    constexpr int pad = 5 - dimCount;

    auto d0 = dstLayout.template GetShapeDim<0, dimCount>();
    auto d1 = dimCount > 1 ? dstLayout.template GetShapeDim<1, dimCount>() : 1;
    auto d2 = dimCount > 2 ? dstLayout.template GetShapeDim<2, dimCount>() : 1;
    auto d3 = dimCount > 3 ? dstLayout.template GetShapeDim<3, dimCount>() : 1;
    auto d4 = dimCount > 4 ? dstLayout.template GetShapeDim<4, dimCount>() : 1;
    auto ds0 = dstLayout.template GetStrideDim<0, dimCount>();
    auto ds1 = dimCount > 1 ? dstLayout.template GetStrideDim<1, dimCount>() : 0;
    auto ds2 = dimCount > 2 ? dstLayout.template GetStrideDim<2, dimCount>() : 0;
    auto ds3 = dimCount > 3 ? dstLayout.template GetStrideDim<3, dimCount>() : 0;

    if constexpr (dimCount == 2) {
        for (LoopVar i0 = 0; i0 < d0; ++i0) {
            for (LoopVar i1 = 0; i1 < d1; ++i1) {
                int64_t sc[5] = {0, 0, 0, 0, 0};
                sc[pad + axis0] = static_cast<int64_t>(i0);
                sc[pad + axis1] = static_cast<int64_t>(i1);
                auto gmOff = sc[0] * srcStride0 + sc[1] * srcStride1 + sc[2] * srcStride2 +
                             sc[3] * srcStride3 + sc[4] * srcStride4;
                dstAddr[i0 * ds0 + i1 * ds1] = srcAddr[gmOff];
            }
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    } else if constexpr (dimCount == 3) {
        for (LoopVar i0 = 0; i0 < d0; ++i0) {
            __ubuf__ dstType *dst1 = dstAddr + i0 * ds0;
            for (LoopVar i1 = 0; i1 < d1; ++i1) {
                __ubuf__ dstType *dst2 = dst1 + i1 * ds1;
                for (LoopVar i2 = 0; i2 < d2; ++i2) {
                    int64_t sc[5] = {0, 0, 0, 0, 0};
                    sc[pad + axis0] = static_cast<int64_t>(i0);
                    sc[pad + axis1] = static_cast<int64_t>(i1);
                    sc[pad + axis2] = static_cast<int64_t>(i2);
                    auto gmOff = sc[0] * srcStride0 + sc[1] * srcStride1 + sc[2] * srcStride2 +
                                 sc[3] * srcStride3 + sc[4] * srcStride4;
                    dst2[i2 * ds2] = srcAddr[gmOff];
                }
            }
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    } else if constexpr (dimCount == 4) {
        for (LoopVar i0 = 0; i0 < d0; ++i0) {
            __ubuf__ dstType *dst1 = dstAddr + i0 * ds0;
            for (LoopVar i1 = 0; i1 < d1; ++i1) {
                __ubuf__ dstType *dst2 = dst1 + i1 * ds1;
                for (LoopVar i2 = 0; i2 < d2; ++i2) {
                    __ubuf__ dstType *dst3 = dst2 + i2 * ds2;
                    for (LoopVar i3 = 0; i3 < d3; ++i3) {
                        int64_t sc[5] = {0, 0, 0, 0, 0};
                        sc[pad + axis0] = static_cast<int64_t>(i0);
                        sc[pad + axis1] = static_cast<int64_t>(i1);
                        sc[pad + axis2] = static_cast<int64_t>(i2);
                        sc[pad + axis3] = static_cast<int64_t>(i3);
                        auto gmOff = sc[0] * srcStride0 + sc[1] * srcStride1 + sc[2] * srcStride2 +
                                     sc[3] * srcStride3 + sc[4] * srcStride4;
                        dst3[i3 * ds3] = srcAddr[gmOff];
                    }
                }
            }
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    } else if constexpr (dimCount == 5) {
        for (LoopVar i0 = 0; i0 < d0; ++i0) {
            __ubuf__ dstType *dst1 = dstAddr + i0 * ds0;
            for (LoopVar i1 = 0; i1 < d1; ++i1) {
                __ubuf__ dstType *dst2 = dst1 + i1 * ds1;
                for (LoopVar i2 = 0; i2 < d2; ++i2) {
                    __ubuf__ dstType *dst3 = dst2 + i2 * ds2;
                    for (LoopVar i3 = 0; i3 < d3; ++i3) {
                        __ubuf__ dstType *dst4 = dst3 + i3 * ds3;
                        for (LoopVar i4 = 0; i4 < d4; ++i4) {
                            int64_t sc[5] = {0, 0, 0, 0, 0};
                            sc[pad + axis0] = static_cast<int64_t>(i0);
                            sc[pad + axis1] = static_cast<int64_t>(i1);
                            sc[pad + axis2] = static_cast<int64_t>(i2);
                            sc[pad + axis3] = static_cast<int64_t>(i3);
                            sc[pad + axis4] = static_cast<int64_t>(i4);
                            auto gmOff = sc[0] * srcStride0 + sc[1] * srcStride1 + sc[2] * srcStride2 +
                                         sc[3] * srcStride3 + sc[4] * srcStride4;
                            dst4[i4] = srcAddr[gmOff];
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
