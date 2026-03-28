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
 * \file permute.h
 * \brief Permute tile operation implementation
 */

#ifndef TILEOP_TILE_OPERATOR_PERMUTE__H
#define TILEOP_TILE_OPERATOR_PERMUTE__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_PERMUTE Tpermute

template <typename DstType, typename SrcType, typename DstLayout, typename SrcLayout>
__aicore__ inline void PermuteCopy4DScalar(
    __ubuf__ DstType *dstAddr, __ubuf__ SrcType *srcAddr, const DstLayout &dstLayout,
    const SrcLayout &srcLayout, int axis0, int axis1, int axis2, int axis3) {
    auto dstShape0 = dstLayout.template GetShapeDim<0, 4>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, 4>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, 4>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, 4>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, 4>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, 4>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, 4>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, 4>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, 4>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, 4>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, 4>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, 4>();
    for (LoopVar i0 = 0; i0 < dstShape0; ++i0) {
        for (LoopVar i1 = 0; i1 < dstShape1; ++i1) {
            for (LoopVar i2 = 0; i2 < dstShape2; ++i2) {
                for (LoopVar i3 = 0; i3 < dstShape3; ++i3) {
                    int64_t dstCoord[4] = {
                        static_cast<int64_t>(i0), static_cast<int64_t>(i1), static_cast<int64_t>(i2),
                        static_cast<int64_t>(i3)};
                    int64_t srcCoord[4] = {0, 0, 0, 0};
                    srcCoord[axis0] = dstCoord[0];
                    srcCoord[axis1] = dstCoord[1];
                    srcCoord[axis2] = dstCoord[2];
                    srcCoord[axis3] = dstCoord[3];
                    auto dstOffset = i0 * dstStride0 + i1 * dstStride1 + i2 * dstStride2 + i3 * dstStride3;
                    auto srcOffset =
                        srcCoord[0] * srcStride0 + srcCoord[1] * srcStride1 + srcCoord[2] * srcStride2 +
                        srcCoord[3] * srcStride3;
                    dstAddr[dstOffset] = srcAddr[srcOffset];
                }
            }
        }
    }
}

template <typename DstType, typename SrcType, typename DstLayout, typename SrcLayout>
__aicore__ inline void PermuteCopy5DScalar(
    __ubuf__ DstType *dstAddr, __ubuf__ SrcType *srcAddr, const DstLayout &dstLayout,
    const SrcLayout &srcLayout, int axis0, int axis1, int axis2, int axis3, int axis4) {
    auto dstShape0 = dstLayout.template GetShapeDim<0, 5>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, 5>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, 5>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, 5>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, 5>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, 5>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, 5>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, 5>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, 5>();
    auto dstStride4 = dstLayout.template GetStrideDim<4, 5>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, 5>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, 5>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, 5>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, 5>();
    auto srcStride4 = srcLayout.template GetStrideDim<4, 5>();
    for (LoopVar i0 = 0; i0 < dstShape0; ++i0) {
        for (LoopVar i1 = 0; i1 < dstShape1; ++i1) {
            for (LoopVar i2 = 0; i2 < dstShape2; ++i2) {
                for (LoopVar i3 = 0; i3 < dstShape3; ++i3) {
                    for (LoopVar i4 = 0; i4 < dstShape4; ++i4) {
                        int64_t dstCoord[5] = {
                            static_cast<int64_t>(i0), static_cast<int64_t>(i1), static_cast<int64_t>(i2),
                            static_cast<int64_t>(i3), static_cast<int64_t>(i4)};
                        int64_t srcCoord[5] = {0, 0, 0, 0, 0};
                        srcCoord[axis0] = dstCoord[0];
                        srcCoord[axis1] = dstCoord[1];
                        srcCoord[axis2] = dstCoord[2];
                        srcCoord[axis3] = dstCoord[3];
                        srcCoord[axis4] = dstCoord[4];
                        auto dstOffset =
                            i0 * dstStride0 + i1 * dstStride1 + i2 * dstStride2 + i3 * dstStride3 + i4 * dstStride4;
                        auto srcOffset = srcCoord[0] * srcStride0 + srcCoord[1] * srcStride1 + srcCoord[2] * srcStride2 +
                                         srcCoord[3] * srcStride3 + srcCoord[4] * srcStride4;
                        dstAddr[dstOffset] = srcAddr[srcOffset];
                    }
                }
            }
        }
    }
}

template <int axis0, int axis1, int axis2, int axis3, int axis4, int dimCount, typename T0, typename T1, typename T2>
TILEOP void Tpermute(T0 dst, T1 src, T2 tmp) {
    (void)tmp;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    __ubuf__ typename T0::Type *dstAddr = (__ubuf__ typename T0::Type *)((uint64_t)(dst.GetAddr()));
    __ubuf__ typename T1::Type *srcAddr = (__ubuf__ typename T1::Type *)((uint64_t)(src.GetAddr()));

    if constexpr (dimCount == 2 && axis0 == 1 && axis1 == 0) {
        constexpr size_t expectSize = 2;
        auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
        auto srcShape0 = srcLayout.template GetShapeDim<0, expectSize>();
        auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
        auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
        constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 0, expectSize>();
        constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 1, expectSize>();
        constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 0, expectSize>();
        constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 1, expectSize>();
        using DstTileDefine =
            pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
        using SrcTileDefine =
            pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
        constexpr unsigned tmpTileW = (sizeof(typename T0::Type) == 1) ? 32 : 16;
        using TmpTileDefine2D =
            pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, tmpTileW, pto::BLayout::RowMajor, dstTileH, tmpTileW>;
        DstTileDefine dstTile(dstLayout.template GetShapeDim<0, expectSize>(), dstLayout.template GetShapeDim<1, expectSize>());
        SrcTileDefine srcTile(srcLayout.template GetShapeDim<0, expectSize>(), srcLayout.template GetShapeDim<1, expectSize>());
        TmpTileDefine2D tmpTile;
        pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
        for (LoopVar i0 = 0; i0 < dstShape0; ++i0) {
            auto dstOffset = i0 * dstStride0;
            auto srcOffset = i0 * srcStride0;
            pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(typename T0::Type)));
            pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(typename T1::Type)));
            pto::TTRANS(dstTile, srcTile, tmpTile);
        }
    } else if constexpr (dimCount == 4) {
        PermuteCopy4DScalar<typename T0::Type, typename T1::Type>(dstAddr, srcAddr, dstLayout, srcLayout, axis0,
            axis1, axis2, axis3);
    } else if constexpr (dimCount == 5) {
        PermuteCopy5DScalar<typename T0::Type, typename T1::Type>(dstAddr, srcAddr, dstLayout, srcLayout, axis0,
            axis1, axis2, axis3, axis4);
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

#endif
