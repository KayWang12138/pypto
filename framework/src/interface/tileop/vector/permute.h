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
 * \brief Permute tile operation implementation using TGATHER instruction
 */

#ifndef TILEOP_TILE_OPERATOR_PERMUTE__H
#define TILEOP_TILE_OPERATOR_PERMUTE__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_PERMUTE Tpermute

template <int axis0, int axis1, int axis2, int axis3, int axis4, int dimCount, typename T0, typename T1, typename T2>
TILEOP void Tpermute(T0 dst, T1 src, T2 tmp) {
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);

    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    auto srcShape0 = srcLayout.template GetShapeDim<0, expectSize>();
    auto srcShape1 = srcLayout.template GetShapeDim<1, expectSize>();
    auto srcShape2 = srcLayout.template GetShapeDim<2, expectSize>();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();

    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();
    auto dstStride4 = dstLayout.template GetStrideDim<4, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();
    auto srcStride4 = srcLayout.template GetStrideDim<4, expectSize>();

    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();

    using DstTileDefine =
        pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcTileDefine =
        pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
    using IdxTileDefine =
        pto::Tile<pto::TileType::Vec, int32_t, 1, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using TmpTileDefine =
        pto::Tile<pto::TileType::Vec, int32_t, 1, dstTileW, pto::BLayout::RowMajor, 1, dstTileW>;

    DstTileDefine dstTile(dstShape3, dstShape4);
    SrcTileDefine srcTile(srcShape3, srcShape4);
    IdxTileDefine idxTile(1, dstShape4);
    TmpTileDefine tmpIdxTile;

    pto::TASSIGN(tmpIdxTile, (uint64_t)(tmp.GetAddr()));

    __ubuf__ typename T0::Type *dstAddr = (__ubuf__ typename T0::Type *)((uint64_t)(dst.GetAddr()));
    __ubuf__ typename T1::Type *srcAddr = (__ubuf__ typename T1::Type *)((uint64_t)(src.GetAddr()));
    __ubuf__ int32_t *idxAddr = (__ubuf__ int32_t *)((uint64_t)(tmp.GetAddr()));
    __ubuf__ int32_t *tmpAddr = (__ubuf__ int32_t *)((uint64_t)(tmp.GetAddr() + dstTileW * sizeof(int32_t)));

    if constexpr (dimCount == 2 && axis0 == 1 && axis1 == 0) {
        constexpr unsigned tmpTileW = (sizeof(typename T0::Type) == 1) ? 32 : 16;
        using TmpTileDefine2D =
            pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, tmpTileW, pto::BLayout::RowMajor, dstTileH, tmpTileW>;
        TmpTileDefine2D tmpTile;
        pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
        
        for (LoopVar i0 = 0; i0 < dstShape0; ++i0) {
            auto dstOffset = i0 * dstStride0;
            auto srcOffset = i0 * srcStride0;
            pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
            pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
            pto::TTRANS(dstTile, srcTile, tmpTile);
        }
    } else if constexpr (dimCount == 2) {
        for (LoopVar i0 = 0; i0 < dstShape0; ++i0) {
            for (LoopVar i1 = 0; i1 < dstShape1; ++i1) {
                auto dstOffset = i0 * dstStride0 + i1 * dstStride1;
                int64_t srcIdx0 = (axis0 == 0) ? i0 : ((axis0 == 1) ? i1 : 0);
                int64_t srcIdx1 = (axis1 == 0) ? i0 : ((axis1 == 1) ? i1 : 0);
                auto srcOffset = srcIdx0 * srcStride0 + srcIdx1 * srcStride1;
                
                for (LoopVar k = 0; k < dstShape4; ++k) {
                    idxAddr[k] = k;
                }
                
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                pto::TASSIGN(idxTile, (uint64_t)(tmp.GetAddr()));
                pto::TASSIGN(tmpIdxTile, (uint64_t)(tmp.GetAddr() + dstTileW * sizeof(int32_t)));
                pto::TGATHER(dstTile, srcTile, idxTile, tmpIdxTile);
            }
        }
    } else if constexpr (dimCount == 3) {
        for (LoopVar i0 = 0; i0 < dstShape0; ++i0) {
            for (LoopVar i1 = 0; i1 < dstShape1; ++i1) {
                for (LoopVar i2 = 0; i2 < dstShape2; ++i2) {
                    auto dstOffset = i0 * dstStride0 + i1 * dstStride1 + i2 * dstStride2;
                    
                    int64_t srcIdx0 = (axis0 == 0) ? i0 : ((axis0 == 1) ? i1 : ((axis0 == 2) ? i2 : 0));
                    int64_t srcIdx1 = (axis1 == 0) ? i0 : ((axis1 == 1) ? i1 : ((axis1 == 2) ? i2 : 0));
                    int64_t srcIdx2 = (axis2 == 0) ? i0 : ((axis2 == 1) ? i1 : ((axis2 == 2) ? i2 : 0));
                    auto srcOffset = srcIdx0 * srcStride0 + srcIdx1 * srcStride1 + srcIdx2 * srcStride2;
                    
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TASSIGN(idxTile, (uint64_t)(tmp.GetAddr()));
                    pto::TASSIGN(tmpIdxTile, (uint64_t)(tmp.GetAddr() + dstTileW * sizeof(int32_t)));
                    
                    for (LoopVar k = 0; k < dstShape4; ++k) {
                        idxAddr[k] = k;
                    }
                    pto::TGATHER(dstTile, srcTile, idxTile, tmpIdxTile);
                }
            }
        }
    } else if constexpr (dimCount == 4) {
        for (LoopVar i0 = 0; i0 < dstShape0; ++i0) {
            for (LoopVar i1 = 0; i1 < dstShape1; ++i1) {
                for (LoopVar i2 = 0; i2 < dstShape2; ++i2) {
                    for (LoopVar i3 = 0; i3 < dstShape3; ++i3) {
                        auto dstOffset = i0 * dstStride0 + i1 * dstStride1 + i2 * dstStride2 + i3 * dstStride3;
                        
                        int64_t srcIdx0 = (axis0 == 0) ? i0 : ((axis0 == 1) ? i1 : ((axis0 == 2) ? i2 : ((axis0 == 3) ? i3 : 0)));
                        int64_t srcIdx1 = (axis1 == 0) ? i0 : ((axis1 == 1) ? i1 : ((axis1 == 2) ? i2 : ((axis1 == 3) ? i3 : 0)));
                        int64_t srcIdx2 = (axis2 == 0) ? i0 : ((axis2 == 1) ? i1 : ((axis2 == 2) ? i2 : ((axis2 == 3) ? i3 : 0)));
                        int64_t srcIdx3 = (axis3 == 0) ? i0 : ((axis3 == 1) ? i1 : ((axis3 == 2) ? i2 : ((axis3 == 3) ? i3 : 0)));
                        auto srcOffset = srcIdx0 * srcStride0 + srcIdx1 * srcStride1 + srcIdx2 * srcStride2 + srcIdx3 * srcStride3;
                        
                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                        pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                        pto::TASSIGN(idxTile, (uint64_t)(tmp.GetAddr()));
                        pto::TASSIGN(tmpIdxTile, (uint64_t)(tmp.GetAddr() + dstTileW * sizeof(int32_t)));
                        
                        for (LoopVar k = 0; k < dstShape4; ++k) {
                            idxAddr[k] = k;
                        }
                        pto::TGATHER(dstTile, srcTile, idxTile, tmpIdxTile);
                    }
                }
            }
        }
    } else if constexpr (dimCount == 5) {
        for (LoopVar i0 = 0; i0 < dstShape0; ++i0) {
            for (LoopVar i1 = 0; i1 < dstShape1; ++i1) {
                for (LoopVar i2 = 0; i2 < dstShape2; ++i2) {
                    for (LoopVar i3 = 0; i3 < dstShape3; ++i3) {
                        for (LoopVar i4 = 0; i4 < dstShape4; ++i4) {
                            auto dstOffset = i0 * dstStride0 + i1 * dstStride1 + i2 * dstStride2 + 
                                              i3 * dstStride3 + i4 * dstStride4;
                            
                            int64_t srcIdx0 = (axis0 == 0) ? i0 : ((axis0 == 1) ? i1 : ((axis0 == 2) ? i2 : ((axis0 == 3) ? i3 : ((axis0 == 4) ? i4 : 0))));
                            int64_t srcIdx1 = (axis1 == 0) ? i0 : ((axis1 == 1) ? i1 : ((axis1 == 2) ? i2 : ((axis1 == 3) ? i3 : ((axis1 == 4) ? i4 : 0))));
                            int64_t srcIdx2 = (axis2 == 0) ? i0 : ((axis2 == 1) ? i1 : ((axis2 == 2) ? i2 : ((axis2 == 3) ? i3 : ((axis2 == 4) ? i4 : 0))));
                            int64_t srcIdx3 = (axis3 == 0) ? i0 : ((axis3 == 1) ? i1 : ((axis3 == 2) ? i2 : ((axis3 == 3) ? i3 : ((axis3 == 4) ? i4 : 0))));
                            int64_t srcIdx4 = (axis4 == 0) ? i0 : ((axis4 == 1) ? i1 : ((axis4 == 2) ? i2 : ((axis4 == 3) ? i3 : ((axis4 == 4) ? i4 : 0))));
                            auto srcOffset = srcIdx0 * srcStride0 + srcIdx1 * srcStride1 + srcIdx2 * srcStride2 + 
                                              srcIdx3 * srcStride3 + srcIdx4 * srcStride4;
                            
                            auto elementOffset = dstOffset;
                            dstAddr[elementOffset] = srcAddr[srcOffset];
                        }
                    }
                }
            }
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    }
}

#endif
