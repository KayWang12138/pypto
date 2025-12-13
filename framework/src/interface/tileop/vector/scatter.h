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
 * \file scatter.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_SCATTER__H
#define TILEOP_TILE_OPERATOR_SCATTER__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

constexpr unsigned SCATTER_MODE_MAX = 3;
template <int axis, int scatterMode, typename T0, typename T1, typename Scalar>
TILEOP void TscatterElementS(T0 dst, T1 src1, Scalar src2) {
    static_assert(scatterMode < SCATTER_MODE_MAX, "Unsupport scatterMode");
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto n0DstStride = dstLayout.template GetStrideDim<0, expectSize>();
    auto n1DstStride = dstLayout.template GetStrideDim<1, expectSize>();
    auto n2DstStride = dstLayout.template GetStrideDim<2, expectSize>();
    auto n3DstStride = dstLayout.template GetStrideDim<3, expectSize>();

    const auto idxLayout = src1.GetLayout();
    auto n0IdxStride = idxLayout.template GetStrideDim<0, expectSize>();
    auto n1IdxStride = idxLayout.template GetStrideDim<1, expectSize>();
    auto n2IdxStride = idxLayout.template GetStrideDim<2, expectSize>();
    auto n3IdxStride = idxLayout.template GetStrideDim<3, expectSize>();
    auto n0IdxShape = idxLayout.template GetShapeDim<0, expectSize>();
    auto n1IdxShape = idxLayout.template GetShapeDim<1, expectSize>();
    auto n2IdxShape = idxLayout.template GetShapeDim<2, expectSize>();
    auto n3IdxShape = idxLayout.template GetShapeDim<3, expectSize>();
    auto n4IdxShape = idxLayout.template GetShapeDim<4, expectSize>();

    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    auto idxAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src1.GetAddr()));
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    for (int i = 0; i < n0IdxShape; ++i) {
        for (int j = 0; j < n1IdxShape; ++j) {
            for (int k = 0; k < n2IdxShape; ++k) {
                for (int l = 0; l < n3IdxShape; ++l) {
                    for (int m = 0; m < n4IdxShape; ++m) {
                        auto index =
                            *(idxAddr + i * n0IdxStride + j * n1IdxStride + k * n2IdxStride + l * n3IdxStride + m);
                        int dstOffset = 0;
                        if constexpr (axis == 0) {
                            dstOffset = index * n0DstStride + j * n1DstStride + k * n2DstStride + l * n3DstStride + m;
                        } else if constexpr (axis == 1) {
                            dstOffset = i * n0DstStride + index * n1DstStride + k * n2DstStride + l * n3DstStride + m;
                        } else if constexpr (axis == 2) {
                            dstOffset = i * n0DstStride + j * n1DstStride + index * n2DstStride + l * n3DstStride + m;
                        } else if constexpr (axis == 3) {
                            dstOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride + index * n3DstStride + m;
                        } else {
                            dstOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride + l * n3DstStride + index;
                        }
                        if constexpr (scatterMode == 0) {
                            dstAddr[dstOffset] = src2;
                        } else if constexpr (scatterMode == 1) {
                            dstAddr[dstOffset] = static_cast<typename T0::Type>(static_cast<float>(src2) +
                                static_cast<float>(dstAddr[dstOffset]));
                        } else {
                            dstAddr[dstOffset] = static_cast<typename T0::Type>(static_cast<float>(src2) *
                                static_cast<float>(dstAddr[dstOffset]));
                        }
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <int axis, int scatterMode, typename T0, typename T1, typename T2, typename T3>
TILEOP void Tscatter(T0 dst, T1 src1, T2 src2, T3 tmp) {
    static_assert(scatterMode < SCATTER_MODE_MAX, "Unsupport scatterMode");
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto n0DstStride = dstLayout.template GetStrideDim<0, expectSize>();
    auto n1DstStride = dstLayout.template GetStrideDim<1, expectSize>();
    auto n2DstStride = dstLayout.template GetStrideDim<2, expectSize>();
    auto n3DstStride = dstLayout.template GetStrideDim<3, expectSize>();

    const auto idxLayout = src1.GetLayout();
    auto n0IdxStride = idxLayout.template GetStrideDim<0, expectSize>();
    auto n1IdxStride = idxLayout.template GetStrideDim<1, expectSize>();
    auto n2IdxStride = idxLayout.template GetStrideDim<2, expectSize>();
    auto n3IdxStride = idxLayout.template GetStrideDim<3, expectSize>();
    auto n0IdxShape = idxLayout.template GetShapeDim<0, expectSize>();
    auto n1IdxShape = idxLayout.template GetShapeDim<1, expectSize>();
    auto n2IdxShape = idxLayout.template GetShapeDim<2, expectSize>();
    auto n3IdxShape = idxLayout.template GetShapeDim<3, expectSize>();
    auto n4IdxShape = idxLayout.template GetShapeDim<4, expectSize>();
    const auto srcLayout = src2.GetLayout();
    auto n0SrcStride = srcLayout.template GetStrideDim<0, expectSize>();
    auto n1SrcStride = srcLayout.template GetStrideDim<1, expectSize>();
    auto n2SrcStride = srcLayout.template GetStrideDim<2, expectSize>();
    auto n3SrcStride = srcLayout.template GetStrideDim<3, expectSize>();

    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();
    constexpr auto idxTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();

    constexpr bool scalarFlag = ((sizeof(typename T1::Type) == 8) || (scatterMode > 0)) ? true : false;
    constexpr auto idxTypeSize = sizeof(typename T1::Type);
    constexpr auto srcTypeSize = sizeof(typename T2::Type);
    constexpr auto dstTileShapeH = TileOp::GetOutterAxisMergeResult<shapeSize, typename T0::TileShape>();
    using dstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileShapeH, dstTileW, pto::BLayout::RowMajor>;
    using idxTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, idxTileW, pto::BLayout::RowMajor, -1, -1>;
    using srcTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, srcTileW, pto::BLayout::RowMajor>;
    dstTileDefine dstTile;
    idxTileDefine idxTile(1, n4IdxShape);
    srcTileDefine srcTile;

    if constexpr (scalarFlag) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    }
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    auto idxAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src1.GetAddr()));
    auto srcAddr = (__ubuf__ typename T2::Type*)((uint64_t)(src2.GetAddr()));
    auto tmpAddr = (__ubuf__ typename T3::Type*)((uint64_t)(tmp.GetAddr()));
    for (int i = 0; i < n0IdxShape; ++i) {
        for (int j = 0; j < n1IdxShape; ++j) {
            for (int k = 0; k < n2IdxShape; ++k) {
                for (int l = 0; l < n3IdxShape; ++l) {
                    if constexpr (scalarFlag == false) {
                        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                    }
                    for (int m = 0; m < n4IdxShape; ++m) {
                        auto index =
                            *(idxAddr + i * n0IdxStride + j * n1IdxStride + k * n2IdxStride + l * n3IdxStride + m);
                        int src2Offset = i * n0SrcStride + j * n1SrcStride + k * n2SrcStride + l * n3SrcStride + m;
                        int dstOffset = 0;
                        if constexpr (axis == 0) {
                            dstOffset = index * n0DstStride + j * n1DstStride + k * n2DstStride + l * n3DstStride + m;
                        } else if constexpr (axis == 1) {
                            dstOffset = i * n0DstStride + index * n1DstStride + k * n2DstStride + l * n3DstStride + m;
                        } else if constexpr (axis == 2) {
                            dstOffset = i * n0DstStride + j * n1DstStride + index * n2DstStride + l * n3DstStride + m;
                        } else if constexpr (axis == 3) {
                            dstOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride + index * n3DstStride + m;
                        } else {
                            dstOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride + l * n3DstStride + index;
                        }
                        /* idx类型为int64或scatter操作为add或multiply，退化为标量实现 */
                        if constexpr (scalarFlag) {
                            if constexpr (scatterMode == 0) {
                                dstAddr[dstOffset] = srcAddr[src2Offset];
                            } else if constexpr (scatterMode == 1) {
                                dstAddr[dstOffset] = srcAddr[src2Offset] + dstAddr[dstOffset];
                            } else {
                                dstAddr[dstOffset] = srcAddr[src2Offset] * dstAddr[dstOffset];
                            }
                        } else {
                            *(tmpAddr + m) = dstOffset;
                        }
                    }
                    if constexpr (scalarFlag == false) {
                        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                        // auto idxOffset = i * n0IdxStride + j * n1IdxStride + k * n2IdxStride;
                        auto srcOffset = i * n0SrcStride + j * n1SrcStride + k * n2SrcStride + l * n3SrcStride;
                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr()));
                        pto::TASSIGN(idxTile, (uint64_t)(tmp.GetAddr()));
                        pto::TASSIGN(srcTile, (uint64_t)(src2.GetAddr() + srcOffset * srcTypeSize));
                        pto::TSCATTER(dstTile, srcTile, idxTile);
                    }
                }
            }
        }
    }
    if constexpr (scalarFlag) {
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    }
}

#endif
