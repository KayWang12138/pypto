/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
TILEOP void TscatterElementS(T0 dst, T1 src1, Scalar src2)
{
    static_assert(scatterMode < SCATTER_MODE_MAX, "Unsupport scatterMode");
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    const auto dstLayout = dst.GetLayout();
    auto n0DstStride = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto n1DstStride = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto n2DstStride = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto n3DstStride = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    const auto idxLayout = src1.GetLayout();
    auto n0IdxStride = idxLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto n1IdxStride = idxLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto n2IdxStride = idxLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto n3IdxStride = idxLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    auto n0IdxShape = idxLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto n1IdxShape = idxLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto n2IdxShape = idxLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto n3IdxShape = idxLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto n4IdxShape = idxLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    auto idxAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src1.GetAddr()));
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    for (LoopVar i = 0; i < n0IdxShape; ++i) {
        for (LoopVar j = 0; j < n1IdxShape; ++j) {
            for (LoopVar k = 0; k < n2IdxShape; ++k) {
                for (LoopVar l = 0; l < n3IdxShape; ++l) {
                    for (LoopVar m = 0; m < n4IdxShape; ++m) {
                        typename T1::Type index =
                            *(idxAddr + i * n0IdxStride + j * n1IdxStride + k * n2IdxStride + l * n3IdxStride + m);
                        typename T1::Type dstOffset = 0;
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
                            dstAddr[dstOffset] = static_cast<typename T0::Type>(
                                static_cast<float>(src2) + static_cast<float>(dstAddr[dstOffset]));
                        } else {
                            dstAddr[dstOffset] = static_cast<typename T0::Type>(
                                static_cast<float>(src2) * static_cast<float>(dstAddr[dstOffset]));
                        }
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

#ifndef __DAV_V220
template <int axis, int scatterMode, typename T1, typename T2, typename T3>
TILEOP void TscatterA5(
    __gm__ typename T2::Type* dstGmBase, typename T1::Type* idxAddr, typename T2::Type* srcAddr,
    typename T3::Type* tmpAddr, int64_t dstStride0, int64_t dstStride1, int64_t dstStride2, int64_t dstStride3,
    int64_t srcStride0, int64_t srcStride1, int64_t srcStride2, int64_t srcStride3, int64_t idxStride0,
    int64_t idxStride1, int64_t idxStride2, int64_t idxStride3, int64_t idxShape0, int64_t idxShape1, int64_t idxShape2,
    int64_t idxShape3, int64_t idxShape4, int64_t d0Shape, int64_t d1Shape, int64_t d2Shape, int64_t d3Shape,
    int64_t d4Shape)
{
    using SrcDtype = typename T2::Type;
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, 1, srcTileW, pto::BLayout::RowMajor>;
    using IdxTileDefine = pto::Tile<pto::TileType::Vec, uint32_t, 1, srcTileW, pto::BLayout::RowMajor>;
    using TableGlobal = pto::GlobalTensor<SrcDtype, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    __ubuf__ uint32_t* elemIdxBuf = (__ubuf__ uint32_t*)((uint64_t)(tmpAddr));
    constexpr auto atomicOp = (scatterMode == 1) ? pto::ScatterAtomicOp::Add : pto::ScatterAtomicOp::None;

    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (LoopVar i = 0; i < idxShape0; ++i) {
        for (LoopVar j = 0; j < idxShape1; ++j) {
            for (LoopVar k = 0; k < idxShape2; ++k) {
                for (LoopVar l = 0; l < idxShape3; ++l) {
                    for (LoopVar m = 0; m < idxShape4; ++m) {
                        typename T1::Type index =
                            *(idxAddr + i * idxStride0 + j * idxStride1 + k * idxStride2 + l * idxStride3 + m);
                        int64_t dstOffset = 0;
                        if constexpr (axis == 0) {
                            dstOffset =
                                (int64_t)(index)*dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == 1) {
                            dstOffset =
                                i * dstStride0 + (int64_t)(index)*dstStride1 + k * dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == 2) {
                            dstOffset =
                                i * dstStride0 + j * dstStride1 + (int64_t)(index)*dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == 3) {
                            dstOffset =
                                i * dstStride0 + j * dstStride1 + k * dstStride2 + (int64_t)(index)*dstStride3 + m;
                        } else {
                            dstOffset =
                                i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + (int64_t)(index);
                        }
                        elemIdxBuf[m] = static_cast<uint32_t>(dstOffset);
                    }

                    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                    uint64_t srcBaseOffset = i * srcStride0 + j * srcStride1 + k * srcStride2 + l * srcStride3;
                    SrcTileDefine srcTile(1, idxShape4);
                    IdxTileDefine idxTile(1, idxShape4);

                    size_t tableTotalSize = static_cast<size_t>(d0Shape) * static_cast<size_t>(d1Shape) *
                                            static_cast<size_t>(d2Shape) * static_cast<size_t>(d3Shape) *
                                            static_cast<size_t>(d4Shape);
                    auto tableShape = pto::Shape(1, 1, 1, 1, tableTotalSize);
                    auto tableStride = pto::Stride(1, 1, 1, 1, 1);
                    TableGlobal tableGM(dstGmBase, tableShape, tableStride);

                    pto::TASSIGN(srcTile, (uint64_t)(srcAddr + srcBaseOffset));
                    pto::TASSIGN(idxTile, (uint64_t)(elemIdxBuf));
                    pto::MSCATTER<atomicOp, pto::ScatterOOB::Skip>(tableGM, srcTile, idxTile);
                    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}
#endif

template <int axis, int scatterMode, typename T0, typename T1, typename T2, typename T3, typename C>
TILEOP void Tscatter(T0 dst, T1 src1, T2 src2, T3 tmp, C coordinate)
{
    static_assert(scatterMode < SCATTER_MODE_MAX, "Unsupport scatterMode");
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();

    const auto idxLayout = src1.GetLayout();
    auto idxStride0 = idxLayout.template GetStrideDim<0, expectSize>();
    auto idxStride1 = idxLayout.template GetStrideDim<1, expectSize>();
    auto idxStride2 = idxLayout.template GetStrideDim<2, expectSize>();
    auto idxStride3 = idxLayout.template GetStrideDim<3, expectSize>();
    auto idxShape0 = idxLayout.template GetShapeDim<0, expectSize>();
    auto idxShape1 = idxLayout.template GetShapeDim<1, expectSize>();
    auto idxShape2 = idxLayout.template GetShapeDim<2, expectSize>();
    auto idxShape3 = idxLayout.template GetShapeDim<3, expectSize>();
    auto idxShape4 = idxLayout.template GetShapeDim<4, expectSize>();

    const auto srcLayout = src2.GetLayout();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();

    auto dstBaseOffset = dstLayout.template GetGmOffset<C, expectSize>(coordinate);
    auto idxAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src1.GetAddr()));
    auto srcAddr = (__ubuf__ typename T2::Type*)((uint64_t)(src2.GetAddr()));
    auto tmpAddr = (__ubuf__ typename T3::Type*)((uint64_t)(tmp.GetAddr()));
    using SrcDtype = typename T2::Type;
    __gm__ SrcDtype* dstGmBase = (__gm__ SrcDtype*)((uint64_t)(dst.GetAddr())) + dstBaseOffset;

    using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
    using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using DstGlobalType = pto::GlobalTensor<SrcDtype, ShapeDim5, StrideDim5>;
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();
    using TileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, 1, srcTileW, pto::BLayout::RowMajor, -1, -1>;
    constexpr auto atomicType = (scatterMode == 1) ? pto::AtomicType::AtomicAdd : pto::AtomicType::AtomicNone;
    auto unitShape = pto::Shape(1, 1, 1, 1, 1);
    auto unitStride = pto::Stride(1, 1, 1, 1, 1);
    __ubuf__ SrcDtype* scratch = (__ubuf__ SrcDtype*)((uint64_t)(tmpAddr));

#ifdef __DAV_V220
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    for (LoopVar i = 0; i < idxShape0; ++i) {
        for (LoopVar j = 0; j < idxShape1; ++j) {
            for (LoopVar k = 0; k < idxShape2; ++k) {
                for (LoopVar l = 0; l < idxShape3; ++l) {
                    for (LoopVar m = 0; m < idxShape4; ++m) {
                        typename T1::Type index =
                            *(idxAddr + i * idxStride0 + j * idxStride1 + k * idxStride2 + l * idxStride3 + m);
                        auto srcOffset = i * srcStride0 + j * srcStride1 + k * srcStride2 + l * srcStride3 + m;
                        int64_t dstOffset = 0;
                        if constexpr (axis == 0) {
                            dstOffset =
                                (int64_t)(index)*dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == 1) {
                            dstOffset =
                                i * dstStride0 + (int64_t)(index)*dstStride1 + k * dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == 2) {
                            dstOffset =
                                i * dstStride0 + j * dstStride1 + (int64_t)(index)*dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == 3) {
                            dstOffset =
                                i * dstStride0 + j * dstStride1 + k * dstStride2 + (int64_t)(index)*dstStride3 + m;
                        } else {
                            dstOffset =
                                i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + (int64_t)(index);
                        }
                        __gm__ SrcDtype* dstGmAddr = dstGmBase + dstOffset;
                        DstGlobalType dstGlobal(dstGmAddr, unitShape, unitStride);
                        TileDefine tile(1, 1);
                        __ubuf__ SrcDtype* srcPtr = srcAddr + srcOffset;
                        scratch[0] = *srcPtr;
                        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                        pto::TASSIGN(tile, (uint64_t)(scratch));
                        pto::TSTORE<TileDefine, DstGlobalType, atomicType>(dstGlobal, tile);
                        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
                        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
#else
    TscatterA5<axis, scatterMode, T1, T2, T3>(
        dstGmBase, idxAddr, srcAddr, tmpAddr, dstStride0, dstStride1, dstStride2, dstStride3, srcStride0, srcStride1,
        srcStride2, srcStride3, idxStride0, idxStride1, idxStride2, idxStride3, idxShape0, idxShape1, idxShape2,
        idxShape3, idxShape4, dstLayout.template GetShapeDim<0, expectSize>(),
        dstLayout.template GetShapeDim<1, expectSize>(), dstLayout.template GetShapeDim<2, expectSize>(),
        dstLayout.template GetShapeDim<3, expectSize>(), dstLayout.template GetShapeDim<4, expectSize>());
#endif
}
#endif