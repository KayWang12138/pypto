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
TILEOP void TscatterElementS(T0 dst, T1 src1, Scalar src2) {
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

    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();
    constexpr auto idxTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();

    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto idxTypeSize = sizeof(typename T1::Type);
    constexpr auto srcTypeSize = sizeof(typename T2::Type);
#ifdef __DAV_V220
    /* A2 A3不支持vscatter指令，调用pto封装接口会导致性能劣化，因此pypto自行用scalar计算实现，A5正常调用pto接口 */
    constexpr bool scalarFlag = true;
#else
    constexpr bool scalarFlag = ((sizeof(typename T1::Type) == 8) || (scatterMode > 0) ||
        (dstTypeSize == 2 && idxTypeSize == 4)) ? true : false;
#endif
    constexpr auto dstTileShapeH = TileOp::GetOutterAxisMergeResult<shapeSize, typename T0::TileShape>();
    using dstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileShapeH, dstTileW, pto::BLayout::RowMajor>;
    using idxTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, idxTileW, pto::BLayout::RowMajor, -1, -1>;
    using srcTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, srcTileW, pto::BLayout::RowMajor>;
    dstTileDefine dstTile;
    idxTileDefine idxTile(1, idxShape4);
    srcTileDefine srcTile;

    if constexpr (scalarFlag) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    }
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    auto idxAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src1.GetAddr()));
    auto srcAddr = (__ubuf__ typename T2::Type*)((uint64_t)(src2.GetAddr()));
    auto tmpAddr = (__ubuf__ typename T3::Type*)((uint64_t)(tmp.GetAddr()));
    typename T1::Type dstOffset = 0;
    for (LoopVar i = 0; i < idxShape0; ++i) {
        for (LoopVar j = 0; j < idxShape1; ++j) {
            for (LoopVar k = 0; k < idxShape2; ++k) {
                for (LoopVar l = 0; l < idxShape3; ++l) {
                    if constexpr (scalarFlag == false) {
                        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                    }
                    for (LoopVar m = 0; m < idxShape4; ++m) {
                        typename T1::Type index =
                            *(idxAddr + i * idxStride0 + j * idxStride1 + k * idxStride2 + l * idxStride3 + m);
                        typename T1::Type src2Offset = 
                            i * srcStride0 + j * srcStride1 + k * srcStride2 + l * srcStride3 + m;
                        if constexpr (axis == 0) {
                            dstOffset = index * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == 1) {
                            dstOffset = i * dstStride0 + index * dstStride1 + k * dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == 2) {
                            dstOffset = i * dstStride0 + j * dstStride1 + index * dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == 3) {
                            dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + index * dstStride3 + m;
                        } else {
                            dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + index;
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
                        auto srcOffset = i * srcStride0 + j * srcStride1 + k * srcStride2 + l * srcStride3;
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
    } else {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    }
}

template <int axis, int scatterMode, typename T0, typename T1, typename T2, typename T3, typename C>
TILEOP void Tscatter(T0 dst, T1 src1, T2 src2, T3 temp, C coordinate) {
    static_assert(scatterMode >= 0 && scatterMode < SCATTER_MODE_MAX, "Unsupported scatterMode");
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto d0Shape = dstLayout.template GetShapeDim<0, expectSize>();
    auto d1Shape = dstLayout.template GetShapeDim<1, expectSize>();
    auto d2Shape = dstLayout.template GetShapeDim<2, expectSize>();
    auto d3Shape = dstLayout.template GetShapeDim<3, expectSize>();
    auto d4Shape = dstLayout.template GetShapeDim<4, expectSize>();
    auto d0Stride = dstLayout.template GetStrideDim<0, expectSize>();
    auto d1Stride = dstLayout.template GetStrideDim<1, expectSize>();
    auto d2Stride = dstLayout.template GetStrideDim<2, expectSize>();
    auto d3Stride = dstLayout.template GetStrideDim<3, expectSize>();

    const auto idxLayout = src1.GetLayout();
    auto i0Shape = idxLayout.template GetShapeDim<0, expectSize>();
    auto i1Shape = idxLayout.template GetShapeDim<1, expectSize>();
    auto i2Shape = idxLayout.template GetShapeDim<2, expectSize>();
    auto i3Shape = idxLayout.template GetShapeDim<3, expectSize>();
    auto i4Shape = idxLayout.template GetShapeDim<4, expectSize>();
    auto i0Stride = idxLayout.template GetStrideDim<0, expectSize>();
    auto i1Stride = idxLayout.template GetStrideDim<1, expectSize>();
    auto i2Stride = idxLayout.template GetStrideDim<2, expectSize>();
    auto i3Stride = idxLayout.template GetStrideDim<3, expectSize>();
    const auto srcLayout = src2.GetLayout();
    auto s0Stride = srcLayout.template GetStrideDim<0, expectSize>();
    auto s1Stride = srcLayout.template GetStrideDim<1, expectSize>();
    auto s2Stride = srcLayout.template GetStrideDim<2, expectSize>();
    auto s3Stride = srcLayout.template GetStrideDim<3, expectSize>();
    
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto idxTypeSize = sizeof(typename T1::Type);
    constexpr auto srcTypeSize = sizeof(typename T2::Type);
    constexpr auto idxTileH = TileOp::GetTensorTileShapeDim<T1, 3, 5>();
    constexpr auto idxTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T2, 3, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();
    //modify
    using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
    using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = pto::GlobalTensor<typename T0::Type, ShapeDim5, StrideDim5>;
 
#ifdef __DAV_V220
    constexpr bool isV220 = true;
#else
    constexpr bool isV220 = false;
#endif
    constexpr auto srcTileShape1 = TileOp::GetOutterAxisMergeResult<shapeSize, typename T1::TileShape>();   
    auto dstBaseOffset = dstLayout.template GetGmOffset<C, expectSize>(coordinate);
    __gm__ typename T0::Type* dstBaseAddr = reinterpret_cast<__gm__ typename T0::Type*>(dst.GetAddr()) + dstBaseOffset;
    __ubuf__ typename T1::Type* idxBaseAddr = reinterpret_cast<__ubuf__ typename T1::Type*>(src1.GetAddr());
    __ubuf__ typename T2::Type* srcBaseAddr = reinterpret_cast<__ubuf__ typename T2::Type*>(src2.GetAddr());
    constexpr bool useScalarPath = isV220 || (scatterMode == 2);
    if constexpr (useScalarPath) {
        for (LoopVar i = 0; i < i0Shape; ++i) {
            for (LoopVar j = 0; j < i1Shape; ++j) {
                for (LoopVar k = 0; k < i2Shape; ++k) {
                    for (LoopVar l = 0; l < i3Shape; ++l) {
                        for (LoopVar m = 0; m < i4Shape; ++m) {
                            typename T1::Type index = *(idxBaseAddr + i * i0Stride + j * i1Stride + k * i2Stride + l * i3Stride + m);
                            typename T2::Type srcVal = *(srcBaseAddr + i * s0Stride + j * s1Stride + k * s2Stride + l * s3Stride + m);
                            int64_t localOffset = 0;
                            if constexpr (axis == 0) {
                                localOffset = static_cast<int64_t>(index) * d0Stride + j * d1Stride + k * d2Stride + l * d3Stride + m;
                            } else if constexpr (axis == 1) {
                                localOffset = i * d0Stride + static_cast<int64_t>(index) * d1Stride + k * d2Stride + l * d3Stride + m;
                            } else if constexpr (axis == 2) {
                                localOffset = i * d0Stride + j * d1Stride + static_cast<int64_t>(index) * d2Stride + l * d3Stride + m;
                            } else if constexpr (axis == 3) {
                                localOffset = i * d0Stride + j * d1Stride + k * d2Stride + static_cast<int64_t>(index) * d3Stride + m;
                            } else {
                                localOffset = i * d0Stride + j * d1Stride + k * d2Stride + l * d3Stride + static_cast<int64_t>(index);
                            }
                            if constexpr (scatterMode == 0) {
                                dstBaseAddr[localOffset] = srcVal;
                            } else if constexpr (scatterMode == 1) {
                                dstBaseAddr[localOffset] += srcVal;
                            } else if constexpr (scatterMode == 2) {
                                dstBaseAddr[localOffset] *= srcVal;
                            }
                        }
                    }
                }
            }
        }
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    } else {
        __ubuf__ uint32_t* offsetBuf = reinterpret_cast<__ubuf__ uint32_t*>(temp.GetAddr());
        constexpr bool useMScatter = (scatterMode == 0 || scatterMode == 1);
        if constexpr (useMScatter) {
            using SrcType = typename T2::Type;
            using IdxType = typename T1::Type;

            using SrcTileType = pto::Tile<pto::TileType::Vec, SrcType, srcTileH, srcTileW, pto::BLayout::RowMajor>;
            using OffsetTileType = pto::Tile<pto::TileType::Vec, IdxType, idxTileH, idxTileW, pto::BLayout::RowMajor>;

            for (LoopVar i = 0; i < i0Shape; ++i) {
                for (LoopVar j = 0; j < i1Shape; ++j) {
                    for (LoopVar k = 0; k < i2Shape; ++k) {
                        for (LoopVar l = 0; l < i3Shape; ++l) {
                            // 填充 offsetBuf: [NUM_COLS] 个偏移量
                            for (LoopVar m = 0; m < i4Shape; ++m) {
                                IdxType localIdx = *(idxBaseAddr + i * i0Stride + j * i1Stride + k * i2Stride + l * i3Stride + m);
                                int64_t elemOffset = 0;
                                if constexpr (axis == 0) {
                                    elemOffset = static_cast<int64_t>(localIdx) * d0Stride + j * d1Stride + k * d2Stride + l * d3Stride + m;
                                } else if constexpr (axis == 1) {
                                    elemOffset = i * d0Stride + static_cast<int64_t>(localIdx) * d1Stride + k * d2Stride + l * d3Stride + m;
                                } else if constexpr (axis == 2) {
                                    elemOffset = i * d0Stride + j * d1Stride + static_cast<int64_t>(localIdx) * d2Stride + l * d3Stride + m;
                                } else if constexpr (axis == 3) {
                                    elemOffset = i * d0Stride + j * d1Stride + k * d2Stride + static_cast<int64_t>(localIdx) * d3Stride + m;
                                } else {
                                    elemOffset = i * d0Stride + j * d1Stride + k * d2Stride + l * d3Stride + static_cast<int64_t>(localIdx);
                                }
                                offsetBuf[l * i3Shape + m] = static_cast<uint32_t>(elemOffset);
                            }   
                        }
                        // 构造 Tile
                        SrcTileType srcTile;
                        OffsetTileType offsetTile;

                        // 地址对齐到当前 (i,j,k,l) 块的起始位置
                        uint64_t srcTileAddr = reinterpret_cast<uint64_t>(
                            srcBaseAddr + i * s0Stride + j * s1Stride + k * s2Stride + l * s3Stride
                        );
                        GlobalData dstGlobal(dstBaseAddr, pto::Shape(1, 1, 1, d3Shape, d4Shape), pto::Stride(0, 0, 0, d3Stride, 1));
                        pto::TASSIGN(srcTile, srcTileAddr);
                        pto::TASSIGN(offsetTile, reinterpret_cast<uint64_t>(offsetBuf));

                        // 调用 MSCATTER
                        if constexpr (scatterMode == 0) {
                            pto::MSCATTER<pto::ScatterAtomicOp::None, pto::ScatterOOB::Skip>(
                                dstGlobal, srcTile, offsetTile
                            );
                        } else if constexpr (scatterMode == 1) {
                            pto::MSCATTER<pto::ScatterAtomicOp::Add, pto::ScatterOOB::Skip>(
                                dstGlobal, srcTile, offsetTile
                            );
                        }
                    }
                }
            }
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
        }
    }  
}                            
#endif
