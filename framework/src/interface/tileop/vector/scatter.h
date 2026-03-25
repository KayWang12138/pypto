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

//原版
// template <int axis, int scatterMode, typename T0, typename T1, typename T2, typename T3>
// TILEOP void Tscatter(T0 dst, T1 src1, T2 src2, T3 tmp) {
//     static_assert(scatterMode < SCATTER_MODE_MAX, "Unsupport scatterMode");
//     constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
//     constexpr size_t expectSize = 5;
//     const auto dstLayout = dst.GetLayout();
//     auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
//     auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
//     auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
//     auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();

//     const auto idxLayout = src1.GetLayout();
//     auto idxStride0 = idxLayout.template GetStrideDim<0, expectSize>();
//     auto idxStride1 = idxLayout.template GetStrideDim<1, expectSize>();
//     auto idxStride2 = idxLayout.template GetStrideDim<2, expectSize>();
//     auto idxStride3 = idxLayout.template GetStrideDim<3, expectSize>();
//     auto idxShape0 = idxLayout.template GetShapeDim<0, expectSize>();
//     auto idxShape1 = idxLayout.template GetShapeDim<1, expectSize>();
//     auto idxShape2 = idxLayout.template GetShapeDim<2, expectSize>();
//     auto idxShape3 = idxLayout.template GetShapeDim<3, expectSize>();
//     auto idxShape4 = idxLayout.template GetShapeDim<4, expectSize>();
//     const auto srcLayout = src2.GetLayout();
//     auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
//     auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
//     auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
//     auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();

//     constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();
//     constexpr auto idxTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
//     constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();

//     constexpr auto dstTypeSize = sizeof(typename T0::Type);
//     constexpr auto idxTypeSize = sizeof(typename T1::Type);
//     constexpr auto srcTypeSize = sizeof(typename T2::Type);
// #ifdef __DAV_V220
//     /* A2 A3不支持vscatter指令，调用pto封装接口会导致性能劣化，因此pypto自行用scalar计算实现，A5正常调用pto接口 */
//     constexpr bool scalarFlag = true;
// #else
//     constexpr bool scalarFlag = ((sizeof(typename T1::Type) == 8) || (scatterMode > 0) ||
//         (dstTypeSize == 2 && idxTypeSize == 4)) ? true : false;
// #endif
//     constexpr auto dstTileShapeH = TileOp::GetOutterAxisMergeResult<shapeSize, typename T0::TileShape>();
//     using dstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileShapeH, dstTileW, pto::BLayout::RowMajor>;
//     using idxTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, idxTileW, pto::BLayout::RowMajor, -1, -1>;
//     using srcTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, srcTileW, pto::BLayout::RowMajor>;
//     dstTileDefine dstTile;
//     idxTileDefine idxTile(1, idxShape4);
//     srcTileDefine srcTile;

//     if constexpr (scalarFlag) {
//         set_flag(PIPE_V, PIPE_S, EVENT_ID7);
//         wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
//     }
//     auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
//     auto idxAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src1.GetAddr()));
//     auto srcAddr = (__ubuf__ typename T2::Type*)((uint64_t)(src2.GetAddr()));
//     auto tmpAddr = (__ubuf__ typename T3::Type*)((uint64_t)(tmp.GetAddr()));
//     typename T1::Type dstOffset = 0;
//     for (LoopVar i = 0; i < idxShape0; ++i) {
//         for (LoopVar j = 0; j < idxShape1; ++j) {
//             for (LoopVar k = 0; k < idxShape2; ++k) {
//                 for (LoopVar l = 0; l < idxShape3; ++l) {
//                     if constexpr (scalarFlag == false) {
//                         set_flag(PIPE_V, PIPE_S, EVENT_ID7);
//                         wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
//                     }
//                     for (LoopVar m = 0; m < idxShape4; ++m) {
//                         typename T1::Type index =
//                             *(idxAddr + i * idxStride0 + j * idxStride1 + k * idxStride2 + l * idxStride3 + m);
//                         typename T1::Type src2Offset = 
//                             i * srcStride0 + j * srcStride1 + k * srcStride2 + l * srcStride3 + m;
//                         if constexpr (axis == 0) {
//                             dstOffset = index * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + m;
//                         } else if constexpr (axis == 1) {
//                             dstOffset = i * dstStride0 + index * dstStride1 + k * dstStride2 + l * dstStride3 + m;
//                         } else if constexpr (axis == 2) {
//                             dstOffset = i * dstStride0 + j * dstStride1 + index * dstStride2 + l * dstStride3 + m;
//                         } else if constexpr (axis == 3) {
//                             dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + index * dstStride3 + m;
//                         } else {
//                             dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + index;
//                         }
//                         /* idx类型为int64或scatter操作为add或multiply，退化为标量实现 */
//                         if constexpr (scalarFlag) {
//                             if constexpr (scatterMode == 0) {
//                                 dstAddr[dstOffset] = srcAddr[src2Offset];
//                             } else if constexpr (scatterMode == 1) {
//                                 dstAddr[dstOffset] = srcAddr[src2Offset] + dstAddr[dstOffset];
//                             } else {
//                                 dstAddr[dstOffset] = srcAddr[src2Offset] * dstAddr[dstOffset];
//                             }
//                         } else {
//                             *(tmpAddr + m) = dstOffset;
//                         }
//                     }
//                     if constexpr (scalarFlag == false) {
//                         set_flag(PIPE_S, PIPE_V, EVENT_ID7);
//                         wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
//                         auto srcOffset = i * srcStride0 + j * srcStride1 + k * srcStride2 + l * srcStride3;
//                         pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr()));
//                         pto::TASSIGN(idxTile, (uint64_t)(tmp.GetAddr()));
//                         pto::TASSIGN(srcTile, (uint64_t)(src2.GetAddr() + srcOffset * srcTypeSize));
//                         pto::TSCATTER(dstTile, srcTile, idxTile);
//                     }
//                 }
//             }
//         }
//     }
//     if constexpr (scalarFlag) {
//         set_flag(PIPE_S, PIPE_V, EVENT_ID7);
//         wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
//     } else {
//         set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
//         wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
//     }
// }


template <int axis, int scatterMode, typename T0, typename T1, typename T2, typename T3, typename C>
TILEOP void Tscatter(T0 dst, T1 src1, T2 src2, T3 tmp, C coordinate) {
    static_assert(scatterMode < SCATTER_MODE_MAX, "Unsupport scatterMode");
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto d0Shape = dstLayout.template GetShapeDim<0, expectSize>();
    auto d1Shape = dstLayout.template GetShapeDim<1, expectSize>();
    auto d2Shape = dstLayout.template GetShapeDim<2, expectSize>();
    auto d3Shape = dstLayout.template GetShapeDim<3, expectSize>();
    auto d4Shape = dstLayout.template GetShapeDim<4, expectSize>();
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

    constexpr auto idxTileH = TileOp::GetTensorTileShapeDim<T1, 3, 5>();
    constexpr auto idxTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T2, 3, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();

#ifdef __DAV_V220
    constexpr bool isV220 = true;
#else
    constexpr bool isV220 = false;
#endif

    // Scalar path conditions
    constexpr bool scalarFlag = isV220 || (scatterMode == 2);
    using DstTileType = pto::Tile<pto::TileType::Vec, float, 1, idxTileW, pto::BLayout::RowMajor, -1, -1>;
    // using IdxTileType = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, idxTileW, pto::BLayout::RowMajor, -1,
    // -1>; using SrcTileType = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, srcTileW, pto::BLayout::RowMajor -1,
    // -1>;

    using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
    using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDstType = pto::GlobalTensor<typename T0::Type, ShapeDim5, StrideDim5>;

    // DstTileType dstTile(1, 1);
    // IdxTileType idxTile(1, idxShape4);
    // SrcTileType srcTile;

    auto dstBaseOffset = dstLayout.template GetGmOffset<C, expectSize>(coordinate);
    __gm__ typename T0::Type *dstGmBase = (__gm__ typename T0::Type *)((uint64_t)(dst.GetAddr())) + dstBaseOffset;
    auto idxAddr = (__ubuf__ typename T1::Type *)((uint64_t)(src1.GetAddr()));
    auto srcAddr = (__ubuf__ typename T2::Type *)((uint64_t)(src2.GetAddr()));
    auto tmpAddr = (__ubuf__ typename T3::Type *)((uint64_t)(tmp.GetAddr()));

    if constexpr (scalarFlag) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        // Use tmpAddr as workspace: [0] for srcVal, [1] for loaded dstVal
        __ubuf__ char *buf = (__ubuf__ char *)((uint64_t)(tmpAddr));
        __ubuf__ typename T2::Type *srcValBuf = (__ubuf__ typename T2::Type *)((uint64_t)(buf));
        __ubuf__ typename T2::Type *dstValBuf = (__ubuf__ typename T2::Type *)((uint64_t)(buf));

        for (LoopVar i = 0; i < idxShape0; ++i) {
            for (LoopVar j = 0; j < idxShape1; ++j) {
                for (LoopVar k = 0; k < idxShape2; ++k) {
                    for (LoopVar l = 0; l < idxShape3; ++l) {
                        for (LoopVar m = 0; m < idxShape4; ++m) {
                            typename T1::Type index =
                                *(idxAddr + i * idxStride0 + j * idxStride1 + k * idxStride2 + l * idxStride3 + m);
                            typename T2::Type srcVal =
                                *(srcAddr + i * srcStride0 + j * srcStride1 + k * srcStride2 + l * srcStride3 + m);

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
                                dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 +
                                            (int64_t)(index);
                            }

                            __gm__ typename T0::Type *dstGmAddr = dstGmBase + dstOffset;
                            GlobalDstType globalDst(dstGmAddr, pto::Shape(1, 1, 1, 1, 1), pto::Stride(1, 1, 1, 1, 1));

                            *srcValBuf = srcVal;

                            if constexpr (scatterMode == 0) {
                                // pto::TASSIGN(dstTile, (uint64_t)(srcValBuf));
                                // pto::TSTORE(globalDst, dstTile);
                                dcci((__gm__ T0 *)dstGmAddr, 0);
                                *dstGmAddr = srcVal;
                                dcci((__gm__ T0 *)dstGmAddr, 0);
                            } else {
                                // pto::TASSIGN(dstTile, (uint64_t)(dstValBuf));
                                // pto::TLOAD(dstTile, globalDst);

                                if constexpr (scatterMode == 1) {
                                
                                    
                                    float val_to_add = static_cast<float>(srcVal);

                                    // 2. 配置原子操作为浮点数加法 (ATOMIC_SUM)
                                    set_st_atomic_cfg(ATOMIC_F32, ATOMIC_SUM);

                                    // 3. 执行原子操作
                                    // dcci: 确保指令顺序，防止乱序
                                    dcci((__gm__ float *)(dstGmAddr), 0);
                                    
                                    // st_atomic: 将 val_to_add 累加到 dstGmAddr 指向的内存地址
                                    // 这里的第二个参数是“加数”，不是“写入值”（因为配置了 SUM 模式）
                                    st_atomic(val_to_add, (__gm__ float *)(dstGmAddr));
                                    
                                    dcci((__gm__ float *)(dstGmAddr), 0);
                                } else {
                                    *srcValBuf = *dstValBuf * srcVal;
                                    dcci((__gm__ T0 *)dstGmAddr, 0);
                                    *dstGmAddr = *srcValBuf;
                                    dcci((__gm__ T0 *)dstGmAddr, 0);
                                }

                                
                            }
                        }
                    }
                }
            }
        }
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    } else {
        for (LoopVar i = 0; i < idxShape0; ++i) {
            for (LoopVar j = 0; j < idxShape1; ++j) {
                for (LoopVar k = 0; k < idxShape2; ++k) {
                    uint64_t srcBaseOffset = i * srcStride0 + j * srcStride1 + k * srcStride2;

                    // uint64_t dstBaseSliceOffset = i * dstStride0 + j * dstStride1 + k * dstStride2;

                    for (LoopVar l = 0; l < idxShape3; ++l) {
                        for (LoopVar m = 0; m < idxShape4; ++m) {
                            typename T1::Type index =
                                *(idxAddr + i * idxStride0 + j * idxStride1 + k * idxStride2 + l * idxStride3 + m);

                            int64_t dstLinearOffset = 0;
                            if constexpr (axis == 0) {
                                dstLinearOffset =
                                    (int64_t)(index)*dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + m;
                            } else if constexpr (axis == 1) {
                                dstLinearOffset =
                                    i * dstStride0 + (int64_t)(index)*dstStride1 + k * dstStride2 + l * dstStride3 + m;
                            } else if constexpr (axis == 2) {
                                dstLinearOffset =
                                    i * dstStride0 + j * dstStride1 + (int64_t)(index)*dstStride2 + l * dstStride3 + m;
                            } else if constexpr (axis == 3) {
                                dstLinearOffset =
                                    i * dstStride0 + j * dstStride1 + k * dstStride2 + (int64_t)(index)*dstStride3 + m;
                            } else { // axis == 4
                                dstLinearOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 +
                                                  (int64_t)(index);
                            }
                            // Store the final GM offset (relative to dstGmBase) into tmpAddr
                            // Assuming tmpAddr has enough space for l*idxShape4 + m entries
                            ((__ubuf__ uint32_t *)((uint64_t)(tmpAddr)))[l * idxShape4 + m] =
                                (uint32_t)(dstLinearOffset);
                        }
                    }

                    using SrcTileType =
                        pto::Tile<pto::TileType::Vec, typename T2::Type, idxShape3, idxShape4, pto::BLayout::RowMajor>;
                    using IdxTileType =
                        pto::Tile<pto::TileType::Vec, uint32_t, srcTileH, srcTileW, pto::BLayout::RowMajor>;
                    // DstTile: placeholder, MSCATTER doesn't use it directly for dst addr
                    using GlobalDstType = pto::GlobalTensor<typename T0::Type, pto::Shape<-1, -1, -1, -1, -1>,
                        pto::Stride<-1, -1, -1, -1, -1>>;

                    SrcTileType srcTile;
                    IdxTileType idxTile;
                    // The destination tensor is defined relative to the base address calculated earlier
                    GlobalDstType globalDst(
                        dstGmBase, pto::Shape(1, 1, 1, d3Shape, d4Shape), pto::Stride(1, 1, 1, dstStride3, 1));

                    pto::TASSIGN(srcTile, (uint64_t)(srcAddr + srcBaseOffset));
                    pto::TASSIGN(idxTile, (uint64_t)(tmpAddr));

                    // if constexpr (scatterMode == 0) {
                    //     pto::MSCATTER<pto::ScatterAtomicOp::None, pto::ScatterOOB::Skip>(dstGlobal, srcTile, idxTile);
                    // } else if constexpr (scatterMode == 1) {
                    //     pto::MSCATTER<pto::ScatterAtomicOp::Add, pto::ScatterOOB::Skip>(dstGlobal, srcTile, idxTile);
                    // }
                }
            }
        }
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    }
}
#endif

