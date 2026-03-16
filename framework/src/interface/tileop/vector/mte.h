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
 * \file mte.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_MTE__H
#define TILEOP_TILE_OPERATOR_MTE__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_UB_COPY_IN TLoad
template <typename T, typename U, typename C>
__aicore__ inline void TLoad(T dst, U src, C coordinate) {
    if constexpr (T::FORMAT == Hardware::UB && U::FORMAT == Hardware::GM) {
        const auto srcLayout = src.GetLayout();
        auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
        auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
        auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

        const auto dstLayout = dst.GetLayout();
        auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
        auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
        auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
        auto gmOffset = srcLayout.template GetGmOffset<C, MAX_DIMS>(coordinate);

        if constexpr (TileOp::IsConstContinous<T>() == true) {
            // 对于静态整块场景，将UB合成二维，GM保持五维
            auto srcGlobal = PtoGlobal<U, typename T::Shape, typename U::Stride>(
                src.GetAddr() + gmOffset, dst.GetShape(), src.GetStride())
                                 .Data();
            auto dstTile = PtoTile<T, pto::BLayout::RowMajor, true>().Data();
            pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
            pto::TLOAD(dstTile, srcGlobal);
            return;
        }

        auto dstTile = PtoTile<T>(dst);
        auto srcGlobal = PtoGlobal<T, typename T::Shape, typename U::Stride, true>(dst.GetShape(), src.GetStride());
        for (LoopVar index0 = 0; index0 < dstShape0; ++index0) {
            for (LoopVar index1 = 0; index1 < dstShape1; ++index1) {
                for (LoopVar index2 = 0; index2 < dstShape2; ++index2) {
                    srcGlobal.Assign(
                        src.GetAddr() + gmOffset + index0 * srcStride0 + index1 * srcStride1 + index2 * srcStride2);
                    auto tileOffsets = TileOffset(index0, index1, index2);
                    dstTile.Assign(dst, tileOffsets);
                    pto::TLOAD(dstTile.Data(), srcGlobal.Data());
                }
            }
        }
    }
}

#define OP_TILE_OP_UB_COPY_OUT TStore
template <typename T, typename U, typename C>
__aicore__ inline void TStore(T dst, U src, C coordinate) {
    if constexpr (U::FORMAT == Hardware::UB && T::FORMAT == Hardware::GM) {
        const auto srcLayout = src.GetLayout();
        auto srcShape0 = srcLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
        auto srcShape1 = srcLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
        auto srcShape2 = srcLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
        const auto dstLayout = dst.GetLayout();
        auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
        auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
        auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
        auto gmOffset = dstLayout.template GetGmOffset<C, MAX_DIMS>(coordinate);
        using SrcDtype = std::conditional_t<std::is_same_v<typename U::Type, bool>, uint8_t, typename U::Type>;

        if constexpr (TileOp::IsConstContinous<U>() == true) {
            // 对于静态整块场景，将UB合成二维，GM保持五维
            auto dstGlobal = PtoGlobal<T, typename U::Shape, typename T::Stride>(
                dst.GetAddr() + gmOffset, src.GetShape(), dst.GetStride())
                                 .Data();
            auto srctTile = PtoTile<U, pto::BLayout::RowMajor, true>().Data();
            pto::TASSIGN(srctTile, (uint64_t)src.GetAddr());
            pto::TSTORE(dstGlobal, srctTile);
            return;
        }

        auto srctTile = PtoTile<U>(src);
        auto dstGlobal = PtoGlobal<T, typename U::Shape, typename T::Stride, true>(src.GetShape(), dst.GetStride());
        for (LoopVar index0 = 0; index0 < srcShape0; ++index0) {
            for (LoopVar index1 = 0; index1 < srcShape1; ++index1) {
                for (LoopVar index2 = 0; index2 < srcShape2; ++index2) {
                    dstGlobal.Assign(
                        dst.GetAddr() + gmOffset + index0 * dstStride0 + index1 * dstStride1 + index2 * dstStride2);
                    auto tileOffsets = TileOffset(index0, index1, index2);
                    srctTile.Assign(src, tileOffsets);
                    pto::TSTORE(dstGlobal.Data(), srctTile.Data());
                }
            }
        }
    }
}

template <bool copyIn, typename GlobalData, typename TileDefine>
__aicore__ inline void ProcessTransMove(GlobalData globalData, TileDefine ubData) {
    if constexpr (copyIn) {
        pto::TLOAD(ubData, globalData);
    } else {
        pto::TSTORE(globalData, ubData);
    }
}

template <typename GMType, typename UBType, bool copyIn, int tileW>
__aicore__ inline void DoTransMove(size_t *srcShape, size_t gmShape4,
    size_t *gmStride, size_t *ubStride, GMType *gmAddr, size_t gmOffset, uint64_t ubAddr) {
    using GlobalData = pto::GlobalTensor<GMType, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    for (LoopVar index0 = 0; index0 < srcShape[0]; ++index0) {
        for (LoopVar index1 = 0; index1 < srcShape[1]; ++index1) {
            for (LoopVar index2 = 0; index2 < srcShape[2]; ++index2) {
                for (LoopVar index3 = 0; index3 < srcShape[3]; ++index3) {
                    GlobalData globalData(gmAddr + gmOffset + index0 * gmStride[0] +
                        index1 * gmStride[1] + index2 * gmStride[2] + index3 * gmStride[3],
                        pto::Shape(1, 1, 1, 1, gmShape4), pto::Stride(0, 0, 0, 0, 0));
                    using TileDefine =
                        pto::Tile<pto::TileType::Vec, UBType, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
                    TileDefine ubData(1, srcShape[4]);
                    auto ubOffset = index0 * ubStride[0] + index1 * ubStride[1] +
                        index2 * ubStride[2] + index3 * ubStride[3];
                    pto::TASSIGN(ubData, ubAddr + ubOffset * sizeof(UBType));
                    ProcessTransMove<copyIn>(globalData, ubData);
                }
            }
        }
    }
}

template <unsigned axis0, unsigned axis1, bool copyIn, typename GM, typename UB, typename C>
__aicore__ inline void CallTransMove(GM gm, UB ub, C coordinate) {
    static_assert(axis0 != 4 && axis1 != 4);
    constexpr auto shapeSize = Std::tuple_size<typename UB::Shape>::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename UB::TileShape>::type::value;
    const auto gmLayout = gm.GetLayout();
    size_t gmShape4 = static_cast<size_t>(gmLayout.template GetShapeDim<4, 5>());
    size_t gmStride[] = {
        static_cast<size_t>(gmLayout.template GetStrideDim<0, 5>()),
        static_cast<size_t>(gmLayout.template GetStrideDim<1, 5>()),
        static_cast<size_t>(gmLayout.template GetStrideDim<2, 5>()),
        static_cast<size_t>(gmLayout.template GetStrideDim<3, 5>())
    };
    size_t gmOffset = static_cast<size_t>(gmLayout.template GetGmOffset<C, 5>(coordinate));
    const auto ubLayout = ub.GetLayout();
    size_t srcShape[] = {
        static_cast<size_t>(ubLayout.template GetShapeDim<0, 5>()),
        static_cast<size_t>(ubLayout.template GetShapeDim<1, 5>()),
        static_cast<size_t>(ubLayout.template GetShapeDim<2, 5>()),
        static_cast<size_t>(ubLayout.template GetShapeDim<3, 5>()),
        static_cast<size_t>(ubLayout.template GetShapeDim<4, 5>())
    };
    size_t ubStride[] = {
        static_cast<size_t>(ubLayout.template GetStrideDim<0, 5>()),
        static_cast<size_t>(ubLayout.template GetStrideDim<1, 5>()),
        static_cast<size_t>(ubLayout.template GetStrideDim<2, 5>()),
        static_cast<size_t>(ubLayout.template GetStrideDim<3, 5>())
    };
    auto exchangeAxis = [](size_t *arr) {
        auto tmp = arr[axis0];
        arr[axis0] = arr[axis1];
        arr[axis1] = tmp;
    };
    if constexpr (copyIn) {
        exchangeAxis(ubStride);
        exchangeAxis(srcShape);
    } else {
        exchangeAxis(gmStride);
    }
    using GMType = std::conditional_t<std::is_same_v<typename GM::Type, bool>, uint8_t, typename GM::Type>;
    using UBType = std::conditional_t<std::is_same_v<typename UB::Type, bool>, uint8_t, typename UB::Type>;
    DoTransMove<GMType, UBType, copyIn, tileW>(
        srcShape, gmShape4, gmStride, ubStride, gm.GetAddr(), gmOffset, ub.GetAddr());
}

#define OP_TILE_OP_TRANSPOSE_MOVEIN TTransMoveIn
template <unsigned axis0, unsigned axis1, typename DST, typename SRC, typename C>
__aicore__ inline void TTransMoveIn(DST dst, SRC src, C coordinate) {
    static_assert(DST::FORMAT == Hardware::UB && SRC::FORMAT == Hardware::GM);
    CallTransMove<axis0, axis1, true>(src, dst, coordinate);
}

#define OP_TILE_OP_TRANSPOSE_MOVEOUT TTransMoveOut
template <unsigned axis0, unsigned axis1, typename DST, typename SRC, typename C>
__aicore__ inline void TTransMoveOut(DST dst, SRC src, C coordinate) {
    static_assert(DST::FORMAT == Hardware::GM && SRC::FORMAT == Hardware::UB);
    CallTransMove<axis0, axis1, false>(dst, src, coordinate);
}

template <size_t current, size_t target = 3>
__aicore__ inline size_t IndexPutGetStride(size_t arr[]) {
    size_t result = 1;
    for (size_t i = current; i < target; ++i) {
        result *= arr[i];
    }
    return result;
}

template <size_t dstShapeSize, size_t indicesSize, typename IndicesPtr>
__aicore__ inline uint64_t GetIndexPutDstOffset(size_t dstShapes[], IndicesPtr indicesPtrs[], size_t i) {
    uint64_t dstOffset = 0;
    if constexpr (indicesSize >= 1) {
        dstOffset += indicesPtrs[0][i] * IndexPutGetStride<4 - dstShapeSize>(dstShapes);
    }
    if constexpr (indicesSize >= 2) {
        dstOffset += indicesPtrs[1][i] * IndexPutGetStride<5 - dstShapeSize>(dstShapes);
    }
    if constexpr (indicesSize >= 3) {
        dstOffset += indicesPtrs[2][i] * IndexPutGetStride<6 - dstShapeSize>(dstShapes);
    }
    if constexpr (indicesSize >= 4) {
        dstOffset += indicesPtrs[3][i] * IndexPutGetStride<7 - dstShapeSize>(dstShapes);
    }
    return dstOffset;
}

template <pto::AtomicType atomicType, size_t dstShapeSize, size_t valuesSize, typename VAL, typename DstType, typename IndicesPtr>
__aicore__ inline void DoIndexPut(size_t indicesLength, size_t dstShapes[], size_t valuesStride, size_t valuesShapes[],
    uint64_t valuesAddr, DstType *dstAddr, IndicesPtr indicesPtrs[]) {
    using ValuesDtype = std::conditional_t<std::is_same_v<typename VAL::Type, bool>, uint8_t, typename VAL::Type>;
    constexpr auto tileW = Std::tuple_element<valuesSize - 1, typename VAL::TileShape>::type::value;
    using ValuesTileDefine = pto::Tile<pto::TileType::Vec, ValuesDtype, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
    using DstData = pto::GlobalTensor<DstType, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    size_t copyShapes[] = {1, 1, valuesShapes[2]};
    size_t copyStrides[] = {0, dstShapes[2]};
    if constexpr (valuesSize == 1) {
        copyShapes[2] = 1;
    } else if constexpr (valuesSize == 3) {
        copyShapes[1] = valuesShapes[1];
    } else if constexpr (valuesSize == 4) {
        copyShapes[0] = valuesShapes[0];
        copyShapes[1] = valuesShapes[1];
        copyStrides[0] = dstShapes[1] * dstShapes[2];
    }
    auto dstDataShape = pto::Shape(1, 1, copyShapes[0], copyShapes[1], copyShapes[2]);
    auto dstDataStride = pto::Stride(0, 0, copyStrides[0], copyStrides[1], 0);
    auto valuesPtr = (__ubuf__ ValuesDtype *) valuesAddr;
    ValuesDtype valuesOrigin;
    if constexpr (valuesSize == 1) {
        valuesOrigin = *valuesPtr;
    }
    for (LoopVar i = 0; i < indicesLength; ++i) {
        uint64_t dstOffset = GetIndexPutDstOffset<dstShapeSize, dstShapeSize - valuesSize + 1>(dstShapes, indicesPtrs, i);
        if constexpr (valuesSize == 1) {
            valuesPtr[0] = valuesPtr[i];
        }
        ValuesTileDefine valuesData(copyShapes[1], copyShapes[2]);
        DstData dstData(dstAddr + dstOffset, dstDataShape, dstDataStride);
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
        pto::TASSIGN(valuesData, (uint64_t) valuesPtr);
        pto::TSTORE<ValuesTileDefine, DstData, atomicType>(dstData, valuesData);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        valuesPtr += valuesStride;
    }
    if constexpr (valuesSize == 1) {
        valuesPtr[0] = valuesOrigin;
    }
}

#define OP_TILE_OP_INDEXPUT TIndexPut
template <bool accumulate, size_t indicesSize, typename DST, typename C, typename VAL, typename IDX>
__aicore__ inline void TIndexPut(DST dst, C coordinate, VAL values, IDX indices0, IDX indices1, IDX indices2, IDX indices3) {
    constexpr auto atomicType = accumulate ? pto::AtomicType::AtomicAdd : pto::AtomicType::AtomicNone;
    constexpr auto dstShapeSize = Std::tuple_size<typename DST::Shape>::value;
    constexpr auto valuesSize = Std::tuple_size<typename VAL::Shape>::value;
    static_assert(dstShapeSize >= indicesSize && dstShapeSize <= 4 && dstShapeSize == valuesSize + indicesSize - 1);
    using IndicesDtype = typename IDX::Type;
    using IndicesPtr = __ubuf__ IndicesDtype *;
    IndicesPtr indicesPtrs[] = {
        (IndicesPtr) indices0.GetAddr(),
        (IndicesPtr) indices1.GetAddr(),
        (IndicesPtr) indices2.GetAddr(),
        (IndicesPtr) indices3.GetAddr()
    };
    const auto indicesLayout = indices0.GetLayout();
    const auto dstLayout = dst.GetLayout();
    const auto valuesLayout = values.GetLayout();
    auto indicesLength = indicesLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    size_t dstShapes[] = {
        static_cast<size_t>(dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>()),
        static_cast<size_t>(dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>()),
        static_cast<size_t>(dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>())
    };
    size_t valuesShapes[] = {
        static_cast<size_t>(valuesLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>()),
        static_cast<size_t>(valuesLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>()),
        static_cast<size_t>(valuesLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>())
    };
    size_t valuesStride = static_cast<size_t>(valuesLayout.template GetStrideDim<MAX_DIMS - valuesSize, MAX_DIMS>());
    if constexpr (valuesSize == 1) {
        valuesStride = 0;
    }
    size_t gmOffset = static_cast<size_t>(dstLayout.template GetGmOffset<C, MAX_DIMS>(coordinate));
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    DoIndexPut<atomicType, dstShapeSize, valuesSize, VAL>(indicesLength, dstShapes, valuesStride, valuesShapes,
        values.GetAddr(), dst.GetAddr() + gmOffset, indicesPtrs);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
}

#define OP_TILE_OP_MSCATTER MScatter
template <int axis, int scatterMode, typename T0, typename T1, typename T2, typename T3, typename C>
TILEOP void MScatter(T0 dst, T1 src1, T2 src2, T3 tmp, C coordinate) {
    static_assert(scatterMode < SCATTER_MODE_MAX, "Unsupport scatterMode");
    constexpr size_t expectSize = 5;

    const auto dstLayout = dst.GetLayout();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();

    auto gmElementOffset = dstLayout.template GetGmOffset<C, expectSize>(coordinate);
    __gm__ typename T0::Type* dstGMBase = reinterpret_cast<__gm__ typename T0::Type*>(dst.GetAddr());
    dstGMBase += gmElementOffset;  

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
    constexpr bool scalarFlag = true;
#else
    constexpr bool scalarFlag = ((sizeof(typename T1::Type) == 8) || (scatterMode > 0) ||
        (dstTypeSize == 2 && idxTypeSize == 4)) ? true : false;
#endif

    constexpr auto dstTileShapeH = TileOp::GetOutterAxisMergeResult<5, typename T0::TileShape>();
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

    // UB 地址不变（src/idx/tmp 仍在 UB）
    auto idxAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src1.GetAddr()));
    auto srcAddr = (__ubuf__ typename T2::Type*)((uint64_t)(src2.GetAddr()));
    auto tmpAddr = (__ubuf__ typename T3::Type*)((uint64_t)(tmp.GetAddr()));

    for (LoopVar i = 0; i < idxShape0; ++i) {
        for (LoopVar j = 0; j < idxShape1; ++j) {
            for (LoopVar k = 0; k < idxShape2; ++k) {
                for (LoopVar l = 0; l < idxShape3; ++l) {
                    if constexpr (!scalarFlag) {
                        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                    }
                    for (LoopVar m = 0; m < idxShape4; ++m) {
                        typename T1::Type index =
                            *(idxAddr + i * idxStride0 + j * idxStride1 + k * idxStride2 + l * idxStride3 + m);
                        typename T1::Type src2Offset = 
                            i * srcStride0 + j * srcStride1 + k * srcStride2 + l * srcStride3 + m;

                        // === 计算 tile 内局部偏移（local offset）===
                        typename T1::Type localOffset;
                        if constexpr (axis == 0) {
                            localOffset = index * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == 1) {
                            localOffset = i * dstStride0 + index * dstStride1 + k * dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == 2) {
                            localOffset = i * dstStride0 + j * dstStride1 + index * dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == 3) {
                            localOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + index * dstStride3 + m;
                        } else { // axis == 4
                            localOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + index;
                        }

                        /* idx类型为int64或scatter操作为add或multiply，退化为标量实现 */
                        if constexpr (scalarFlag) {
                            if constexpr (scatterMode == 0) { // 直接赋值
                                dstGMBase[localOffset] = srcAddr[src2Offset];
                            } else if constexpr (scatterMode == 1) {
                                dstGMBase[localOffset] = srcAddr[src2Offset] + dstGMBase[localOffset];
                            } else {
                                dstGMBase[localOffset] = srcAddr[src2Offset] * dstGMBase[localOffset];
                            }
                        } else {
                            // TSCATTER 是 UB→UB，仍使用 localOffset（相对于当前 tile 起点）
                            *(tmpAddr + m) = localOffset;
                        }
                    }
                    if constexpr (!scalarFlag) {
                        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                        auto srcOffset = i * srcStride0 + j * srcStride1 + k * srcStride2 + l * srcStride3;
                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr())); // 注意：TSCATTER 用原始 dst（UB 视图）
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
#endif