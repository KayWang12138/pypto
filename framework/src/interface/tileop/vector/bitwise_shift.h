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
 * \file bitwise_shift.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_BITWISE_SHIFT__H
#define TILEOP_TILE_OPERATOR_BITWISE_SHIFT__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include "tileop_common.h"

template <BitwiseShiftOp op, typename T0, typename T1, typename T2>
TILEOP void BitwiseShiftComputeImpl(T0 dst, T1 src0, T2 src1) {
    if constexpr (op == BitwiseShiftOp::BITWISERIGHTSHIFT) {
        pto::TSHR(dst, src0, src1);
        return;
    }

    if constexpr (op == BitwiseShiftOp::BITWISELEFTSHIFT) {
        pto::TSHL(dst, src0, src1);
        return;
    }
}

template <BitwiseShiftScalarOp op, size_t MAX_SHIFT_NUM, typename T0, typename T1, typename Scalar>
TILEOP void BitwiseShiftScalarComputeImpl(T0 dst, T1 src0, Scalar src1) {
    if (src1 < 0 || src1 > MAX_SHIFT_NUM) {
        pto::TEXPANDS(dst, 0);
        return;
    }
    
    if constexpr (op == BitwiseShiftScalarOp::BITWISERIGHTSHIFT) {
        pto::TSHRS(dst, src0, src1);
        return;
    }

    if constexpr (op == BitwiseShiftScalarOp::BITWISELEFTSHIFT) {
        pto::TSHLS(dst, src0, src1);
        return;
    }
}

template <size_t MAX_SHIFT_NUM, typename T, typename U0, typename U1, typename V0, typename V1>
TILEOP void GetValidShiftTile(T &dst, U0 &src1, U1 &src1_int16, V0 &tmp, V1 &tmp_int16) {
    if constexpr (std::is_same_v<int16_t, typename U0::DType>) {
        pto::TOR(tmp, tmp, src1);
    } else {
        pto::TOR(tmp_int16, tmp_int16, src1_int16);
    }
    #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
    #endif
    pto::TSHRS(dst, tmp, MAX_SHIFT_NUM);
    #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
    #endif
    pto::TEXPANDS(tmp, 1);
    #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
    #endif
    pto::TADD(tmp, tmp, dst);
    #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
    #endif
    pto::TMUL(src1, tmp, src1);
}

template <BitwiseShiftOp op, size_t MAX_SHIFT_NUM, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
TILEOP void BitwiseShiftImpl(T0 &dst, T1 &src0, T2 &src1, T3 &src1_int16, T4 &tmp, T5 &tmp_int16) {
    pto::TEXPANDS(tmp, MAX_SHIFT_NUM);
    #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
    #endif
    pto::TSUB(tmp, tmp, src1);
    #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
    #endif
    GetValidShiftTile<MAX_SHIFT_NUM>(dst, src1, src1_int16, tmp, tmp_int16);
    #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
    #endif
    pto::TMUL(src0, tmp, src0);
    #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
    #endif
    BitwiseShiftComputeImpl<op>(dst, src0, src1);
}

template <BitwiseShiftOp op, typename T0, typename T1, typename T2, typename T3>
TILEOP void BitwiseShiftCompute(T0 dst, T1 src0, T2 src1, T3 tmp) {
    constexpr auto MAX_SHIFT_NUM = sizeof(typename T0::Type) * TileOp::BLOCK_NELEM_B32 - 1;
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    
    using T2_int16 = TileTensor<int16_t, typename T2::LayoutType, Hardware::UB>;
    using T3_int16 = TileTensor<int16_t, typename T3::LayoutType, Hardware::UB>;

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    auto src1Tile = PtoTile<T2>(src1);
    auto src1Tile_int16 = PtoTile<T2_int16>(src1);
    auto tmpTile = PtoTile<T3>(tmp);
    auto tmpTile_int16 = PtoTile<T3_int16>(tmp);

    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                src1Tile.Assign(src1, tileOffsets);
                src1Tile_int16.Assign(src1, tileOffsets);
                tmpTile.Assign(tmp, tileOffsets);
                tmpTile_int16.Assign(tmp, tileOffsets);
                BitwiseShiftImpl<op, MAX_SHIFT_NUM>(dstTile.Data(), src0Tile.Data(), src1Tile.Data(),
                                                    src1Tile_int16.Data(), tmpTile.Data(), tmpTile_int16.Data());
            }
        }
    }
}

template <BitwiseShiftScalarOp op, typename T0, typename T1, typename Scalar>
TILEOP void BitwiseShiftScalarCompute(T0 dst, T1 src0, Scalar src1) {
    constexpr uint64_t BLOCK_NELEM_B32 = 8;
    constexpr auto MAX_SHIFT_NUM = sizeof(typename T0::Type) * TileOp::BLOCK_NELEM_B32 - 1;
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);

    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                BitwiseShiftScalarComputeImpl<op, MAX_SHIFT_NUM>(dstTile.Data(), src0Tile.Data(), src1);
            }
        }
    }
}

#define OP_TILE_OP_BITWISERIGHTSHIFT TBitrshift
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TBitrshift(T0 dst, T1 src0, T2 src1, T3 tmp) {
    BitwiseShiftCompute<BitwiseShiftOp::BITWISERIGHTSHIFT>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_BITWISELEFTSHIFT TBitlshift
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TBitlshift(T0 dst, T1 src0, T2 src1, T3 tmp) {
    BitwiseShiftCompute<BitwiseShiftOp::BITWISELEFTSHIFT>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_BITWISERIGHTSHIFTS TBitrshiftS
template <typename Scalar, typename T0, typename T1>
TILEOP void TBitrshiftS(T0 dst, T1 src0, Scalar src1) {
    BitwiseShiftScalarCompute<BitwiseShiftScalarOp::BITWISERIGHTSHIFT>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISELEFTSHIFTS TBitlshiftS
template <typename Scalar, typename T0, typename T1>
TILEOP void TBitlshiftS(T0 dst, T1 src0, Scalar src1) {
    BitwiseShiftScalarCompute<BitwiseShiftScalarOp::BITWISELEFTSHIFT>(dst, src0, src1);
}
#endif