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
 * \file binary_brcinline.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_BINARY_BRCINLINE__H
#define TILEOP_TILE_OPERATOR_BINARY_BRCINLINE__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include "binary.h"

#define BINARY_EXPAND_DISPATCH(PREFIX)                                                  \
    if constexpr (op == BinaryOp::ADD) {                                                \
        PTO_WITH_LAST_USE(pto::T##PREFIX##ADD(dst, src0, src1), n1, n2, n3); return;    \
    } else if constexpr (op == BinaryOp::SUB) {                                         \
        PTO_WITH_LAST_USE(pto::T##PREFIX##SUB(dst, src0, src1), n1, n2, n3); return;    \
    } else if constexpr (op == BinaryOp::MUL) {                                         \
        PTO_WITH_LAST_USE(pto::T##PREFIX##MUL(dst, src0, src1), n1, n2, n3); return;    \
    } else if constexpr (op == BinaryOp::DIV) {                                         \
        PTO_WITH_LAST_USE(pto::T##PREFIX##DIV(dst, src0, src1), n1, n2, n3); return;    \
    } else if constexpr (op == BinaryOp::MAX) {                                         \
        PTO_WITH_LAST_USE(pto::T##PREFIX##MAX(dst, src0, src1), n1, n2, n3); return;    \
    } else if constexpr (op == BinaryOp::MIN) {                                         \
        PTO_WITH_LAST_USE(pto::T##PREFIX##MIN(dst, src0, src1), n1, n2, n3); return;    \
    }

template <BinaryOp op, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryRowExpandComputeImpl(T0 dst, T1 src0, T2 src1) {
    EXTRACT_LAST_USE_3DIM(LastUse)
    BINARY_EXPAND_DISPATCH(ROWEXPAND)
}

template <BinaryOp op, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryColExpandComputeImpl(T0 dst, T1 src0, T2 src1) {
    EXTRACT_LAST_USE_3DIM(LastUse)
    BINARY_EXPAND_DISPATCH(COLEXPAND)
}

template <BinaryOp op, TileOp::BroadcastOperand tailBrcSide, BinaryLayoutInfo info, 
          TensorTileInfo src0TileInfo, TensorTileInfo src1TileInfo,
          typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryMixBrcCompute(T0 dst, T1 src0, T2 src1) {
    using Src0PtoTile = typename std::conditional<(Src0TileInfo::tileW == 1 && tailBrcSide == TileOp::BroadcastOperand::LEFT_OPERAND), 
        PtoTile<T1, pto::BLayout::ColMajor>, PtoTile<T1>>::type;
    using Src1PtoTile = typename std::conditional<(Src1TileInfo::tileW == 1 && tailBrcSide == TileOp::BroadcastOperand::RIGHT_OPERAND), 
        PtoTile<T2, pto::BLayout::ColMajor>, PtoTile<T2>>::type;
    auto dstTile = PtoTile<T0>(1, info.shape4).Data();
    auto src0Tile = Src0PtoTile(1, tailBrcSide == TileOp::BroadcastOperand::LEFT_OPERAND ? 1 : info.shape4).Data();
    auto src1Tile = Src1PtoTile(1, tailBrcSide == TileOp::BroadcastOperand::RIGHT_OPERAND ? 1 : info.shape4).Data();
    for (LoopVar n0Index = 0; n0Index < info.shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < info.shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < info.shape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < info.shape3; ++n3Index) {
                    auto dsttileOffsets = n0Index * info.dstStride0 + n1Index * info.dstStride1
                                            + n2Index * info.dstStride2 + n3Index * info.dstStride3;
                    auto src0tileOffsets = (Src0TileInfo::tile0 == 1 ? 0 : n0Index) * info.src0Stride0
                                            + (Src0TileInfo::tile1 == 1 ? 0 : n1Index) * info.src0Stride1
                                            + (Src0TileInfo::tile2 == 1 ? 0 : n2Index) * info.src0Stride2
                                            + (Src0TileInfo::tileH == 1 ? 0 : n3Index) * info.src0Stride3;
                    auto src1tileOffsets = (Src1TileInfo::tile0 == 1 ? 0 : n0Index) * info.src1Stride0
                                            + (Src1TileInfo::tile1 == 1 ? 0 : n1Index) * info.src1Stride1
                                            + (Src1TileInfo::tile2 == 1 ? 0 : n2Index) * info.src1Stride2
                                            + (Src1TileInfo::tileH == 1 ? 0 : n3Index) * info.src1Stride3;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dsttileOffsets * sizeof(typename T0::Type)));
                    pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + src0tileOffsets * sizeof(typename T1::Type)));
                    pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1tileOffsets * sizeof(typename T2::Type)));
                    BinaryRowExpandComputeImpl<op, LastUse>(dstTile, src0Tile, src1Tile);
                }
            }
        }
    }
}
#endif
