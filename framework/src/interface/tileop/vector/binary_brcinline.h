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

enum class BrcMode : uint8_t {
    NONE,
    TAIL_LEFT,      // [m, 1] [m, n]
    TAIL_RIGHT,     // [m, n] [m, 1]
    PENU_LEFT,      // [1, n] [m, n]
    PENU_RIGHT,     // [m, n] [1, n]
    SCALAR_LEFT,    // [1, 1] [m ,n]
    SCALAR_RIGHT,   // [m, n] [1, 1]
    MIX_LEFT_TAIL,  // [m, 1] [1, n]
    MIX_RIGHT_TAIL  // [1, n] [m, 1]
};

template <TileOp::BroadcastOperand tailBrcSide, TileOp::PenuBroadcastOperand penuBrcSide>
TILEOP constexpr BrcMode GetBrcMode() {
    if constexpr (tailBrcSide == TileOp::BroadcastOperand::LEFT_OPERAND &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::NONE) {
        return BrcMode::TAIL_LEFT;
    } else if constexpr (tailBrcSide == TileOp::BroadcastOperand::RIGHT_OPERAND &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::NONE) {
        return BrcMode::TAIL_RIGHT;
    } else if constexpr (tailBrcSide == TileOp::BroadcastOperand::NONE &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::LEFT_OPERAND) {
        return BrcMode::PENU_LEFT;
    } else if constexpr (tailBrcSide == TileOp::BroadcastOperand::NONE &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::RIGHT_OPERAND) {
        return BrcMode::PENU_RIGHT;
    } else if constexpr (tailBrcSide == TileOp::BroadcastOperand::LEFT_OPERAND &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::LEFT_OPERAND) {
        return BrcMode::SCALAR_LEFT;
    } else if constexpr (tailBrcSide == TileOp::BroadcastOperand::RIGHT_OPERAND &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::RIGHT_OPERAND) {
        return BrcMode::SCALAR_RIGHT;
    } else if constexpr (tailBrcSide == TileOp::BroadcastOperand::LEFT_OPERAND &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::RIGHT_OPERAND) {
        return BrcMode::MIX_LEFT_TAIL;
    } else if constexpr (tailBrcSide == TileOp::BroadcastOperand::RIGHT_OPERAND &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::LEFT_OPERAND) {
        return BrcMode::MIX_RIGHT_TAIL;
    } else {
        return BrcMode::NONE;
    }
}

#define EXTRACT_LAST_USE_3DIM(LastUse)                                           \
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;       \
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;       \
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;

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

#define BINARY_SCALAR_EXPAND_DISPATCH(dst, tensor, scalar)                              \
    if constexpr (op == BinaryOp::ADD) {                                                \
        PTO_WITH_LAST_USE(pto::TADDS(dst, tensor, scalar), n1, n2, n3); return;         \
    } else if constexpr (op == BinaryOp::SUB) {                                         \
        PTO_WITH_LAST_USE(pto::TSUBS(dst, tensor, scalar), n1, n2, n3); return;         \
    } else if constexpr (op == BinaryOp::MUL) {                                         \
        PTO_WITH_LAST_USE(pto::TMULS(dst, tensor, scalar), n1, n2, n3); return;         \
    } else if constexpr (op == BinaryOp::DIV) {                                         \
        PTO_WITH_LAST_USE(pto::TDIVS(dst, tensor, scalar), n1, n2, n3); return;         \
    } else if constexpr (op == BinaryOp::MAX) {                                         \
        PTO_WITH_LAST_USE(pto::TMAXS(dst, tensor, scalar), n1, n2, n3); return;         \
    } else if constexpr (op == BinaryOp::MIN) {                                         \
        PTO_WITH_LAST_USE(pto::TMINS(dst, tensor, scalar), n1, n2, n3); return;         \
    }

template <BinaryOp op, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryLeftScalarComputeImpl(T0 dst, T1 src1, Scalar src0) {
    EXTRACT_LAST_USE_3DIM(LastUse)

    if constexpr (op == BinaryOp::SUB) {
        PTO_WITH_LAST_USE(pto::TNEG(dst, src1), n1, n2, n3);
        #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
        #endif
        PTO_WITH_LAST_USE(pto::TADDS(dst, dst, src0), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::DIV) {
        PTO_WITH_LAST_USE(pto::TDIVS(dst, src1, src0), n1, n2, n3);   
        // 存在问题，scalar/tensor不支持，如果要用TRECIP 不允许 dst 和 src 指向同一块内存，这里地址一定会复用的
        return;
    }

    BINARY_SCALAR_EXPAND_DISPATCH(dst, src1, src0)
}

template <BinaryOp op, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryRightScalarComputeImpl(T0 dst, T1 src0, Scalar src1) {
    EXTRACT_LAST_USE_3DIM(LastUse)
    BINARY_SCALAR_EXPAND_DISPATCH(dst, src0, src1)
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
#endif
