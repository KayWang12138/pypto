/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file brcinline.h
 * \brief
 */

#ifndef TILEOP_UTILS_BRCINLINE_H
#define TILEOP_UTILS_BRCINLINE_H

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
constexpr BrcMode GetBrcMode() {
    if constexpr (tailBrcSide == TileOp::BroadcastOperand::LEFT_OPERAND &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::NONE) {
        return BrcMode::TAIL_LEFT;
    }
    if constexpr (tailBrcSide == TileOp::BroadcastOperand::RIGHT_OPERAND &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::NONE) {
        return BrcMode::TAIL_RIGHT;
    }
    if constexpr (tailBrcSide == TileOp::BroadcastOperand::NONE &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::LEFT_OPERAND) {
        return BrcMode::PENU_LEFT;
    }
    if constexpr (tailBrcSide == TileOp::BroadcastOperand::NONE &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::RIGHT_OPERAND) {
        return BrcMode::PENU_RIGHT;
    }
    if constexpr (tailBrcSide == TileOp::BroadcastOperand::LEFT_OPERAND &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::LEFT_OPERAND) {
        return BrcMode::SCALAR_LEFT;
    }
    if constexpr (tailBrcSide == TileOp::BroadcastOperand::RIGHT_OPERAND &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::RIGHT_OPERAND) {
        return BrcMode::SCALAR_RIGHT;
    }
    if constexpr (tailBrcSide == TileOp::BroadcastOperand::LEFT_OPERAND &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::RIGHT_OPERAND) {
        return BrcMode::MIX_LEFT_TAIL;
    }
    if constexpr (tailBrcSide == TileOp::BroadcastOperand::RIGHT_OPERAND &&
                  penuBrcSide == TileOp::PenuBroadcastOperand::LEFT_OPERAND) {
        return BrcMode::MIX_RIGHT_TAIL;
    }
    return BrcMode::NONE;
}

#endif // TILEOP_UTILS_BRCINLINE_H