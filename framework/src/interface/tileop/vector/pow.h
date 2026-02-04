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
 * \file pow.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_POW__H
#define TILEOP_TILE_OPERATOR_POW__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_POW TPow
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2,  typename T3>
TILEOP void TPow(T0 dst, T1 src0, T2 src1, T3 tmp) {

}

#endif
