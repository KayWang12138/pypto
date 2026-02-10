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
 * \file conv_pto.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_PTO__H
#define TILEOP_TILE_OPERATOR_CONV_PTO__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

namespace TileOp {

// Copy data from DDR to L1
template <CopyInMode mode, typename Coord, typename T, typename U>
TILEOP void TLoadConv(T &dst, U &src, const Coord &coord, const int64_t &curH, const int64_t &curW) {
    return;
}

} // namespace TileOp
#endif // TILEOP_TILE_OPERATOR_CONV_PTO__H