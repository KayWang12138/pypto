/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cube_operation.h
 * \brief
 */

#pragma once

#include "tilefwk/tensor.h"
#include "common/tile_shape.h"
namespace npu {
namespace tile_fwk {
namespace Matrix {
const int64_t M_INDEX = 0;
const int64_t K_INDEX = 1;
const int64_t N_INDEX = 2;
const int64_t MATRIX_MAXSIZE = 3;
const std::string OP_ATTR_PREFIX = "op_attr_";
const std::string ACC_A_MUL_B = OP_ATTR_PREFIX + "atomic_add";
const std::string A_MUL_B_NZ_ATTR = OP_ATTR_PREFIX + "matmul_nz_attr";
const std::string A_MUL_B_ACT_M = OP_ATTR_PREFIX + "act_m";
const std::string A_MUL_B_ACT_K = OP_ATTR_PREFIX + "act_k";
const std::string A_MUL_B_ACT_N = OP_ATTR_PREFIX + "act_n";

template <bool isTransA = false, bool isTransB = false>
void TiledInnerAMulB(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &operandVec, const LogicalTensorPtr &result,
    const std::vector<int32_t> &matmulSize);

} // namespace Matrix
} // namespace tile_fwk
} // namespace npu
