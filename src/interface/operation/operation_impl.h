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
 * \file operation_impl.h
 * \brief
 */

#pragma once
#include <tuple>
#include <vector>
#include <string>
#include <unordered_set>
#include "tilefwk/tensor.h"
#include "interface/inner/config.h"
#include "opcode.h"
#include "tilefwk/tile_shape.h"

namespace npu::tile_fwk {

class Function;
class Operation;
using LogicalTensorPtr = std::shared_ptr<LogicalTensor>;
enum class CastOpType {
    CAST,
    // FLOOR,
    // ROUND,
};
enum class BinaryOpType {
    ADD,
    SUB,
    MUL,
    DIV,
    MAX,
    MIN,
    ADD_BRC,
    SUB_BRC,
    MUL_BRC,
    DIV_BRC,
    MAX_BRC,
    S_ADD,
    S_SUB,
    S_MUL,
    S_DIV,
    S_MAX,
    MAXIMUM,
};
enum class UnaryOpType {
    EXP,
    NEG,
    RSQRT,
    SQRT,
    RECIPROCAL,
    DUPLICATE,
    ABS,
    LN,
};
enum class ReduceType {
    NORMAL,
    EXPAND,
    SINGLE,
};

constexpr int32_t NUM_VALUE_0 = 0;
constexpr int32_t NUM_VALUE_1 = 1;
constexpr int32_t NUM_VALUE_2 = 2;
constexpr int32_t NUM_VALUE_3 = 3;
constexpr int32_t NUM_VALUE_4 = 4;
constexpr int32_t NUM_VALUE_5 = 5;
constexpr int32_t NUM_VALUE_8 = 8;
constexpr int32_t NUM_VALUE_10 = 10;
constexpr int32_t NUM_VALUE_16 = 16;
constexpr int32_t NUM_VALUE_31 = 31;
constexpr int32_t NUM_VALUE_32 = 32;
constexpr int32_t NUM_VALUE_64 = 64;
const int INT_MAX_VALUE = 2147483647;
const int INT_MIN_VALUE = -2147483648;

struct ExpandInfo {
    const std::shared_ptr<LogicalTensor> &srcTensor;
    const std::shared_ptr<LogicalTensor> &result;
    std::vector<int64_t> &viewShape;
    std::vector<int64_t> &offset;
    const int expandDim;
    ExpandInfo(const std::shared_ptr<LogicalTensor> &srcTensor0, const std::shared_ptr<LogicalTensor> &result0,
        std::vector<int64_t> &viewShape0, std::vector<int64_t> &offset0, const int expandDim0)
        : srcTensor(srcTensor0), result(result0), viewShape(viewShape0), offset(offset0), expandDim(expandDim0) {}
};

void ExpandOperationInto(Function &function, const TileShape &tileShape, Opcode opCode,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);


namespace Matrix {
const size_t M_INDEX = 0;
const size_t K_INDEX = 1;
const size_t N_INDEX = 2;
const int32_t MATRIX_MAXSIZE = 3;

const std::string OP_ATTR_PREFIX = "op_attr_";
const std::string ACC_A_MUL_B = OP_ATTR_PREFIX + "atomic_add";
const std::string MATMUL_NZ_ATTR = OP_ATTR_PREFIX + "matmul_nz_attr";
const std::string A_MUL_B_ACT_M = OP_ATTR_PREFIX + "act_m";
const std::string A_MUL_B_ACT_K = OP_ATTR_PREFIX + "act_k";
const std::string A_MUL_B_ACT_N = OP_ATTR_PREFIX + "act_n";

struct L1DataLoadParam {
    const LogicalTensorPtr &cTilePtr;
    const int64_t mL1Idx;
    const int64_t nL1Idx;
    const int64_t stepK;
    const int64_t mL1Size;
    const int64_t nL1Size;
    const int64_t orgK;
};

struct CollectSubAMulBPara {
    const TileShape &tileShape;
    const std::array<int64_t, 3> &posK;
    const LogicalTensorPtr &aTensorPtr;
    const LogicalTensorPtr &bTensorPtr;
    const LogicalTensorPtr &cTensorPtr;
};

struct DoAMulBParam {
    const TileShape &tileShape;
    const LogicalTensorPtr &cTensorPtr;
};

template <bool isTransA = false, bool isTransB = false>
void TiledInnerAMulB(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec,
    const LogicalTensorPtr &result, const std::vector<int64_t> &matmulSize);

} // namespace Matrix
} // namespace npu::tile_fwk
