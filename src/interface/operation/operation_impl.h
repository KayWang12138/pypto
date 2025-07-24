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
#include <vector>
#include <string>
#include <unordered_set>
#include "tilefwk/tensor.h"
#include "interface/configs/config_storage.h"
#include "opcode.h"
#include "common/tile_shape.h"

namespace npu::tile_fwk {
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
    SQRT,
    RECIPROCAL,
    DUPLICATE,
    ABS,
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
constexpr int32_t NUM_VALUE_16 = 16;
constexpr int32_t NUM_VALUE_31 = 31;
constexpr int32_t NUM_VALUE_32 = 32;
constexpr int32_t NUM_VALUE_64 = 64;

struct ExpandInfo {
    const std::shared_ptr<LogicalTensor> &srcTensor;
    const std::shared_ptr<LogicalTensor> &result;
    std::vector<int> &viewShape;
    std::vector<int> &offset;
    const int expandDim;
    ExpandInfo(const std::shared_ptr<LogicalTensor> &srcTensor0, const std::shared_ptr<LogicalTensor> &result0,
        std::vector<int> &viewShape0, std::vector<int> &offset0, const int expandDim0)
        : srcTensor(srcTensor0), result(result0), viewShape(viewShape0), offset(offset0), expandDim(expandDim0) {}
};

void ExpandOperationInto(Function &function, const TileShape &tileShape, Opcode opCode,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
} // namespace npu::tile_fwk

