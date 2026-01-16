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
 * \file opcode.h
 * \brief
 */

#pragma once

#include "utils_defop.h"
#include "value.h"

namespace pto {

enum class Opcode : int64_t {
    OP_INVALID,

#define DEFOP DEFOP_OPCODE

#include "operation.def"
    OP_END_COMMON,

#include "tile_graph.def"
    OP_END_TILE_GRAPH,

#undef DEFOP

    OP_COMMON_BEGIN = OP_INVALID + 1,
    OP_COMMON_END = OP_END_COMMON,
    OP_TILE_GRAPH_BEGIN = OP_END_COMMON + 1,
    OP_TILE_GRAPH_END = OP_END_TILE_GRAPH - 1,
};

std::string GetOpcodeName(Opcode opcode);

enum class CoreType { AIV = 0, AIC = 1, MIX = 2, AICPU = 3, HUB = 4, GMATOMIC = 5, INVALID = 20 };

enum class OpCalcType {
    ELMWISE,
    CAST,
    BROADCAST,
    OTHER,
    REDUCE,
    MATMUL,
    CONV,
    MOVE_IN,
    MOVE_OUT,
    MOVE_LOCAL,
    SYNC,        // 同步
    DISTRIBUTED, // 通信
    SYS,         // 框架
    CALC_TYPE_BOTTOM
};

// hardware pipeline
enum PipeType {
    PIPE_S = 0, // Scalar Pipe
    PIPE_V,     // Vector Pipe, including{VectorOP write UB,  L0C->UB write}
    PIPE_M,     // Matrix Pipe, including{}
    PIPE_MTE1,  // L1->L0{A,B}
    PIPE_MTE2,  // OUT ->{L1, L0{A,B}, UB}
    PIPE_MTE3,  // UB ->{OUT,L1}
    PIPE_ALL,
    PIPE_MTE4 = 7, // MOV_UB_TO_OUT
    PIPE_MTE5 = 8, // MOV_OUT_TO_UB
    PIPE_V2 = 9,   // Lower priority vector pipe,
    PIPE_FIX = 10, // {L0C} ->{L1,UB,L1UB}
};

struct OpClassInfo {
    std::vector<MemSpaceKind> inputsMemType_;
    std::vector<MemSpaceKind> outputsMemType_;
    PipeType pipeIdStart_;
    PipeType pipeIdEnd_;
    CoreType coreType_;
    OpCalcType calcType_;
};

class Operation;

OpClassInfo GetOpClassInfo(std::shared_ptr<Operation> op);

} // namespace pto
