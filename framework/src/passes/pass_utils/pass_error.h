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
 * \file pass_error.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk {
enum class PassError : uint32_t {
    PUBLIC_ERROR_TENSOR    = 40000U, // 前端传入公共Tensor错误
    PUBLIC_ERROR_OPERATION = 40200U, // 前端传入公共Operation错误
    PUBLIC_ERROR_FUNCTION  = 40400U, // 前端传入公共Function错误
    TENSOR_GRAPH   = 41000U, // TensorGraph阶段
    TILE_GRAPH     = 42000U, // TileGraph阶段
    BLOCK_GRAPH    = 43000U, // BlockGraph阶段
    UNKNOWN        = 49000U  // 未知错误
};

enum class PublicTensorErr : uint32_t {
    TENSOR_NULL_POINTER         = 40001U,
    TENSOR_INVALID_MEMORY_TYPE  = 40002U,
    TENSOR_SUBGRAPH_BOUNDARY    = 40003U,
    TENSOR_SHAPE_MISMATCH       = 40004U,
    TENSOR_UNSUPPORTED_DATATYPE = 40005U,
    TENSOR_MEMORY_ALLOCATION    = 40006U,
    TENSOR_DYNAMIC_ATTR         = 40007U,
    TENSOR_MEMORY_CORRUPTION    = 40008U,
    UNKNOWN                     = 40099U
};

enum class PublicOperationErr : uint32_t {
    OP_INVALID_OPERAND_COUNT = 40201U,
    OP_NULL_POINTER          = 40202U,
    OP_INVALID_OPCODE        = 40203U,
    OP_PRODUCER_CONSUMER     = 40204U,
    OP_SPECIAL_CONSTRAINT    = 40205U,
    OP_NESTING_DEPTH         = 40206U,
    OP_SEQUENCE_ERROR        = 40207U,
    UNKNOWN                  = 40299U
};

enum class PublicFunctionErr : uint32_t {
    FUNCTION_GRAPH_STRUCTURE       = 40401U,
    FUNCTION_BOUNDARY_COMPLETENESS = 40402U,
    FUNCTION_GRAPH_CONNECTION      = 40403U,
    FUNCTION_EXPAND_FEATURE        = 40404U,
    FUNCTION_MEMORY_REACHABILITY   = 40405U,
    FUNCTION_UNIQUENESS            = 40406U,
    FUNCTION_SPECIAL_STRUCTURE     = 40407U,
    UNKNOWN                        = 40499U
};

enum class TensorGraphErr : uint32_t {
    UNKNOWN           = 41099U
};

enum class TileGraphErr : uint32_t {
    UNKNOWN           = 42099U
};

enum class BlockGraphErr : uint32_t {
    BLOCK_MEMORY_BUBBLE   = 43001U,
    UB_OVER_LIMIT         = 43002U,
    L1_OVER_LIMIT         = 43003U,
    UNKNOWN               = 43099U
};

}  // namespace npu::tile_fwk

