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
    PUBLIC_ERROR_OPERATION = 41000U, // 前端传入公共Operation错误
    PUBLIC_ERROR_FUNCTION  = 42000U, // 前端传入公共Function错误
    PUBLIC_ERROR_GRAPH     = 43000U, // 前端传入公共Graph错误
    PUBLIC_ERROR_CONFIG    = 44000U, // 前端传入公共Config错误
    PUBLIC_ERROR_MANAGER   = 45000U, // 前端传入公共Manager错误
    UNKNOWN                = 49999U  // 未知错误
};

enum class PublicTensorErr : uint32_t {
    TENSOR_NULL_POINTER         = 40001U,
    TENSOR_INVALID_MEMORY_TYPE  = 40002U,
    TENSOR_SUBGRAPH_BOUNDARY    = 40003U,
    TENSOR_SHAPE_MISMATCH       = 40004U,
    TENSOR_UNSUPPORTED_DATATYPE = 40005U,
    TENSOR_MEMORY_ALLOCATION    = 40006U,
    TENSOR_DYNAMIC_ATTR         = 40007U,
    TENSOR_MEMORY_CORRUPTION    = 40008U
};

enum class PublicOperationErr : uint32_t {
    OP_INVALID_OPERAND_COUNT = 41001U,
    OP_NULL_POINTER          = 41002U,
    OP_INVALID_OPCODE        = 41003U,
    OP_PRODUCER_CONSUMER     = 41004U,
    OP_SPECIAL_CONSTRAINT    = 41005U,
    OP_NESTING_DEPTH         = 41006U,
    OP_SEQUENCE_ERROR        = 41007U
};

enum class PublicFunctionErr : uint32_t {
    FUNCTION_GRAPH_STRUCTURE       = 42001U,
    FUNCTION_BOUNDARY_COMPLETENESS = 42002U,
    FUNCTION_GRAPH_CONNECTION      = 42003U,
    FUNCTION_EXPAND_FEATURE        = 42004U,
    FUNCTION_MEMORY_REACHABILITY   = 42005U,
    FUNCTION_UNIQUENESS            = 42006U,
    FUNCTION_SPECIAL_STRUCTURE     = 42007U
};

enum class PublicGraphErr : uint32_t {
    GRAPH_LOOP_DETECTION       = 43001U,
    GRAPH_TOPOLOGY_STRUCTURE   = 43002U,
    GRAPH_SUBGRAPH_EMPTY       = 43003U,
    GRAPH_SUBGRAPH_ID_INVALID  = 43004U,
    GRAPH_EDGE_CONSISTENCY     = 43005U,
    GRAPH_COLOR_CONSISTENCY    = 43006U,
    GRAPH_READY_STATE          = 43007U,
    GRAPH_AIV_AIC_MIX          = 43009U
};

enum class PublicConfigErr : uint32_t {
    CONFIG_MEMORY_TYPE_REACHABLE  = 44001U,
    CONFIG_SUBGRAPH_BOUNDARY      = 44002U,
    CONFIG_TENSOR_MEMORY_TYPE     = 44003U
};

enum class PublicManagerErr : uint32_t {
};

}  // namespace npu::tile_fwk

