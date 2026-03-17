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
    TENSOR_GRAPH   = 40000U, // TensorGraph阶段
    TILE_GRAPH     = 41000U, // TileGraph阶段
    BLOCK_GRAPH    = 42000U, // BlockGraph阶段
    UNKNOWN        = 49000U
};

enum class TensorGraphErr : uint32_t {
    ASSEMBLE_OVERLAP_REGION_UNDEFINED_BEHAVIOR = 40001U,
    SCATTER_UPDATE_INPUT_REUSED_INVALID        = 40002U,
    UNKNOWN                                    = 40099U
};

enum class TileGraphErr : uint32_t {
    UNKNOWN           = 41099U
};

enum class BlockGraphErr : uint32_t {
    BLOCK_MEMORY_BUBBLE   = 42001U,
    UB_OVER_LIMIT         = 42002U,
    L1_OVER_LIMIT         = 42003U,
    UNKNOWN               = 42099U
};

}  // namespace npu::tile_fwk

