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
 * \file pass_type.h
 * \brief
 */

#ifndef PASSES_PASS_TYPE_H_
#define PASSES_PASS_TYPE_H_
#include <cstdint>
namespace npu::tile_fwk {
enum class PassType : int32_t {
    TYPE_INVALID = -1,
    TYPE_TENSOR_GRAPH = 0,
    TYPE_TILE_GRAPH = 1,
    TYPE_BLOCK_GRAPH = 2,
    TYPE_BOTTOM
};
}
#endif  // PASSES_PASS_TYPE_H_