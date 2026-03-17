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
 * \file calc_error.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk::calc_error {

// Calculator 层错误码从 0xB5000U 开始，按语义分类。

enum class CalculatorErrorScene : uint32_t {
    INVALID_TENSOR_SIZE   = 0xB5000U, // 张量元素个数/尺寸不匹配（如 Range 生成 numel 与 out.shape 不一致）
    INVALID_TENSOR_SHAPE  = 0xB5001U, // 维度数量/axis 等 shape 相关约束不满足
    INVALID_TENSOR_DTYPE  = 0xB5002U, // dtype 组合不合法（如 QuantPreCompute 的 out/self 类型约束）
    INVALID_OP_CONTEXT    = 0xB5003U, // 运算上下文不合法（如 patternMode 范围、前置状态检查等）
    UNSUPPORTED_OPCODE    = 0xB5004U, // 不支持的比较类型或其它操作类型
};

} // namespace npu::tile_fwk::calc_error

