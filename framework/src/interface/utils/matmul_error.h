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
 * \file matmul_error.h
 * \brief MATMUL 组件错误分类、场景枚举与错误码常量。
 *        - F3XXXX-F5XXX：MATMUL 内部已梳理报错（大流程 Category -> 子流程 Scene）。
 */

#pragma once
#include <cstdint>

namespace npu::tile_fwk {

enum class MatmulErrorCode : int32_t {
    
    // FC3xxx: 参数错误
    ERR_PARAM_INVALID       = 3000,  // 参数无效（shape、dtype、format等）
    ERR_PARAM_MISMATCH      = 3001,  // 参数不匹配（维度、类型、K轴等）
    ERR_PARAM_UNSUPPORTED   = 3002,  // 不支持的参数
    
    // FC4xxx: 配置错误
    ERR_CONFIG_TILE         = 4000,  // Tile 配置错误
    ERR_CONFIG_ALIGNMENT    = 4001,  // 对齐错误（16B/32B/64元素）
    ERR_CONFIG_UNSUPPORTED  = 4002,  // 不支持的配置组合
    
    // FC5xxx: 运行时错误
    ERR_RUNTIME_NULLPTR     = 5000,  // 空指针错误
    ERR_RUNTIME_STATE       = 5001,  // 内部状态错误
    ERR_RUNTIME_LOGIC       = 5002,  // 逻辑不变量错误
};

#define MATMUL_CHECK(error_code, cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            MATMUL_LOGE("[ERR-FC%d] " fmt, static_cast<int>(error_code), ##__VA_ARGS__); \
            return FAILED; \
        } \
    } while (0)

#define MATMUL_ASSERT(error_code, cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            MATMUL_LOGE("[ERR-FC%d] " fmt, static_cast<int>(error_code), ##__VA_ARGS__); \
            ASSERT(false) << "[ERR-FC" << static_cast<int>(error_code) << "] " << fmt; \
        } \
    } while (0)

}  // namespace npu::tile_fwk
