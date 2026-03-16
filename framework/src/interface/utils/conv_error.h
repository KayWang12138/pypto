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
 * \file conv_error.h
 * \brief CONV 组件错误分类、场景枚举与错误码常量。
 *        - F6XXXX-F8XXX：CONV 内部已梳理报错。
 */

#pragma once
#include <cstdint>

namespace npu::tile_fwk {

enum class ConvOperationError : uint32_t {
    // FC61xx: Operation非法拦截类报错 
    FMAP_PARMS_INVALID     = 6101U,
    Weight_PARMS_INVALID     = 6102U,
    BIAS_PARMS_INVALID     = 6103U,
    OUTPUT_PARMS_INVALID    = 6104U,
    UNKNOWN                   = 6199U
};

enum class ConvExpandFuncError : uint32_t {
    EXPANDFUNC_TENSOR_OP_NULLPTR          = 6201U,
    EXPANDFUNC_TENSOR_ATTR_GET_FAILED     = 6202U,
    EXPANDFUNC_TILE_OP_NULLPTR            = 6203U,
    EXPANDFUNC_PARAMS_INVALID             = 6204U,
    EXPANDFUNC_INNER_STATUS_FAILED        = 6205U,
    UNKNOWN                               = 6299U
};

enum class CodenGenError : uint32_t {
};


// 临时，框架会提供公共的ASSERT，CHECK，xxx_LOGE 宏
static inline void conv_snprintf(char* buf, size_t bufSize, const char* fmt, ...) {
    if (buf == nullptr || bufSize == 0 || fmt == nullptr) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    int ret = vsnprintf_s(buf, bufSize, bufSize - 1, fmt, ap);
    va_end(ap);
    if (ret < 0) {
        buf[0] = '\0';
    }
}

#define CONV_CHECK(error_code, cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            CONV_LOGE("[ERR-FC%d] " fmt, static_cast<int>(error_code), ##__VA_ARGS__); \
            return FAILED; \
        } \
    } while (0)


#define CONV_ASSERT(error_code, cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            CONV_LOGE("[FC%d] " fmt, static_cast<int>(error_code), ##__VA_ARGS__); \
            char err_msg[1024] = {0}; \
            conv_snprintf(err_msg, sizeof(err_msg), fmt, ##__VA_ARGS__); \
            ASSERT(false) << "[FC" << static_cast<int>(error_code) << "] " << err_msg; \
        } \
    } while (0)

}  // namespace npu::tile_fwk
