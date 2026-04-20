/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "aikernel_define.h"
#include "aikernel_data.h"

// FP8/HF8 打印支持判定
#if defined(__DAV_M300__) || defined(__DAV_310R6__) || defined(__DAV_L510__) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3801)) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101))
#define SUPPORT_FP8_HF8_PRINT 1
#else
#define SUPPORT_FP8_HF8_PRINT 0
#endif

// ==================== 常量定义 ====================

namespace AicorePrintConst {
    constexpr size_t INDEXED_INDEX_SIZE = 8;
    constexpr size_t TENSOR_RANGE_SIZE = 8;
    constexpr size_t NAMELEN_FIELD_SIZE = 2;
    constexpr size_t TYPE_FIELD_SIZE = 1;
    constexpr size_t MAX_SHAPE_DIMS = 6;
    constexpr size_t REMOTE_HEADER_SIZE = 16;
    constexpr size_t WARNING_RESERVE_SPACE = 10;
    constexpr size_t MIN_BUFFER_TOTAL_SIZE = WARNING_RESERVE_SPACE + REMOTE_HEADER_SIZE;
    constexpr size_t SHORT_MAX_VALUE = 32767;
}

// ---------- FP32 浮点常量 ----------
namespace Fp32Const {
    constexpr uint32_t SIGN_MASK = 0x80000000u;
    constexpr uint32_t EXP_MASK = 0x7F800000u;
    constexpr uint32_t MANT_MASK = 0x007FFFFFu;
    constexpr uint32_t EXP_BIAS = 127;
    constexpr uint32_t EXP_INF_NAN = 0xFFu;
    constexpr uint32_t SIGN_SHIFT = 31;
    constexpr uint32_t EXP_SHIFT = 23;
    constexpr uint32_t MANT_BITS = 23;
}

// ---------- FP16 浮点常量 ----------
namespace Fp16Const {
    constexpr uint16_t SIGN_MASK = 0x8000u;
    constexpr uint16_t EXP_MASK = 0x7C00u;
    constexpr uint16_t MANT_MASK = 0x03FFu;
    constexpr uint16_t NORM_HIDDEN = 0x0400u;
    constexpr uint16_t EXP_INF_NAN = 0x1Fu;
    constexpr uint16_t SIGN_SHIFT = 15;
    constexpr uint16_t EXP_SHIFT = 10;
    constexpr uint16_t EXP_BIAS = 15;
    constexpr uint32_t MANT_TO_FP32_SHIFT = 13;
    constexpr uint32_t SUBNORMAL_EXP_BASE = Fp32Const::EXP_BIAS - (EXP_BIAS - 1);
}

// ---------- BF16 浮点常量 ----------
namespace Bf16Const {
    constexpr uint32_t TO_FP32_SHIFT = 16;
}

#if SUPPORT_FP8_HF8_PRINT
// ---------- FP8 E4M3 浮点常量 ----------
namespace Fp8E4M3Const {
    constexpr uint8_t EXP_BITS = 4;
    constexpr uint8_t MANT_BITS = 3;
    constexpr uint8_t EXP_BIAS = 7;
    constexpr uint8_t EXP_MAX = 0xF;
    constexpr uint8_t SIGN_SHIFT = 7;
    constexpr uint8_t EXP_SHIFT = 3;
    constexpr uint8_t MANT_HIDDEN = 0x4;
}

// ---------- FP8 E5M2 浮点常量 ----------
namespace Fp8E5M2Const {
    constexpr uint8_t EXP_BITS = 5;
    constexpr uint8_t MANT_BITS = 2;
    constexpr uint8_t EXP_BIAS = 15;
    constexpr uint8_t EXP_MAX = 0x1F;
    constexpr uint8_t SIGN_SHIFT = 7;
    constexpr uint8_t EXP_SHIFT = 2;
    constexpr uint8_t MANT_HIDDEN = 0x2;
}

// ---------- FP8 E8M0 浮点常量 ----------
namespace Fp8E8M0Const {
    constexpr uint8_t EXP_BITS = 7;
    constexpr uint8_t MANT_BITS = 0;
    constexpr uint8_t EXP_BIAS = 127;
    constexpr uint8_t SIGN_SHIFT = 7;
}
#endif

// ==================== 数据类型枚举 ====================

namespace AicorePrint {
    enum class DataType : uint8_t {
        End             = 0,
        Normal          = 1,
        Fp32            = 2,
        Int64           = 3,
        Char            = 4,
        String          = 5,
        Pointer         = 6,
        Bf16            = 7,
        Fp16            = 8,
        TensorHeader    = 9,
        IndexedFp32     = 10,
        IndexedInt64    = 11,
        IndexedBf16     = 12,
        IndexedFp16     = 13,
        OverflowWarning = 14,
#if SUPPORT_FP8_HF8_PRINT
        Fp8E4M3         = 15,
        Fp8E5M2         = 16,
        Fp8E8M0         = 17,
        Hf8             = 18,
        IndexedFp8E4M3  = 19,
        IndexedFp8E5M2  = 20,
        IndexedFp8E8M0  = 21,
        IndexedHf8      = 22,
#endif
    };
}

// ==================== LogContext 结构体 ====================

struct LogContext {
    void (*PrintInt64)(LogContext* ctx, __gm__ const char** fmt, int64_t val);
    void (*PrintFp32)(LogContext* ctx, __gm__ const char** fmt, float val);
    void (*PrintBf16)(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits);
    void (*PrintFp16)(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits);
    void (*PrintRaw)(LogContext* ctx, __gm__ const char* fmt);
#if SUPPORT_FP8_HF8_PRINT
    void (*PrintFp8E4M3)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintFp8E5M2)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintFp8E8M0)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintHf8)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
#endif
};

// ==================== IndexedTypeInfo 模板 ====================

template <typename ElemT>
struct IndexedTypeInfo {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::End;
};

template<> struct IndexedTypeInfo<float> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedFp32;
};

template<> struct IndexedTypeInfo<int64_t> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedInt64;
};

#if IS_AICORE
template<> struct IndexedTypeInfo<bfloat16_t> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedBf16;
};

template<> struct IndexedTypeInfo<half> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedFp16;
};

#if SUPPORT_FP8_HF8_PRINT
template<> struct IndexedTypeInfo<float8_e4m3_t> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedFp8E4M3;
};

template<> struct IndexedTypeInfo<float8_e5m2_t> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedFp8E5M2;
};

template<> struct IndexedTypeInfo<float8_e8m0_t> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedFp8E8M0;
};

template<> struct IndexedTypeInfo<hifloat8_t> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedHf8;
};
#endif
#endif