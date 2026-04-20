/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#if defined(__DAV_M300__) || defined(__DAV_310R6__) || defined(__DAV_L510__) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3801)) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101))
#define SUPPORT_FP8_HF8_PRINT 1
#else
#define SUPPORT_FP8_HF8_PRINT 0
#endif

namespace AicorePrintConst {
    constexpr size_t INDEXED_INDEX_SIZE = 8;
    constexpr size_t TENSOR_RANGE_SIZE = 8;
    constexpr size_t NAMELEN_FIELD_SIZE = 2;
    constexpr size_t TYPE_FIELD_SIZE   = 1;
    constexpr size_t MAX_SHAPE_DIMS    = 6;
    constexpr size_t REMOTE_HEADER_SIZE = 16;
    constexpr size_t WARNING_RESERVE_SPACE = 10;
    constexpr size_t MIN_BUFFER_TOTAL_SIZE = WARNING_RESERVE_SPACE + REMOTE_HEADER_SIZE;
    constexpr int16_t SHORT_MAX_VALUE = 32767;
}

namespace EncodeSizes {
    constexpr int64_t IndexedFp32         = 1 + 8 + 4;
    constexpr int64_t IndexedInt64        = 1 + 8 + 8;
    constexpr int64_t IndexedBf16         = 1 + 8 + 2;
    constexpr int64_t IndexedFp16         = 1 + 8 + 2;
    constexpr int64_t EndMarker           = 1;
    
    constexpr int64_t IndexedFp32WithEnd  = IndexedFp32 + EndMarker;
    constexpr int64_t IndexedInt64WithEnd = IndexedInt64 + EndMarker;
    constexpr int64_t IndexedBf16WithEnd  = IndexedBf16 + EndMarker;
    constexpr int64_t IndexedFp16WithEnd  = IndexedFp16 + EndMarker;
    constexpr int64_t OverflowWarningSize = 1 + 8 + 1;
    
#if SUPPORT_FP8_HF8_PRINT
    constexpr int64_t IndexedFp8          = 1 + 8 + 1;
    constexpr int64_t IndexedFp8WithEnd   = IndexedFp8 + EndMarker;
#endif
}

namespace Fp32Const {
    constexpr uint32_t SIGN_MASK     = 0x80000000u;
    constexpr uint32_t EXP_MASK      = 0x7F800000u;
    constexpr uint32_t MANT_MASK     = 0x007FFFFFu;
    constexpr uint32_t EXP_BIAS      = 127;
    constexpr uint32_t EXP_INF_NAN   = 0xFFu;
    constexpr uint32_t SIGN_SHIFT    = 31;
    constexpr uint32_t EXP_SHIFT     = 23;
    constexpr uint32_t MANT_BITS     = 23;
}

namespace Fp16Const {
    constexpr uint16_t SIGN_MASK     = 0x8000u;
    constexpr uint16_t EXP_MASK      = 0x7C00u;
    constexpr uint16_t MANT_MASK     = 0x03FFu;
    constexpr uint16_t NORM_HIDDEN   = 0x0400u;
    constexpr uint16_t EXP_INF_NAN   = 0x1Fu;
    constexpr uint16_t SIGN_SHIFT    = 15;
    constexpr uint16_t EXP_SHIFT     = 10;
    constexpr uint16_t EXP_BIAS      = 15;
    constexpr uint32_t MANT_TO_FP32_SHIFT = 13;
    constexpr uint32_t SUBNORMAL_EXP_BASE = Fp32Const::EXP_BIAS - (EXP_BIAS - 1);
}

namespace Bf16Const {
    constexpr uint32_t TO_FP32_SHIFT = 16;
}

#if SUPPORT_FP8_HF8_PRINT
namespace Fp8E4M3Const {
    constexpr uint8_t EXP_BITS      = 4;
    constexpr uint8_t MANT_BITS     = 3;
    constexpr uint8_t EXP_BIAS      = 7;
    constexpr uint8_t EXP_MAX       = 0xF;
    constexpr uint8_t SIGN_SHIFT    = 7;
    constexpr uint8_t EXP_SHIFT     = 3;
    constexpr uint8_t MANT_HIDDEN   = 0x4;
}

namespace Fp8E5M2Const {
    constexpr uint8_t EXP_BITS      = 5;
    constexpr uint8_t MANT_BITS     = 2;
    constexpr uint8_t EXP_BIAS      = 15;
    constexpr uint8_t EXP_MAX       = 0x1F;
    constexpr uint8_t SIGN_SHIFT    = 7;
    constexpr uint8_t EXP_SHIFT     = 2;
    constexpr uint8_t MANT_HIDDEN   = 0x2;
}

namespace Fp8E8M0Const {
    constexpr uint8_t EXP_BITS      = 7;
    constexpr uint8_t MANT_BITS     = 0;
    constexpr uint8_t EXP_BIAS      = 127;
    constexpr uint8_t SIGN_SHIFT    = 7;
}
#endif