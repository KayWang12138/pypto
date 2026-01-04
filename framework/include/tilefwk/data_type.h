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
 * \file data_type.h
 * \brief
 */

#pragma once

#include <string>
#include <stdexcept>

namespace npu::tile_fwk {

#define DATA_TYPE_ALL                                   \
    DTYPE_DESC(DT_INT4, 1, 4, false, int4)              \
    DTYPE_DESC(DT_INT8, 1, 8, false, int8_t)            \
    DTYPE_DESC(DT_INT16, 2, 16, false, int16_t)         \
    DTYPE_DESC(DT_INT32, 4, 32, false, int32_t)         \
    DTYPE_DESC(DT_INT64, 8, 64, false, int64_t)         \
    DTYPE_DESC(DT_FP8, 1, 8, true, float8_t)            \
    DTYPE_DESC(DT_FP16, 2, 16, true, half)              \
    DTYPE_DESC(DT_FP32, 4, 32, true, float)             \
    DTYPE_DESC(DT_BF16, 2, 16, true, bfloat16_t)        \
    DTYPE_DESC(DT_HF4, 1, 4, true, hfloat4)             \
    DTYPE_DESC(DT_HF8, 1, 8, true, hfloat8)             \
    DTYPE_DESC(DT_UINT8, 1, 8, false, uint8_t)          \
    DTYPE_DESC(DT_UINT16, 2, 16, false, uint16_t)       \
    DTYPE_DESC(DT_UINT32, 4, 32, false, uint32_t)       \
    DTYPE_DESC(DT_UINT64, 8, 64, false, uint64_t)       \
    DTYPE_DESC(DT_BOOL, 1, 8, false, bool)              \
    DTYPE_DESC(DT_DOUBLE, 8, 64, true, double)          \
    DTYPE_DESC(DT_FP8E4M3FN, 1, 8, true, float8_e4m3_t) \
    DTYPE_DESC(DT_FP8E5M2, 1, 8, true, float8_e5m2_t)

enum DataType {
#define DTYPE_DESC(name, byte, bit, is_float, type) name,
    DATA_TYPE_ALL
#undef DTYPE_DESC
    DT_BOTTOM
};

inline bool IsFloat(DataType t) {
    switch (t) {
#define DTYPE_DESC(name, byte, bit, is_float, type) case name: return is_float;
    DATA_TYPE_ALL
#undef DTYPE_DESC
    default:
        throw std::invalid_argument("Unknown DataType");
    }
}

inline const char *DataType2String(DataType t, bool brief = false) {
    const char *ret = nullptr;
    switch (t) {
#define DTYPE_DESC(name, byte, bit, is_float, type) case name: ret = #name; break;
    DATA_TYPE_ALL
#undef DTYPE_DESC
    default:
        throw std::invalid_argument("Unknown DataType");
    }
    return brief ? (ret + 0x3) : ret; // brief skip "DT_"
}

inline size_t BytesOf(DataType t) {
    switch (t) {
#define DTYPE_DESC(name, byte, bit, is_float, type) case name: return byte;
    DATA_TYPE_ALL
#undef DTYPE_DESC
    default:
        throw std::invalid_argument("Unknown DataType");
    }
}

inline int64_t BitsOf(DataType t) {
    switch (t) {
#define DTYPE_DESC(name, byte, bit, is_float, type) case name: return bit;
    DATA_TYPE_ALL
#undef DTYPE_DESC
    default:
        throw std::invalid_argument("Unknown DataType");
    }
}

inline const char *DataType2CCEStr(DataType t) {
    switch (t) {
#define DTYPE_DESC(name, byte, bit, is_float, type) case name: return #type;
    DATA_TYPE_ALL
#undef DTYPE_DESC
    default:
        throw std::invalid_argument("Unknown DataType");
    }
}

enum class NodeType {
    LOCAL = 0,
    INCAST = 1,
    OUTCAST = 2,
};

enum class TileOpFormat {
    TILEOP_ND = 0,
    TILEOP_NZ = 1,
    TILEOP_FORMAT_NUM [[maybe_unused]],
};

enum MemoryType {
    MEM_UB = 0,
    MEM_L1 = 1,
    MEM_L0A = 2,
    MEM_L0B = 3,
    MEM_L0C = 4,
    MEM_FIX = 5,
    MEM_FIX_QUANT_PRE = 6,
    MEM_FIX_RELU_PRE = 7,
    MEM_FIX_RELU_POST = 8,
    MEM_FIX_QUANT_POST = 9,
    MEM_FIX_ELT_ANTIQ = 10,
    MEM_FIX_MTE2_ANTIQ = 11,
    MEM_BT = 12,
    MEM_L2 = 13,
    MEM_L3 = 14,
    MEM_DEVICE_DDR = 15,
    MEM_HOST1 = 16,
    MEM_FAR1 = 17,
    MEM_FAR2 = 18,
    MEM_WORKSPACE = 19,
    MEM_VECTOR_REG = 20,
    MEM_UNKNOWN
};

enum CastMode {
    CAST_NONE = 0,
    CAST_RINT = 1,  // round to nearest, tie to even
    CAST_ROUND = 2, // round to nearest, tie away from zero
    CAST_FLOOR = 3, // round to minus infinity
    CAST_CEIL = 4,  // round to positive infinity
    CAST_TRUNC = 5, // round to zero
    CAST_ODD = 6,   // round to odd (Von Neumann rounding)
};

inline std::string NodeType2String(NodeType n) {
    switch (n) {
        case NodeType::LOCAL: return "LOCAL";
        case NodeType::INCAST: return "INCAST";
        case NodeType::OUTCAST: return "OUTCAST";
        default: throw std::invalid_argument("Unknown NodeType");
    }
}

inline std::string MemoryTypeToString(MemoryType mt) {
    switch (mt) {
        case MEM_UB: return "MEM_UB";
        case MEM_L1: return "MEM_L1";
        case MEM_L0A: return "MEM_L0A";
        case MEM_L0B: return "MEM_L0B";
        case MEM_L0C: return "MEM_L0C";
        case MEM_L2: return "MEM_L2";
        case MEM_L3: return "MEM_L3";
        case MEM_DEVICE_DDR: return "MEM_DEVICE_DDR";
        case MEM_HOST1: return "MEM_HOST1";
        case MEM_FAR1: return "MEM_FAR1";
        case MEM_FAR2: return "MEM_FAR2";
        case MEM_BT: return "MEM_BT";
        case MEM_VECTOR_REG: return "MEM_VECTOR_REG";
        case MEM_FIX: return "MEM_FIX";
        case MEM_FIX_QUANT_PRE: return "MEM_FIX_QUANT_PRE";
        case MEM_FIX_RELU_PRE: return "MEM_FIX_RELU_PRE";
        case MEM_FIX_RELU_POST: return "MEM_FIX_RELU_POST";
        case MEM_FIX_QUANT_POST: return "MEM_FIX_QUANT_POST";
        case MEM_FIX_ELT_ANTIQ: return "MEM_FIX_ELT_ANTIQ";
        case MEM_FIX_MTE2_ANTIQ: return "MEM_FIX_MTE2_ANTIQ";
        default: return "MEM_UNKNOWN";
    }
#undef CASE
}

[[maybe_unused]] inline std::string BriefMemoryTypeToString(MemoryType mt) {
    switch (mt) {
        case MEM_UB: return "UB";
        case MEM_L1: return "L1";
        case MEM_L0A: return "L0A";
        case MEM_L0B: return "L0B";
        case MEM_L0C: return "L0C";
        case MEM_L2: return "L2";
        case MEM_L3: return "L3";
        case MEM_DEVICE_DDR: return "DDR";
        case MEM_HOST1: return "HOST1";
        case MEM_FAR1: return "FAR1";
        case MEM_FAR2: return "FAR2";
        case MEM_BT: return "BT";
        default: return "MEM_UNKNOWN";
    }
}

[[maybe_unused]] inline std::string CastMode2String(CastMode mode) {
    switch (mode) {
        case CastMode::CAST_NONE: return "CAST_NONE";
        case CastMode::CAST_RINT: return "CAST_RINT";
        case CastMode::CAST_ROUND: return "CAST_ROUND";
        case CastMode::CAST_FLOOR: return "CAST_FLOOR";
        case CastMode::CAST_CEIL: return "CAST_CEIL";
        case CastMode::CAST_TRUNC: return "CAST_TRUNC";
        case CastMode::CAST_ODD: return "CAST_ODD";
        default: throw std::invalid_argument("Unknown CastMode");
    }
}

enum class CachePolicy {
    PREFETCH,
    NONE_CACHEABLE,
    MAX_NUM
};
} // namespace npu::tile_fwk

namespace std {
inline std::string to_string(npu::tile_fwk::TileOpFormat format) {
    switch (format) {
        case npu::tile_fwk::TileOpFormat::TILEOP_ND: return "TILEOP_ND";
        case npu::tile_fwk::TileOpFormat::TILEOP_NZ: return "TILEOP_NZ";
        default: return "UNKNOWN";
    }
}
} // namespace std
