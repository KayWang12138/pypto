/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "aicore_print_common.h"

// ==================== SafeBitCast 位转换函数 ====================

template <typename T, typename U>
INLINE void SafeBitCast(T& dst, const U& src) {
    static_assert(sizeof(T) >= sizeof(U), "Target type too small");
    const unsigned char* srcBytes = reinterpret_cast<const unsigned char*>(&src);
    unsigned char* dstBytes = reinterpret_cast<unsigned char*>(&dst);
    for (std::size_t i = 0; i < sizeof(U); ++i) { dstBytes[i] = srcBytes[i]; }
    for (std::size_t i = sizeof(U); i < sizeof(T); ++i) { dstBytes[i] = 0; }
}

template <typename T, typename U>
INLINE T SafeBitCast(const U& src) {
    T dst{};
    SafeBitCast(dst, src);
    return dst;
}

// ==================== BF16/FP16 解码函数 ====================

INLINE float DecodeBf16(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << Bf16Const::TO_FP32_SHIFT;
    return SafeBitCast<float>(u);
}

INLINE float DecodeF16(uint16_t bits) {
    const uint16_t sign = (bits & Fp16Const::SIGN_MASK) >> Fp16Const::SIGN_SHIFT;
    const uint16_t exp = (bits & Fp16Const::EXP_MASK) >> Fp16Const::EXP_SHIFT;
    const uint16_t mant = bits & Fp16Const::MANT_MASK;

    uint32_t sign32 = static_cast<uint32_t>(sign) << Fp32Const::SIGN_SHIFT;
    uint32_t exp32 = 0;
    uint32_t mant32 = 0;

    if (exp == 0) {
        if (mant == 0) {
            exp32 = 0;
            mant32 = 0;
        } else {
            exp32 = Fp16Const::SUBNORMAL_EXP_BASE;
            uint16_t normMant = mant;
            while ((normMant & Fp16Const::NORM_HIDDEN) == 0) { normMant <<= 1; --exp32; }
            normMant &= Fp16Const::MANT_MASK;
            mant32 = static_cast<uint32_t>(normMant) << Fp16Const::MANT_TO_FP32_SHIFT;
        }
    } else if (exp == Fp16Const::EXP_INF_NAN) {
        exp32 = Fp32Const::EXP_INF_NAN;
        mant32 = static_cast<uint32_t>(mant) << Fp16Const::MANT_TO_FP32_SHIFT;
    } else {
        exp32 = static_cast<uint32_t>(exp) - Fp16Const::EXP_BIAS + Fp32Const::EXP_BIAS;
        mant32 = static_cast<uint32_t>(mant) << Fp16Const::MANT_TO_FP32_SHIFT;
    }

    uint32_t result = sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32;
    return SafeBitCast<float>(result);
}

// ==================== FP8/HF8 解码函数 ====================

#if SUPPORT_FP8_HF8_PRINT

// ---------- FP8 通用解码模板 ----------
template<uint8_t ExpBits, uint8_t MantBits, uint8_t ExpBias, bool HasInf>
INLINE float DecodeFp8Common(uint8_t bits) {
    constexpr uint8_t SignShift = 7;
    constexpr uint8_t ExpShift = MantBits;
    constexpr uint8_t ExpMax = (1u << ExpBits) - 1;
    constexpr uint8_t MantHidden = (1u << (MantBits - 1));
    constexpr uint8_t MantMask = (1u << MantBits) - 1;

    const uint8_t sign = (bits >> SignShift) & 0x1;
    const uint8_t exp = (bits >> ExpShift) & ((1u << ExpBits) - 1);
    const uint8_t mant = bits & MantMask;

    uint32_t sign32 = static_cast<uint32_t>(sign) << Fp32Const::SIGN_SHIFT;
    uint32_t exp32 = 0;
    uint32_t mant32 = 0;

    if (exp == 0) {
        if (mant == 0) {
            exp32 = 0;
            mant32 = 0;
        } else {
            exp32 = Fp32Const::EXP_BIAS - (ExpBias - 1);
            uint8_t normMant = mant;
            while ((normMant & MantHidden) == 0) { normMant <<= 1; --exp32; }
            normMant &= MantMask;
            mant32 = static_cast<uint32_t>(normMant) << (Fp32Const::MANT_BITS - MantBits + 1);
        }
    } else if (HasInf && exp == ExpMax) {
        exp32 = Fp32Const::EXP_INF_NAN;
        mant32 = static_cast<uint32_t>(mant) << (Fp32Const::MANT_BITS - MantBits);
    } else {
        exp32 = static_cast<uint32_t>(exp) - ExpBias + Fp32Const::EXP_BIAS;
        mant32 = static_cast<uint32_t>(mant) << (Fp32Const::MANT_BITS - MantBits);
    }

    uint32_t result = sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32;
    return SafeBitCast<float>(result);
}

// ---------- FP8 E4M3/E5M2/E8M0 解码 ----------
INLINE float DecodeFp8E4M3(uint8_t bits) { return DecodeFp8Common<4, 3, 7, false>(bits); }
INLINE float DecodeFp8E5M2(uint8_t bits) { return DecodeFp8Common<5, 2, 15, true>(bits); }

INLINE float DecodeFp8E8M0(uint8_t bits) {
    const uint8_t sign = (bits >> Fp8E8M0Const::SIGN_SHIFT) & 0x1;
    const int8_t exp = static_cast<int8_t>(bits & 0x7F);

    uint32_t sign32 = static_cast<uint32_t>(sign) << Fp32Const::SIGN_SHIFT;
    uint32_t exp32 = (exp == 0) ? 0 : (exp - Fp8E8M0Const::EXP_BIAS + Fp32Const::EXP_BIAS);
    uint32_t mant32 = 0;

    uint32_t result = sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32;
    return SafeBitCast<float>(result);
}

// ---------- HF8 解码函数系列 ----------
INLINE float DecodeHf8Tiny(int signBit, int mv) {
    if (mv == 0) {
        uint32_t result = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
        return SafeBitCast<float>(result);
    }
    const uint32_t sign32 = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
    const int fp32Exp = mv - 23 + Fp32Const::EXP_BIAS;
    const uint32_t result = sign32 | (static_cast<uint32_t>(fp32Exp) << Fp32Const::EXP_SHIFT);
    return SafeBitCast<float>(result);
}

INLINE float DecodeHf8Small(int signBit, int mv) {
    const uint32_t sign32 = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
    const uint32_t exp32 = Fp32Const::EXP_BIAS;
    const uint32_t mant32 = static_cast<uint32_t>(mv) << (Fp32Const::MANT_BITS - 3);
    return SafeBitCast<float>(sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32);
}

INLINE float DecodeHf8Medium(int signBit, int lower7) {
    const int eb = (lower7 >> 3) & 0x1;
    const int ev = (eb == 0) ? 1 : -1;
    const int mv = lower7 & 0x7;
    const uint32_t sign32 = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
    const uint32_t exp32 = static_cast<uint32_t>(ev + Fp32Const::EXP_BIAS);
    const uint32_t mant32 = static_cast<uint32_t>(mv) << (Fp32Const::MANT_BITS - 3);
    return SafeBitCast<float>(sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32);
}

INLINE float DecodeHf8Large(int signBit, int lower7) {
    const int eb = (lower7 >> 3) & 0x3;
    const int evSign = (eb >> 1) & 0x1;
    const int evAbs = 2 + (eb & 0x1);
    const int ev = evSign ? -evAbs : evAbs;
    const int mv = lower7 & 0x7;
    const uint32_t sign32 = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
    const uint32_t exp32 = static_cast<uint32_t>(ev + Fp32Const::EXP_BIAS);
    const uint32_t mant32 = static_cast<uint32_t>(mv) << (Fp32Const::MANT_BITS - 3);
    return SafeBitCast<float>(sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32);
}

INLINE float DecodeHf8Huge(int signBit, int lower7) {
    const int eb = (lower7 >> 2) & 0x7;
    const int evSign = (eb >> 2) & 0x1;
    const int evAbs = 4 + (eb & 0x3);
    const int ev = evSign ? -evAbs : evAbs;
    const int mv = lower7 & 0x3;
    const uint32_t sign32 = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
    const uint32_t exp32 = static_cast<uint32_t>(ev + Fp32Const::EXP_BIAS);
    const uint32_t mant32 = static_cast<uint32_t>(mv) << (Fp32Const::MANT_BITS - 2);
    return SafeBitCast<float>(sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32);
}

INLINE float DecodeHf8Max(int signBit, int lower7) {
    const int eb = (lower7 >> 1) & 0xF;
    const int evSign = (eb >> 3) & 0x1;
    const int evAbs = 8 + (eb & 0x7);
    const int ev = evSign ? -evAbs : evAbs;
    const int mv = lower7 & 0x1;
    const uint32_t sign32 = static_cast<uint32_t>(signBit << Fp32Const::SIGN_SHIFT);
    const uint32_t exp32 = static_cast<uint32_t>(ev + Fp32Const::EXP_BIAS);
    const uint32_t mant32 = static_cast<uint32_t>(mv) << (Fp32Const::MANT_BITS - 1);
    return SafeBitCast<float>(sign32 | (exp32 << Fp32Const::EXP_SHIFT) | mant32);
}

INLINE float DecodeHf8(uint8_t bits) {
    const int signBit = (bits >> 7) & 0x1;
    const int lower7 = bits & 0x7F;
    const int top4 = lower7 >> 3;

    if (top4 == 0) { return DecodeHf8Tiny(signBit, lower7 & 0x7); }
    if (top4 == 1) { return DecodeHf8Small(signBit, lower7 & 0x7); }

    const int top3 = lower7 >> 4;
    if (top3 == 1) { return DecodeHf8Medium(signBit, lower7); }

    const int top2 = lower7 >> 5;
    if (top2 == 1) { return DecodeHf8Large(signBit, lower7); }
    if (top2 == 2) { return DecodeHf8Huge(signBit, lower7); }

    return DecodeHf8Max(signBit, lower7);
}

#endif

// ==================== Host端解码辅助 ====================

#ifdef __TILE_FWK_HOST__

#include <string>
#include <securec.h>

// ---------- DecodeState 结构体 ----------
struct DecodeState {
    int64_t tail_;
    int64_t head_;
    int64_t size_;
    __gm__ uint8_t* data_;
    std::string lastTensorName_;
};

// ---------- 基础读取函数 ----------
inline uint8_t ReadDecodeByte(DecodeState& state, int64_t off) {
    return state.data_[off % state.size_];
}

template <typename T>
inline T ReadDecodeValue(DecodeState& state, int64_t off) {
    T val{};
    auto* bytes = reinterpret_cast<uint8_t*>(&val);
    for (size_t i = 0; i < sizeof(T); i++) { bytes[i] = ReadDecodeByte(state, off + i); }
    return val;
}

inline std::string ReadDecodeString(DecodeState& state, int64_t off) {
    std::string result;
    result.reserve(64);
    while (off < state.head_) {
        char c = ReadDecodeValue<char>(state, off++);
        if (c == '\0') break;
        result.push_back(c);
    }
    return result;
}

// ---------- Tensor Header 解码 ----------
inline int DecodeTensorHeader(DecodeState& state, char* buf, size_t maxSize) {
    short nameLen = ReadDecodeValue<short>(state, state.tail_);
    state.tail_ += AicorePrintConst::NAMELEN_FIELD_SIZE;

    std::string name = ReadDecodeString(state, state.tail_);
    state.tail_ += nameLen;

    int64_t begin = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::TENSOR_RANGE_SIZE;
    int64_t end = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::TENSOR_RANGE_SIZE;

    state.lastTensorName_ = name;

    return snprintf_s(buf, maxSize, maxSize - 1, "tensor '%s', range=[%ld, %ld)\n", name.c_str(), begin, end);
}

// ---------- Indexed 类型解码 ----------
inline int DecodeIndexedFp32(DecodeState& state, char* buf, size_t maxSize) {
    int64_t index = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
    float value = ReadDecodeValue<float>(state, state.tail_);
    state.tail_ += sizeof(float);

    return snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n", state.lastTensorName_.c_str(), index, value);
}

inline int DecodeIndexedInt64(DecodeState& state, char* buf, size_t maxSize) {
    int64_t index = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
    int64_t value = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::TENSOR_RANGE_SIZE;

    return snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %lld\n", state.lastTensorName_.c_str(), index, value);
}

inline int DecodeIndexedBf16(DecodeState& state, char* buf, size_t maxSize) {
    int64_t index = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
    uint16_t bits = ReadDecodeValue<uint16_t>(state, state.tail_);
    state.tail_ += sizeof(uint16_t);

    float value = DecodeBf16(bits);
    return snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n", state.lastTensorName_.c_str(), index, value);
}

inline int DecodeIndexedFp16(DecodeState& state, char* buf, size_t maxSize) {
    int64_t index = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
    uint16_t bits = ReadDecodeValue<uint16_t>(state, state.tail_);
    state.tail_ += sizeof(uint16_t);

    float value = DecodeF16(bits);
    return snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n", state.lastTensorName_.c_str(), index, value);
}

#if SUPPORT_FP8_HF8_PRINT
inline int DecodeIndexedFp8E4M3(DecodeState& state, char* buf, size_t maxSize) {
    int64_t index = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
    uint8_t bits = ReadDecodeValue<uint8_t>(state, state.tail_);
    state.tail_ += sizeof(uint8_t);

    float value = DecodeFp8E4M3(bits);
    return snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n", state.lastTensorName_.c_str(), index, value);
}

inline int DecodeIndexedFp8E5M2(DecodeState& state, char* buf, size_t maxSize) {
    int64_t index = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
    uint8_t bits = ReadDecodeValue<uint8_t>(state, state.tail_);
    state.tail_ += sizeof(uint8_t);

    float value = DecodeFp8E5M2(bits);
    return snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n", state.lastTensorName_.c_str(), index, value);
}

inline int DecodeIndexedFp8E8M0(DecodeState& state, char* buf, size_t maxSize) {
    int64_t index = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
    uint8_t bits = ReadDecodeValue<uint8_t>(state, state.tail_);
    state.tail_ += sizeof(uint8_t);

    float value = DecodeFp8E8M0(bits);
    return snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n", state.lastTensorName_.c_str(), index, value);
}

inline int DecodeIndexedHf8(DecodeState& state, char* buf, size_t maxSize) {
    int64_t index = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
    uint8_t bits = ReadDecodeValue<uint8_t>(state, state.tail_);
    state.tail_ += sizeof(uint8_t);

    float value = DecodeHf8(bits);
    return snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n", state.lastTensorName_.c_str(), index, value);
}
#endif

// ---------- Overflow Warning 解码 ----------
inline int DecodeOverflowWarning(DecodeState& state, char* buf, size_t maxSize) {
    int64_t bufferSize = ReadDecodeValue<int64_t>(state, state.tail_);
    state.tail_ += sizeof(int64_t);

    constexpr int64_t remoteHeaderSize = static_cast<int64_t>(AicorePrintConst::REMOTE_HEADER_SIZE);
    int64_t fullBufferSize = bufferSize + remoteHeaderSize;
    int64_t recommendedSize = fullBufferSize * 2;

    return snprintf_s(buf, maxSize, maxSize - 1,
        "[WARNING] The PRINT_BUFFER_SIZE (ring buffer) is full! "
        "Current buffer: %ld bytes (%ld KB). "
        "Recommend: set PRINT_BUFFER_SIZE >= %ld (%ld KB, double current size) "
        "in framework/src/interface/machine/device/tilefwk/aicpu_common.h, "
        "then rebuild and reinstall.\n",
        fullBufferSize, fullBufferSize / 1024, recommendedSize, recommendedSize / 1024);
}

// ---------- Legacy Record 解码 ----------
inline int DecodeLegacyRecord(DecodeState& state, AicorePrint::DataType type, char* buf, size_t maxSize) {
    auto valOff = state.tail_ + AicorePrintConst::NAMELEN_FIELD_SIZE;
    state.tail_ += ReadDecodeValue<short>(state, state.tail_) + AicorePrintConst::NAMELEN_FIELD_SIZE;
    auto fmtOff = state.tail_ + AicorePrintConst::NAMELEN_FIELD_SIZE;
    std::string fmt = ReadDecodeString(state, fmtOff);
    state.tail_ += ReadDecodeValue<short>(state, state.tail_) + AicorePrintConst::NAMELEN_FIELD_SIZE;

    switch (type) {
        case AicorePrint::DataType::Normal:
            return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), 0);
        case AicorePrint::DataType::Fp32:
            return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), ReadDecodeValue<float>(state, valOff));
        case AicorePrint::DataType::Int64:
            return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), ReadDecodeValue<int64_t>(state, valOff));
        case AicorePrint::DataType::Char:
            return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), ReadDecodeValue<char>(state, valOff));
        case AicorePrint::DataType::String:
            return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), ReadDecodeString(state, valOff).c_str());
        case AicorePrint::DataType::Pointer:
            return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), ReadDecodeValue<int64_t>(state, valOff));
        case AicorePrint::DataType::Bf16: {
            uint16_t bits = ReadDecodeValue<uint16_t>(state, valOff);
            return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), DecodeBf16(bits));
        }
        case AicorePrint::DataType::Fp16: {
            uint16_t bits = ReadDecodeValue<uint16_t>(state, valOff);
            return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), DecodeF16(bits));
        }
#if SUPPORT_FP8_HF8_PRINT
        case AicorePrint::DataType::Fp8E4M3: {
            uint8_t bits = ReadDecodeValue<uint8_t>(state, valOff);
            return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), DecodeFp8E4M3(bits));
        }
        case AicorePrint::DataType::Fp8E5M2: {
            uint8_t bits = ReadDecodeValue<uint8_t>(state, valOff);
            return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), DecodeFp8E5M2(bits));
        }
        case AicorePrint::DataType::Fp8E8M0: {
            uint8_t bits = ReadDecodeValue<uint8_t>(state, valOff);
            return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), DecodeFp8E8M0(bits));
        }
        case AicorePrint::DataType::Hf8: {
            uint8_t bits = ReadDecodeValue<uint8_t>(state, valOff);
            return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), DecodeHf8(bits));
        }
#endif
        default:
            buf[0] = '?';
            return 1;
    }
}

// ---------- 统一解码入口 ----------
inline int DecodeRecordImpl(DecodeState& state, AicorePrint::DataType type, char* buf, size_t maxSize) {
    switch (type) {
        case AicorePrint::DataType::TensorHeader:
            return DecodeTensorHeader(state, buf, maxSize);
        case AicorePrint::DataType::IndexedFp32:
            return DecodeIndexedFp32(state, buf, maxSize);
        case AicorePrint::DataType::IndexedInt64:
            return DecodeIndexedInt64(state, buf, maxSize);
        case AicorePrint::DataType::IndexedBf16:
            return DecodeIndexedBf16(state, buf, maxSize);
        case AicorePrint::DataType::IndexedFp16:
            return DecodeIndexedFp16(state, buf, maxSize);
#if SUPPORT_FP8_HF8_PRINT
        case AicorePrint::DataType::IndexedFp8E4M3:
            return DecodeIndexedFp8E4M3(state, buf, maxSize);
        case AicorePrint::DataType::IndexedFp8E5M2:
            return DecodeIndexedFp8E5M2(state, buf, maxSize);
        case AicorePrint::DataType::IndexedFp8E8M0:
            return DecodeIndexedFp8E8M0(state, buf, maxSize);
        case AicorePrint::DataType::IndexedHf8:
            return DecodeIndexedHf8(state, buf, maxSize);
#endif
        case AicorePrint::DataType::OverflowWarning:
            return DecodeOverflowWarning(state, buf, maxSize);
        default:
            return DecodeLegacyRecord(state, type, buf, maxSize);
    }
}

#endif