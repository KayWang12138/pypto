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
 * \file aicore_print.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <cstring>
#include "aikernel_data.h"

#ifdef __TILE_FWK_AICORE__
#include "tileop/utils/layout.h"
#endif

#define ENABLE_AICORE_PRINT 0

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

// FP8/HF8 打印支持宏（仅 __DAV_V310 平台可用）
#ifdef __DAV_V310
#define SUPPORT_FP8_HF8_PRINT 1
#else
#define SUPPORT_FP8_HF8_PRINT 0
#endif

#ifdef __TILE_FWK_HOST__
#include <string>
#include <sstream>
#include <securec.h>
#endif

#define BF16_TO_FP32_SHIFT 16
#define F16_SIGN_MASK 0x8000u
#define F16_EXP_MASK 0x7C00u
#define F16_MANT_MASK 0x03FFu
#define F16_NORM_HIDDEN_BIT 0x0400u
#define F16_EXP_INF_NAN 0x1Fu
#define FP32_EXP_INF_NAN 0xFFu
#define F16_SIGN_SHIFT 15
#define F16_EXP_SHIFT 10
#define FP32_SIGN_SHIFT 31
#define FP32_EXP_SHIFT 23
#define F16_TO_FP32_MANT_SHIFT 13
#define F16_EXP_BIAS 15
#define FP32_EXP_BIAS 127
#define F16_SUBNORMAL_FP32_EXP_BASE (FP32_EXP_BIAS - (F16_EXP_BIAS - 1))

template <typename T, typename U>
INLINE void SafeBitCast(T& dst, const U& src)
{
    const unsigned char* srcBytes = reinterpret_cast<const unsigned char*>(&src);
    unsigned char* dstBytes = reinterpret_cast<unsigned char*>(&dst);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        dstBytes[i] = srcBytes[i];
    }
}

template <typename T, typename U>
INLINE T SafeBitCast(const U& src)
{
    T dst;
    SafeBitCast(dst, src);
    return dst;
}

INLINE float DecodeBf16(uint16_t bits)
{
    uint32_t u = static_cast<uint32_t>(bits) << BF16_TO_FP32_SHIFT;
    return SafeBitCast<float>(u);
}

INLINE float DecodeF16(uint16_t bits)
{
    uint16_t sign = static_cast<uint16_t>((bits & F16_SIGN_MASK) >> F16_SIGN_SHIFT);
    uint16_t exp = static_cast<uint16_t>((bits & F16_EXP_MASK) >> F16_EXP_SHIFT);
    uint16_t mant = static_cast<uint16_t>(bits & F16_MANT_MASK);

    uint32_t sign32 = static_cast<uint32_t>(sign) << FP32_SIGN_SHIFT;
    uint32_t exp32;
    uint32_t mant32;

    if (exp == 0) {
        if (mant == 0) {
            exp32 = 0;
            mant32 = 0;
        } else {
            exp32 = F16_SUBNORMAL_FP32_EXP_BASE;
            while ((mant & F16_NORM_HIDDEN_BIT) == 0) {
                mant <<= 1;
                --exp32;
            }
            mant &= F16_MANT_MASK;
            mant32 = static_cast<uint32_t>(mant) << F16_TO_FP32_MANT_SHIFT;
        }
    } else if (exp == F16_EXP_INF_NAN) {
        exp32 = FP32_EXP_INF_NAN;
        mant32 = static_cast<uint32_t>(mant) << F16_TO_FP32_MANT_SHIFT;
    } else {
        exp32 = static_cast<uint32_t>(exp) - F16_EXP_BIAS + FP32_EXP_BIAS;
        mant32 = static_cast<uint32_t>(mant) << F16_TO_FP32_MANT_SHIFT;
    }

    uint32_t u = sign32 | (exp32 << FP32_EXP_SHIFT) | mant32;
    return SafeBitCast<float>(u);
}

#if SUPPORT_FP8_HF8_PRINT
INLINE float DecodeFp8E4M3(uint8_t bits)
{
    uint8_t sign = (bits >> 7) & 0x1;
    uint8_t exp = (bits >> 3) & 0xF;
    uint8_t mant = bits & 0x7;
    
    const uint32_t fp32_exp_bias = 127;
    const uint8_t fp8_exp_bias = 7;
    const uint8_t fp8_exp_max = 0xF;
    
    uint32_t sign32 = sign << 31;
    uint32_t exp32;
    uint32_t mant32;
    
    if (exp == 0) {
        if (mant == 0) {
            exp32 = 0;
            mant32 = 0;
        } else {
            exp32 = fp32_exp_bias - (fp8_exp_bias - 1);
            while ((mant & 0x4) == 0) {
                mant <<= 1;
                --exp32;
            }
            mant &= 0x3;
            mant32 = mant << (23 - 3 + 1);
        }
    } else if (exp == fp8_exp_max) {
        exp32 = 0xFF;
        mant32 = mant << (23 - 3);
    } else {
        exp32 = exp - fp8_exp_bias + fp32_exp_bias;
        mant32 = mant << (23 - 3);
    }
    
    uint32_t u = sign32 | (exp32 << 23) | mant32;
    return SafeBitCast<float>(u);
}

INLINE float DecodeFp8E5M2(uint8_t bits)
{
    uint8_t sign = (bits >> 7) & 0x1;
    uint8_t exp = (bits >> 2) & 0x1F;
    uint8_t mant = bits & 0x3;
    
    const uint32_t fp32_exp_bias = 127;
    const uint8_t fp8_exp_bias = 15;
    const uint8_t fp8_exp_max = 0x1F;
    
    uint32_t sign32 = sign << 31;
    uint32_t exp32;
    uint32_t mant32;
    
    if (exp == 0) {
        if (mant == 0) {
            exp32 = 0;
            mant32 = 0;
        } else {
            exp32 = fp32_exp_bias - (fp8_exp_bias - 1);
            while ((mant & 0x2) == 0) {
                mant <<= 1;
                --exp32;
            }
            mant &= 0x1;
            mant32 = mant << (23 - 2 + 1);
        }
    } else if (exp == fp8_exp_max) {
        exp32 = 0xFF;
        mant32 = mant << (23 - 2);
    } else {
        exp32 = exp - fp8_exp_bias + fp32_exp_bias;
        mant32 = mant << (23 - 2);
    }
    
    uint32_t u = sign32 | (exp32 << 23) | mant32;
    return SafeBitCast<float>(u);
}

INLINE float DecodeFp8E8M0(uint8_t bits)
{
    uint8_t sign = (bits >> 7) & 0x1;
    int8_t exp = static_cast<int8_t>(bits & 0x7F);
    
    const uint32_t fp32_exp_bias = 127;
    const uint8_t fp8_exp_bias = 127;
    
    uint32_t sign32 = sign << 31;
    uint32_t exp32 = (exp == 0) ? 0 : (exp - fp8_exp_bias + fp32_exp_bias);
    uint32_t mant32 = 0;
    
    uint32_t u = sign32 | (exp32 << 23) | mant32;
    return SafeBitCast<float>(u);
}

INLINE float DecodeHf8(uint8_t bits)
{
    const int signBit = (bits >> 7) & 0x1;
    const int lower7 = bits & 0x7F;
    
    const int top4 = lower7 >> 3;
    if (top4 == 0) {
        const int mv = lower7 & 0x7;
        if (mv == 0) {
            return SafeBitCast<float>(static_cast<uint32_t>(signBit << 31));
        }
        const uint32_t sign32 = static_cast<uint32_t>(signBit << 31);
        const int fp32_exp = mv - 23 + 127;
        const uint32_t u = sign32 | (static_cast<uint32_t>(fp32_exp) << 23);
        return SafeBitCast<float>(u);
    }
    
    if (top4 == 1) {
        const int mv = lower7 & 0x7;
        const uint32_t sign32 = static_cast<uint32_t>(signBit << 31);
        const uint32_t exp32 = 127;
        const uint32_t mant32 = static_cast<uint32_t>(mv) << (23 - 3);
        return SafeBitCast<float>(sign32 | (exp32 << 23) | mant32);
    }
    
    const int top3 = lower7 >> 4;
    if (top3 == 1) {
        const int eb = (lower7 >> 3) & 0x1;
        const int ev = (eb == 0) ? 1 : -1;
        const int mv = lower7 & 0x7;
        const uint32_t sign32 = static_cast<uint32_t>(signBit << 31);
        const uint32_t exp32 = static_cast<uint32_t>(ev + 127);
        const uint32_t mant32 = static_cast<uint32_t>(mv) << (23 - 3);
        return SafeBitCast<float>(sign32 | (exp32 << 23) | mant32);
    }
    
    const int top2 = lower7 >> 5;
    if (top2 == 1) {
        const int eb = (lower7 >> 3) & 0x3;
        const int evSign = (eb >> 1) & 0x1;
        const int evAbs = 2 + (eb & 0x1);
        const int ev = evSign ? -evAbs : evAbs;
        const int mv = lower7 & 0x7;
        const uint32_t sign32 = static_cast<uint32_t>(signBit << 31);
        const uint32_t exp32 = static_cast<uint32_t>(ev + 127);
        const uint32_t mant32 = static_cast<uint32_t>(mv) << (23 - 3);
        return SafeBitCast<float>(sign32 | (exp32 << 23) | mant32);
    }
    
    if (top2 == 2) {
        const int eb = (lower7 >> 2) & 0x7;
        const int evSign = (eb >> 2) & 0x1;
        const int evAbs = 4 + (eb & 0x3);
        const int ev = evSign ? -evAbs : evAbs;
        const int mv = lower7 & 0x3;
        const uint32_t sign32 = static_cast<uint32_t>(signBit << 31);
        const uint32_t exp32 = static_cast<uint32_t>(ev + 127);
        const uint32_t mant32 = static_cast<uint32_t>(mv) << (23 - 2);
        return SafeBitCast<float>(sign32 | (exp32 << 23) | mant32);
    }
    
    const int eb = (lower7 >> 1) & 0xF;
    const int evSign = (eb >> 3) & 0x1;
    const int evAbs = 8 + (eb & 0x7);
    const int ev = evSign ? -evAbs : evAbs;
    const int mv = lower7 & 0x1;
    const uint32_t sign32 = static_cast<uint32_t>(signBit << 31);
    const uint32_t exp32 = static_cast<uint32_t>(ev + 127);
    const uint32_t mant32 = static_cast<uint32_t>(mv) << (23 - 1);
    return SafeBitCast<float>(sign32 | (exp32 << 23) | mant32);
}
#endif

enum NodeTy { END, NORMAL, FP32, INT, CHAR, STRING, POINTER, BF16, FP16,
              TENSOR_HEADER, INDEXED_FP32, INDEXED_INT64, INDEXED_BF16, INDEXED_FP16,
              OVERFLOW_WARNING
#if SUPPORT_FP8_HF8_PRINT
              , FP8E4M3, FP8E5M2, FP8E8M0, HF8,
              INDEXED_FP8E4M3, INDEXED_FP8E5M2, INDEXED_FP8E8M0, INDEXED_HF8
#endif
              };

struct LogContext {
    void (*PrintInt)(LogContext* ctx, __gm__ const char** fmt, int64_t val);
    void (*PrintFp32)(LogContext* ctx, __gm__ const char** fmt, float val);
    void (*PrintBf16)(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits);
    void (*PrintFp16)(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits);
    void (*Print)(LogContext* ctx, __gm__ const char* fmt);
#if SUPPORT_FP8_HF8_PRINT
    void (*PrintFp8E4M3)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintFp8E5M2)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintFp8E8M0)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintHf8)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
#endif
};

template <typename T>
INLINE void __AiCorePrint(LogContext* ctx, __gm__ const char** fmt, T val)
{
    if constexpr (std::is_integral_v<T>) {
        ctx->PrintInt(ctx, fmt, static_cast<int64_t>(val));
    } else if constexpr (std::is_floating_point_v<T>) {
        ctx->PrintFp32(ctx, fmt, static_cast<float>(val));
    } else if constexpr (std::is_pointer_v<T>) {
        ctx->PrintInt(ctx, fmt, reinterpret_cast<int64_t>(val));
#if IS_AICORE
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        ctx->PrintBf16(ctx, fmt, SafeBitCast<uint16_t>(val));
    } else if constexpr (std::is_same_v<T, half>) {
        ctx->PrintFp16(ctx, fmt, SafeBitCast<uint16_t>(val));
#if SUPPORT_FP8_HF8_PRINT
    } else if constexpr (std::is_same_v<T, float8_e4m3_t>) {
        ctx->PrintFp8E4M3(ctx, fmt, SafeBitCast<uint8_t>(val));
    } else if constexpr (std::is_same_v<T, float8_e5m2_t>) {
        ctx->PrintFp8E5M2(ctx, fmt, SafeBitCast<uint8_t>(val));
    } else if constexpr (std::is_same_v<T, float8_e8m0_t>) {
        ctx->PrintFp8E8M0(ctx, fmt, SafeBitCast<uint8_t>(val));
    } else if constexpr (std::is_same_v<T, hifloat8_t>) {
        ctx->PrintHf8(ctx, fmt, SafeBitCast<uint8_t>(val));
#endif
#endif
    }
}

template <typename... Ts>
INLINE void AiCoreLogF(LogContext* ctx, __gm__ const char* fmt, Ts... Args)
{
    if (ctx && fmt) {
        (__AiCorePrint(ctx, &fmt, Args), ...);
        ctx->Print(ctx, fmt);
    }
}

struct AicoreLogger {
    struct Remote {
        int64_t head_;
        int64_t tail_;
    };

    static __aicore__ void __PrintInt(LogContext* ctx, __gm__ const char** fmt, int64_t val)
    {
        auto self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->PrintInt(fmt, val);
        }
    }

    static __aicore__ void __PrintFloat(LogContext* ctx, __gm__ const char** fmt, float val)
    {
        auto self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->PrintFp32(fmt, val);
        }
    }

    static __aicore__ void __PrintBf16(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits)
    {
        auto self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->PrintBf16(fmt, rawBits);
        }
    }

    static __aicore__ void __PrintF16(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits)
    {
        auto self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->PrintFp16(fmt, rawBits);
        }
    }

#if SUPPORT_FP8_HF8_PRINT
    static __aicore__ void __PrintFp8E4M3(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits)
    {
        auto self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->PrintFp8E4M3(fmt, rawBits);
        }
    }

    static __aicore__ void __PrintFp8E5M2(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits)
    {
        auto self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->PrintFp8E5M2(fmt, rawBits);
        }
    }

    static __aicore__ void __PrintFp8E8M0(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits)
    {
        auto self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->PrintFp8E8M0(fmt, rawBits);
        }
    }

    static __aicore__ void __PrintHf8(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits)
    {
        auto self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->PrintHf8(fmt, rawBits);
        }
    }
#endif

    static __aicore__ void __Print(LogContext* ctx, __gm__ const char* fmt)
    {
        auto self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->Print(fmt);
        }
    }

    __aicore__ void Init(__gm__ uint8_t* buf, size_t n)
    {
        remote_ = reinterpret_cast<volatile __gm__ Remote*>(buf);
        remote_->head_ = remote_->tail_ = 0;
        head_ = tail_ = 0;
        size_ = n - sizeof(Remote);
        data_ = buf + sizeof(Remote);
        ctx.PrintInt = __PrintInt;
        ctx.PrintFp32 = __PrintFloat;
        ctx.PrintBf16 = __PrintBf16;
        ctx.PrintFp16 = __PrintF16;
        ctx.Print = __Print;
#if SUPPORT_FP8_HF8_PRINT
        ctx.PrintFp8E4M3 = __PrintFp8E4M3;
        ctx.PrintFp8E5M2 = __PrintFp8E5M2;
        ctx.PrintFp8E8M0 = __PrintFp8E8M0;
        ctx.PrintHf8 = __PrintHf8;
#endif
    }

    __aicore__ __gm__ uint8_t* GetBuffer() { return data_ - sizeof(Remote); }

    __aicore__ void PrintInt(__gm__ const char** fmt, int64_t val)
    {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);
        if (idx == -1) {
            return;
        }
        switch (curFmt[idx++]) {
            case 's': {
                auto tmp = reinterpret_cast<__gm__ const char*>(val);
                if (tmp == nullptr) {
                    tmp = "<null>";
                }
                Encode(STRING, reinterpret_cast<__gm__ const uint8_t*>(tmp), Length(tmp), *fmt, idx);
                break;
            }
            case 'd':
            case 'i':
            case 'x':
            case 'X':
            case 'o':
            case 'u': {
                Encode(INT, reinterpret_cast<uint8_t*>(&val), sizeof(val), *fmt, idx);
                break;
            }
            case 'p': {
                Encode(POINTER, reinterpret_cast<uint8_t*>(&val), sizeof(val), *fmt, idx);
                break;
            }
            case 'c': {
                char c = static_cast<char>(val);
                Encode(CHAR, reinterpret_cast<uint8_t*>(&c), 1, *fmt, idx);
                break;
            }
            default:
                Encode(NORMAL, static_cast<uint8_t*>(nullptr), 0, *fmt, idx);
                break;
        }

        *fmt = *fmt + idx;
    }

    __aicore__ void PrintFp32(__gm__ const char** fmt, float val)
    {
        EncodeFloatType(fmt, FP32, reinterpret_cast<uint8_t*>(&val), sizeof(val));
    }

    __aicore__ void PrintBf16(__gm__ const char** fmt, uint16_t rawBits)
    {
        EncodeFloatType(fmt, BF16, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintFp16(__gm__ const char** fmt, uint16_t rawBits)
    {
        EncodeFloatType(fmt, FP16, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

#if SUPPORT_FP8_HF8_PRINT
    __aicore__ void PrintFp8E4M3(__gm__ const char** fmt, uint8_t rawBits)
    {
        EncodeFloatType(fmt, FP8E4M3, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintFp8E5M2(__gm__ const char** fmt, uint8_t rawBits)
    {
        EncodeFloatType(fmt, FP8E5M2, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintFp8E8M0(__gm__ const char** fmt, uint8_t rawBits)
    {
        EncodeFloatType(fmt, FP8E8M0, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintHf8(__gm__ const char** fmt, uint8_t rawBits)
    {
        EncodeFloatType(fmt, HF8, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }
#endif

    __aicore__ void Print(__gm__ const char* str)
    {
        auto n = Length(str);
        if (n) {
            Encode(NORMAL, reinterpret_cast<const __gm__ uint8_t*>(str), n, str, n);
        }
        Encode(END);
        Sync();
    }

    __aicore__ void Sync()
    {
#ifndef __TILE_FWK_HOST__
        int64_t delta = (int64_t)(&data_[remote_->head_ % size_]) & (CACHE_LINE_SIZE - 1);
        int64_t off = remote_->head_ - delta;
        while (off < head_) {
            dcci(&data_[off % size_], SINGLE_CACHE_LINE, CACHELINE_OUT);
            off += CACHE_LINE_SIZE;
        }
        remote_->head_ = head_;
        remote_->tail_ = tail_;
        dcci(remote_, SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
        remote_->head_ = head_;
        remote_->tail_ = tail_;
#endif
    }

    // 编码 tensor 头部（名称 + 范围）
    __aicore__ void EncodeTensorHeader(__gm__ const char* name, int64_t begin, int64_t end)
    {
        // 1. 编码 type
        Encode(static_cast<uint8_t>(TENSOR_HEADER));

        // 2. 编码 nameLen
        short nameLen = Length(name) + 1;  // 包含 '\0'
        auto bytes = reinterpret_cast<uint8_t*>(&nameLen);
        Encode(bytes[0]);
        Encode(bytes[1]);

        // 3. 编码 name + '\0'
        for (auto i = 0; i < nameLen; i++) {
            Encode(name[i]);
        }

        // 4. 编码 begin
        bytes = reinterpret_cast<uint8_t*>(&begin);
        for (auto i = 0; i < 8; i++) {
            Encode(bytes[i]);
        }

        // 5. 编码 end
        bytes = reinterpret_cast<uint8_t*>(&end);
        for (auto i = 0; i < 8; i++) {
            Encode(bytes[i]);
        }
    }

    // 编码单条紧凑数据行（index + value，无 fmt 字段）
    __aicore__ void EncodeIndexed(NodeTy ty, int64_t index, const uint8_t* val, short valLen)
    {
        // 1. 编码 type
        Encode(static_cast<uint8_t>(ty));

        // 2. 编码 index（8 字节）
        auto bytes = reinterpret_cast<uint8_t*>(&index);
        for (auto i = 0; i < 8; i++) {
            Encode(bytes[i]);
        }

        // 3. 编码 value（valLen 字节）
        for (auto i = 0; i < valLen; i++) {
            Encode(val[i]);
        }
    }

    // 写入 END 标记
    __aicore__ void EncodeEnd()
    {
        Encode(END);
    }

    // 获取 ring buffer 有效数据区大小
    __aicore__ int64_t GetBufferSize() const { return size_; }

    // 获取当前 tail_ 位置（用于检测覆盖）
    __aicore__ int64_t GetTail() const { return tail_; }

    // 编码溢出警告信息
    template <typename NamePtrT>
    __aicore__ void EncodeOverflowWarning(NamePtrT name, int64_t totalBytes, int64_t bufferSize, int64_t tailBefore, int64_t tailAfter)
    {
        // 1. 编码 type
        Encode(static_cast<uint8_t>(OVERFLOW_WARNING));

        // 2. 编码 nameLen
        short nameLen = 0;
        while (name[nameLen]) ++nameLen;
        nameLen += 1;  // 包含 '\0'
        auto nlBytes = reinterpret_cast<uint8_t*>(&nameLen);
        Encode(nlBytes[0]);
        Encode(nlBytes[1]);

        // 3. 编码 name + '\0'
        for (short i = 0; name[i]; ++i) {
            Encode(static_cast<uint8_t>(name[i]));
        }
        Encode('\0');

        // 4. 编码 totalBytes
        auto tbBytes = reinterpret_cast<uint8_t*>(&totalBytes);
        for (size_t i = 0; i < sizeof(int64_t); ++i) {
            Encode(tbBytes[i]);
        }

        // 5. 编码 bufferSize
        auto bsBytes = reinterpret_cast<uint8_t*>(&bufferSize);
        for (size_t i = 0; i < sizeof(int64_t); ++i) {
            Encode(bsBytes[i]);
        }

        // 6. 编码 tailBefore
        auto tbBeforeBytes = reinterpret_cast<uint8_t*>(&tailBefore);
        for (size_t i = 0; i < sizeof(int64_t); ++i) {
            Encode(tbBeforeBytes[i]);
        }

        // 7. 编码 tailAfter
        auto tbAfterBytes = reinterpret_cast<uint8_t*>(&tailAfter);
        for (size_t i = 0; i < sizeof(int64_t); ++i) {
            Encode(tbAfterBytes[i]);
        }
    }

    INLINE LogContext* context() { return &ctx; }

#ifdef __TILE_FWK_HOST__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
    int Read(char* buf, size_t maxSize)
    {
        size_t size = 0;
        head_ = remote_->head_;
        if (tail_ < remote_->tail_) {
            // lose some data
            tail_ = remote_->tail_;
        }
        while (tail_ != head_) {
            auto type = Read<uint8_t>(tail_++);
            if (type == END) {
                if (size == 0)
                    continue;
                else
                    return size;
            } else if (maxSize == 0) {
                continue;
            }

            int n = 0;

            switch (type) {
                case TENSOR_HEADER: {
                    // 1. 读取 nameLen
                    auto nameLen = Read<short>(tail_);
                    tail_ += sizeof(short);

                    // 2. 读取 name + '\0'
                    std::string name = ReadString(tail_);
                    tail_ += nameLen;

                    // 3. 读取 begin 和 end
                    auto begin = Read<int64_t>(tail_);
                    tail_ += sizeof(int64_t);
                    auto end = Read<int64_t>(tail_);
                    tail_ += sizeof(int64_t);

                    // 4. 缓存 tensor 名称
                    lastTensorName_ = name;

                    // 5. 输出头部信息
                    n = snprintf_s(buf, maxSize, maxSize - 1, "tensor '%s', range=[%ld, %ld)\n",
                                   name.c_str(), begin, end);
                    break;
                }

                case INDEXED_FP32: {
                    auto index = Read<int64_t>(tail_);
                    tail_ += sizeof(int64_t);
                    float value = Read<float>(tail_);
                    tail_ += sizeof(float);
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                                   lastTensorName_.c_str(), index, value);
                    break;
                }

                case INDEXED_INT64: {
                    auto index = Read<int64_t>(tail_);
                    tail_ += sizeof(int64_t);
                    auto value = Read<int64_t>(tail_);
                    tail_ += sizeof(int64_t);
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %lld\n",
                                   lastTensorName_.c_str(), index, value);
                    break;
                }

                case INDEXED_BF16: {
                    auto index = Read<int64_t>(tail_);
                    tail_ += sizeof(int64_t);
                    uint16_t bits = Read<uint16_t>(tail_);
                    tail_ += sizeof(uint16_t);
                    float value = DecodeBf16(bits);
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                                   lastTensorName_.c_str(), index, value);
                    break;
                }

                case INDEXED_FP16: {
                    auto index = Read<int64_t>(tail_);
                    tail_ += sizeof(int64_t);
                    uint16_t bits = Read<uint16_t>(tail_);
                    tail_ += sizeof(uint16_t);
                    float value = DecodeF16(bits);
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                                   lastTensorName_.c_str(), index, value);
                    break;
                }

#if SUPPORT_FP8_HF8_PRINT
                case INDEXED_FP8E4M3: {
                    auto index = Read<int64_t>(tail_);
                    tail_ += sizeof(int64_t);
                    uint8_t bits = Read<uint8_t>(tail_);
                    tail_ += sizeof(uint8_t);
                    float value = DecodeFp8E4M3(bits);
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                                   lastTensorName_.c_str(), index, value);
                    break;
                }

                case INDEXED_FP8E5M2: {
                    auto index = Read<int64_t>(tail_);
                    tail_ += sizeof(int64_t);
                    uint8_t bits = Read<uint8_t>(tail_);
                    tail_ += sizeof(uint8_t);
                    float value = DecodeFp8E5M2(bits);
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                                   lastTensorName_.c_str(), index, value);
                    break;
                }

                case INDEXED_FP8E8M0: {
                    auto index = Read<int64_t>(tail_);
                    tail_ += sizeof(int64_t);
                    uint8_t bits = Read<uint8_t>(tail_);
                    tail_ += sizeof(uint8_t);
                    float value = DecodeFp8E8M0(bits);
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                                   lastTensorName_.c_str(), index, value);
                    break;
                }

                case INDEXED_HF8: {
                    auto index = Read<int64_t>(tail_);
                    tail_ += sizeof(int64_t);
                    uint8_t bits = Read<uint8_t>(tail_);
                    tail_ += sizeof(uint8_t);
                    float value = DecodeHf8(bits);
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                                   lastTensorName_.c_str(), index, value);
                    break;
                }
#endif

                case OVERFLOW_WARNING: {
                    // 1. 读取 nameLen
                    auto nameLen = Read<short>(tail_);
                    tail_ += sizeof(short);

                    // 2. 读取 name + '\0'
                    std::string name;
                    for (short i = 0; i < nameLen - 1; ++i) {
                        name += Read<char>(tail_++);
                    }
                    tail_++;  // skip '\0'

                    // 3. 读取 totalBytes
                    auto totalBytes = Read<int64_t>(tail_);
                    tail_ += 8;

                    // 4. 读取 bufferSize
                    auto bufferSize = Read<int64_t>(tail_);
                    tail_ += 8;

                    // 5. 读取 tailBefore
                    auto tailBefore = Read<int64_t>(tail_);
                    tail_ += 8;

                    // 6. 读取 tailAfter
                    auto tailAfter = Read<int64_t>(tail_);
                    tail_ += 8;

                    // 7. 计算推荐 buffer 大小（含 Remote 开销，向上取整到 KB，留 20% 余量）
                    int64_t recommendedBytes = totalBytes + 16;  // sizeof(Remote)
                    recommendedBytes = ((recommendedBytes * 12 / 10 + 1023) / 1024) * 1024;

                    // 8. 输出 warning 信息
                    n = snprintf_s(buf, maxSize, maxSize - 1,
                        "[WARNING] Ring buffer overflow detected for tensor '%s'! "
                        "This tensor's encoding (%ld bytes) caused earlier data to be overwritten. "
                        "Buffer tail advanced from %ld to %ld (delta=%ld bytes), indicating data loss. "
                        "Buffer data area size is %ld bytes. "
                        "To avoid overflow, set PRINT_BUFFER_SIZE >= %ld (%ld KB) "
                        "in framework/src/interface/machine/device/tilefwk/aicpu_common.h:52, "
                        "then rebuild and reinstall.\n",
                        name.c_str(), totalBytes, tailBefore, tailAfter, (tailAfter - tailBefore),
                        bufferSize, recommendedBytes, recommendedBytes / 1024);
                    break;
                }

                default: {
                    // 原有类型解码逻辑
                    auto valOff = tail_ + sizeof(short);
                    tail_ += Read<short>(tail_) + sizeof(short);
                    auto fmtOff = tail_ + sizeof(short);
                    std::string fmt = ReadString(fmtOff);
                    tail_ += Read<short>(tail_) + sizeof(short);

                    switch (type) {
                        case NORMAL:
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), 0);
                            break;
                        case FP32:
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), Read<float>(valOff));
                            break;
                        case INT:
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), Read<int64_t>(valOff));
                            break;
                        case CHAR:
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), Read<char>(valOff));
                            break;
                        case STRING:
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), ReadString(valOff).c_str());
                            break;
                        case POINTER:
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), Read<int64_t>(valOff));
                            break;
                        case BF16: {
                            uint16_t bits = Read<uint16_t>(valOff);
                            float fv = DecodeBf16(bits);
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
                            break;
                        }
                        case FP16: {
                            uint16_t bits = Read<uint16_t>(valOff);
                            float fv = DecodeF16(bits);
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
                            break;
                        }
#if SUPPORT_FP8_HF8_PRINT
                        case FP8E4M3: {
                            uint8_t bits = Read<uint8_t>(valOff);
                            float fv = DecodeFp8E4M3(bits);
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
                            break;
                        }
                        case FP8E5M2: {
                            uint8_t bits = Read<uint8_t>(valOff);
                            float fv = DecodeFp8E5M2(bits);
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
                            break;
                        }
                        case FP8E8M0: {
                            uint8_t bits = Read<uint8_t>(valOff);
                            float fv = DecodeFp8E8M0(bits);
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
                            break;
                        }
                        case HF8: {
                            uint8_t bits = Read<uint8_t>(valOff);
                            float fv = DecodeHf8(bits);
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
                            break;
                        }
#endif
                        default:
                            if (n) {
                                buf[0] = '?';
                                n = 1;
                            }
                            break;
                    }
                    break;
                }
            }
            buf += n;
            size += n;
            maxSize -= n;
        }
        return 0;
    }
#pragma GCC diagnostic pop
#endif

private:
    __aicore__ void EncodeFloatType(__gm__ const char** fmt, NodeTy ty, uint8_t* val, short valLen)
    {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);
        if (idx == -1) {
            return;
        }
        switch (curFmt[idx++]) {
            case 'f': {
                Encode(ty, val, valLen, *fmt, idx);
                break;
            }
            default:
                Encode(NORMAL, static_cast<uint8_t*>(nullptr), 0, *fmt, idx);
                break;
        }
        *fmt = *fmt + idx;
    }
    __aicore__ int64_t ParseNextFormat(__gm__ const char* fmt)
    {
        int64_t idx = 0;
        while (fmt[idx]) {
            if (fmt[idx] == '%') {
                if (fmt[idx + 1] == '%') {
                    idx += 2;
                } else {
                    break;
                }
            } else {
                idx++;
            }
        }

        if (!fmt[idx]) {
            return -1;
        }

        idx++;

        // skip fmt
        while (fmt[idx]) {
            if (fmt[idx] != '0' && fmt[idx] != '+' && fmt[idx] != '-' && fmt[idx] != ' ' && fmt[idx] != '#') {
                break;
            }
            idx++;
        }

        // width
        while (IsDigit(fmt[idx])) {
            idx++;
        }

        // precision
        if (fmt[idx] == '.') {
            idx++;
            while (IsDigit(fmt[idx])) {
                idx++;
            }
        }

        // Length
        if (fmt[idx] == 'l' || fmt[idx] == 'z' || fmt[idx] == 'h') {
            idx++;
            if (fmt[idx] == 'l')
                idx++;
        }

        return fmt[idx] ? idx : -1;
    }

    template <typename T>
    INLINE T Read(int64_t off)
    {
        T val;
        char tmp[sizeof(T)];
        for (size_t i = 0; i < sizeof(T); i++) {
            tmp[i] = data_[(off + i) % size_];
        }
        val = *reinterpret_cast<T*>(tmp);
        return val;
    }

#ifdef __TILE_FWK_HOST__
    std::string ReadString(int64_t off)
    {
        std::stringstream ss;
        while (off < head_) {
            auto c = Read<char>(off++);
            if (c == '\0')
                break;
            ss << c;
        }
        return ss.str();
    }
#endif

    __aicore__ void Encode(uint8_t val)
    {
        if (head_ == tail_ + size_) {
            while (Read<uint8_t>(tail_) != END) {
                auto segType = Read<uint8_t>(tail_);
                tail_++;

                switch (segType) {
                    case TENSOR_HEADER: {
                        // 跳过 nameLen + name + '\0'
                        auto nameLen = Read<short>(tail_);
                        tail_ += sizeof(short) + nameLen;
                        // 跳过 begin + end
                        tail_ += 8 + 8;
                        break;
                    }
                    case INDEXED_FP32:
                        tail_ += 8 + 4;  // index + float
                        break;
                    case INDEXED_INT64:
                        tail_ += 8 + 8;  // index + int64
                        break;
                    case INDEXED_BF16:
                    case INDEXED_FP16:
                        tail_ += 8 + 2;  // index + bf16/fp16
                        break;
#if SUPPORT_FP8_HF8_PRINT
                    case INDEXED_FP8E4M3:
                    case INDEXED_FP8E5M2:
                    case INDEXED_FP8E8M0:
                    case INDEXED_HF8:
                        tail_ += 8 + 1;  // index + fp8/hf8
                        break;
#endif
                    case OVERFLOW_WARNING: {
                        // 跳过 nameLenLen + name + '\0'
                        auto nl = Read<short>(tail_);
                        tail_ += sizeof(short) + nl;
                        // 跳过 totalBytes + bufferSize + tailBefore + tailAfter
                        tail_ += 8 + 8 + 8 + 8;
                        break;
                    }
                    default: {
                        // 原有类型：valLen + val + fmtLen + fmt
                        tail_ += Read<short>(tail_) + sizeof(short);
                        tail_ += Read<short>(tail_) + sizeof(short);
                        break;
                    }
                }
            }
            tail_++;
        }
        volatile __gm__ uint8_t* p = &data_[head_++ % size_];
        *p = val;
    }

    template <typename T>
    __aicore__ void Encode(NodeTy ty, const T* val, short valLen, __gm__ const char* fmt, int fmtLen)
    {
        Encode(ty);

        auto bytes = reinterpret_cast<uint8_t*>(&valLen);
        Encode(bytes[0]);
        Encode(bytes[1]);
        for (auto i = 0; i < valLen; i++) {
            Encode(val[i]);
        }

        fmtLen += 1; // pad '\0'
        bytes = reinterpret_cast<uint8_t*>(&fmtLen);
        Encode(bytes[0]);
        Encode(bytes[1]);
        for (auto i = 0; i < fmtLen - 1; i++) {
            Encode(fmt[i]);
        }
        Encode('\0');
    }

    INLINE size_t Length(__gm__ const char* str)
    {
        size_t n = 0;
        while (*str++) {
            n++;
        }
        return n;
    }

    INLINE bool IsDigit(char c) { return c >= '0' && c <= '9'; }

private:
    LogContext ctx;
    int64_t head_;
    int64_t tail_;
    int64_t size_;
    volatile __gm__ Remote* remote_;
    __gm__ uint8_t* data_;
#ifdef __TILE_FWK_HOST__
    std::string lastTensorName_;
#endif
};

#if defined(__TILE_FWK_AICORE__) && defined(TILEOP_UTILS_TUPLE_H)
constexpr size_t AICORE_PRINT_SHAPE_MAX_DIMS = 6;
template <size_t I, typename ShapeTuple>
INLINE void __AiCoreFillShapeDims(int64_t (&d)[AICORE_PRINT_SHAPE_MAX_DIMS], const ShapeTuple& shape)
{
    constexpr size_t n = Std::tuple_size<ShapeTuple>::value;
    constexpr size_t m = (n < AICORE_PRINT_SHAPE_MAX_DIMS) ? n : AICORE_PRINT_SHAPE_MAX_DIMS;
    if constexpr (I < m) {
        d[I] = static_cast<int64_t>(Std::get<I>(shape));
        __AiCoreFillShapeDims<I + 1>(d, shape);
    }
}

template <size_t N>
INLINE void __AiCoreLogShapeDims(LogContext* ctx, const int64_t (&d)[6])
{
    if constexpr (N == 1) {
        AiCoreLogF(ctx, "shape=[%ld]\n", d[0]);
    } else if constexpr (N == 2) {
        AiCoreLogF(ctx, "shape=[%ld,%ld]\n", d[0], d[1]);
    } else if constexpr (N == 3) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld]\n", d[0], d[1], d[2]);
    } else if constexpr (N == 4) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld]\n", d[0], d[1], d[2], d[3]);
    } else if constexpr (N == 5) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld,%ld]\n", d[0], d[1], d[2], d[3], d[4]);
    } else if constexpr (N == 6) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld,%ld,%ld]\n", d[0], d[1], d[2], d[3], d[4], d[5]);
    }
}

template <typename... Dims>
INLINE void AiCorePrintShape(LogContext* ctx, const TileOp::Shape<Dims...>& shape)
{
    constexpr size_t N = Std::tuple_size<TileOp::Shape<Dims...>>::value;
    if constexpr (N == 0 || N > AICORE_PRINT_SHAPE_MAX_DIMS) {
        return;
    }
    int64_t d[AICORE_PRINT_SHAPE_MAX_DIMS]{};
    __AiCoreFillShapeDims<0>(d, shape);
    __AiCoreLogShapeDims<Std::tuple_size<TileOp::Shape<Dims...>>::value>(ctx, d);
}
#endif

template <typename T, typename PtrT>
INLINE void __AiCorePrintTensorImpl(LogContext* ctx, PtrT data, int64_t end,
                                         int64_t begin, __gm__ const char* name)
{
    using ElemT = std::remove_cv_t<T>;
    auto* logger = reinterpret_cast<AicoreLogger*>(ctx);

    // 记录编码前的 buffer 状态
    int64_t tailBefore = logger->GetTail();

    // 预计算当前 tensor 的编码量
    int64_t count = end - begin;

    // 计算每元素编码开销（INDEXED_type + END）
    int64_t perElement = 0;
    if constexpr (std::is_floating_point_v<ElemT>) {
        perElement = 13 + 1;   // INDEXED_FP32 (13B) + END (1B) = 14
    } else if constexpr (std::is_integral_v<ElemT>) {
        perElement = 17 + 1;   // INDEXED_INT64 (17B) + END (1B) = 18
#if IS_AICORE
    } else if constexpr (std::is_same_v<ElemT, bfloat16_t>) {
        perElement = 11 + 1;   // INDEXED_BF16 (11B) + END (1B) = 12
    } else if constexpr (std::is_same_v<ElemT, half>) {
        perElement = 11 + 1;   // INDEXED_FP16 (11B) + END (1B) = 12
#if SUPPORT_FP8_HF8_PRINT
    } else if constexpr (std::is_same_v<ElemT, float8_e4m3_t> ||
                         std::is_same_v<ElemT, float8_e5m2_t> ||
                         std::is_same_v<ElemT, float8_e8m0_t> ||
                         std::is_same_v<ElemT, hifloat8_t>) {
        perElement = 10 + 1;   // type(1) + index(8) + value(1) + END(1) = 11B
#endif
#endif
    }

    // 计算 nameLen
    int64_t nameLen = 0;
    while (name[nameLen]) ++nameLen;
    nameLen += 1;  // 含 '\0'

    // 计算总字节数
    //    TENSOR_HEADER = type(1) + nameLen_field(2) + name(N) + begin(8) + end(8)
    //    完整编码 = TENSOR_HEADER + END + count × (INDEXED + END)
    int64_t headerSize = 1 + 2 + nameLen + 8 + 8;
    int64_t totalBytes = headerSize + 1 + (perElement * count);

    // 正常编码路径（无论是否溢出都执行）

    // 1. 编码 TENSOR_HEADER + END + Sync
    logger->EncodeTensorHeader(name, begin, end);
    logger->EncodeEnd();
    logger->Sync();

    // 2. 逐元素编码 INDEXED_* + END + Sync
    for (int64_t i = begin; i < end; ++i) {
        ElemT tmp = data[i];
        if constexpr (std::is_floating_point_v<ElemT>) {
            float v = static_cast<float>(tmp);
            logger->EncodeIndexed(INDEXED_FP32, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else if constexpr (std::is_integral_v<ElemT>) {
            int64_t v = static_cast<int64_t>(tmp);
            logger->EncodeIndexed(INDEXED_INT64, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else if constexpr (std::is_pointer_v<ElemT>) {           
            int64_t v = reinterpret_cast<int64_t>(tmp);
            logger->EncodeIndexed(INDEXED_INT64, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
#if IS_AICORE
        } else if constexpr (std::is_same_v<ElemT, bfloat16_t>) {
            uint16_t v = SafeBitCast<uint16_t>(tmp);
            logger->EncodeIndexed(INDEXED_BF16, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else if constexpr (std::is_same_v<ElemT, half>) {
            uint16_t v = SafeBitCast<uint16_t>(tmp);
            logger->EncodeIndexed(INDEXED_FP16, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
#if SUPPORT_FP8_HF8_PRINT
        } else if constexpr (std::is_same_v<ElemT, float8_e4m3_t>) {
            uint8_t v = SafeBitCast<uint8_t>(tmp);
            logger->EncodeIndexed(INDEXED_FP8E4M3, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else if constexpr (std::is_same_v<ElemT, float8_e5m2_t>) {
            uint8_t v = SafeBitCast<uint8_t>(tmp);
            logger->EncodeIndexed(INDEXED_FP8E5M2, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else if constexpr (std::is_same_v<ElemT, float8_e8m0_t>) {
            uint8_t v = SafeBitCast<uint8_t>(tmp);
            logger->EncodeIndexed(INDEXED_FP8E8M0, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else if constexpr (std::is_same_v<ElemT, hifloat8_t>) {
            uint8_t v = SafeBitCast<uint8_t>(tmp);
            logger->EncodeIndexed(INDEXED_HF8, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
#endif
#endif
        }
        logger->EncodeEnd();
        logger->Sync();
    }

    // 检查是否发生了覆盖
    int64_t tailAfter = logger->GetTail();
    bool overflowOccurred = (tailAfter > tailBefore);

    // 如果发生了覆盖，追加 Warning
    if (overflowOccurred) {
        logger->EncodeOverflowWarning(name, totalBytes, logger->GetBufferSize(), tailBefore, tailAfter);
        logger->EncodeEnd();
        logger->Sync();
    }

    logger->EncodeEnd();
    logger->Sync();
}

template <typename T>
INLINE void AiCorePrintGmTensor(LogContext* ctx, __gm__ const T* data,
                                      int64_t end, int64_t begin, __gm__ const char* name)
{
    __AiCorePrintTensorImpl<T>(ctx, data, end, begin, name);
}

#if IS_AICORE
template <typename T>
INLINE void AiCorePrintUbTensor(LogContext* ctx, __ubuf__ const T* data,
                                      int64_t end, int64_t begin, __ubuf__ const char* name)
{
    __AiCorePrintTensorImpl<T>(ctx, data, end, begin, name);
}

/**
 * 将 L1 (__cbuf__) 数据通过 DMA 搬运到 GM 暂存缓冲区。
 * 内部使用 copy_cbuf_to_gm，逻辑与 DynL1CopyOutND(cube_dyn.h:675) 一致。
 *
 * @param dst   GM 目标地址（必须 32B 对齐）
 * @param src   L1 源地址
 * @param count 元素数量
 *
 * @note DMA 最小搬运单位为 32 字节（BLOCK_SIZE，tileop_common.h:100）。
 * @note 搬运量向上取整到 32B 边界，尾部可能包含多余数据。
 */
template <typename T>
__aicore__ void L1RawCopyToGM(__gm__ T* dst, __cbuf__ const T* src, int64_t count)
{
    int64_t totalBytes = count * sizeof(T);
    if (totalBytes == 0) {
        return;
    }

    uint16_t nBurst;
    uint16_t lenBurst;
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;

    // 参照 DynL1CopyOutND(cube_dyn.h:679-693)：
    //   正常模式：lenBurst = TShape1 * sizeof(GMT) / BLOCK_SIZE
    //   fallback 模式（< 32B）：按字节搬运
    if (totalBytes >= 32) {
        // 正常模式：以 32B 块为单位的突发传输
        nBurst = 1;
        lenBurst = static_cast<uint16_t>((totalBytes + 31) / 32);
    } else {
        // Fallback 模式：与 DynL1CopyOutND(cube_dyn.h:684-691) 一致
        nBurst = 1;
        lenBurst = static_cast<uint16_t>(totalBytes);
        if (lenBurst == 0) {
            lenBurst = 1;
        }
    }

    copy_cbuf_to_gm(dst, src, 0 /*sid*/, nBurst, lenBurst, srcStride, dstStride);
}

/**
 * 打印 L1 (__cbuf__) tensor 数据（Copy-then-Print, Named 版本）。
 *
 * 完整流程：
 *   1. DMA 搬运：L1 data -> GM staging buffer
 *   2. 流水线同步：等待 MTE3 完成
 *   3. 逐值打印：从 GM staging 读值，编码到 print ring buffer
 *
 * @param ctx     LogContext 指针（来自 param->ctx）
 * @param data    L1 数据指针（__cbuf__ 地址空间）
 * @param end     打印结束索引（不包含）
 * @param begin   打印起始索引
 * @param staging GM 暂存缓冲区（通过 workspace offset 传入）
 * @param name    tensor 名称（__gm__ 字符串字面量）
 *
 * @note staging 缓冲区大小必须 >= (end - begin) * sizeof(T) 字节。
 * @note staging 地址建议 32B 对齐。
 * @note 应在 TLoad + wait_flag 完成后调用。
 * @note 使用 EVENT_ID7 进行 MTE3->S 同步（与 PipeSync() 一致，避免冲突）。
 */
template <typename T>
INLINE void AiCorePrintL1Tensor(LogContext* ctx, __cbuf__ const T* data,
                                 int64_t end, int64_t begin,
                                 __gm__ T* staging, __gm__ const char* name)
{
    int64_t count = end - begin;
    if (count <= 0) {
        return;
    }

    // Step 1: DMA L1 -> GM staging
    L1RawCopyToGM(staging, data + begin, count);

    // Step 2: 同步 MTE3 -> S（等待 DMA 完成）
    // 使用 EVENT_ID7：与 PipeSync()(aicore_entry.h) 一致，
    // 避免与 kernel 中常见的 MTE2->MTE1(EVENT_ID0) 冲突
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);

    // Step 3: 从 GM staging 打印（复用现有 GM 打印 API）
    AiCorePrintGmTensor<T>(ctx, staging, count, 0, name);
}
#endif
