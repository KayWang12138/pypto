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
#ifdef __DAV_V310
#define __FP8_EXPLICIT_TYPES_AVAILABLE__ 1
#endif
#endif

#define ENABLE_AICORE_PRINT 1

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
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

// FP8E4M3 (E4M3FN): 1 sign + 4 exp + 3 mant, bias=7
#define FP8E4M3_SIGN_MASK       0x80u
#define FP8E4M3_EXP_MASK        0x78u
#define FP8E4M3_MANT_MASK       0x07u
#define FP8E4M3_SIGN_SHIFT      7
#define FP8E4M3_EXP_SHIFT       3
#define FP8E4M3_HIDDEN_BIT      0x08u
#define FP8E4M3_EXP_BIAS        7
#define FP8E4M3_EXP_SATURATE    0xFu
#define FP8E4M3_TO_FP32_MANT_SHIFT 20
#define FP8E4M3_SUBNORMAL_FP32_EXP_BASE (FP32_EXP_BIAS - (FP8E4M3_EXP_BIAS - 1))

INLINE float DecodeFloat8E4M3(uint8_t bits)
{
    uint32_t sign = (bits & FP8E4M3_SIGN_MASK) >> FP8E4M3_SIGN_SHIFT;
    uint32_t exp  = (bits & FP8E4M3_EXP_MASK) >> FP8E4M3_EXP_SHIFT;
    uint32_t mant = bits & FP8E4M3_MANT_MASK;

    uint32_t sign32 = sign << FP32_SIGN_SHIFT;

    if (exp == 0 && mant == 0) {
        return SafeBitCast<float>(sign32);
    }

    if (exp == FP8E4M3_EXP_SATURATE) {
        uint32_t satBits = sign32 | ((FP32_EXP_BIAS + 7) << FP32_EXP_SHIFT)
                                  | (0x7u << FP8E4M3_TO_FP32_MANT_SHIFT);
        return SafeBitCast<float>(satBits);
    }

    uint32_t exp32;
    uint32_t mant32;
    if (exp == 0) {
        exp32 = FP8E4M3_SUBNORMAL_FP32_EXP_BASE;
        while ((mant & FP8E4M3_HIDDEN_BIT) == 0) {
            mant <<= 1;
            --exp32;
        }
        mant &= FP8E4M3_MANT_MASK;
        mant32 = mant << FP8E4M3_TO_FP32_MANT_SHIFT;
    } else {
        exp32 = exp - FP8E4M3_EXP_BIAS + FP32_EXP_BIAS;
        mant32 = mant << FP8E4M3_TO_FP32_MANT_SHIFT;
    }

    return SafeBitCast<float>(sign32 | (exp32 << FP32_EXP_SHIFT) | mant32);
}

// FP8E5M2: 1 sign + 5 exp + 2 mant, bias=15
#define FP8E5M2_SIGN_MASK       0x80u
#define FP8E5M2_EXP_MASK        0x7Cu
#define FP8E5M2_MANT_MASK       0x03u
#define FP8E5M2_SIGN_SHIFT      7
#define FP8E5M2_EXP_SHIFT       2
#define FP8E5M2_HIDDEN_BIT      0x04u
#define FP8E5M2_EXP_BIAS        15
#define FP8E5M2_EXP_INF_NAN     0x1Fu
#define FP8E5M2_TO_FP32_MANT_SHIFT 21
#define FP8E5M2_SUBNORMAL_FP32_EXP_BASE (FP32_EXP_BIAS - (FP8E5M2_EXP_BIAS - 1))

INLINE float DecodeFloat8E5M2(uint8_t bits)
{
    uint32_t sign = (bits & FP8E5M2_SIGN_MASK) >> FP8E5M2_SIGN_SHIFT;
    uint32_t exp  = (bits & FP8E5M2_EXP_MASK) >> FP8E5M2_EXP_SHIFT;
    uint32_t mant = bits & FP8E5M2_MANT_MASK;

    uint32_t sign32 = sign << FP32_SIGN_SHIFT;

    if (exp == 0 && mant == 0) {
        return SafeBitCast<float>(sign32);
    }

    if (exp == FP8E5M2_EXP_INF_NAN) {
        uint32_t exp32 = FP32_EXP_INF_NAN;
        uint32_t mant32 = mant << FP8E5M2_TO_FP32_MANT_SHIFT;
        return SafeBitCast<float>(sign32 | (exp32 << FP32_EXP_SHIFT) | mant32);
    }

    uint32_t exp32;
    uint32_t mant32;
    if (exp == 0) {
        exp32 = FP8E5M2_SUBNORMAL_FP32_EXP_BASE;
        while ((mant & FP8E5M2_HIDDEN_BIT) == 0) {
            mant <<= 1;
            --exp32;
        }
        mant &= FP8E5M2_MANT_MASK;
        mant32 = mant << FP8E5M2_TO_FP32_MANT_SHIFT;
    } else {
        exp32 = exp - FP8E5M2_EXP_BIAS + FP32_EXP_BIAS;
        mant32 = mant << FP8E5M2_TO_FP32_MANT_SHIFT;
    }

    return SafeBitCast<float>(sign32 | (exp32 << FP32_EXP_SHIFT) | mant32);
}

// FP8E8M0: 1 sign + 7 exp + 0 mant, bias=63
#define FP8E8M0_SIGN_MASK       0x80u
#define FP8E8M0_EXP_MASK        0x7Fu
#define FP8E8M0_SIGN_SHIFT      7
#define FP8E8M0_EXP_BIAS        63

INLINE float DecodeFloat8E8M0(uint8_t bits)
{
    uint32_t sign = (bits & FP8E8M0_SIGN_MASK) >> FP8E8M0_SIGN_SHIFT;
    uint32_t exp  = bits & FP8E8M0_EXP_MASK;

    if (exp == 0) {
        uint32_t sign32 = sign << FP32_SIGN_SHIFT;
        return SafeBitCast<float>(sign32);
    }

    uint32_t sign32 = sign << FP32_SIGN_SHIFT;
    uint32_t exp32 = exp - FP8E8M0_EXP_BIAS + FP32_EXP_BIAS;
    return SafeBitCast<float>(sign32 | (exp32 << FP32_EXP_SHIFT));
}

enum NodeTy { END, NORMAL, FP32, INT, CHAR, STRING, POINTER, BF16, FP16,
              TENSOR_HEADER,
              INDEXED_FP32, INDEXED_INT64, INDEXED_BF16, INDEXED_FP16,
              FP8E4M3, INDEXED_FP8E4M3,
              FP8E5M2, INDEXED_FP8E5M2,
              FP8E8M0, INDEXED_FP8E8M0,
              OVERFLOW_WARNING };

struct LogContext {
    void (*PrintInt)(LogContext* ctx, __gm__ const char** fmt, int64_t val);
    void (*PrintFp32)(LogContext* ctx, __gm__ const char** fmt, float val);
    void (*PrintBf16)(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits);
    void (*PrintFp16)(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits);
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
    void (*PrintFp8e4m3)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintFp8e5m2)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintFp8e8m0)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
#endif
    void (*Print)(LogContext* ctx, __gm__ const char* fmt);
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
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
    } else if constexpr (std::is_same_v<T, float8_e4m3_t>) {
        ctx->PrintFp8e4m3(ctx, fmt, SafeBitCast<uint8_t>(val));
    } else if constexpr (std::is_same_v<T, float8_e5m2_t>) {
        ctx->PrintFp8e5m2(ctx, fmt, SafeBitCast<uint8_t>(val));
    } else if constexpr (std::is_same_v<T, float8_e8m0_t>) {
        ctx->PrintFp8e8m0(ctx, fmt, SafeBitCast<uint8_t>(val));
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

#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
    static __aicore__ void __PrintFp8e4m3(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits)
    {
        auto self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->PrintFp8e4m3(fmt, rawBits);
        }
    }

    static __aicore__ void __PrintFp8e5m2(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits)
    {
        auto self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->PrintFp8e5m2(fmt, rawBits);
        }
    }

    static __aicore__ void __PrintFp8e8m0(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits)
    {
        auto self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) {
            self->PrintFp8e8m0(fmt, rawBits);
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
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
        ctx.PrintFp8e4m3 = __PrintFp8e4m3;
        ctx.PrintFp8e5m2 = __PrintFp8e5m2;
        ctx.PrintFp8e8m0 = __PrintFp8e8m0;
#endif
        ctx.Print = __Print;
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

#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
    __aicore__ void PrintFp8e4m3(__gm__ const char** fmt, uint8_t rawBits)
    {
        EncodeFloatType(fmt, FP8E4M3, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintFp8e5m2(__gm__ const char** fmt, uint8_t rawBits)
    {
        EncodeFloatType(fmt, FP8E5M2, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintFp8e8m0(__gm__ const char** fmt, uint8_t rawBits)
    {
        EncodeFloatType(fmt, FP8E8M0, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
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

    INLINE LogContext* context() { return &ctx; }

    template <typename NamePtrT>
    __aicore__ void EncodeTensorHeader(NamePtrT name, int64_t begin, int64_t end)
    {
        Encode(static_cast<uint8_t>(TENSOR_HEADER));
        short nameLen = 0;
        while (name[nameLen]) ++nameLen;
        nameLen += 1;
        auto nlBytes = reinterpret_cast<uint8_t*>(&nameLen);
        Encode(nlBytes[0]);
        Encode(nlBytes[1]);
        for (short i = 0; name[i]; ++i) {
            Encode(static_cast<uint8_t>(name[i]));
        }
        Encode('\0');
        auto bBytes = reinterpret_cast<uint8_t*>(&begin);
        for (size_t i = 0; i < sizeof(int64_t); ++i) {
            Encode(bBytes[i]);
        }
        auto eBytes = reinterpret_cast<uint8_t*>(&end);
        for (size_t i = 0; i < sizeof(int64_t); ++i) {
            Encode(eBytes[i]);
        }
    }

    __aicore__ void EncodeIndexed(NodeTy ty, int64_t index, const uint8_t* val, short valLen)
    {
        Encode(static_cast<uint8_t>(ty));
        auto idxBytes = reinterpret_cast<const uint8_t*>(&index);
        for (size_t i = 0; i < sizeof(int64_t); ++i) {
            Encode(idxBytes[i]);
        }
        for (short i = 0; i < valLen; ++i) {
            Encode(val[i]);
        }
    }

    __aicore__ void EncodeEnd()
    {
        Encode(static_cast<uint8_t>(END));
    }

    __aicore__ int64_t GetBufferSize() const { return size_; }

    __aicore__ int64_t GetTail() const { return tail_; }

    template <typename NamePtrT>
    __aicore__ void EncodeOverflowWarning(NamePtrT name, int64_t totalBytes, int64_t tailBefore, int64_t tailAfter)
    {
        Encode(static_cast<uint8_t>(OVERFLOW_WARNING));
        short nameLen = 0;
        while (name[nameLen]) ++nameLen;
        nameLen += 1;
        auto nlBytes = reinterpret_cast<uint8_t*>(&nameLen);
        Encode(nlBytes[0]);
        Encode(nlBytes[1]);
        for (short i = 0; name[i]; ++i) {
            Encode(static_cast<uint8_t>(name[i]));
        }
        Encode('\0');
        auto tbBytes = reinterpret_cast<uint8_t*>(&totalBytes);
        for (size_t i = 0; i < sizeof(int64_t); ++i) {
            Encode(tbBytes[i]);
        }
        auto bsBytes = reinterpret_cast<uint8_t*>(&size_);
        for (size_t i = 0; i < sizeof(int64_t); ++i) {
            Encode(bsBytes[i]);
        }
        auto tbBeforeBytes = reinterpret_cast<uint8_t*>(&tailBefore);
        for (size_t i = 0; i < sizeof(int64_t); ++i) {
            Encode(tbBeforeBytes[i]);
        }
        auto tbAfterBytes = reinterpret_cast<uint8_t*>(&tailAfter);
        for (size_t i = 0; i < sizeof(int64_t); ++i) {
            Encode(tbAfterBytes[i]);
        }
    }

#ifdef __TILE_FWK_HOST__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
    int Read(char* buf, size_t maxSize)
    {
        size_t size = 0;
        head_ = remote_->head_;
        if (tail_ < remote_->tail_) {
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
                    auto nameLen = Read<short>(tail_);
                    tail_ += sizeof(short);
                    std::string name;
                    for (short i = 0; i < nameLen - 1; ++i) {
                        name += Read<char>(tail_++);
                    }
                    tail_++;
                    auto begin = Read<int64_t>(tail_);
                    tail_ += 8;
                    auto end = Read<int64_t>(tail_);
                    tail_ += 8;
                    lastTensorName_ = name;
                    n = snprintf_s(buf, maxSize, maxSize - 1, "tensor '%s', range=[%ld, %ld)\n",
                                   name.c_str(), begin, end);
                    break;
                }
                case INDEXED_FP32: {
                    int64_t idx = Read<int64_t>(tail_);
                    tail_ += 8;
                    float fv = Read<float>(tail_);
                    tail_ += 4;
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                                   lastTensorName_.c_str(), idx, fv);
                    break;
                }
                case INDEXED_INT64: {
                    int64_t idx = Read<int64_t>(tail_);
                    tail_ += 8;
                    int64_t iv = Read<int64_t>(tail_);
                    tail_ += 8;
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %ld\n",
                                   lastTensorName_.c_str(), idx, iv);
                    break;
                }
                case INDEXED_BF16: {
                    int64_t idx = Read<int64_t>(tail_);
                    tail_ += 8;
                    uint16_t bits = Read<uint16_t>(tail_);
                    tail_ += 2;
                    float fv = DecodeBf16(bits);
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                                   lastTensorName_.c_str(), idx, fv);
                    break;
                }
                case INDEXED_FP16: {
                    int64_t idx = Read<int64_t>(tail_);
                    tail_ += 8;
                    uint16_t bits = Read<uint16_t>(tail_);
                    tail_ += 2;
                    float fv = DecodeF16(bits);
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                                   lastTensorName_.c_str(), idx, fv);
                    break;
                }
                case INDEXED_FP8E4M3: {
                    int64_t idx = Read<int64_t>(tail_);
                    tail_ += 8;
                    uint8_t bits = Read<uint8_t>(tail_);
                    tail_ += 1;
                    float fv = DecodeFloat8E4M3(bits);
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                                   lastTensorName_.c_str(), idx, fv);
                    break;
                }
                case INDEXED_FP8E5M2: {
                    int64_t idx = Read<int64_t>(tail_);
                    tail_ += 8;
                    uint8_t bits = Read<uint8_t>(tail_);
                    tail_ += 1;
                    float fv = DecodeFloat8E5M2(bits);
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                                   lastTensorName_.c_str(), idx, fv);
                    break;
                }
                case INDEXED_FP8E8M0: {
                    int64_t idx = Read<int64_t>(tail_);
                    tail_ += 8;
                    uint8_t bits = Read<uint8_t>(tail_);
                    tail_ += 1;
                    float fv = DecodeFloat8E8M0(bits);
                    n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                                   lastTensorName_.c_str(), idx, fv);
                    break;
                }
                case OVERFLOW_WARNING: {
                    auto nameLen = Read<short>(tail_);
                    tail_ += sizeof(short);
                    std::string name;
                    for (short i = 0; i < nameLen - 1; ++i) {
                        name += Read<char>(tail_++);
                    }
                    tail_++;  // skip '\0'
                    auto totalBytes = Read<int64_t>(tail_);
                    tail_ += 8;
                    auto bufferSize = Read<int64_t>(tail_);
                    tail_ += 8;
                    auto tailBefore = Read<int64_t>(tail_);
                    tail_ += 8;
                    auto tailAfter = Read<int64_t>(tail_);
                    tail_ += 8;
                    int64_t recommendedBytes = totalBytes + 16;  // sizeof(Remote)
                    recommendedBytes = ((recommendedBytes * 12 / 10 + 1023) / 1024) * 1024;
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
                        case FP8E4M3: {
                            uint8_t bits = Read<uint8_t>(valOff);
                            float fv = DecodeFloat8E4M3(bits);
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
                            break;
                        }
                        case FP8E5M2: {
                            uint8_t bits = Read<uint8_t>(valOff);
                            float fv = DecodeFloat8E5M2(bits);
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
                            break;
                        }
                        case FP8E8M0: {
                            uint8_t bits = Read<uint8_t>(valOff);
                            float fv = DecodeFloat8E8M0(bits);
                            n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
                            break;
                        }
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
                        auto nl = Read<short>(tail_);
                        tail_ += sizeof(short) + nl;
                        tail_ += 8 + 8;
                        break;
                    }
                    case INDEXED_FP32:  tail_ += 8 + 4; break;
                    case INDEXED_INT64: tail_ += 8 + 8; break;
                    case INDEXED_BF16:
                    case INDEXED_FP16:  tail_ += 8 + 2; break;
                    case INDEXED_FP8E4M3:
                    case INDEXED_FP8E5M2:
                    case INDEXED_FP8E8M0: tail_ += 8 + 1; break;
                    case OVERFLOW_WARNING: {
                        auto nl = Read<short>(tail_);
                        tail_ += sizeof(short) + nl;
                        tail_ += 8 + 8 + 8 + 8;  // totalBytes + bufferSize + tailBefore + tailAfter
                        break;
                    }
                    default:
                        tail_ += Read<short>(tail_) + sizeof(short);
                        tail_ += Read<short>(tail_) + sizeof(short);
                        break;
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
INLINE void __AiCorePrintTensorImpl(LogContext* ctx, PtrT data, int64_t end, int64_t begin = 0)
{
    using ElemT = std::remove_cv_t<T>;
    AiCoreLogF(ctx, "tensor data, range=[%ld, %ld)\n", begin, end);
    for (int64_t i = begin; i < end; ++i) {
        if constexpr (std::is_integral_v<ElemT>) {
            AiCoreLogF(ctx, "%lld\n", data[i]);
        } else if constexpr (std::is_floating_point_v<ElemT>) {
            AiCoreLogF(ctx, "%f\n", data[i]);
        } else if constexpr (std::is_pointer_v<ElemT>) {
            AiCoreLogF(ctx, "%p\n", data[i]);
#if IS_AICORE
        } else if constexpr (std::is_same_v<ElemT, bfloat16_t>) {
            AiCoreLogF(ctx, "%f\n", data[i]);
        } else if constexpr (std::is_same_v<ElemT, half>) {
            AiCoreLogF(ctx, "%f\n", data[i]);
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
        } else if constexpr (std::is_same_v<ElemT, float8_e4m3_t>) {
            AiCoreLogF(ctx, "%f\n", data[i]);
        } else if constexpr (std::is_same_v<ElemT, float8_e5m2_t>) {
            AiCoreLogF(ctx, "%f\n", data[i]);
        } else if constexpr (std::is_same_v<ElemT, float8_e8m0_t>) {
            AiCoreLogF(ctx, "%f\n", data[i]);
#endif
#endif
        }
    }
}

template <typename T, typename PtrT, typename NamePtrT>
INLINE void __AiCorePrintTensorImpl(LogContext* ctx, PtrT data, int64_t end,
                                    int64_t begin, NamePtrT name)
{
    using ElemT = std::remove_cv_t<T>;
    auto* logger = reinterpret_cast<AicoreLogger*>(ctx);

    int64_t tailBefore = logger->GetTail();

    int64_t count = end - begin;
    int64_t perElement = 0;
    if constexpr (std::is_floating_point_v<ElemT>) {
        perElement = 13 + 1;   // INDEXED_FP32 (13B) + END (1B) = 14
    } else if constexpr (std::is_integral_v<ElemT>) {
        perElement = 17 + 1;   // INDEXED_INT64 (17B) + END (1B) = 18
    } else if constexpr (std::is_pointer_v<ElemT>) {
        perElement = 17 + 1;
#if IS_AICORE
    } else if constexpr (std::is_same_v<ElemT, bfloat16_t>) {
        perElement = 11 + 1;
    } else if constexpr (std::is_same_v<ElemT, half>) {
        perElement = 11 + 1;
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
    } else if constexpr (std::is_same_v<ElemT, float8_e4m3_t>) {
        perElement = 10 + 1;
    } else if constexpr (std::is_same_v<ElemT, float8_e5m2_t>) {
        perElement = 10 + 1;
    } else if constexpr (std::is_same_v<ElemT, float8_e8m0_t>) {
        perElement = 10 + 1;
#endif
#endif
    }

    int64_t nameLen = 0;
    while (name[nameLen]) ++nameLen;
    nameLen += 1;
    int64_t headerSize = 1 + 2 + nameLen + 8 + 8;
    int64_t totalBytes = headerSize + 1 + (perElement * count);

    logger->EncodeTensorHeader(name, begin, end);
    logger->EncodeEnd();
    logger->Sync();

    for (int64_t i = begin; i < end; ++i) {
        if constexpr (std::is_floating_point_v<ElemT>) {
            float v = static_cast<float>(data[i]);
            logger->EncodeIndexed(INDEXED_FP32, i, reinterpret_cast<uint8_t*>(&v), sizeof(float));
        } else if constexpr (std::is_integral_v<ElemT>) {
            int64_t v = static_cast<int64_t>(data[i]);
            logger->EncodeIndexed(INDEXED_INT64, i, reinterpret_cast<uint8_t*>(&v), sizeof(int64_t));
        } else if constexpr (std::is_pointer_v<ElemT>) {
            int64_t v = reinterpret_cast<int64_t>(data[i]);
            logger->EncodeIndexed(INDEXED_INT64, i, reinterpret_cast<uint8_t*>(&v), sizeof(int64_t));
#if IS_AICORE
        } else if constexpr (std::is_same_v<ElemT, bfloat16_t>) {
            uint16_t rawBits = SafeBitCast<uint16_t>(data[i]);
            logger->EncodeIndexed(INDEXED_BF16, i, reinterpret_cast<uint8_t*>(&rawBits), sizeof(uint16_t));
        } else if constexpr (std::is_same_v<ElemT, half>) {
            uint16_t rawBits = SafeBitCast<uint16_t>(data[i]);
            logger->EncodeIndexed(INDEXED_FP16, i, reinterpret_cast<uint8_t*>(&rawBits), sizeof(uint16_t));
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
        } else if constexpr (std::is_same_v<ElemT, float8_e4m3_t>) {
            uint8_t rawBits = SafeBitCast<uint8_t>(data[i]);
            logger->EncodeIndexed(INDEXED_FP8E4M3, i, &rawBits, sizeof(uint8_t));
        } else if constexpr (std::is_same_v<ElemT, float8_e5m2_t>) {
            uint8_t rawBits = SafeBitCast<uint8_t>(data[i]);
            logger->EncodeIndexed(INDEXED_FP8E5M2, i, &rawBits, sizeof(uint8_t));
        } else if constexpr (std::is_same_v<ElemT, float8_e8m0_t>) {
            uint8_t rawBits = SafeBitCast<uint8_t>(data[i]);
            logger->EncodeIndexed(INDEXED_FP8E8M0, i, &rawBits, sizeof(uint8_t));
#endif
#endif
        }
        logger->EncodeEnd();
        logger->Sync();
    }

    int64_t tailAfter = logger->GetTail();
    bool overflowOccurred = (tailAfter > tailBefore);

    if (overflowOccurred) {
        logger->EncodeOverflowWarning(name, totalBytes, tailBefore, tailAfter);
        logger->EncodeEnd();
        logger->Sync();
    }
}

template <typename T>
INLINE void AiCorePrintGmTensor(LogContext* ctx, __gm__ const T* data, int64_t end, int64_t begin = 0)
{
    __AiCorePrintTensorImpl<T>(ctx, data, end, begin);
}

template <typename T>
INLINE void AiCorePrintGmTensorNamed(LogContext* ctx, __gm__ const T* data,
                                      int64_t end, int64_t begin, __gm__ const char* name)
{
    __AiCorePrintTensorImpl<T>(ctx, data, end, begin, name);
}

#if IS_AICORE
template <typename T>
INLINE void AiCorePrintUbTensor(LogContext* ctx, __ubuf__ const T* data, int64_t end, int64_t begin = 0)
{
    __AiCorePrintTensorImpl<T>(ctx, data, end, begin);
}

template <typename T>
INLINE void AiCorePrintUbTensorNamed(LogContext* ctx, __ubuf__ const T* data,
                                      int64_t end, int64_t begin, __ubuf__ const char* name)
{
    __AiCorePrintTensorImpl<T>(ctx, data, end, begin, name);
}

/**
 * 将 L1 (__cbuf__) 数据通过 DMA 搬运到 GM 暂存缓冲区。
 * 内部使用 copy_cbuf_to_gm，支持任意元素类型和数量。
 *
 * @param dst   GM 目标地址
 * @param src   L1 源地址
 * @param count 元素数量
 *
 * @note DMA 最小搬运单位为 32 字节，不足部分会向上取整，
 *       拷贝的数据可能比请求的略多（尾部填充字节），不影响正确性。
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

    if (totalBytes >= 32) {
        // 正常模式：以 32B 块为单位的突发传输
        nBurst = 1;
        lenBurst = static_cast<uint16_t>((totalBytes + 31) / 32);
    } else {
        // Fallback 模式：直接按字节传输
        nBurst = 1;
        lenBurst = static_cast<uint16_t>(totalBytes);
    }

    copy_cbuf_to_gm(dst, src, 0, nBurst, lenBurst, 0, 0);
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
 * @param staging GM 暂存缓冲区（通过 workspace + rawTensorDesc 传入）
 * @param name    tensor 名称（__gm__ 字符串字面量）
 *
 * @note staging 缓冲区大小必须 >= (end - begin) * sizeof(T) 字节。
 * @note 应在 TLoad + wait_flag 完成后调用。
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
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);

    // Step 3: 从 GM staging 打印（复用现有 GM 打印 API）
    AiCorePrintGmTensorNamed<T>(ctx, staging, count, 0, name);
}

/**
 * 打印 L1 (__cbuf__) tensor 数据（Copy-then-Print, 无名称版本）。
 *
 * @param ctx     LogContext 指针（来自 param->ctx）
 * @param data    L1 数据指针（__cbuf__ 地址空间）
 * @param end     打印结束索引（不包含）
 * @param begin   打印起始索引
 * @param staging GM 暂存缓冲区（通过 workspace + rawTensorDesc 传入）
 */
template <typename T>
INLINE void AiCorePrintL1Tensor(LogContext* ctx, __cbuf__ const T* data,
                                 int64_t end, int64_t begin,
                                 __gm__ T* staging)
{
    int64_t count = end - begin;
    if (count <= 0) {
        return;
    }

    L1RawCopyToGM(staging, data + begin, count);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);

    AiCorePrintGmTensor<T>(ctx, staging, count, 0);
}
#endif
