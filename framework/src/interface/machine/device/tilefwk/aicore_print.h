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

#include <type_traits>
#include "aicore_print_common.h"
#include "aicore_print_decode.h"

#define ENABLE_AICORE_PRINT 0

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#ifdef __TILE_FWK_AICORE__
#include "tileop/utils/layout.h"
#endif

// ==================== AicoreLogger 类 ====================

class AicoreLogger {
public:
    struct RemoteHeader { int64_t head_; int64_t tail_; };

    // ---------- 静态打印包装器宏 ----------
#define AICORE_STATIC_PRINT_WRAPPER(Name, Func, T) \
    static __aicore__ void Name(LogContext* ctx, __gm__ const char** fmt, T val) { \
        auto* self = reinterpret_cast<AicoreLogger*>(ctx); if (self) { self->Func(fmt, val); } \
    }

    AICORE_STATIC_PRINT_WRAPPER(StaticPrintInt64, PrintInt64, int64_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp32, PrintFp32, float)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintBf16, PrintBf16, uint16_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp16, PrintFp16, uint16_t)
    static __aicore__ void StaticPrintRaw(LogContext* ctx, __gm__ const char* fmt) {
        auto* self = reinterpret_cast<AicoreLogger*>(ctx); if (self) { self->PrintRaw(fmt); }
    }
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp8E4M3, PrintFp8E4M3, uint8_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp8E5M2, PrintFp8E5M2, uint8_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintFp8E8M0, PrintFp8E8M0, uint8_t)
    AICORE_STATIC_PRINT_WRAPPER(StaticPrintHf8, PrintHf8, uint8_t)
#undef AICORE_STATIC_PRINT_WRAPPER

    // ---------- 初始化和状态管理 ----------
    __aicore__ void Init(__gm__ uint8_t* buf, size_t n) {
        if (n < AicorePrintConst::MIN_BUFFER_TOTAL_SIZE) {
            overflowed_ = true;
            size_ = 0;
            remote_ = nullptr;
            data_ = nullptr;
            head_ = tail_ = 0;
            return;
        }
        remote_ = reinterpret_cast<volatile __gm__ RemoteHeader*>(buf);
        remote_->head_ = remote_->tail_ = 0;
        head_ = tail_ = 0;
        size_ = n - sizeof(RemoteHeader);
        data_ = buf + sizeof(RemoteHeader);
        overflowed_ = false;
        ctx_.PrintInt64 = StaticPrintInt64;
        ctx_.PrintFp32 = StaticPrintFp32;
        ctx_.PrintBf16 = StaticPrintBf16;
        ctx_.PrintFp16 = StaticPrintFp16;
        ctx_.PrintRaw = StaticPrintRaw;
        ctx_.PrintFp8E4M3 = StaticPrintFp8E4M3;
        ctx_.PrintFp8E5M2 = StaticPrintFp8E5M2;
        ctx_.PrintFp8E8M0 = StaticPrintFp8E8M0;
        ctx_.PrintHf8 = StaticPrintHf8;
    }

    __aicore__ __gm__ uint8_t* GetBuffer() const { return data_ - sizeof(RemoteHeader); }
    INLINE LogContext* Context() { return &ctx_; }

    // ---------- 数据打印方法 ----------
    __aicore__ void PrintInt64(__gm__ const char** fmt, int64_t val) {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);
        if (idx == -1) { return; }

        switch (curFmt[idx++]) {
            case 's': {
                auto tmp = reinterpret_cast<__gm__ const char*>(val);
                if (tmp == nullptr) { tmp = "<null>"; }
                EncodeTyped(AicorePrint::DataType::String,
                            reinterpret_cast<__gm__ const uint8_t*>(tmp), StringLength(tmp), *fmt, idx);
                break;
            }
            case 'd': case 'i': case 'x': case 'X': case 'o': case 'u':
                EncodeTyped(AicorePrint::DataType::Int64, reinterpret_cast<uint8_t*>(&val), sizeof(val), *fmt, idx);
                break;
            case 'p':
                EncodeTyped(AicorePrint::DataType::Pointer, reinterpret_cast<uint8_t*>(&val), sizeof(val), *fmt, idx);
                break;
            case 'c': {
                char c = static_cast<char>(val);
                EncodeTyped(AicorePrint::DataType::Char, reinterpret_cast<uint8_t*>(&c), 1, *fmt, idx);
                break;
            }
            default:
                EncodeTyped(AicorePrint::DataType::Normal, static_cast<uint8_t*>(nullptr), 0, *fmt, idx);
                break;
        }
        *fmt = *fmt + idx;
    }

    __aicore__ void PrintFp32(__gm__ const char** fmt, float val) {
        EncodeFloatType(fmt, AicorePrint::DataType::Fp32, reinterpret_cast<uint8_t*>(&val), sizeof(val));
    }

    __aicore__ void PrintBf16(__gm__ const char** fmt, uint16_t rawBits) {
        EncodeFloatType(fmt, AicorePrint::DataType::Bf16, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintFp16(__gm__ const char** fmt, uint16_t rawBits) {
        EncodeFloatType(fmt, AicorePrint::DataType::Fp16, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintFp8E4M3(__gm__ const char** fmt, uint8_t rawBits) {
        EncodeFloatType(fmt, AicorePrint::DataType::Fp8E4M3, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintFp8E5M2(__gm__ const char** fmt, uint8_t rawBits) {
        EncodeFloatType(fmt, AicorePrint::DataType::Fp8E5M2, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintFp8E8M0(__gm__ const char** fmt, uint8_t rawBits) {
        EncodeFloatType(fmt, AicorePrint::DataType::Fp8E8M0, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintHf8(__gm__ const char** fmt, uint8_t rawBits) {
        EncodeFloatType(fmt, AicorePrint::DataType::Hf8, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
    }

    __aicore__ void PrintRaw(__gm__ const char* str) {
        auto n = StringLength(str);
        if (n) { EncodeTyped(AicorePrint::DataType::Normal, reinterpret_cast<const __gm__ uint8_t*>(str), n, str, n); }
        Sync();
    }

    // ---------- 数据编码方法 ----------
    __aicore__ void EncodeTensorHeader(__gm__ const char* name, int64_t begin, int64_t end) {
        size_t nameLenRaw = StringLength(name) + 1;
        if (nameLenRaw > static_cast<size_t>(AicorePrintConst::SHORT_MAX_VALUE)) { return; }

        short nameLen = static_cast<short>(nameLenRaw);
        int64_t recordSize = 1 + sizeof(short) + nameLen + sizeof(int64_t) * 2 + 1;
        if (!CheckSpaceForRecord(recordSize)) { return; }

        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::TensorHeader));
        EncodeValue<short>(nameLen);
        for (short i = 0; i < nameLen; i++) { EncodeByte(static_cast<uint8_t>(name[i])); }
        EncodeValue<int64_t>(begin);
        EncodeValue<int64_t>(end);
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }

    __aicore__ void EncodeIndexed(AicorePrint::DataType ty, int64_t index, const uint8_t* val, short valLen) {
        int64_t recordSize = 1 + sizeof(int64_t) + valLen + 1;
        if (!CheckSpaceForRecord(recordSize)) { return; }

        EncodeByte(static_cast<uint8_t>(ty));
        EncodeValue<int64_t>(index);
        for (short i = 0; i < valLen; i++) { EncodeByte(val[i]); }
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }

    __aicore__ void EncodeOverflowWarning(int64_t bufferSize) {
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::OverflowWarning));
        EncodeValue<int64_t>(bufferSize);
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }

    template<typename PtrT>
    __aicore__ void EncodeTyped(AicorePrint::DataType ty, PtrT val, short valLen, __gm__ const char* fmt, int fmtLen) {
        short paddedFmtLen = fmtLen + 1;
        int64_t recordSize = 1 + sizeof(short) + valLen + sizeof(short) + paddedFmtLen + 1;
        if (!CheckSpaceForRecord(recordSize)) { return; }

        EncodeByte(static_cast<uint8_t>(ty));
        EncodeValue<short>(valLen);
        if constexpr (!std::is_same_v<std::remove_cv_t<PtrT>, std::nullptr_t>) {
            if (val) { for (short i = 0; i < valLen; i++) { EncodeByte(val[i]); } }
        } else { (void)val; }

        EncodeValue<short>(paddedFmtLen);
        for (int i = 0; i < fmtLen; i++) { EncodeByte(static_cast<uint8_t>(fmt[i])); }
        EncodeByte('\0');
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }

    template<typename PtrT>
    __aicore__ void EncodeFloatType(__gm__ const char** fmt, AicorePrint::DataType ty, PtrT val, short valLen) {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);
        if (idx == -1) { return; }
        if (curFmt[idx] == 'f') {
            EncodeTyped(ty, val, valLen, *fmt, idx + 1);
        } else {
            EncodeTyped(AicorePrint::DataType::Normal, nullptr, 0, *fmt, idx + 1);
        }
        *fmt = *fmt + idx + 1;
    }

    // ---------- 同步和读取方法 ----------
    __aicore__ void Sync() {
        if (remote_ == nullptr || data_ == nullptr) { return; }
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

#ifdef __TILE_FWK_HOST__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
    int Read(char* buf, size_t maxSize, uint32_t maxIterations = 1000) {
        if (remote_ == nullptr || data_ == nullptr || size_ == 0) { return 0; }
        size_t totalWritten = 0;
        head_ = remote_->head_;
        if (tail_ < remote_->tail_) { tail_ = remote_->tail_; }
        DecodeState state{tail_, head_, size_, data_, lastTensorName_};
        uint32_t iterationCount = 0;
        while (state.tail_ != state.head_ && iterationCount < maxIterations) {
            iterationCount++;
            AicorePrint::DataType type = static_cast<AicorePrint::DataType>(ReadDecodeByte(state, state.tail_++));
            if (type == AicorePrint::DataType::End) {
                if (totalWritten > 0) {
                    tail_ = state.tail_;
                    lastTensorName_ = state.lastTensorName_;
                    return static_cast<int>(totalWritten);
                }
                continue;
            }
            if (maxSize == 0) { continue; }

            int written = DecodeRecordImpl(state, type, buf, maxSize);
            if (written > 0) {
                buf += written;
                totalWritten += written;
                maxSize -= written;
            }
        }
        tail_ = state.tail_;
        lastTensorName_ = state.lastTensorName_;
        return 0;
    }
#pragma GCC diagnostic pop
#endif

private:
    // ---------- 空间检查 ----------
    __aicore__ bool CheckSpaceForRecord(int64_t recordSize) {
        if (overflowed_ || size_ == 0) { return false; }
        int64_t freeSpace = size_ - (head_ - tail_);
        int64_t requiredSpace = recordSize + AicorePrintConst::WARNING_RESERVE_SPACE;
        if (freeSpace < requiredSpace) {
            EncodeOverflowWarning(size_);
            Sync();
            overflowed_ = true;
            return false;
        }
        return true;
    }

    // ---------- 底层编码辅助 ----------
    __aicore__ void EncodeByte(uint8_t val) {
        volatile __gm__ uint8_t* p = &data_[head_++ % size_];
        *p = val;
    }

    template<typename T>
    __aicore__ void EncodeValue(T value) {
        auto* bytes = reinterpret_cast<uint8_t*>(&value);
        for (size_t i = 0; i < sizeof(T); i++) { EncodeByte(bytes[i]); }
    }

    // ---------- 格式解析辅助 ----------
    INLINE bool IsFormatFlagChar(char c) { return c == '0' || c == '+' || c == '-' || c == ' ' || c == '#'; }

    INLINE int64_t SkipFormatFlags(__gm__ const char* fmt, int64_t idx) {
        while (fmt[idx] && IsFormatFlagChar(fmt[idx])) { idx++; }
        return idx;
    }

    INLINE int64_t SkipLengthModifier(__gm__ const char* fmt, int64_t idx) {
        if (fmt[idx] == 'l' || fmt[idx] == 'z' || fmt[idx] == 'h') { idx++; if (fmt[idx] == 'l') idx++; }
        return idx;
    }

    __aicore__ int64_t ParseNextFormat(__gm__ const char* fmt) {
        int64_t idx = 0;
        while (fmt[idx]) {
            if (fmt[idx] == '%') {
                if (fmt[idx + 1] == '%') { idx += 2; } else { break; }
            } else { idx++; }
        }
        if (!fmt[idx]) { return -1; }
        idx++;
        idx = SkipFormatFlags(fmt, idx);
        while (IsDigit(fmt[idx])) { idx++; }
        if (fmt[idx] == '.') { idx++; while (IsDigit(fmt[idx])) { idx++; } }
        idx = SkipLengthModifier(fmt, idx);
        return fmt[idx] ? idx : -1;
    }

    INLINE size_t StringLength(__gm__ const char* str) { size_t n = 0; while (*str++) { n++; } return n; }
    INLINE bool IsDigit(char c) { return c >= '0' && c <= '9'; }

    // ---------- 成员变量 ----------
    LogContext ctx_{};
    int64_t head_ = 0;
    int64_t tail_ = 0;
    int64_t size_ = 0;
    volatile __gm__ RemoteHeader* remote_ = nullptr;
    __gm__ uint8_t* data_ = nullptr;
    bool overflowed_ = false;
#ifdef __TILE_FWK_HOST__
    std::string lastTensorName_{};
#endif
};

// ==================== 内部辅助函数 ====================

template <typename T>
INLINE void DispatchPrint(LogContext* ctx, __gm__ const char** fmt, T val) {
    if constexpr (std::is_integral_v<T>) {
        ctx->PrintInt64(ctx, fmt, static_cast<int64_t>(val));
    } else if constexpr (std::is_floating_point_v<T>) {
        ctx->PrintFp32(ctx, fmt, static_cast<float>(val));
    } else if constexpr (std::is_pointer_v<T>) {
        ctx->PrintInt64(ctx, fmt, reinterpret_cast<int64_t>(val));
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
INLINE void AiCoreLogF(LogContext* ctx, __gm__ const char* fmt, Ts... args) {
    if (ctx && fmt) {
        (DispatchPrint(ctx, &fmt, args), ...);
        ctx->PrintRaw(ctx, fmt);
    }
}

#if defined(__TILE_FWK_AICORE__) && defined(TILEOP_UTILS_TUPLE_H)

template <size_t I, typename ShapeTuple>
INLINE void FillShapeDims(int64_t (&dims)[AicorePrintConst::MAX_SHAPE_DIMS], const ShapeTuple& shape) {
    constexpr size_t n = Std::tuple_size<ShapeTuple>::value;
    constexpr size_t m = (n < AicorePrintConst::MAX_SHAPE_DIMS) ? n : AicorePrintConst::MAX_SHAPE_DIMS;
    if constexpr (I < m) {
        dims[I] = static_cast<int64_t>(Std::get<I>(shape));
        FillShapeDims<I + 1>(dims, shape);
    }
}

template <size_t N>
INLINE void LogShapeDims(LogContext* ctx, const int64_t (&dims)[AicorePrintConst::MAX_SHAPE_DIMS]) {
    if constexpr (N == 1) { AiCoreLogF(ctx, "shape=[%ld]\n", dims[0]); }
    else if constexpr (N == 2) { AiCoreLogF(ctx, "shape=[%ld,%ld]\n", dims[0], dims[1]); }
    else if constexpr (N == 3) { AiCoreLogF(ctx, "shape=[%ld,%ld,%ld]\n", dims[0], dims[1], dims[2]); }
    else if constexpr (N == 4) { AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld]\n", dims[0], dims[1], dims[2], dims[3]); }
    else if constexpr (N == 5) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld,%ld]\n", dims[0], dims[1], dims[2], dims[3], dims[4]);
    }
    else if constexpr (N == 6) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld,%ld,%ld]\n", dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
    }
}
#endif

template <typename T, typename PtrT>
INLINE void PrintTensorImpl(LogContext* ctx, PtrT data, int64_t end, int64_t begin, __gm__ const char* name) {
    using ElemT = std::remove_cv_t<T>;
    auto* logger = reinterpret_cast<AicoreLogger*>(ctx);
    logger->EncodeTensorHeader(name, begin, end);
    logger->Sync();

    for (int64_t i = begin; i < end; ++i) {
        ElemT tmp = data[i];
        if constexpr (std::is_floating_point_v<ElemT>) {
            float v = static_cast<float>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<float>::Type, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else if constexpr (std::is_integral_v<ElemT>) {
            int64_t v = static_cast<int64_t>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<int64_t>::Type, i, reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else if constexpr (sizeof(ElemT) == 2) {
            auto bits = SafeBitCast<uint16_t>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<ElemT>::Type, i, reinterpret_cast<uint8_t*>(&bits), sizeof(bits));
        } else if constexpr (sizeof(ElemT) == 1) {
            uint8_t bits = SafeBitCast<uint8_t>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<ElemT>::Type, i, reinterpret_cast<uint8_t*>(&bits), sizeof(bits));
        }
        logger->Sync();
    }
}

#if IS_AICORE
template <typename T>
__aicore__ void L1RawCopyToGM(__gm__ T* dst, __cbuf__ const T* src, int64_t count) {
    int64_t totalBytes = count * sizeof(T);
    if (totalBytes == 0) { return; }
    uint16_t nBurst = 1;
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    uint16_t lenBurst = (totalBytes >= 32) ? static_cast<uint16_t>((totalBytes + 31) / 32)
                                           : static_cast<uint16_t>(totalBytes > 0 ? totalBytes : 1);
    copy_cbuf_to_gm(dst, src, 0, nBurst, lenBurst, srcStride, dstStride);
}
#endif

// ==================== 对外接口函数 ====================

#if defined(__TILE_FWK_AICORE__) && defined(TILEOP_UTILS_TUPLE_H)
template <typename... Dims>
INLINE void AiCorePrintShape(LogContext* ctx, const TileOp::Shape<Dims...>& shape) {
    constexpr size_t N = Std::tuple_size<TileOp::Shape<Dims...>>::value;
    if constexpr (N == 0 || N > AicorePrintConst::MAX_SHAPE_DIMS) { return; }
    int64_t dims[AicorePrintConst::MAX_SHAPE_DIMS]{};
    FillShapeDims<0>(dims, shape);
    LogShapeDims<N>(ctx, dims);
}
#endif

template <typename T>
INLINE void AiCorePrintGmTensor(LogContext* ctx, __gm__ const T* data, int64_t end, int64_t begin, __gm__ const char* name) {
    PrintTensorImpl<T>(ctx, data, end, begin, name);
}

#if IS_AICORE
template <typename T>
INLINE void AiCorePrintUbTensor(LogContext* ctx, __ubuf__ const T* data, int64_t end, int64_t begin, __ubuf__ const char* name) {
    PrintTensorImpl<T>(ctx, data, end, begin, name);
}

template <typename T>
INLINE void AiCorePrintL1Tensor(LogContext* ctx, __cbuf__ const T* data, int64_t end,
                                int64_t begin, __gm__ T* staging, __gm__ const char* name) {
    int64_t count = end - begin;
    if (count <= 0) { return; }
    pipe_barrier(PIPE_ALL);
    L1RawCopyToGM(staging, data + begin, count);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    pipe_barrier(PIPE_ALL);
    // 强制清除 staging buffer 的 cache line（CACHELINE_OUT = clean + invalidate）
    // 这确保 Scalar Processor 能看到 DMA 写入的新数据
    int64_t totalBytes = count * sizeof(T);
    int64_t offset = 0;
    while (offset < totalBytes) {
        dcci((__gm__ uint8_t*)staging + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
        offset += CACHE_LINE_SIZE;
    }
    AiCorePrintGmTensor<T>(ctx, staging, count, 0, name);
}
#endif