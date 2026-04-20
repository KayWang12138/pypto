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
 * \brief AiCore环形缓冲区日志器 - 设备侧编码/Host侧解码
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <cstring>
#include "aikernel_data.h"
#include "aicore_print_consts.h"
#include "aicore_print_decode.h"

#ifdef __TILE_FWK_AICORE__
#include "tileop/utils/layout.h"
#endif

#define ENABLE_AICORE_PRINT 0

#ifdef __TILE_FWK_HOST__
#include <string>
#include <sstream>
#include <securec.h>
#endif

// ============================================================================
// Section 8: 数据类型枚举（类型安全）
// ============================================================================

namespace AicorePrint {
    enum class DataType : uint8_t {
        End                 = 0,
        Normal              = 1,
        Fp32                = 2,
        Int64               = 3,
        Char                = 4,
        String              = 5,
        Pointer             = 6,
        Bf16                = 7,
        Fp16                = 8,
        TensorHeader        = 9,
        IndexedFp32         = 10,
        IndexedInt64        = 11,
        IndexedBf16         = 12,
        IndexedFp16         = 13,
        OverflowWarning     = 14,
#if SUPPORT_FP8_HF8_PRINT
        Fp8E4M3             = 15,
        Fp8E5M2             = 16,
        Fp8E8M0             = 17,
        Hf8                 = 18,
        IndexedFp8E4M3      = 19,
        IndexedFp8E5M2      = 20,
        IndexedFp8E8M0      = 21,
        IndexedHf8          = 22,
#endif
    };
}

// ============================================================================
// Section 9: LogContext 结构定义
// ============================================================================

/**
 * \struct LogContext
 * \brief 打印上下文，桥接AicoreLogger与外部调用者
 */
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

// ============================================================================
// Section 12: AicoreLogger 类定义
// ============================================================================

class AicoreLogger {
public:
    struct RemoteHeader {
        int64_t head_;
        int64_t tail_;
    };

    static __aicore__ void StaticPrintInt64(LogContext* ctx, __gm__ const char** fmt, int64_t val) {
        auto* self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) { self->PrintInt64(fmt, val); }
    }
    
    static __aicore__ void StaticPrintFp32(LogContext* ctx, __gm__ const char** fmt, float val) {
        auto* self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) { self->PrintFp32(fmt, val); }
    }
    
    static __aicore__ void StaticPrintBf16(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits) {
        auto* self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) { self->PrintBf16(fmt, rawBits); }
    }
    
    static __aicore__ void StaticPrintFp16(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits) {
        auto* self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) { self->PrintFp16(fmt, rawBits); }
    }
    
    static __aicore__ void StaticPrintRaw(LogContext* ctx, __gm__ const char* fmt) {
        auto* self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) { self->PrintRaw(fmt); }
    }
    
#if SUPPORT_FP8_HF8_PRINT
    static __aicore__ void StaticPrintFp8E4M3(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits) {
        auto* self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) { self->PrintFp8E4M3(fmt, rawBits); }
    }
    
    static __aicore__ void StaticPrintFp8E5M2(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits) {
        auto* self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) { self->PrintFp8E5M2(fmt, rawBits); }
    }
    
    static __aicore__ void StaticPrintFp8E8M0(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits) {
        auto* self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) { self->PrintFp8E8M0(fmt, rawBits); }
    }
    
    static __aicore__ void StaticPrintHf8(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits) {
        auto* self = reinterpret_cast<AicoreLogger*>(ctx);
        if (self) { self->PrintHf8(fmt, rawBits); }
    }
#endif

    __aicore__ void Init(__gm__ uint8_t* buf, size_t n) {
        if (n < AicorePrintConst::MIN_BUFFER_TOTAL_SIZE) {
            overflowed_ = true;
            size_ = 0;
            return;
        }
        
        remote_ = reinterpret_cast<volatile __gm__ RemoteHeader*>(buf);
        remote_->head_ = remote_->tail_ = 0;
        head_ = tail_ = 0;
        size_ = n - sizeof(RemoteHeader);
        data_ = buf + sizeof(RemoteHeader);
        
        overflowed_ = false;
        
        ctx_.PrintInt64   = StaticPrintInt64;
        ctx_.PrintFp32    = StaticPrintFp32;
        ctx_.PrintBf16    = StaticPrintBf16;
        ctx_.PrintFp16    = StaticPrintFp16;
        ctx_.PrintRaw     = StaticPrintRaw;
#if SUPPORT_FP8_HF8_PRINT
        ctx_.PrintFp8E4M3 = StaticPrintFp8E4M3;
        ctx_.PrintFp8E5M2 = StaticPrintFp8E5M2;
        ctx_.PrintFp8E8M0 = StaticPrintFp8E8M0;
        ctx_.PrintHf8     = StaticPrintHf8;
#endif
    }

    __aicore__ __gm__ uint8_t* GetBuffer() const { 
        return data_ - sizeof(RemoteHeader); 
    }
    
    __aicore__ int64_t GetBufferSize() const { return size_; }
    __aicore__ int64_t GetTail() const { return tail_; }
    
    /**
     * \brief 重置overflow状态，允许继续写入
     * 
     * 当ring buffer溢出后，overflowed_被设为true，阻止后续写入。
     * 调用此接口可清除overflow状态，允许在分配新buffer后继续写入。
     * 
     * \note 调用前应确保buffer已重新分配或扩容，否则会再次触发overflow
     */
    __aicore__ void ClearOverflow() { overflowed_ = false; }
    
    /**
     * \brief 查询当前overflow状态
     * \return true if ring buffer已溢出，false otherwise
     */
    __aicore__ bool IsOverflowed() const { return overflowed_; }
    
    INLINE LogContext* Context() { return &ctx_; }

    __aicore__ void EncodeTensorHeader(__gm__ const char* name, int64_t begin, int64_t end) {
        size_t nameLenRaw = StringLength(name) + 1;
        
        if (nameLenRaw > static_cast<size_t>(AicorePrintConst::SHORT_MAX_VALUE)) {
            return;
        }
        
        short nameLen = static_cast<short>(nameLenRaw);
        int64_t recordSize = 1 + sizeof(short) + nameLen + sizeof(int64_t) * 2 + 1;
        
        if (!CheckSpaceForRecord(recordSize)) {
            return;
        }
        
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::TensorHeader));
        EncodeValue<short>(nameLen);
        
        for (short i = 0; i < nameLen; i++) {
            EncodeByte(static_cast<uint8_t>(name[i]));
        }
        
        EncodeValue<int64_t>(begin);
        EncodeValue<int64_t>(end);
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }
    
    __aicore__ void EncodeIndexed(AicorePrint::DataType ty, int64_t index, const uint8_t* val, short valLen) {
        int64_t recordSize = 1 + sizeof(int64_t) + valLen + 1;
        
        if (!CheckSpaceForRecord(recordSize)) {
            return;
        }
        
        EncodeByte(static_cast<uint8_t>(ty));
        EncodeValue<int64_t>(index);
        
        for (short i = 0; i < valLen; i++) {
            EncodeByte(val[i]);
        }
        
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }
    
    __aicore__ void EncodeOverflowWarning(int64_t bufferSize) {
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::OverflowWarning));
        EncodeValue<int64_t>(bufferSize);
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }

    __aicore__ void PrintInt64(__gm__ const char** fmt, int64_t val) {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);
        if (idx == -1) return;
        
        switch (curFmt[idx++]) {
            case 's': {
                auto tmp = reinterpret_cast<__gm__ const char*>(val);
                if (tmp == nullptr) tmp = "<null>";
                EncodeTyped(AicorePrint::DataType::String, reinterpret_cast<__gm__ const uint8_t*>(tmp), 
                           StringLength(tmp), *fmt, idx);
                break;
            }
            case 'd': case 'i': case 'x': case 'X': case 'o': case 'u': {
                EncodeTyped(AicorePrint::DataType::Int64, reinterpret_cast<uint8_t*>(&val), sizeof(val), *fmt, idx);
                break;
            }
            case 'p': {
                EncodeTyped(AicorePrint::DataType::Pointer, reinterpret_cast<uint8_t*>(&val), sizeof(val), *fmt, idx);
                break;
            }
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

#if SUPPORT_FP8_HF8_PRINT
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
#endif

    __aicore__ void PrintRaw(__gm__ const char* str) {
        auto n = StringLength(str);
        if (n) {
            EncodeTyped(AicorePrint::DataType::Normal, reinterpret_cast<const __gm__ uint8_t*>(str), n, str, n);
        }
        Sync();
    }

    __aicore__ void Sync() {
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

/**
     * \brief 从ring buffer读取解码数据到输出缓冲区
     * 
     * \param buf 输出缓冲区指针
     * \param maxSize 输出缓冲区最大容量（字节）
     * \param maxIterations 最大迭代次数限制，防止极端情况卡死（默认1000）
     * \return 成功读取的字节数，无数据时返回0
     * 
     * \note 添加maxIterations参数防止head_/tail_异常导致无限循环
     */
    int Read(char* buf, size_t maxSize, uint32_t maxIterations = 1000) {
        size_t totalWritten = 0;
        head_ = remote_->head_;
        
        if (tail_ < remote_->tail_) {
            tail_ = remote_->tail_;
        }
        
        uint32_t iterationCount = 0;
        while (tail_ != head_ && iterationCount < maxIterations) {
            iterationCount++;
            AicorePrint::DataType type = static_cast<AicorePrint::DataType>(ReadByte(tail_++));
            
            if (type == AicorePrint::DataType::End) {
                if (totalWritten > 0) {
                    return static_cast<int>(totalWritten);
                }
                continue;
            }
            
            if (maxSize == 0) {
                continue;
            }
            
            int written = DecodeRecord(type, buf, maxSize);
            if (written > 0) {
                buf += written;
                totalWritten += written;
                maxSize -= written;
            }
        }
        
        return 0;
    }

#pragma GCC diagnostic pop
#endif

private:
    __aicore__ bool CheckSpaceForRecord(int64_t recordSize) {
        if (overflowed_) {
            return false;
        }
        
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
    
    __aicore__ void EncodeByte(uint8_t val) {
        volatile __gm__ uint8_t* p = &data_[head_++ % size_];
        *p = val;
    }
    
    template<typename T>
    __aicore__ void EncodeValue(T value) {
        auto* bytes = reinterpret_cast<uint8_t*>(&value);
        for (size_t i = 0; i < sizeof(T); i++) {
            EncodeByte(bytes[i]);
        }
    }
    
    template<typename PtrT>
    __aicore__ void EncodeTyped(AicorePrint::DataType ty, PtrT val, short valLen, 
                                __gm__ const char* fmt, int fmtLen) {
        short paddedFmtLen = fmtLen + 1;
        int64_t recordSize = 1 + sizeof(short) + valLen + sizeof(short) + paddedFmtLen + 1;
        
        if (!CheckSpaceForRecord(recordSize)) {
            return;
        }
        
        EncodeByte(static_cast<uint8_t>(ty));
        EncodeValue<short>(valLen);
        
        if constexpr (!std::is_same_v<std::remove_cv_t<PtrT>, std::nullptr_t>) {
            if (val) {
                for (short i = 0; i < valLen; i++) {
                    EncodeByte(val[i]);
                }
            }
        } else {
            (void)val;
        }

        
        EncodeValue<short>(paddedFmtLen);
        for (int i = 0; i < fmtLen; i++) {
            EncodeByte(static_cast<uint8_t>(fmt[i]));
        }
        EncodeByte('\0');
        EncodeByte(static_cast<uint8_t>(AicorePrint::DataType::End));
    }
    
    template<typename PtrT>
    __aicore__ void EncodeFloatType(__gm__ const char** fmt, AicorePrint::DataType ty, 
                                    PtrT val, short valLen) {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);
        if (idx == -1) return;
        
        if (curFmt[idx] == 'f') {
            EncodeTyped(ty, val, valLen, *fmt, idx + 1);
        } else {
            EncodeTyped(AicorePrint::DataType::Normal, nullptr, 0, *fmt, idx + 1);
        }
        *fmt = *fmt + idx + 1;
    }

    /**
     * \brief 检查字符是否为printf格式标志字符(0, +, -, 空格, #)
     */
    INLINE bool IsFormatFlagChar(char c) {
        return c == '0' || c == '+' || c == '-' || c == ' ' || c == '#';
    }
    
    /**
     * \brief 跳过printf格式标志字符，返回跳过后的索引位置
     */
    INLINE int64_t SkipFormatFlags(__gm__ const char* fmt, int64_t idx) {
        while (fmt[idx] && IsFormatFlagChar(fmt[idx])) {
            idx++;
        }
        return idx;
    }
    
    /**
     * \brief 跳过printf长度修饰符(l, z, h)，返回跳过后的索引位置
     */
    INLINE int64_t SkipLengthModifier(__gm__ const char* fmt, int64_t idx) {
        if (fmt[idx] == 'l' || fmt[idx] == 'z' || fmt[idx] == 'h') {
            idx++;
            if (fmt[idx] == 'l') idx++;
        }
        return idx;
    }
    
    __aicore__ int64_t ParseNextFormat(__gm__ const char* fmt) {
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
        
        if (!fmt[idx]) return -1;
        idx++;
        
        idx = SkipFormatFlags(fmt, idx);
        while (IsDigit(fmt[idx])) idx++;
        
        if (fmt[idx] == '.') {
            idx++;
            while (IsDigit(fmt[idx])) idx++;
        }
        
        idx = SkipLengthModifier(fmt, idx);
        return fmt[idx] ? idx : -1;
    }
    
    INLINE size_t StringLength(__gm__ const char* str) {
        size_t n = 0;
        while (*str++) { n++; }
        return n;
    }
    
    INLINE bool IsDigit(char c) { return c >= '0' && c <= '9'; }
    
    __aicore__ uint8_t ReadByte(int64_t off) {
        return data_[off % size_];
    }
    
    template <typename T>
    __aicore__ T ReadValue(int64_t off) {
        T val{};
        auto* bytes = reinterpret_cast<uint8_t*>(&val);
        for (size_t i = 0; i < sizeof(T); i++) {
            bytes[i] = ReadByte(off + i);
        }
        return val;
    }
    
#ifdef __TILE_FWK_HOST__
    std::string ReadString(int64_t off) {
        std::string result;
        result.reserve(64);
        while (off < head_) {
            char c = ReadValue<char>(off++);
            if (c == '\0') break;
            result.push_back(c);
        }
        return result;
    }
    
    int DecodeRecord(AicorePrint::DataType type, char* buf, size_t maxSize) {
        switch (type) {
            case AicorePrint::DataType::TensorHeader:
                return DecodeTensorHeader(buf, maxSize);
            case AicorePrint::DataType::IndexedFp32:
                return DecodeIndexedFp32(buf, maxSize);
            case AicorePrint::DataType::IndexedInt64:
                return DecodeIndexedInt64(buf, maxSize);
            case AicorePrint::DataType::IndexedBf16:
                return DecodeIndexedBf16(buf, maxSize);
            case AicorePrint::DataType::IndexedFp16:
                return DecodeIndexedFp16(buf, maxSize);
#if SUPPORT_FP8_HF8_PRINT
            case AicorePrint::DataType::IndexedFp8E4M3:
                return DecodeIndexedFp8E4M3(buf, maxSize);
            case AicorePrint::DataType::IndexedFp8E5M2:
                return DecodeIndexedFp8E5M2(buf, maxSize);
            case AicorePrint::DataType::IndexedFp8E8M0:
                return DecodeIndexedFp8E8M0(buf, maxSize);
            case AicorePrint::DataType::IndexedHf8:
                return DecodeIndexedHf8(buf, maxSize);
#endif
            case AicorePrint::DataType::OverflowWarning:
                return DecodeOverflowWarning(buf, maxSize);
            default:
                return DecodeLegacyRecord(type, buf, maxSize);
        }
    }
    
    int DecodeTensorHeader(char* buf, size_t maxSize) {
        short nameLen = ReadValue<short>(tail_);
        tail_ += AicorePrintConst::NAMELEN_FIELD_SIZE;
        
        std::string name = ReadString(tail_);
        tail_ += nameLen;
        
        int64_t begin = ReadValue<int64_t>(tail_);
        tail_ += AicorePrintConst::TENSOR_RANGE_SIZE;
        int64_t end = ReadValue<int64_t>(tail_);
        tail_ += AicorePrintConst::TENSOR_RANGE_SIZE;
        
        lastTensorName_ = name;
        
        return snprintf_s(buf, maxSize, maxSize - 1,
            "tensor '%s', range=[%ld, %ld)\n",
            name.c_str(), begin, end);
    }
    
    int DecodeIndexedFp32(char* buf, size_t maxSize) {
        int64_t index = ReadValue<int64_t>(tail_);
        tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
        float value = ReadValue<float>(tail_);
        tail_ += sizeof(float);
        
        return snprintf_s(buf, maxSize, maxSize - 1,
            "%s[%ld] %f\n", lastTensorName_.c_str(), index, value);
    }
    
    int DecodeIndexedInt64(char* buf, size_t maxSize) {
        int64_t index = ReadValue<int64_t>(tail_);
        tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
        int64_t value = ReadValue<int64_t>(tail_);
        tail_ += AicorePrintConst::TENSOR_RANGE_SIZE;
        
        return snprintf_s(buf, maxSize, maxSize - 1,
            "%s[%ld] %lld\n", lastTensorName_.c_str(), index, value);
    }
    
    int DecodeIndexedBf16(char* buf, size_t maxSize) {
        int64_t index = ReadValue<int64_t>(tail_);
        tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
        uint16_t bits = ReadValue<uint16_t>(tail_);
        tail_ += sizeof(uint16_t);
        
        float value = DecodeBf16(bits);
        return snprintf_s(buf, maxSize, maxSize - 1,
            "%s[%ld] %f\n", lastTensorName_.c_str(), index, value);
    }
    
    int DecodeIndexedFp16(char* buf, size_t maxSize) {
        int64_t index = ReadValue<int64_t>(tail_);
        tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
        uint16_t bits = ReadValue<uint16_t>(tail_);
        tail_ += sizeof(uint16_t);
        
        float value = DecodeF16(bits);
        return snprintf_s(buf, maxSize, maxSize - 1,
            "%s[%ld] %f\n", lastTensorName_.c_str(), index, value);
    }
    
#if SUPPORT_FP8_HF8_PRINT
    int DecodeIndexedFp8E4M3(char* buf, size_t maxSize) {
        int64_t index = ReadValue<int64_t>(tail_);
        tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
        uint8_t bits = ReadValue<uint8_t>(tail_);
        tail_ += sizeof(uint8_t);
        
        float value = DecodeFp8E4M3(bits);
        return snprintf_s(buf, maxSize, maxSize - 1,
            "%s[%ld] %f\n", lastTensorName_.c_str(), index, value);
    }
    
    int DecodeIndexedFp8E5M2(char* buf, size_t maxSize) {
        int64_t index = ReadValue<int64_t>(tail_);
        tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
        uint8_t bits = ReadValue<uint8_t>(tail_);
        tail_ += sizeof(uint8_t);
        
        float value = DecodeFp8E5M2(bits);
        return snprintf_s(buf, maxSize, maxSize - 1,
            "%s[%ld] %f\n", lastTensorName_.c_str(), index, value);
    }
    
    int DecodeIndexedFp8E8M0(char* buf, size_t maxSize) {
        int64_t index = ReadValue<int64_t>(tail_);
        tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
        uint8_t bits = ReadValue<uint8_t>(tail_);
        tail_ += sizeof(uint8_t);
        
        float value = DecodeFp8E8M0(bits);
        return snprintf_s(buf, maxSize, maxSize - 1,
            "%s[%ld] %f\n", lastTensorName_.c_str(), index, value);
    }
    
    int DecodeIndexedHf8(char* buf, size_t maxSize) {
        int64_t index = ReadValue<int64_t>(tail_);
        tail_ += AicorePrintConst::INDEXED_INDEX_SIZE;
        uint8_t bits = ReadValue<uint8_t>(tail_);
        tail_ += sizeof(uint8_t);
        
        float value = DecodeHf8(bits);
        return snprintf_s(buf, maxSize, maxSize - 1,
            "%s[%ld] %f\n", lastTensorName_.c_str(), index, value);
    }
#endif
    
    int DecodeOverflowWarning(char* buf, size_t maxSize) {
        int64_t bufferSize = ReadValue<int64_t>(tail_);
        tail_ += sizeof(int64_t);
        
        int64_t fullBufferSize = bufferSize + sizeof(RemoteHeader);
        int64_t recommendedSize = fullBufferSize * 2;
        
        return snprintf_s(buf, maxSize, maxSize - 1,
            "[WARNING] The PRINT_BUFFER_SIZE (ring buffer) is full! "
            "Current buffer: %ld bytes (%ld KB). "
            "Recommend: set PRINT_BUFFER_SIZE >= %ld (%ld KB, double current size) "
            "in framework/src/interface/machine/device/tilefwk/aicpu_common.h, "
 	        "then rebuild and reinstall.\n",
            fullBufferSize, fullBufferSize / 1024,
            recommendedSize, recommendedSize / 1024);
    }
    
    int DecodeLegacyRecord(AicorePrint::DataType type, char* buf, size_t maxSize) {
        auto valOff = tail_ + AicorePrintConst::NAMELEN_FIELD_SIZE;
        tail_ += ReadValue<short>(tail_) + AicorePrintConst::NAMELEN_FIELD_SIZE;
        auto fmtOff = tail_ + AicorePrintConst::NAMELEN_FIELD_SIZE;
        std::string fmt = ReadString(fmtOff);
        tail_ += ReadValue<short>(tail_) + AicorePrintConst::NAMELEN_FIELD_SIZE;
        
        switch (type) {
            case AicorePrint::DataType::Normal:
                return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), 0);
            case AicorePrint::DataType::Fp32:
                return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), ReadValue<float>(valOff));
            case AicorePrint::DataType::Int64:
                return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), ReadValue<int64_t>(valOff));
            case AicorePrint::DataType::Char:
                return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), ReadValue<char>(valOff));
            case AicorePrint::DataType::String:
                return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), ReadString(valOff).c_str());
            case AicorePrint::DataType::Pointer:
                return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), ReadValue<int64_t>(valOff));
            case AicorePrint::DataType::Bf16: {
                uint16_t bits = ReadValue<uint16_t>(valOff);
                return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), DecodeBf16(bits));
            }
            case AicorePrint::DataType::Fp16: {
                uint16_t bits = ReadValue<uint16_t>(valOff);
                return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), DecodeF16(bits));
            }
#if SUPPORT_FP8_HF8_PRINT
            case AicorePrint::DataType::Fp8E4M3: {
                uint8_t bits = ReadValue<uint8_t>(valOff);
                return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), DecodeFp8E4M3(bits));
            }
            case AicorePrint::DataType::Fp8E5M2: {
                uint8_t bits = ReadValue<uint8_t>(valOff);
                return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), DecodeFp8E5M2(bits));
            }
            case AicorePrint::DataType::Fp8E8M0: {
                uint8_t bits = ReadValue<uint8_t>(valOff);
                return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), DecodeFp8E8M0(bits));
            }
            case AicorePrint::DataType::Hf8: {
                uint8_t bits = ReadValue<uint8_t>(valOff);
                return snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), DecodeHf8(bits));
            }
#endif
            default:
                buf[0] = '?';
                return 1;
        }
    }
#endif

    LogContext ctx_;
    int64_t head_ = 0;
    int64_t tail_ = 0;
    int64_t size_ = 0;
    volatile __gm__ RemoteHeader* remote_;
    __gm__ uint8_t* data_;
    
    bool overflowed_ = false;
    
#ifdef __TILE_FWK_HOST__
    std::string lastTensorName_;
#endif
};

// ============================================================================
// Section 13: IndexedTypeInfo 模板特化
// ============================================================================

template <typename ElemT>
struct IndexedTypeInfo {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::End;
    static constexpr int64_t Size  = 0;
};

template<> struct IndexedTypeInfo<float> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedFp32;
    static constexpr int64_t Size  = EncodeSizes::IndexedFp32WithEnd;
};

template<> struct IndexedTypeInfo<int64_t> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedInt64;
    static constexpr int64_t Size  = EncodeSizes::IndexedInt64WithEnd;
};

#if IS_AICORE
template<> struct IndexedTypeInfo<bfloat16_t> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedBf16;
    static constexpr int64_t Size  = EncodeSizes::IndexedBf16WithEnd;
};

template<> struct IndexedTypeInfo<half> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedFp16;
    static constexpr int64_t Size  = EncodeSizes::IndexedFp16WithEnd;
};

#if SUPPORT_FP8_HF8_PRINT
template<> struct IndexedTypeInfo<float8_e4m3_t> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedFp8E4M3;
    static constexpr int64_t Size  = EncodeSizes::IndexedFp8WithEnd;
};

template<> struct IndexedTypeInfo<float8_e5m2_t> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedFp8E5M2;
    static constexpr int64_t Size  = EncodeSizes::IndexedFp8WithEnd;
};

template<> struct IndexedTypeInfo<float8_e8m0_t> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedFp8E8M0;
    static constexpr int64_t Size  = EncodeSizes::IndexedFp8WithEnd;
};

template<> struct IndexedTypeInfo<hifloat8_t> {
    static constexpr AicorePrint::DataType Type = AicorePrint::DataType::IndexedHf8;
    static constexpr int64_t Size  = EncodeSizes::IndexedFp8WithEnd;
};
#endif
#endif

// ============================================================================
// Section 14: 通用打印分发器
// ============================================================================

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

// ============================================================================
// Section 15: Shape 打印支持
// ============================================================================

#if defined(__TILE_FWK_AICORE__) && defined(TILEOP_UTILS_TUPLE_H)

constexpr size_t AICORE_PRINT_SHAPE_MAX_DIMS = 6;

template <size_t I, typename ShapeTuple>
INLINE void FillShapeDims(int64_t (&dims)[AICORE_PRINT_SHAPE_MAX_DIMS], const ShapeTuple& shape) {
    constexpr size_t n = Std::tuple_size<ShapeTuple>::value;
    constexpr size_t m = (n < AICORE_PRINT_SHAPE_MAX_DIMS) ? n : AICORE_PRINT_SHAPE_MAX_DIMS;
    
    if constexpr (I < m) {
        dims[I] = static_cast<int64_t>(Std::get<I>(shape));
        FillShapeDims<I + 1>(dims, shape);
    }
}

template <size_t N>
INLINE void LogShapeDims(LogContext* ctx, const int64_t (&dims)[AICORE_PRINT_SHAPE_MAX_DIMS]) {
    if constexpr (N == 1) {
        AiCoreLogF(ctx, "shape=[%ld]\n", dims[0]);
    } else if constexpr (N == 2) {
        AiCoreLogF(ctx, "shape=[%ld,%ld]\n", dims[0], dims[1]);
    } else if constexpr (N == 3) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld]\n", dims[0], dims[1], dims[2]);
    } else if constexpr (N == 4) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld]\n", dims[0], dims[1], dims[2], dims[3]);
    } else if constexpr (N == 5) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld,%ld]\n", dims[0], dims[1], dims[2], dims[3], dims[4]);
    } else if constexpr (N == 6) {
        AiCoreLogF(ctx, "shape=[%ld,%ld,%ld,%ld,%ld,%ld]\n",
                   dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
    }
}

template <typename... Dims>
INLINE void AiCorePrintShape(LogContext* ctx, const TileOp::Shape<Dims...>& shape) {
    constexpr size_t N = Std::tuple_size<TileOp::Shape<Dims...>>::value;
    if constexpr (N == 0 || N > AICORE_PRINT_SHAPE_MAX_DIMS) {
        return;
    }
    
    int64_t dims[AICORE_PRINT_SHAPE_MAX_DIMS]{};
    FillShapeDims<0>(dims, shape);
    LogShapeDims<N>(ctx, dims);
}

#endif

// ============================================================================
// Section 16: Tensor 打印实现
// ============================================================================

template <typename T, typename PtrT>
INLINE void PrintTensorImpl(LogContext* ctx, PtrT data, int64_t end,
                            int64_t begin, __gm__ const char* name) {
    using ElemT = std::remove_cv_t<T>;
    auto* logger = reinterpret_cast<AicoreLogger*>(ctx);
    
    logger->EncodeTensorHeader(name, begin, end);
    logger->Sync();
    
    for (int64_t i = begin; i < end; ++i) {
        ElemT tmp = data[i];
        
        if constexpr (std::is_floating_point_v<ElemT>) {
            float v = static_cast<float>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<float>::Type, i,
                                 reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else if constexpr (std::is_integral_v<ElemT>) {
            int64_t v = static_cast<int64_t>(tmp);
            logger->EncodeIndexed(IndexedTypeInfo<int64_t>::Type, i,
                                 reinterpret_cast<uint8_t*>(&v), sizeof(v));
        } else {
            if constexpr (sizeof(ElemT) == 2) {
                auto bits = SafeBitCast<uint16_t>(tmp);
                logger->EncodeIndexed(IndexedTypeInfo<ElemT>::Type, i,
                                     reinterpret_cast<uint8_t*>(&bits), sizeof(bits));
            } else if constexpr (sizeof(ElemT) == 1) {
                uint8_t bits = SafeBitCast<uint8_t>(tmp);
                logger->EncodeIndexed(IndexedTypeInfo<ElemT>::Type, i,
                                     reinterpret_cast<uint8_t*>(&bits), sizeof(bits));
            }
        }
        logger->Sync();
    }
}

template <typename T>
INLINE void AiCorePrintGmTensor(LogContext* ctx, __gm__ const T* data,
                                int64_t end, int64_t begin, __gm__ const char* name) {
    PrintTensorImpl<T>(ctx, data, end, begin, name);
}

#if IS_AICORE

template <typename T>
INLINE void AiCorePrintUbTensor(LogContext* ctx, __ubuf__ const T* data,
                                int64_t end, int64_t begin, __ubuf__ const char* name) {
    PrintTensorImpl<T>(ctx, data, end, begin, name);
}

template <typename T>
__aicore__ void L1RawCopyToGM(__gm__ T* dst, __cbuf__ const T* src, int64_t count) {
    int64_t totalBytes = count * sizeof(T);
    if (totalBytes == 0) return;
    
    uint16_t nBurst = 1;
    uint16_t lenBurst;
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    
    if (totalBytes >= 32) {
        lenBurst = static_cast<uint16_t>((totalBytes + 31) / 32);
    } else {
        lenBurst = static_cast<uint16_t>(totalBytes > 0 ? totalBytes : 1);
    }
    
    copy_cbuf_to_gm(dst, src, 0, nBurst, lenBurst, srcStride, dstStride);
}

template <typename T>
INLINE void AiCorePrintL1Tensor(LogContext* ctx, __cbuf__ const T* data,
                                int64_t end, int64_t begin,
                                __gm__ T* staging, __gm__ const char* name) {
    int64_t count = end - begin;
    if (count <= 0) return;
    
    L1RawCopyToGM(staging, data + begin, count);
    
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    
    AiCorePrintGmTensor<T>(ctx, staging, count, 0, name);
}

#endif
