/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "aicore_data.h"

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#ifdef __TILE_FWK_HOST__
#include <string>
#include <sstream>
#include <securec.h>
#endif

enum NodeTy { END, NORMAL, FLOAT, INT, CHAR, STRING, POINTER };

struct LogContext {
    void (*print_int)(LogContext *ctx, __gm__ const char **fmt, int64_t val);
    void (*print_float)(LogContext *ctx, __gm__ const char **fmt, float val);
    void (*print)(LogContext *ctx, __gm__ const char *fmt);
};

template <typename T>
INLINE void __aicore_print(LogContext *ctx, __gm__ const char **fmt, T val) {
    if constexpr (std::is_integral_v<T>) {
        ctx->print_int(ctx, fmt, static_cast<int64_t>(val));
    } else if constexpr (std::is_floating_point_v<T>) {
        ctx->print_float(ctx, fmt, static_cast<float>(val));
    } else if constexpr (std::is_pointer_v<T>) {
        ctx->print_int(ctx, fmt, reinterpret_cast<int64_t>(val));
    }
}

template <typename... Ts>
INLINE void aicore_printf(LogContext *ctx, __gm__ const char *fmt, Ts... Args) {
    if (ctx && fmt) {
        (__aicore_print(ctx, &fmt, Args), ...);
        ctx->print(ctx, fmt);
    }
}

struct AicoreLogger {
    struct Remote {
        int64_t head_;
        int64_t tail_;
    };

    static __aicore__ void __print_int(LogContext *ctx, __gm__ const char **fmt, int64_t val) {
        auto self = reinterpret_cast<AicoreLogger *>(ctx);
        if (self) {
            self->print_int(fmt, val);
        }
    }

    static __aicore__ void __print_float(LogContext *ctx, __gm__ const char **fmt, float val) {
        auto self = reinterpret_cast<AicoreLogger *>(ctx);
        if (self) {
            self->print_float(fmt, val);
        }
    }

    static __aicore__ void __print(LogContext *ctx, __gm__ const char *fmt) {
        auto self = reinterpret_cast<AicoreLogger *>(ctx);
        if (self) {
            self->print(fmt);
        }
    }

    __aicore__ void Init(__gm__ uint8_t *buf, size_t n) {
        remote_ = (volatile __gm__ Remote *)buf;
        remote_->head_ = remote_->tail_ = 0;
        head_ = tail_ = 0;
        size_ = n - sizeof(Remote);
        data_ = buf + sizeof(Remote);
        ctx.print_int = __print_int;
        ctx.print_float = __print_float;
        ctx.print = __print;
    }

    __aicore__ __gm__ uint8_t *GetBuffer()  {
        return data_ - sizeof(Remote);
    }

    __aicore__ void print_int(__gm__ const char **fmt, int64_t val) {
        auto curFmt = *fmt;
        auto idx = parse_next_format(*fmt);
        if (idx == -1) {
            return;
        }
        switch (curFmt[idx++]) {
            case 's': {
                auto tmp = reinterpret_cast<__gm__ const char *>(val);
                if (tmp == nullptr) {
                    tmp = "<null>";
                }
                encode(STRING, (__gm__ const uint8_t *)tmp, length(tmp), *fmt, idx);
                break;
            }
            case 'd':
            case 'i':
            case 'x':
            case 'X':
            case 'o':
            case 'u': {
                encode(INT, (uint8_t *)&val, sizeof(val), *fmt, idx);
                break;
            }
            case 'p': {
                encode(POINTER, (uint8_t *)&val, sizeof(val), *fmt, idx);
                break;
            }
            case 'c': {
                char c = static_cast<char>(val);
                encode(POINTER, (uint8_t *)&c, 1, *fmt, idx);
                break;
            }
            default: encode(NORMAL, (uint8_t *)0, 0, *fmt, idx); break;
        }

        *fmt = *fmt + idx;
    }

    __aicore__ void print_float(__gm__ const char **fmt, float val) {
        auto curFmt = *fmt;
        auto idx = parse_next_format(*fmt);
        if (idx == -1) {
            return;
        }
        switch (curFmt[idx++]) {
            case 'u': {
                encode(FLOAT, (uint8_t *)&val, sizeof(val), *fmt, idx);
                break;
                default: encode(NORMAL, (uint8_t *)0, 0, *fmt, idx); break;
            }
        }
        *fmt = *fmt + idx;
    }

    __aicore__ void print(__gm__ const char *str) {
        auto n = length(str);
        if (n) {
            encode(NORMAL, (uint8_t *)0, 0, str, n);
        }
        encode(END);
        sync();
    }

    __aicore__ void sync() {
#ifndef __TILE_FWK_HOST__
        int64_t delta = (int64_t)(&data_[remote_->head_ % size_]) & (CACHE_LINE_SIZE -1);
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

    INLINE LogContext *context() { return &ctx; }

#ifdef __TILE_FWK_HOST__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
    int read(char *buf, size_t maxSize) {
        size_t size = 0;
        head_ = remote_->head_;
        if (tail_ < remote_->tail_) {
            // lose some data
            tail_ = remote_->tail_;
        }
        while (tail_ != head_) {
            auto type = read<uint8_t>(tail_++);
            if (type == END) {
                if (size == 0)
                    continue;
                else
                    return size;
            } else if (maxSize == 0) {
                continue;
            }

            auto valOff = tail_ + sizeof(short);
            tail_ += read<short>(tail_) + sizeof(short);
            auto fmtOff = tail_ + sizeof(short);
            std::string fmt = readString(fmtOff);
            tail_ += read<short>(tail_) + sizeof(short);
            int n = 0;
            switch (type) {
                case NORMAL: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), 0); break;
                case FLOAT: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), read<float>(valOff)); break;
                case INT: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), read<int64_t>(valOff)); break;
                case CHAR: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), read<char>(valOff)); break;
                case STRING: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), readString(valOff).c_str()); break;
                case POINTER: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), read<int64_t>(valOff)); break;
                default: if (n) { buf[0] = '?'; n = 1;} break;
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
    __aicore__ int64_t parse_next_format(__gm__ const char *fmt) {
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
        while (isdigit(fmt[idx])) {
            idx++;
        }

        // precision
        if (fmt[idx] == '.') {
            idx++;
            while (isdigit(fmt[idx])) {
                idx++;
            }
        }

        // length
        if (fmt[idx] == 'l' || fmt[idx] == 'z' || fmt[idx] == 'h') {
            idx++;
            if (fmt[idx] == 'l')
                idx++;
        }

        return fmt[idx] ? idx : -1;
    }

    template <typename T>
    INLINE T read(int64_t off) {
        T val;
        char tmp[sizeof(T)];
        for (size_t i = 0; i < sizeof(T); i++) {
            tmp[i] = data_[(off + i) % size_];
        }
        val = *(T *)(tmp);
        return val;
    }

#ifdef __TILE_FWK_HOST__
    std::string readString(int64_t off) {
        std::stringstream ss;
        while (off < head_) {
            auto c = read<char>(off++);
            if (c == '\0') break;
            ss << c;
        }
        return ss.str();
    }
#endif

    __aicore__ void encode(uint8_t val) {
        if (head_ == tail_ + size_) {
            while (read<uint8_t>(tail_) != END) {
                tail_++;
                tail_ += read<short>(tail_) + sizeof(short);
                tail_ += read<short>(tail_) + sizeof(short);
            }
            tail_++;
        }
        volatile __gm__ uint8_t *p = &data_[head_++ % size_];
        *p = val;
    }

    template<typename T>
    __aicore__ void encode(NodeTy ty, const T *val, short valLen, __gm__ const char *fmt, int fmtLen) {
        encode(ty);

        auto bytes = (uint8_t *)(&valLen);
        encode(bytes[0]);
        encode(bytes[1]);
        for (auto i = 0; i < valLen; i++) {
            encode(val[i]);
        }

        fmtLen += 1; // pad '\0'
        bytes = (uint8_t *)(&fmtLen);
        encode(bytes[0]);
        encode(bytes[1]);
        for (auto i = 0; i < fmtLen - 1; i++) {
            encode(fmt[i]);
        }
        encode('\0');
    }

    INLINE size_t length(__gm__ const char *str) {
        size_t n = 0;
        while (*str++) {
            n++;
        }
        return n;
    }

    INLINE bool isdigit(char c) {
        return c >= '0' && c <= '9';
    }

private:
    LogContext ctx;
    int64_t head_;
    int64_t tail_;
    int64_t size_;
    volatile __gm__ Remote *remote_;
    __gm__ uint8_t *data_;
};
