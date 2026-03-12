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
 * \file error.h
 * \brief
 */

#pragma once

#include <cstddef>
#include <exception>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <cstring>
#include <cassert>

#include "lazy.h"

namespace npu::tile_fwk {
using Backtrace = std::shared_ptr<LazyValue<std::string>>;

Backtrace GetBacktrace(size_t skipFrames, size_t maxFrames);

struct ErrorMessage {
public:
    std::string Message() { return ss.str(); }

    template <typename T>
    ErrorMessage &operator<<(const T &value) {
        ss << value;
        return *this;
    }

    template <typename T>
    ErrorMessage &operator<<(const std::vector<T> &vec) {
        ss << "[";
        for (auto iter = vec.begin(); iter != vec.end(); ++iter) {
            if (iter != vec.begin()) {
                ss << ", ";
            }
            ss << *iter;
        }
        ss << "]";
        return *this;
    }

    // support std::endl etc.
    ErrorMessage &operator<<(std::ostream &(*manipulator)(std::ostream &)) {
        ss << manipulator;
        return *this;
    }

    std::stringstream ss;
};

class Error : public std::exception {
public:
    Error(const char *func, const char *file, size_t line, const std::string &msg, Backtrace backtrace)
        : func_(func), file_(file), line_(line), msg_(msg), backtrace_(backtrace) {}

    Error(const char *func, const char *file, size_t line, Backtrace backtrace)
        : func_(func), file_(file), line_(line), backtrace_(backtrace) {}

    const char *what() const noexcept override;

    int operator=(ErrorMessage &msg) {
        msg_ = msg.Message();
        /* avoid nested throw */
        if (std::uncaught_exceptions() == 0) {
            throw *this;
        }
        return 0;
    }

private:
    const char *func_;
    const char *file_;
    size_t line_;
    std::string msg_;
    std::string umsg_;
    Backtrace backtrace_;
    mutable LazyShared<std::string> what_;
};

class AssertInfo {
public:
    [[noreturn]] int operator=(ErrorMessage &msg) {
        (void)fprintf(stderr, "%s\n", msg.Message().c_str());
        abort();
    }
};

namespace detail {
/* Host 侧流式 ASSERT/CHECK 构建器 */
class HostAssertStream {
public:
    HostAssertStream(bool enabled,
                     const char *func,
                     const char *file,
                     size_t line,
                     const char *prefix)
        : enabled_(enabled),
          err_(func, file, line, GetBacktrace(0, /* 64 is maxFrames */ 64)) {
        if (enabled_) {
            msg_ << prefix;
        }
    }

    HostAssertStream(bool enabled,
                     const char *func,
                     const char *file,
                     size_t line,
                     const char *prefix,
                     uint32_t errCode)
        : enabled_(enabled),
          err_(func, file, line, GetBacktrace(0, /* 64 is maxFrames */ 64)) {
        if (enabled_) {
            msg_ << prefix << " ErrCode: F" << errCode << "!";
        }
    }

    template <typename T>
    HostAssertStream &operator<<(const T &value) {
        if (enabled_) {
            msg_ << value;
        }
        return *this;
    }

    HostAssertStream &operator<<(std::ostream &(*manip)(std::ostream &)) {
        if (enabled_) {
            msg_ << manip;
        }
        return *this;
    }

    ~HostAssertStream() {
        if (enabled_) {
            err_ = msg_;
        }
    }

    HostAssertStream(const HostAssertStream &) = delete;
    HostAssertStream &operator=(const HostAssertStream &) = delete;

private:
    bool enabled_{false};
    Error err_;
    ErrorMessage msg_;
};
} // namespace detail

#ifndef __DEVICE__
/* host 侧 ASSERT：支持可选 errCode + 流式输出 */
#define ASSERT_IMPL_NO_ERRCODE(cond)                                                                                          \
    npu::tile_fwk::detail::HostAssertStream(!(cond), __func__, __FILE__, __LINE__,                                           \
        "ASSERTION FAILED: " #cond)

#define ASSERT_IMPL_WITH_ERRCODE(cond, errCode)                                                                               \
    npu::tile_fwk::detail::HostAssertStream(!(cond), __func__, __FILE__, __LINE__,                                           \
        "ASSERTION FAILED: " #cond, static_cast<uint32_t>(errCode))

#define ASSERT_SELECT_MACRO(_1, _2, NAME, ...) NAME
#define ASSERT(...) ASSERT_SELECT_MACRO(__VA_ARGS__,                                      \
    ASSERT_IMPL_WITH_ERRCODE, ASSERT_IMPL_NO_ERRCODE)(__VA_ARGS__)

#define ASSERT_C(cond, errCode) ASSERT(cond, errCode)

/* host 侧 CHECK：支持可选 errCode + 流式输出 */
#define CHECK_IMPL_NO_ERRCODE(cond)                                                                                           \
    npu::tile_fwk::detail::HostAssertStream(!(cond), __func__, __FILE__, __LINE__,                                           \
        "CHECK FAILED: " #cond)

#define CHECK_IMPL_WITH_ERRCODE(cond, errCode)                                                                                \
    npu::tile_fwk::detail::HostAssertStream(!(cond), __func__, __FILE__, __LINE__,                                           \
        "CHECK FAILED: " #cond, static_cast<uint32_t>(errCode))

#define CHECK_SELECT_MACRO(_1, _2, NAME, ...) NAME
#define CHECK(...) CHECK_SELECT_MACRO(__VA_ARGS__,                                        \
    CHECK_IMPL_WITH_ERRCODE, CHECK_IMPL_NO_ERRCODE)(__VA_ARGS__)

#define CHECK_C(cond, errCode) CHECK(cond, errCode)

#define TILEFWK_ERROR()                                                                                            \
    npu::tile_fwk::Error(__func__, __FILE__, __LINE__, npu::tile_fwk::GetBacktrace(0, /* 64 is maxFrames */ 64)) = \
        npu::tile_fwk::ErrorMessage()
#else

/* device 侧仍使用简单的非流式实现 */
#define ASSERT(cond)                             \
    (cond) ? 0 : AssertInfo() = npu::tile_fwk::ErrorMessage() \
        << "ASSERTION FAILED: " #cond " file " << __FILE__ << ", line " << __LINE__ << "\n"

#define ASSERT_C(cond, errCode)                             \
    (cond) ? 0 : AssertInfo() = npu::tile_fwk::ErrorMessage() \
        << "ASSERTION FAILED: " #cond " ErrCode: F" << static_cast<uint32_t>(errCode) \
        << "! file " << __FILE__ << ", line " << __LINE__ << "\n"

#define CHECK(cond)                             \
    (cond) ? 0 : AssertInfo() = npu::tile_fwk::ErrorMessage() \
        << "CHECK FAILED: " #cond " file " << __FILE__ << ", line " << __LINE__ << "\n"

#define CHECK_C(cond, errCode)                             \
    (cond) ? 0 : AssertInfo() = npu::tile_fwk::ErrorMessage() \
        << "CHECK FAILED: " #cond " ErrCode: F" << static_cast<uint32_t>(errCode) \
        << "! file " << __FILE__ << ", line " << __LINE__ << "\n"
#endif
} // namespace npu::tile_fwk
