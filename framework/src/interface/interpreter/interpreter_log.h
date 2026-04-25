/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file interpreter_log.h
 * \brief Interpreter logging helpers and macros.
 */

#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <sys/stat.h>

#include "securec.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::interpreter_log {
inline std::mutex& LogMutex()
{
    static std::mutex logMutex;
    return logMutex;
}

inline std::string& LogFilePath()
{
    static std::string path = "output/interpreter.log";
    return path;
}

inline void SetLogFilePath(const std::string& path)
{
    std::lock_guard<std::mutex> lock(LogMutex());
    LogFilePath() = path;
}

inline void EnsureLogDir()
{
    static std::once_flag once;
    std::call_once(once, []() { (void)mkdir("output", 0755); });
}

inline bool ShouldWriteLevel(const char* level)
{
    return std::strcmp(level, "ERROR") == 0 || std::strcmp(level, "EVENT") == 0;
}

inline bool ShouldPrintToStdout()
{
    const char* value = std::getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

inline void WriteLine(const char* level, const char* fmt, va_list args) __attribute__((format(gnu_printf, 2, 0)));
inline void WriteLine(const char* level, const char* fmt, va_list args)
{
    if (!ShouldWriteLevel(level)) {
        return;
    }
    EnsureLogDir();
    std::lock_guard<std::mutex> lock(LogMutex());
    std::string filePath = LogFilePath();
    FILE* fp = fopen(filePath.c_str(), "a");
    if (fp == nullptr) {
        return;
    }

    constexpr int kInitialBufSize = 1024;
    std::string msgBuf(static_cast<size_t>(kInitialBufSize), '\0');

    va_list argsCopy;
    va_copy(argsCopy, args);
    int msgLength = vsnprintf_s(msgBuf.data(), msgBuf.size(), msgBuf.size() - 1, fmt, argsCopy);
    va_end(argsCopy);
    if (msgLength < 0) {
        fclose(fp);
        return;
    }
    if (msgLength > kInitialBufSize) {
        msgBuf.resize(static_cast<size_t>(msgLength) + 1, '\0');
        va_copy(argsCopy, args);
        const int ret = vsnprintf_s(msgBuf.data(), msgBuf.size(), msgBuf.size() - 1, fmt, argsCopy);
        va_end(argsCopy);
        if (ret < 0) {
            fclose(fp);
            return;
        }
    }

    std::time_t now = std::time(nullptr);
    std::tm localTm {};
    (void)localtime_r(&now, &localTm);
    char timeBuf[32] = {0};
    (void)std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", &localTm);

    fprintf(fp, "[%s][%s] %s\n", timeBuf, level, msgBuf.c_str());
    if (ShouldPrintToStdout()) {
        fprintf(stdout, "[%s][%s] %s\n", timeBuf, level, msgBuf.c_str());
        fflush(stdout);
    }
    fclose(fp);
}

inline void Log(const char* level, const char* fmt, ...) __attribute__((format(gnu_printf, 2, 3)));
inline void Log(const char* level, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    WriteLine(level, fmt, args);
    va_end(args);
}
} // namespace npu::tile_fwk::interpreter_log

#define INTERPRETER_LOGD(...) VERIFY_LOGD(__VA_ARGS__)
#define INTERPRETER_LOGI(...) VERIFY_LOGI(__VA_ARGS__)
#define INTERPRETER_LOGW(...) VERIFY_LOGW(__VA_ARGS__)
#define INTERPRETER_EVENT(...) npu::tile_fwk::interpreter_log::Log("EVENT", __VA_ARGS__)

#define INTERPRETER_LOGE(errCode, fmt, ...)                                                                  \
    npu::tile_fwk::interpreter_log::Log(                                                                     \
        "ERROR", "ErrCode: F%05X! Enum: %s " fmt, static_cast<uint32_t>(errCode) & 0xFFFFF, #errCode,      \
        ##__VA_ARGS__)

#define INTERPRETER_LOGE_FULL(errCode, fmt, ...) INTERPRETER_LOGE(errCode, fmt, ##__VA_ARGS__)
