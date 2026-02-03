/**
* Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file log_manager.cpp
 * \brief
 */

#include "log_manager.h"
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <map>
#include "securec.h"

namespace tile::fwk {
namespace {
constexpr const char *kEnvGlobalLogLevel = "ASCEND_GLOBAL_LOG_LEVEL";
constexpr const char *kEnvModuleLogLevel = "ASCEND_MODULE_LOG_LEVEL";
constexpr const char *kEnvProcessLogPath = "ASCEND_PROCESS_LOG_PATH";
constexpr const char *kModuleName = "PYPTO";
constexpr const char *kLogFilePrefix = "pypto-log-";

const std::map<LogLevel, std::string> logLevelStrMap = {
    {LogLevel::NONE, "NONE"},
    {LogLevel::DEBUG, "DEBUG"},
    {LogLevel::INFO, "INFO"},
    {LogLevel::WARN, "WARN"},
    {LogLevel::ERROR, "ERROR"}
};

const std::string& GetLogLevelStr(const LogLevel logLevel) {
    const auto iter = logLevelStrMap.find(logLevel);
    return iter != logLevelStrMap.end() ? iter->second : logLevelStrMap.begin()->second;
}

std::string GetCurrentTime() {
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm* nowTm = std::localtime(&nowTime);
    std::stringstream ss;
    ss << std::put_time(nowTm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}
}
LogManager &LogManager::Instance() {
    static LogManager instance;
    return instance;
}

LogManager::LogManager() {
    const char *envGlobalLogLevel = getenv(kEnvGlobalLogLevel);
    if (envGlobalLogLevel != nullptr) {
        try {
            if (int logLevel = std::stoi(envGlobalLogLevel); logLevel >= 0 && logLevel < 4) {
                level_ = static_cast<LogLevel>(logLevel);
            }
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument: " << ia.what() << '\n';
        } catch (const std::out_of_range& oor) {
            std::cerr << "Out of Range error: " << oor.what() << '\n';
        }
    }
    const char *envModuleLogLevel = getenv(kEnvModuleLogLevel);
    if (envModuleLogLevel != nullptr) {

    }
    const char *envProcessLogPath = getenv(kEnvProcessLogPath);
    if (envProcessLogPath != nullptr) {
        fileDir_.assign(envProcessLogPath);
        // enableStdOut_ = false;
    }
}

LogManager::~LogManager() {
    level_ = LogLevel::ERROR;
    enableStdOut_ = true;
    fileDir_.clear();
    std::queue<std::string> tmp_files;
    logFiles_.swap(tmp_files);
}

void LogManager::Record(const LogLevel logLevel, const char *fmt, va_list list) {
    LogMsg logMsg{};
    ConstructMessage(logLevel, fmt, list, logMsg);
    WriteMessage(logMsg);
}

void LogManager::ConstructMessage(const LogLevel logLevel, const char *fmt, va_list list, LogMsg &logMsg) {
    ConstructMsgHeader(logLevel, logMsg);
    int ret = vsnprintf_truncated_s(logMsg.msg + logMsg.length, kMsgMaxLen - logMsg.length, fmt, list);
    if (ret < 0) {
        std::cerr << "Constrcut message failed: " << ret << std::endl;
    }
    logMsg.length = std::strlen(logMsg.msg);
    ConstructMsgTail(logMsg);
}

void LogManager::ConstructMsgHeader(const LogLevel logLevel, LogMsg &logMsg) {
    int ret = snprintf_s(logMsg.msg, kMsgMaxLen, kMsgMaxLen - 1, "[%s] %s:%s ",
                         GetLogLevelStr(logLevel).c_str(), kModuleName, GetCurrentTime().c_str());
    if (ret != 0) {
        std::cerr << "Construct log msg hader failed: " << ret << std::endl;
    }
    logMsg.length = std::strlen(logMsg.msg);
}

void LogManager::ConstructMsgTail(LogMsg &logMsg) {
    if (logMsg.msg[logMsg.length - 1] != '\n') {
        if (logMsg.length < kMsgMaxLen) {
            logMsg.msg[logMsg.length] = '\n';
        } else {
            logMsg.msg[kMsgMaxLen - 1] = '\n';
        }
    }
}

void LogManager::WriteMessage(const LogMsg &logMsg) {
    const std::lock_guard<std::mutex> lockGuard(writeMutex_);
    if (enableStdOut_) {
        WriteToStdOut(logMsg);
    } else {
        WriteToFile(logMsg);
    }
}

void LogManager::WriteToStdOut(const LogMsg &logMsg) {
    int fd = fileno(stdout);
    if (fd <= 0) {
        std::cerr << "Cannot get fileno of stdout: " << strerror(errno) << std::endl;
    }
    int ret = write(fd, logMsg.msg, logMsg.length);
    if (ret < 0) {
        std::cerr << "Cannot write to stdout: " << ret << std::endl;
    }
}

void LogManager::WriteToFile(const LogMsg &logMsg) {

}
}
