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
 * \file log_api.cpp
 * \brief
 */

#include <map>
#include <cstdarg>

#include "tilefwk/tilefwk_log.h"
#include "log_manager.h"
#include "dlog_handler.h"

namespace tile::fwk {
namespace {
const std::map<int32_t, LogLevel> LOG_LEVEL_MAP = {
    {DLOG_DEBUG, LogLevel::DEBUG},
    {DLOG_INFO, LogLevel::INFO},
    {DLOG_WARN, LogLevel::WARN},
    {DLOG_ERROR, LogLevel::ERROR}
};
LogLevel GetLogLevel(const int32_t logLevel) {
    auto iter = LOG_LEVEL_MAP.find(logLevel);
    return iter == LOG_LEVEL_MAP.end() ? LogLevel::ERROR : iter->second;
}
}

bool PyptoCheckLogLevel(const int32_t logLevel) {
    if (DLogHandler::Instance().Enable()) {
        return DLogHandler::Instance().CheckLogLevel(logLevel);
    } else {
        return LogManager::Instance().CheckLevel(GetLogLevel(logLevel));
    }
}

void PyptoLogRecord(const int32_t logLevel, const char *fmt, ...) {
    va_list list;
    va_start(list, fmt);
    if (DLogHandler::Instance().Enable()) {
        DLogHandler::Instance().LogRecord(logLevel, fmt, list);
    } else {
        LogManager::Instance().Record(GetLogLevel(logLevel), fmt, list);
    }
    va_end(list);
}
}
