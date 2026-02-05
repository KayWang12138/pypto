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
 * \file tilefwk_log.h
 * \brief
 */

#pragma once

#include <sys/syscall.h>
#include <cstdint>

#ifndef __DEVICE__
#include "dlog_pub.h"
#else
#define DLOG_DEBUG 0x0
#define DLOG_INFO  0x1
#define DLOG_WARN  0x2
#define DLOG_ERROR 0x3
#define DLOG_NULL  0x4
#endif

// only using slog inside device
#ifdef __DEVICE__
#define PYPTO_RECORD_SLOG(level, fmt, ...)                                                              \
    do {                                                                                                \
        if (CheckLogLevel(FE, level) == 1) {                                                            \
            DlogRecord(FE, level, "[%s:%d][%s]:" fmt, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
        }                                                                                               \
    } while (0)

#else
namespace tile::fwk {
bool PyptoCheckLogLevel(const int32_t logLevel);
void PyptoLogRecord(const int32_t logLevel, const char *fmt, ...);
}

#define PYPTO_RECORD_TKLOG(level, fmt, ...)                                                                        \
    do {                                                                                                           \
        if (tile::fwk::PyptoCheckLogLevel(level)) {                                                                \
            tile::fwk::PyptoLogRecord(level, "[%s:%d][%s]:" fmt, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
        }                                                                                                          \
    } while (0)

#endif

#if defined(__DEVICE__)
#define PYPTO_LOG_INNER(level, fmt, ...) PYPTO_RECORD_SLOG(level, fmt, ##__VA_ARGS__)
#else
#define PYPTO_LOG_INNER(level, fmt, ...) PYPTO_RECORD_TKLOG(level, fmt, ##__VA_ARGS__)
#endif

#define PYPTO_LOGD(...) PYPTO_LOG_INNER(DLOG_DEBUG, __VA_ARGS__)
#define PYPTO_LOGI(...) PYPTO_LOG_INNER(DLOG_INFO, __VA_ARGS__)
#define PYPTO_LOGW(...) PYPTO_LOG_INNER(DLOG_WARN, __VA_ARGS__)
#define PYPTO_LOGE(...) PYPTO_LOG_INNER(DLOG_ERROR, __VA_ARGS__)
