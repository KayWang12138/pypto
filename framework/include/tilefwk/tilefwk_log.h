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

#include <cstdint>
#include <cstring>

#ifdef __DEVICE__
#include "dlog_pub.h"
#define PYPTO AICPU
#else
#define PYPTO 36
#define DLOG_DEBUG 0x0
#define DLOG_INFO  0x1
#define DLOG_WARN  0x2
#define DLOG_ERROR 0x3
#endif

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// only using slog inside device
#ifdef __DEVICE__
#define INNFER_PYPTO_LOG(level, fmt, ...)                                                                      \
    do {                                                                                                       \
        if (CheckLogLevel(PYPTO, level) == 1) {                                                                \
            DlogRecord(PYPTO, level, "[%s:%d][%s]:" fmt, __FILENAME__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
        }                                                                                                      \
    } while (0)

#else
class PyptoLogFuncInstance {
public:
    PyptoLogFuncInstance();
    ~PyptoLogFuncInstance();
    int32_t(*checkLevel)(int32_t, int32_t);
    void(*record)(int32_t, int32_t, const char *, ...);
};
inline PyptoLogFuncInstance logFuncInst;

#define INNFER_PYPTO_LOG(level, fmt, ...)                                                                                  \
    do {                                                                                                                   \
        if (logFuncInst.checkLevel != nullptr && logFuncInst.record != nullptr) {                                          \
            if (logFuncInst.checkLevel(PYPTO, level)) {                                                                    \
                logFuncInst.record(PYPTO, level, "[%s:%d][%s]:" fmt, __FILENAME__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
            }                                                                                                              \
        }                                                                                                                  \
    } while (0)

#endif

#define PYPTO_LOGD(...) INNFER_PYPTO_LOG(DLOG_DEBUG, __VA_ARGS__)
#define PYPTO_LOGI(...) INNFER_PYPTO_LOG(DLOG_INFO, __VA_ARGS__)
#define PYPTO_LOGW(...) INNFER_PYPTO_LOG(DLOG_WARN, __VA_ARGS__)
#define PYPTO_LOGE(...) INNFER_PYPTO_LOG(DLOG_ERROR, __VA_ARGS__)
