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
 * \file dlog_handler.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstdarg>

namespace tile::fwk {
class DLogHandler {
public:
    static DLogHandler &Instance();
    DLogHandler();
    ~DLogHandler();
    bool Enable() const { return enable_; }
    bool CheckLogLevel(int32_t level);
    void LogRecord(int32_t level, const char *fmt, va_list list);
private:
    void CloseHandle();
private:
    bool enable_{false};
    void *handle_{nullptr};
    int32_t(*checkLevelFunc_)(int32_t, int32_t);
    void(*logRecordFunc_)(int32_t, int32_t, const char *, ...);
};
}
