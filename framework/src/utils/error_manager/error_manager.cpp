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
 * \file error_manager.cpp
 * \brief
 */

#include "tilefwk/error_manager.h"

#include <cstdarg>
#include <iostream>
#include "securec.h"

namespace npu::tile_fwk {
namespace {
constexpr size_t MAX_MSG_LENGTH = 1024;
}
ErrorManager::ErrorManager() {}

ErrorManager::~ErrorManager() {}

ErrorManager& ErrorManager::Instance()
{
    static ErrorManager instance;
    return instance;
}

void ErrorManager::ReportErrorMessage(const char *fmt, ...)
{
    va_list list;
    va_start(list, fmt);
    std::string errMsg = ConstructMessage(fmt, list);
    va_end(list);
    if (errMsg.empty()) {
        return;
    }
    const std::lock_guard<std::mutex> lockGuard(reportMutex_);
    errorMsgQueue_.push(errMsg);
}

std::string ErrorManager::ConstructMessage(const char *fmt, va_list list)
{
    char msg[MAX_MSG_LENGTH];
    int ret = vsnprintf_truncated_s(msg, MAX_MSG_LENGTH, fmt, list);
    if (ret < 0) {
        std::cerr << "Construct error message failed: " << ret << std::endl;
        return std::string();
    }
    return std::string(msg);
}

void ErrorManager::OutputErrorMessage(const bool outputAll)
{
    const std::lock_guard<std::mutex> lockGuard(reportMutex_);
    if (outputAll) {
        while (!errorMsgQueue_.empty()) {
            std::cerr << errorMsgQueue_.front() << std::endl;
            errorMsgQueue_.pop();
        }
    } else {
        if (!errorMsgQueue_.empty()) {
            std::cerr << errorMsgQueue_.front() << std::endl;
            errorMsgQueue_.pop();
        }
    }
}
} // namespace npu::tile_fwk
