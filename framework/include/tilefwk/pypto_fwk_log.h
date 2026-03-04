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
 * \file pypto_fwk_log.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>

#define DLOG_DEBUG 0x0
#define DLOG_INFO  0x1
#define DLOG_WARN  0x2
#define DLOG_ERROR 0x3

#define PYPTO 59

#ifndef __DEVICE__
#ifndef __FILE_NAME__
#define __FILE_NAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

namespace npu::tile_fwk {
enum class LogModule {
    FUNCTION = 0,
    PASS,
    CODEGEN,
    MACHINE,
    DISTRIBUTED,
    SIMULATION,
    VERIFY,
    BOTTOM
};
const std::string& GetLogModuleName(const LogModule logModule);

class LogFuncInfo {
public:
    static LogFuncInfo &Instance();
    int32_t(*checkLevel)(int32_t, int32_t, LogModule);
    void(*record)(int32_t, int32_t, const char *, ...);
    void(*pyptoRecord)(int32_t, int32_t, const char *, ...);
    void(*setAttr)(bool);
private:
    LogFuncInfo();
    ~LogFuncInfo();
};
}

#define PYPTO_HOST_LOG(level, module, fmt, ...)                                                                                                  \
    do {                                                                                                                                         \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                                                         \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(false);                                                                               \
        }                                                                                                                                        \
        if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel != nullptr && npu::tile_fwk::LogFuncInfo::Instance().record != nullptr) {          \
            if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel(PYPTO, level)) {                                                               \
                npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d][%s]:" fmt, __FILE_NAME__, __LINE__, GetLogModuleName(module).c_str(), ##__VA_ARGS__); \
            }                                                                                                                                    \
        }                                                                                                                                        \
    } while (0)

#define MAX_LOG_LENGTH 880

#define PYPTO_HOST_SPLIT_LOG(level, module, fmt, ...)                                                                                            \
    do {                                                                                                                                         \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                                                         \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(false);                                                                               \
        }                                                                                                                                        \
        if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel == nullptr || npu::tile_fwk::LogFuncInfo::Instance().record == nullptr) {          \
            return;                                                                                                                              \
        }                                                                                                                                        \
        if  (!npu::tile_fwk::LogFuncInfo::Instance().checkLevel(PYPTO, level)) {                                                                 \
            return;                                                                                                                              \
        }                                                                                                                                        \
        char *formatStr = nullptr;                                                                                                               \
        int len = asprintf(&formatStr, fmt, ##__VA_ARGS__);                                                                                      \
        if (len <= 0 || formatStr == nullptr) {                                                                                                  \
            return;                                                                                                                              \
        }                                                                                                                                        \
        if (len < MAX_LOG_LENGTH) {                                                                                                              \
            npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d][%s]:%s", __FILE_NAME__, __LINE__, module, formatStr);           \
        } else {                                                                                                                                 \
            char *msgBegin = formatStr;                                                                                                          \
            char *msgEnd = formatStr + len;                                                                                                      \
            while (msgBegin < msgEnd) {                                                                                                          \
                std::string logMsg(msgBegin, std::min(static_cast<size_t>(MAX_LOG_LENGTH), static_cast<size_t>(msgEnd - msgBegin)));             \
                npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d][%s]:%s", __FILE_NAME__, __LINE__, module, logMsg.c_str());  \
                msgBegin += logMsg.size();                                                                                                       \
            }                                                                                                                                    \
        }                                                                                                                                        \
        free(formatStr);                                                                                                                         \
    } while (0)

#define PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(level, module, fmt, ...)                                                                              \
    do {                                                                                                                                         \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                                                         \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(false);                                                                               \
        }                                                                                                                                        \
        if (npu::tile_fwk::LogFuncInfo::Instance().record != nullptr) {                                                                          \
            npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d][%s]:" fmt, __FILE_NAME__, __LINE__, module, ##__VA_ARGS__);     \
        }                                                                                                                                        \
    } while (0)

#define PYPTO_SIM_LOG(level, module, fmt, ...)                                                                                                   \
    do {                                                                                                                                         \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                                                         \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(true);                                                                                \
        }                                                                                                                                        \
        if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel != nullptr && npu::tile_fwk::LogFuncInfo::Instance().record != nullptr) {          \
            if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel(PYPTO, level)) {                                                               \
                npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d][%s]:" fmt, __FILE_NAME__, __LINE__, module, ##__VA_ARGS__); \
            }                                                                                                                                    \
        }                                                                                                                                        \
    } while (0)

#define FUNCTION_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, LogModule::FUNCTION, __VA_ARGS__)
#define FUNCTION_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, LogModule::FUNCTION, __VA_ARGS__)
#define FUNCTION_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, LogModule::FUNCTION, __VA_ARGS__)
#define FUNCTION_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, LogModule::FUNCTION, __VA_ARGS__)
#define FUNCTION_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, LogModule::FUNCTION, __VA_ARGS__)
#define FUNCTION_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, LogModule::FUNCTION, __VA_ARGS__)

#define PASS_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, LogModule::PASS, __VA_ARGS__)
#define PASS_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, LogModule::PASS, __VA_ARGS__)
#define PASS_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, LogModule::PASS, __VA_ARGS__)
#define PASS_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, LogModule::PASS, __VA_ARGS__)
#define PASS_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, LogModule::PASS, __VA_ARGS__)
#define PASS_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, LogModule::PASS, __VA_ARGS__)

#define CODEGEN_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, LogModule::CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, LogModule::CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, LogModule::CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, LogModule::CODEGEN, __VA_ARGS__)
#define CODEGEN_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, LogModule::CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, LogModule::CODEGEN, __VA_ARGS__)

#define MACHINE_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, LogModule::MACHINE, __VA_ARGS__)
#define MACHINE_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, LogModule::MACHINE, __VA_ARGS__)
#define MACHINE_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, LogModule::MACHINE, __VA_ARGS__)
#define MACHINE_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, LogModule::MACHINE, __VA_ARGS__)
#define MACHINE_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, LogModule::MACHINE, __VA_ARGS__)
#define MACHINE_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, LogModule::MACHINE, __VA_ARGS__)

#define DISTRIBUTED_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, LogModule::DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, LogModule::DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, LogModule::DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, LogModule::DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, LogModule::DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, LogModule::DISTRIBUTED, __VA_ARGS__)

#define SIMULATION_LOGD(...) PYPTO_SIM_LOG(DLOG_DEBUG, LogModule::SIMULATION, __VA_ARGS__)
#define SIMULATION_LOGI(...) PYPTO_SIM_LOG(DLOG_INFO, LogModule::SIMULATION, __VA_ARGS__)
#define SIMULATION_LOGW(...) PYPTO_SIM_LOG(DLOG_WARN, LogModule::SIMULATION, __VA_ARGS__)
#define SIMULATION_LOGE(...) PYPTO_SIM_LOG(DLOG_ERROR, LogModule::SIMULATION, __VA_ARGS__)

#define VERIFY_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, LogModule::VERIFY, __VA_ARGS__)
#define VERIFY_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, LogModule::VERIFY, __VA_ARGS__)
#define VERIFY_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, LogModule::VERIFY, __VA_ARGS__)
#define VERIFY_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, LogModule::VERIFY, __VA_ARGS__)
#define VERIFY_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, LogModule::VERIFY, __VA_ARGS__)
#define VERIFY_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, LogModule::VERIFY, __VA_ARGS__)

#endif