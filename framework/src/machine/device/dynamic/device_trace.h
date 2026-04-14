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
 * \file device_trace.h
 * \brief Device trace functionality with dynamic library loading
 *
 * This header provides device trace capabilities by dynamically loading
 * trace libraries (libascend_trace.so or libutrace.so) and providing
 * unified interfaces for trace operations.
 *
 * Key features:
 * - Automatic library detection and loading
 * - Thread-safe singleton pattern
 * - Support for both Atrace and Utrace backends
 * - PyPTO-style error handling and logging
 */

#pragma once
#ifndef DEVICE_TRACE_H
#define DEVICE_TRACE_H

#include <functional>
#include <mutex>
#include <string>
#include <cstdint>
#include "machine/utils/device_log.h"
#include "machine/utils/machine_error.h"

#define MAX_MSG_LEN 112
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
static void TraceInfo(const char* fmt, ...)
{
    char traceBuf[MAX_MSG_LEN] = {0};

    auto ret = snprintf_s(traceBuf, MAX_MSG_LEN, MAX_MSG_LEN - 1, fmt);
    if (ret != 0) {
        DEV_WARN("Trace info is not success");
        return;
    }
    npu::tile_fwk::dynamic::DeviceTrace::GetInstance().SubmitTraceMsg(traceBuf);
}

// 上层包装：自动带入 文件名、行号、函数名
#define TRACE_INFO(fmt, ...) \
    TraceInfo("[%s:%d] %s: " fmt, FILENAME, __LINE__, __func__, ##__VA_ARGS__) \

// #define TRACE_INFO(fmt, ...)            \
// do {                                    \
//     char traceBuf[MAX_MSG_LEN] = {0};          \
//     snprintf_s(traceBuf, MAX_MSG_LEN, MAX_MSG_LEN - 1, "[%s:%d] %s: " fmt, \
//              FILENAME, __LINE__, __func__, ##__VA_ARGS__);                       \
//     npu::tile_fwk::dynamic::DeviceTrace::GetInstance().SubmitTraceMsg(traceBuf); \
// } while(0)                                                                       \

#ifdef __DEVICE__
#include "trace/atrace_types.h"
#include "trace/atrace_pub.h"

namespace npu::tile_fwk::dynamic {

enum class TraceError : uint32_tDEVICE_MACHINE_OK
    PYPTO_TRACE_SUCCESS = 0,
    LOAD_LIBRARY_FAILED = 1,
    SYMBOL_NOT_FOUND = 2,
    INVALID_HANDLE = 3,
    SUBMIT_FAILED = 4,
    SAVE_FAILED = 5,
};

/**
 * \brief Device trace manager with dynamic library loading
 *
 * This singleton class manages device trace functionality by:
 * - Dynamically loading trace libraries at runtime
 * - Providing unified interfaces for trace operations
 * - Supporting both Atrace and Utrace backends
 * - Ensuring thread-safe initialization and access
 *
 * Thread-safety: All public methods are thread-safe through mutex protection.
 */
class DeviceTrace {
public:
    ~DeviceTrace();
    TraceError CreateTraceHandle(void *targ);
    void SubmitTraceMsg(const std::string &traceMsg);
    void ReportTraceMsg();
    /**
     * \brief Get the singleton instance of DeviceTrace
     * \return Reference to the singleton DeviceTrace instance
     */
    static DeviceTrace& GetInstance();

    /**
     * \brief Initialize the device trace manager
     *
     * Attempts to load trace libraries in the following order:
     * 1. libascend_trace.so (preferred)
     * 2. libutrace.so (fallback)
     *
     * \return SUCCESS if initialization succeeds, error code otherwise
     */
    TraceError Initialize(void *targ);

private:

    /**
     * \brief Destroy a trace handle
     * \param handle Trace handle to destroy
     */
    std::function<void(TraHandle handle)> TraceDestroy{};

    /**
     * \brief Submit trace data
     * \param handle Trace handle
     * \param buffer Pointer to trace data buffer
     * \param bufSize Size of the trace data buffer
     * \return SUCCESS if submission succeeds, error code otherwise
     */
    std::function<TraStatus(TraHandle handle, const void* buffer, uint32_t bufSize)> TraceSubmit{};

    /**
     * \brief Create a trace handle
     * \param tracerType Type of tracer to create
     * \param objName Name of the trace object
     * \return Trace handle on success, nullptr on failure
     */
    std::function<TraHandle(TracerType tracerType, const char* objName)> TraceCreate{};

    /**
     * \brief Save trace data
     * \param tracerType Type of tracer to save
     * \param syncFlag Whether to save synchronously
     * \return Trace status
     */
    std::function<TraStatus(TracerType tracerType, bool syncFlag)> TraceSave{};

    /**
     * \brief Create a trace event handle
     * \param eventName Name of the event
     * \return Event handle on success, nullptr on failure
     */
    std::function<TraEventHandle(const char* eventName)> TraceEventCreate{};

    /**
     * \brief Bind a trace event to a trace handle
     * \param eventHandle Event handle to bind
     * \param handle Trace handle to bind to
     * \return Trace status
     */
    std::function<TraStatus(TraEventHandle eventHandle, TraHandle handle)> TraceEventBindTrace{};

    /**
     * \brief Report a trace event
     * \param eventHandle Event handle to report
     * \return Trace status
     */
    std::function<TraStatus(TraEventHandle eventHandle)> TraceEventReport{};

    /**
     * \brief Report a trace event synchronously
     * \param eventHandle Event handle to report
     * \return Trace status
     */
    std::function<TraStatus(TraEventHandle eventHandle)> TraceEventReportSync{};

    /**
     * \brief Destroy a trace event handle
     * \param eventHandle Event handle to destroy
     */
    std::function<void(TraEventHandle eventHandle)> TraceEventDestroy{};

    std::function<TraStatus(const TraceGlobalAttr *attr)> TraceSetGlobalAttr{};

private:
    void* handle_{nullptr};
    std::mutex mutex_;
    std::mutex pyptoTraceMutex_;
    TraHandle pyptoHandle_{-1};
    TraEventHandle eventHandle_{-1};

    DeviceTrace(const DeviceTrace&) = delete;
    DeviceTrace& operator=(const DeviceTrace&) = delete;
    DeviceTrace() = default;
    TraceError UtraceInitialize();
    TraceError AtraceInitialize();

    /**
     * \brief Initialize Atrace function pointers
     * \return SUCCESS if successful, error code otherwise
     */
    TraceError InitializeAtraceFunctions();

    /**
     * \brief Initialize Utrace function pointers
     * \return SUCCESS if successful, error code otherwise
     */
    TraceError InitializeUtraceFunctions();

    /**
     * \brief Load a dynamic library
     * \param libraryName Name of the library to load
     * \return Handle to the loaded library, nullptr on failure
     */
    void* LoadLibrary(const std::string& libraryName);

    /**
     * \brief Get a symbol from the loaded library
     * \param symbolName Name of the symbol to retrieve
     * \return Pointer to the symbol, nullptr on failure
     */
    void* GetSymbol(const std::string& symbolName);
};
}
#else
namespace npu::tile_fwk::dynamic {
class DeviceTrace {
public:
    static DeviceTrace& GetInstance() {
        static DeviceTrace deviceTrace;
        return deviceTrace;
    }
    void SubmitTraceMsg(const std::string &traceMsg) {
        DEV_INFO("Trace submit msg: %s", traceMsg.c_str());
    }
    void ReportTraceMsg() {}
};
} // namespace npu::tile_fwk::dynamic
#endif
#endif // DEVICE_TRACE_H
