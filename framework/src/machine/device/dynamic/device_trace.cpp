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
 * \file device_trace.cpp
 * \brief
 */

#ifdef __DEVICE__
#include "device_trace.h"
#include "machine/utils/device_log.h"
#include "machine/utils/machine_error.h"

namespace npu::tile_fwk::dynamic {
namespace {
constexpr char const *GlobalTraceHandleName = "PYPTO_Global_Trace";
constexpr char const *EventTraceHandleName = "PYPTO_Event_Trace";
constexpr uint32_t MAX_MSG_LEN = 112;
}
DeviceTraceManager& DeviceTraceManager::GetInstance() {
    static DeviceTraceManager deviceTraceManager;
    std::call_once(deviceTraceManager.once_, [&]() { deviceTraceManager.Initialize(); });
    return deviceTraceManager;
}

void DeviceTraceManager::Initialize() {
    DEV_INFO("====Start Atrace Create ====");
    if (AtraceCreate == nullptr) {
        DEV_INFO("====Atrace Create is null");
        return;
    }
    pyptoHandle_ = AtraceCreate(TracerType::TRACER_TYPE_SCHEDULE, GlobalTraceHandleName);
    if (pyptoHandle_ < 0) {
        DEV_ERROR(DevCommonErr::GET_HANDLE_FAILED, "Create pypto trace failed");
        return;
    }
    eventHandle_ = AtraceEventCreate(EventTraceHandleName);
    if (eventHandle_ < 0) {
        DEV_ERROR(DevCommonErr::GET_HANDLE_FAILED, "Create pypto event trace failed");
        return;
    }
    auto status = AtraceEventBindTrace(eventHandle_, pyptoHandle_);
    if (status < 0) {
        DEV_ERROR(DevCommonErr::PARAM_CHECK_FAILED, "Bind pypto trace handle to pypto event trace failed, error status: %d", status);
    }
}

void DeviceTraceManager::SubmitPyptoTrace(const std::string &traceMsg) const {
    std::lock_guard<std::mutex> lock_guard(pyptoTraceMutex_);
    if (pyptoHandle_  < 0 || traceMsg.empty()) {
        DEV_WARN("pypto Handle is null or traceMsg is empty, cann't to submit");
        return;
    }
    uint32_t msgSize = static_cast<uint32_t>(traceMsg.size());
    const void *buffer = reinterpret_cast<const void *>(traceMsg.c_str());
    uint32_t bufSize = msgSize > MAX_MSG_LEN ? MAX_MSG_LEN : msgSize;
    auto ret = AtraceSubmit(pyptoHandle_, buffer, bufSize); 
    if (ret < 0) {
        DEV_ERROR(DevCommonErr::CMD_ERROR, "Submit pyptoHandle buffer info failed, ret: %d", ret);
    }
}

void DeviceTraceManager::ReportPyptoTrace() {
    if (eventHandle_ < 0) {
        DEV_WARN("Pypto event handle is invalid");
        return;
    }
    auto ret = AtraceEventReportSync(eventHandle_);
    if (ret < 0) {
        DEV_ERROR(DevCommonErr::CMD_ERROR, "Report pypto envet Handle buffer info failed, ret: %d", ret);
    }
    DestroyTraceHandle();
}

void DeviceTraceManager::DestroyTraceHandle() {
    AtraceDestroy(pyptoHandle_);
    pyptoHandle_ = -1;
    AtraceEventDestroy(eventHandle_);
    eventHandle_ = -1;
}
} // namespace
#endif