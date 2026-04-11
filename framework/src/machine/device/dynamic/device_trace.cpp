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
extern "C" {
__attribute__((weak)) TraHandle UtraceCreate(TracerType tracerType, const char* objName);
__attribute__((weak)) void UtraceDestroy(TraHandle handle);
__attribute__((weak)) TraStatus UtraceSubmit(TraHandle handle, const void* buffer, uint32_t bufSize);
__attribute__((weak)) TraEventHandle UtraceEventCreate(const char* eventName);
__attribute__((weak)) TraStatus UtraceEventBindTrace(TraEventHandle eventHandle, TraHandle handle);
__attribute__((weak)) TraStatus UtraceEventReport(TraEventHandle eventHandle);
__attribute__((weak)) void UtraceEventDestroy(TraEventHandle eventHandle);
}
DeviceTraceManager& DeviceTraceManager::GetInstance() {
    static DeviceTraceManager deviceTraceManager;
    std::call_once(deviceTraceManager.once_, [&]() {
        deviceTraceManager.GetTraceFunPtr();
        deviceTraceManager.Initialize(); });
    return deviceTraceManager;
}

void DeviceTraceManager::GetTraceFunPtr() {
    if (UtraceCreate != nullptr) {
        traceCreatePtr_ = UtraceCreate;
        DEV_INFO("====Trace Ctreate using api is  UtraceCreate====");
    }
    if (UtraceDestroy != nullptr) {
        traceDestoryPtr_ = UtraceDestroy;
        DEV_INFO("====Trace Ctreate using api is  UtraceDestroy====");
    }
    if (UtraceSubmit != nullptr) {
        traceSubmitPtr_ = UtraceSubmit;
        DEV_INFO("====Trace Ctreate using api is  UtraceSubmit====");
    }
    if (UtraceEventCreate != nullptr) {
        traceEventCreatePtr_ = UtraceEventCreate;
        DEV_INFO("====Trace Ctreate using api is  UtraceEventCreate====");
    }
    if (UtraceEventBindTrace != nullptr) {
        traceEventBindPtr_ = UtraceEventBindTrace;
        DEV_INFO("====Trace Ctreate using api is  UtraceEventBindTrace====");
    }
    if (UtraceEventReport != nullptr) {
        traceEventReport_ = UtraceEventReport;
        DEV_INFO("====Trace Ctreate using api is  UtraceEventReport====");
    }
    if (UtraceEventDestroy != nullptr) {
        traceEventDestory_ = UtraceEventDestroy;
        DEV_INFO("====Trace Ctreate using api is  UtraceEventDestroy====");
    }
}

void DeviceTraceManager::Initialize() {
    DEV_INFO("====Start Atrace Create ====");
    if (traceCreatePtr_ == nullptr || traceEventCreatePtr_ == nullptr || traceEventBindPtr_ == nullptr) {
        DEV_INFO("====UtraceCreate is null");
        return;
    }
    // if (AtraceCreate == nullptr) {
    //     DEV_INFO("====Atrace Create is null");
    //     return;
    // }
    pyptoHandle_ = traceCreatePtr_(TracerType::TRACER_TYPE_SCHEDULE, GlobalTraceHandleName);
    if (pyptoHandle_ < 0) {
        DEV_ERROR(DevCommonErr::GET_HANDLE_FAILED, "Create pypto trace failed");
        return;
    }
    eventHandle_ = traceEventCreatePtr_(EventTraceHandleName);
    if (eventHandle_ < 0) {
        DEV_ERROR(DevCommonErr::GET_HANDLE_FAILED, "Create pypto event trace failed");
        return;
    }
    auto status = traceEventBindPtr_(eventHandle_, pyptoHandle_);
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
    if (traceSubmitPtr_ == nullptr) {
        DEV_INFO("====UtraceSubmit is null");
        return; 
    }
    auto ret = traceSubmitPtr_(pyptoHandle_, buffer, bufSize); 
    if (ret < 0) {
        DEV_ERROR(DevCommonErr::CMD_ERROR, "Submit pyptoHandle buffer info failed, ret: %d", ret);
    }
    DEV_INFO("Submit msg: %s success", traceMsg.c_str());
}

void DeviceTraceManager::ReportPyptoTrace() {
    if (eventHandle_ < 0 || traceEventReport_ == nullptr) {
        DEV_WARN("Pypto event handle is invalid");
        return;
    }
    auto ret = traceEventReport_(eventHandle_);
    if (ret < 0) {
        DEV_ERROR(DevCommonErr::CMD_ERROR, "Report pypto envet Handle buffer info failed, ret: %d", ret);
    }
    DEV_INFO("reprt Submit Msg success");
    DestroyTraceHandle();
}

void DeviceTraceManager::DestroyTraceHandle() {
    if (traceDestoryPtr_ == nullptr || traceEventDestory_ == nullptr) {
        DEV_INFO("Destor ptr is null");
    }
    traceDestoryPtr_(pyptoHandle_);
    pyptoHandle_ = -1;
    traceDestoryPtr_(eventHandle_);
    eventHandle_ = -1;
}
} // namespace
#endif