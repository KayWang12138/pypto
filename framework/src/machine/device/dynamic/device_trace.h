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
 * \brief
 */
#pragma once
#ifndef DEVICE_TRACE_H
#define DEVICE_TRACE_H

#include <string>
#include <mutex>
// #ifdef BUILD_WITH_CANN
#ifdef __DEVICE__
#include "trace/atrace_types.h"
#include "trace/atrace_pub.h"


namespace npu::tile_fwk::dynamic {
class DeviceTraceManager {
public:
    static DeviceTraceManager &GetInstance();
    DeviceTraceManager(const DeviceTraceManager &) = delete;
    DeviceTraceManager &operator=(const DeviceTraceManager &) = delete;
    void Initialize();
    void Finalize();
    void SubmitPyptoTrace(const std::string &traceMsg) const;
    void ReportPyptoTrace();
    

private:
    DeviceTraceManager() = default;
    ~DeviceTraceManager() = default;
    void DestroyTraceHandle();
    TraHandle pyptoHandle_{-1};
    TraEventHandle eventHandle_{-1};
    mutable std::mutex pyptoTraceMutex_;
    std::once_flag once_;

};
} // namespace
#endif // build with can
#endif