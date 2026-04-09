/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trace_stub.cpp
 * \brief
 */

#ifdef __DEVICE__
#include <stdint.h>

typedef int32_t TraStatus;
typedef intptr_t TraHandle;
typedef intptr_t TraEventHandle;

extern "C" {
TraHandle AtraceCreate(int tracerType, const char* objName);
TraStatus AtraceSubmit(TraHandle handle, const void* buffer, uint32_t bufSize);
void AtraceDestroy(TraHandle handle);
TraEventHandle AtraceEventCreate(const char* eventName);
TraStatus AtraceEventBindTrace(TraEventHandle eventHandle, TraHandle handle);
TraStatus AtraceEventReportSync(TraEventHandle eventHandle);
void AtraceEventDestroy(TraEventHandle eventHandle);
}

TraHandle AtraceCreate([[maybe_unused]] int tracerType, [[maybe_unused]] const char* objName) { return 0; }

TraStatus AtraceSubmit(
    [[maybe_unused]] TraHandle handle, [[maybe_unused]] const void* buffer, [[maybe_unused]] uint32_t bufSize)
{
    return 0;
}

void AtraceDestroy([[maybe_unused]] TraHandle handle) {}

TraEventHandle AtraceEventCreate([[maybe_unused]] const char* eventName) { return 0; }

TraStatus AtraceEventBindTrace([[maybe_unused]] TraEventHandle eventHandle, [[maybe_unused]] TraHandle handle)
{
    return 0;
}

TraStatus AtraceEventReportSync([[maybe_unused]] TraEventHandle eventHandle) { return 0; }

void AtraceEventDestroy([[maybe_unused]] TraEventHandle eventHandle) {}

#endif
