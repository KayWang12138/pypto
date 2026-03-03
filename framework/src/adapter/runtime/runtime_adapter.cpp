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
 * \file runtime_adapter.cpp
 * \brief
 */

#include "runtime/runtime_adapter.h"
#include <string>
#include <map>

namespace npu::tile_fwk {
namespace {
const std::string kRuntimeLibName = "libruntime.so";
const std::map<RuntimeFunc, std::string> kRuntimeFuncStrMap {
    {RuntimeFunc::Malloc, "rtMalloc"},
    {RuntimeFunc::Memset, "rtMemset"},
    {RuntimeFunc::Memcpy, "rtMemcpy"},
    {RuntimeFunc::MemcpyAsync, "rtMemcpyAsync"},
    {RuntimeFunc::Free, "rtFree"},
    {RuntimeFunc::SetDevice, "rtSetDevice"},
    {RuntimeFunc::GetDevice, "rtGetDevice"},
    {RuntimeFunc::GetSocSpec, "rtGetSocSpec"},
    {RuntimeFunc::GetSocVersion, "rtGetSocVersion"},
    {RuntimeFunc::GetAiCpuCount, "rtGetAiCpuCount"},
    {RuntimeFunc::GetL2CacheOffset, "rtGetL2CacheOffset"},
    {RuntimeFunc::GetLogicDevIdByUserDevId, "rtGetLogicDevIdByUserDevId"},
    {RuntimeFunc::StreamCreate, "rtStreamCreate"},
    {RuntimeFunc::StreamDestroy, "rtStreamDestroy"},
    {RuntimeFunc::StreamAddToModel, "rtStreamAddToModel"},
    {RuntimeFunc::StreamSynchronize, "rtStreamSynchronize"},
    {RuntimeFunc::DevBinaryUnRegister, "rtDevBinaryUnRegister"},
    {RuntimeFunc::RegisterAllKernel, "rtRegisterAllKernel"},
    {RuntimeFunc::KernelLaunchWithHandleV2, "rtKernelLaunchWithHandleV2"},
    {RuntimeFunc::AicpuKernelLaunchExWithArgs, "rtAicpuKernelLaunchExWithArgs"}
};
}
RuntimeAdapter& RuntimeAdapter::Instance() {
    static RuntimeAdapter runtimeAdapter;
    return runtimeAdapter;
}

RuntimeAdapter::RuntimeAdapter() {
    if (!libHandler_.OpenHandler(kRuntimeLibName)) {
        return;
    }
    functions_.fill(nullptr);
    for (const std::pair<const RuntimeFunc, std::string> &item : kRuntimeFuncStrMap) {
        void *func = libHandler_.GetFunction(item.second);
        if (func == nullptr) {
            continue;
        }
        functions_[static_cast<size_t>(item.first)] = func;
    }
}

RuntimeAdapter::~RuntimeAdapter() {
    libHandler_.CloseHandler();
    functions_.fill(nullptr);
}
}