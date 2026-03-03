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
 * \file runtime_adapter.h
 * \brief
 */

#pragma once

#include <array>
#include "adapter/manager/plugin_handler.h"

namespace npu::tile_fwk {
enum class RuntimeFunc {
    Malloc = 0,
    Memset,
    Memcpy,
    MemcpyAsync,
    Free,
    SetDevice,
    GetDevice,
    GetSocSpec,
    GetSocVersion,
    GetAiCpuCount,
    GetL2CacheOffset,
    GetLogicDevIdByUserDevId,
    StreamCreate,
    StreamDestroy,
    StreamAddToModel,
    StreamSynchronize,
    DevBinaryUnRegister,
    RegisterAllKernel,
    KernelLaunchWithHandleV2,
    AicpuKernelLaunchExWithArgs,
    Bottom
};
class RuntimeAdapter {
public:
    static RuntimeAdapter& Instance();
    void *GetFunction(const RuntimeFunc func) const {
        if (func < RuntimeFunc::Malloc || func >= RuntimeFunc::Bottom) {
            return nullptr;
        }
        return functions_[static_cast<size_t>(func)];
    }

private:
    RuntimeAdapter();
    ~RuntimeAdapter();

    PluginHandler libHandler_;
    std::array<void*, static_cast<size_t>(RuntimeFunc::Bottom)> functions_;
};
}