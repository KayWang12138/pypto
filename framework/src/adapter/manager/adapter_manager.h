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
 * \file adapter_manager.h
 * \brief
 */

#pragma once

#include "adapter/manager/cann_adapter.h"

namespace npu::tile_fwk {
enum class HcclFunc {
    GetCommName = 0,
    GetL0TopoTypeEx,
    GetCommHandleByGroup,
    GetRootInfo,
    CommInitRootInfo,
    AllocComResourceByTiling,
    Bottom
};
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
    FuncGetByName,
    BinaryLoadFromFile,
    StreamCreate,
    StreamDestroy,
    StreamAddToModel,
    StreamSynchronize,
    DevBinaryUnRegister,
    RegisterAllKernel,
    LaunchCpuKernel,
    KernelLaunchWithHandleV2,
    AicpuKernelLaunchExWithArgs,
    Bottom
};
class AdapterManager {
public:
    static AdapterManager& Instance();
    const CannAdapter<RuntimeFunc>& GetRuntimeAdapter() const {
        return runtimeAdapter_;
    }
    const CannAdapter<HcclFunc>& GetHcclAdapter() const {
        return hcclAdapter_;
    }

private:
    AdapterManager();
    ~AdapterManager();
    CannAdapter<HcclFunc> hcclAdapter_;
    CannAdapter<RuntimeFunc> runtimeAdapter_;
};
}
