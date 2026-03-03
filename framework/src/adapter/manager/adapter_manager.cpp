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
 * \file adapter_manager.cpp
 * \brief
 */

#include "adapter/manager/adapter_manager.h"

namespace npu::tile_fwk {
AdapterManager& AdapterManager::Instance() {
    static AdapterManager adapterManager;
    return adapterManager;
}

AdapterManager::AdapterManager() {
    if (!aclAdapter_.Initialize(kAclLibName, kAclFuncStrMap)) {
        ADAPTER_LOGI("Acl adapter has not been initialized from library[%s].", kAclLibName.c_str());
    }
    ADAPTER_LOGI("Acl adapter has been initialized from library[%s] successfully.", kAclLibName.c_str());
    if (!hcclAdapter_.Initialize(kHcclLibName, kHcclFuncStrMap)) {
        ADAPTER_LOGI("Hccl adapter has not been initialized from library[%s].", kHcclLibName.c_str());
    }
    ADAPTER_LOGI("Hccl adapter has been initialized from library[%s] successfully.", kHcclLibName.c_str());
    if (!msprofAdapter_.Initialize(kMsprofLibName, kMsprofFuncStrMap)) {
        ADAPTER_LOGI("Msprof adapter has not been initialized from library[%s].", kMsprofLibName.c_str());
    }
    ADAPTER_LOGI("Msprof adapter has been initialized from library[%s] successfully.", kMsprofLibName.c_str());
    if (!runtimeAdapter_.Initialize(kRuntimeLibName, kRuntimeFuncStrMap)) {
        ADAPTER_LOGI("Runtime adapter has not been initialized from library[%s].", kRuntimeLibName.c_str());
    }
    ADAPTER_LOGI("Runtime adapter has been initialized from library[%s] successfully.", kRuntimeLibName.c_str());
}

AdapterManager::~AdapterManager() {}
}
