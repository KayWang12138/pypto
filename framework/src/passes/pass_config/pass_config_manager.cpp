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
 * \file pass_config_manager.cpp
 * \brief
 */
#include "passes/pass_config/pass_config_manager.h"
namespace npu {
namespace tile_fwk {
Status PassConfigManager::Initialize(DPlatform id) {
    passPlatformInfo_.InitPlatformConfig(id);
    return SUCCESS;
}

PassConfigManager &PassConfigManager::Instance() {
    static PassConfigManager instance;
    return instance;
}
} // namespace tile_fwk
} // namespace npu