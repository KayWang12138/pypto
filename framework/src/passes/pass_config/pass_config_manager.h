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
 * \file pass_config_manager.h
 * \brief
 */
#ifndef PASS_CONFIG_MANAGER_H_
#define PASS_CONFIG_MANAGER_H_
#include <string>
#include "interface/utils/common.h"
#include "passes/pass_config/platform_config.h"
namespace npu{
namespace tile_fwk {
class PassConfigManager {
  public:
    static PassConfigManager &Instance();
    Status Initialize(DPlatform id);
    PassConfigManager(const PassConfigManager &) = delete;
    PassConfigManager &operator=(const PassConfigManager &) = delete;
    const PlatformConfig &GetPlatformConfig() const {
      return passPlatformInfo_;
    };
  private:
    PlatformConfig passPlatformInfo_;
    PassConfigManager() = default;
    ~PassConfigManager() = default;
};
} // namespace tile_fwk
} // namespace npu
#endif