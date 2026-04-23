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
 * \file pmu_ko_loader_example.h
 * \brief PMU 内核模块加载使用示例
 *
 *  演示如何将 pmu_user_access.ko 嵌入代码并在 device 启动时加载
 */

#pragma once

#ifdef __DEVICE__

#include "machine/utils/device_switch.h"
#if defined(__has_include)
#if __has_include("machine/utils/pmu_user_access_ko_embedded.h")
#include "machine/utils/pmu_user_access_ko_embedded.h"
#endif
#endif
#include "machine/utils/pmu_ko_loader.h"

namespace npu::tile_fwk {

/**
 * @brief 初始化 PMU 用户态访问能力
 *
 *  在 device 启动时调用，加载内嵌的内核模块。
 *  成功后，ARM PMU 直读模式即可使用。
 *
 * @return 0 成功，负值失败
 *
 * 使用示例（在 device_ctrl.h 的初始化阶段调用）：
 *
 * @code
 * #ifdef __DEVICE__
 * #if ARM_PMU_DIRECT_ENABLE
 *     // 加载 PMU 内核模块（首次启动时）
 *     PmuKoLoaderExample::InitPmuUserAccess();
 * #endif
 * #endif
 * @endcode
 */
struct PmuKoLoaderExample {
    static bool& Loaded()
    {
        static bool loaded = false;
        return loaded;
    }

    static int InitPmuUserAccess()
    {
        if (Loaded()) {
            DEV_INFO("[PMU_KO] Already loaded, skip");
            return 0;
        }

        int ret = PmuInitEmbeddedKo();

        if (ret == 0) {
            Loaded() = true;
            DEV_INFO("[PMU_KO] PMU user access enabled successfully");
        } else {
            DEV_WARN("[PMU_KO] Failed to load PMU module, ARM direct PMU may not work");
        }

        return ret;
    }

    /**
     * @brief 检查内核模块是否已加载
     */
    static bool IsLoaded() { return Loaded(); }
};

} // namespace npu::tile_fwk

#endif // __DEVICE__