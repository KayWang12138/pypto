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
 * \file arch_config.h
 * \brief 编译时架构配置宏定义，用于消除运行时版本判断
 */

#pragma once

#include <cstdint>

#if defined(DAV_ARCH) && DAV_ARCH == 3510
    #define PTO_ARCH_DAV_3510 1
    #define PTO_ARCH_DAV_2210 0
    #define PTO_ARCH_NAME "DAV_3510"
#elif defined(DAV_ARCH) && DAV_ARCH == 2210
    #define PTO_ARCH_DAV_3510 0
    #define PTO_ARCH_DAV_2210 1
    #define PTO_ARCH_NAME "DAV_2210"
#else
    #define PTO_ARCH_DAV_3510 0
    #define PTO_ARCH_DAV_2210 1
    #define PTO_ARCH_NAME "DAV_2210"
#endif

#if PTO_ARCH_DAV_3510
    constexpr int32_t PTO_MAX_AICORE_NUM = 108;
    constexpr uint32_t PTO_REG_SPR_DATA_MAIN_BASE = 0xD0;
    constexpr uint32_t PTO_REG_SPR_COND = 0x5108;
    constexpr int32_t PTO_PMU_CNT_NUM = 10;
    constexpr uint32_t PTO_MAX_CORE_PER_DIE = 36;
    constexpr uint32_t PTO_AICORE_PER_DIE = 18;
    constexpr uint32_t PTO_AIV_BASE_OFFSET = 18;
#else
    constexpr int32_t PTO_MAX_AICORE_NUM = 75;
    constexpr uint32_t PTO_REG_SPR_DATA_MAIN_BASE = 0xA0;
    constexpr uint32_t PTO_REG_SPR_COND = 0x4C8;
    constexpr int32_t PTO_PMU_CNT_NUM = 8;
    constexpr uint32_t PTO_MAX_CORE_PER_DIE = 25;
#endif

#define PTO_SUPPORT_VF_FUSION    PTO_ARCH_DAV_3510
#define PTO_SUPPORT_MULTI_DIE    PTO_ARCH_DAV_3510
#define PTO_SUPPORT_WRAP         PTO_ARCH_DAV_3510
#define PTO_SUPPORT_MIX_ARCH     PTO_ARCH_DAV_3510