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
 * \file platform_ops.cpp
 * \brief
 */

#include "tilefwk/platform.h"

#ifdef BUILD_WITH_CANN
#include "runtime/rt.h"
#endif

namespace npu::tile_fwk {
const uint32_t kMaxLength = 50;

extern "C" bool GetrtSocSpec(const std::string& column, const std::string& key, std::string& val) {
#ifdef BUILD_WITH_CANN
    char charVal[kMaxLength] = {0};
    if (rtGetSocSpec(column.c_str(), key.c_str(), charVal, kMaxLength) == 0) {
        val = std::string(charVal);
        return true;
    }
#endif
    (void)column;
    (void)key;
    (void)val;
    return false;
}

extern "C" bool GetrtAICPUNum(size_t &aiCpuNum) {
   uint32_t cpuNum = 0;
   int ret = 1;
#ifdef BUILD_WITH_CANN
    ret = rtGetAiCpuCount(&cpuNum);
#endif
    if (ret == 0) {
        aiCpuNum = static_cast<size_t>(cpuNum);
        return true;
    } else {
        return false;
    }
}

extern "C" bool GetrtSocVersion(std::string& socVerString) {
    int ret = 1;
    char socVer[kMaxLength] = {0x00};
#ifdef BUILD_WITH_CANN
    ret = rtGetSocVersion(socVer, kMaxLength);
#endif
    if (ret == 0) {
        socVerString = std::string(socVer);
        return true;
    }
    return false;
}
}