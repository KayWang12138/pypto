/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdio>
#include <string>

namespace npu::tile_fwk {

inline std::string FormatElapsed(double seconds) {
    if (seconds < 60.0) {
        char buf[32];
        (void)snprintf(buf, sizeof(buf), "%.1fs", seconds);
        return buf;
    }
    int total_sec = static_cast<int>(seconds);
    int min = total_sec / 60;
    int sec = total_sec % 60;
    char buf[64];
    if (min >= 60) {
        int hour = min / 60;
        min = min % 60;
        (void)snprintf(buf, sizeof(buf), "%dh %dm %ds (%ds)", hour, min, sec, total_sec);
    } else {
        (void)snprintf(buf, sizeof(buf), "%dmin %ds (%ds)", min, sec, total_sec);
    }
    return buf;
}

}  // namespace npu::tile_fwk
