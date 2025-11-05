/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file device_log.cpp
 * \brief
 */

#include "machine/utils/device_log.h"
namespace npu::tile_fwk {
#if DEBUG_PLOG && defined(__DEVICE__)

bool g_isLogDEnable = false;
bool g_isLogIEnable = false;
bool g_isLogWEnable = false;
bool g_isLogEEnable = false;

void InitLogSwitch() {
    g_isLogDEnable = CheckLogLevel(AICPU, DLOG_DEBUG);
    g_isLogIEnable = CheckLogLevel(AICPU, DLOG_INFO);
    g_isLogWEnable = CheckLogLevel(AICPU, DLOG_WARN);
    g_isLogEEnable = CheckLogLevel(AICPU, DLOG_ERROR);
}
#endif
} // namespace npu::tile_fwk