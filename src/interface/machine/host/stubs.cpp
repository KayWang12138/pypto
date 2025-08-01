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
 * \file stubs.cpp
 * \brief
 */

#include <iostream>
#include "interface/machine/host/stubs.h"

using namespace npu::tile_fwk::stubs;

bool DeviceStub::mInited = false;
int32_t DeviceStub::mDevId = 0;

int32_t DeviceStub::GetCurrentDeviceId() {
    if (!mInited) {
        int32_t invalidDevId = 0;
        int32_t devId = invalidDevId;
        const char *devIdPtr = getenv("TILE_FWK_STEST_DEVICE_ID");
        if (devIdPtr != nullptr) {
            try {
                // 将字符串转换为 int32_t 类型
                devId = static_cast<int32_t>(std::stoi(devIdPtr));
            } catch (const std::invalid_argument &) {
                devId = invalidDevId;
            } catch (const std::out_of_range &) {
                devId = invalidDevId;
            }
        }
        mInited = true;
        mDevId = devId;
    }
    return mDevId;
}
