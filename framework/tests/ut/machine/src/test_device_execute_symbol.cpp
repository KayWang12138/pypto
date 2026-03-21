/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "interface/tensor/symbol_handler.h"
#include "machine/device/dynamic/context/device_execute_context.h"

using npu::tile_fwk::SymbolHandlerId;
using npu::tile_fwk::dynamic::DeviceExecuteContext;

TEST(DeviceExecuteSymbolTest, SymbolHandlerIdToHandler_InvalidId_AbortsOnAssert) {
    EXPECT_DEATH(
        (void)DeviceExecuteContext::SymbolHandlerIdToHandler(static_cast<SymbolHandlerId>(99999ULL)),
        ".*");
}
