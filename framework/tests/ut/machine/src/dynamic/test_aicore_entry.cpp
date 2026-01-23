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
 * \file test_aicore_entry.h
 * \brief
 */


#include "interface/utils/string_utils.h"

#include "test_machine_common.h"
#include "aicore_emulation.h"

struct AiCoreTest : UnitTestBase {};

TEST_F(AiCoreTest, InitGoodbye) {
    unsigned char sharedBuffer[0x1][SHARED_BUFFER_SIZE];
    KernelArgs *args = reinterpret_cast<KernelArgs *>(&sharedBuffer);
    args->waveBufferCpuToCore[0] = AICORE_SAY_GOODBYE;

    DeviceArgs devArgs;
    devArgs.sharedBuffer = (uint64_t)(uintptr_t)&sharedBuffer;

    KernelEntry(0, 0, 0, 0, 0, (uint64_t)(uintptr_t)&devArgs);
}