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

#define dcci(...)
#define dsb(...)
#define set_flag(...)
#define wait_flag(...)
#define set_mask_norm(...)

uint64_t get_sys_cnt() {
    auto now = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    return ns;
}

int64_t get_coreid() {
    return 0;
}

int get_block_idx() {
    return 0;
}

void set_cond(uint64_t cond) {
    UNUSED(cond);
}

uint64_t GetDataMainBase() {
    return 0;
}

#include "tilefwk/aicore_entry.h"
#include "tilefwk/aicore_print.h"

struct AiCoreTest : UnitTestBase {};

TEST_F(AiCoreTest, TimeOut) {
    unsigned char sharedBuffer[0x1][SHARED_BUFFER_SIZE];
    KernelArgs *args = reinterpret_cast<KernelArgs *>(&sharedBuffer);
    args->waveBufferCpuToCore[0] = AICORE_SAY_GOODBYE;

    DeviceArgs devArgs;
    devArgs.sharedBuffer = (uint64_t)(uintptr_t)&sharedBuffer;

    KernelEntry(0, 0, 0, 0, 0, (uint64_t)(uintptr_t)&devArgs);
}