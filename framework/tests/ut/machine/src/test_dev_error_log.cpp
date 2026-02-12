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
#include <vector>
#include <string>
#include <memory>
#include <cstring>

#include "machine/device/aicore_manager.h"
#include "machine/device/distributed/shmem_wait_until.h"
#include "interface/machine/device/tilefwk/core_func_data.h"

namespace npu::tile_fwk {

class TestDevErrorLog : public testing::Test {
public:
    void SetUp() override {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
        g_isLogEnableError = true;
    }

    void TearDown() override {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    }
};

TEST_F(TestDevErrorLog, DataValid_SdmaPrefetch_InvalidPrefetchNum) {
    DeviceTask devTask;
    
    devTask.l2Info.prefetchNum = MAX_PREFETCH_NUM + 1;
    
    SdmaPrefetch(&devTask);
}

} // namespace npu::tile_fwk
