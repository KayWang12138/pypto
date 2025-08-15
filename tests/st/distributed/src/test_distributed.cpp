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
 * \file test_distributed.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "test_common.h"
#include "distributed_op_test_suite.h"
#include "distributed_test_framework.h"

namespace npu::tile_fwk {
namespace Distributed {
class DistributedTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override
    {
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
        Distributed::TestFrameworkInit(testParam, hcomTestParam);
    }

    void TearDown() override
    {
        DistributedTestDestroy();
        Distributed::TestFrameworkDestroy(timeout);
    }

    // 暴露超时设置接口
    void SetDestroyTimeout(int32_t destroyTimeout)
    {
        timeout = destroyTimeout;
    }

protected:
    void DistributedTestDestroy()
    {
        // 销毁集合通信域
        ASSERT(HcclCommDestroy(hcomTestParam.hcclComm) == 0);
        // 重置设备
        ASSERT(aclrtResetDevice(testParam.rankId) == 0);
        // 设备去初始化
        ASSERT(aclFinalize() == 0);
    }

    Distributed::OpTestParam testParam;
    Distributed::HcomTestParam hcomTestParam;
    int32_t timeout = 10;
};

TEST_F(DistributedTest, aicpuWaitFlag_single_test_reduce_scatter_int32_32_32_4)
{
    Distributed::TestReduceScatter(testParam);
}

TEST_F(DistributedTest, aicpuWaitFlag_multi_test_reduce_scatter_float32_128_256_4)
{
    Distributed::TestReduceScatterEx(testParam);
}

TEST_F(DistributedTest, aicpuWaitFlag_single_test_all_gather_bfloat16_256_256_4)
{
    Distributed::TestAllGather(testParam);
}

TEST_F(DistributedTest, aicpuWaitFlag_multi_test_all_gather_float16_32_32_4)
{
    Distributed::TestAllGatherEx(testParam);
}

TEST_F(DistributedTest, aivWaitFlag_single_test_reduce_scatter_int32_128_256_4)
{
    config::SetDistConfig(KEY_AICPU_WAIT_FLAG_ENABLE, false);
    Distributed::TestReduceScatter(testParam);
}

TEST_F(DistributedTest, aivWaitFlag_multi_test_reduce_scatter_float32_128_256_4)
{
    config::SetDistConfig(KEY_AICPU_WAIT_FLAG_ENABLE, false);
    Distributed::TestReduceScatterEx(testParam);
}

TEST_F(DistributedTest, aivWaitFlag_single_test_all_gather_bfloat16_256_128_4)
{
    config::SetDistConfig(KEY_AICPU_WAIT_FLAG_ENABLE, false);
    Distributed::TestAllGather(testParam);
}

TEST_F(DistributedTest, aivWaitFlag_multi_test_all_gather_float16_32_32_4)
{
    config::SetDistConfig(KEY_AICPU_WAIT_FLAG_ENABLE, false);
    Distributed::TestAllGatherEx(testParam);
}

TEST_F(DistributedTest, aivWaitFlag_single_test_moe_dispatch_bfloat16_rank_size_4)
{
    config::SetDistConfig(KEY_AICPU_WAIT_FLAG_ENABLE, false);
    Distributed::TestMoeDispatch(testParam);
}

TEST_F(DistributedTest, aivWaitFlag_single_test_moe_combine_bfloat16_rank_size_4)
{
    config::SetDistConfig(KEY_AICPU_WAIT_FLAG_ENABLE, false);
    Distributed::TestMoeCombine(testParam);
}

TEST_F(DistributedTest, allgather_attn_post_reducescatter_b64_s1_n32_lora256_dim128_h128_rank4_bf16)
{
    config::SetDistConfig(KEY_AICPU_WAIT_FLAG_ENABLE, false);
    Distributed::TestAllGatherAttentionPostReducescatter(testParam);
}

} // namespace Distributed
} // namespace npu::tile_fwk
