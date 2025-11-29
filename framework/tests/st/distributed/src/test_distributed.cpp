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
#include <unistd.h>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "test_common.h"
#include "distributed_op_test_suite.h"
#include "distributed_test_framework.h"

namespace npu::tile_fwk::Distributed {
class DistributedTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override
    {   
        Distributed::TestFrameworkInit(testParam, hcomTestParam, physicalDeviceId);
        std::string folderPath = "output/output_" + getTimeStamp() + "_" + std::to_string(getpid());
        setenv("TILE_FWK_OUTPUT_DIR", folderPath.c_str(), 0);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
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
        ASSERT(aclrtResetDevice(physicalDeviceId) == 0);
        // 设备去初始化
        ASSERT(aclFinalize() == 0);
    }

    Distributed::OpTestParam testParam;
    Distributed::HcomTestParam hcomTestParam;
    int32_t timeout = 10;
    int physicalDeviceId;
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

TEST_F(DistributedTest, aivWaitFlag_single_test_moe_dispatch_bfloat16_8_7168_1_3_2_4)
{
    config::SetDistConfig(KEY_AICPU_WAIT_FLAG_ENABLE, false);
    Distributed::TestMoeDispatch(testParam);
}

TEST_F(DistributedTest, shmem_allgather_attn_post_reducescatter_bfloat16_64_1_32_256_128_128_4)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    Distributed::TestAllGatherAttentionPostReducescatter(testParam);
}

TEST_F(DistributedTest, shmem_all_gather_int32_128_256_4)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    Distributed::TestDynAllGather<int32_t>(testParam);
}

TEST_F(DistributedTest, shmem_reduce_scatter_int32_128_256_4)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    Distributed::TestShmemReduceScatter<int32_t>(testParam);
}

TEST_F(DistributedTest, shmem_allgather_matmul_reducescatter_int32_128_256_4)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    Distributed::TestDynAllGatherMatmulReducescatter(testParam);
}

TEST_F(DistributedTest, shmem_reduce_scatter_float16_128_256_4)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    Distributed::TestShmemReduceScatter<npu::tile_fwk::float16>(testParam);
}

TEST_F(DistributedTest, shmem_reduce_scatter_bfloat16_32_32_4)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    Distributed::TestShmemReduceScatter<npu::tile_fwk::bfloat16>(testParam);
}

TEST_F(DistributedTest, shmem_add_all_reduce_int32_64_256_4)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    Distributed::TestShmemAddAndAllReduce<int32_t>(testParam);
}

TEST_F(DistributedTest, shmem_moe_combine_bfloat16_8_5120_0_160_8_4)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    Distributed::TestShmemMoeCombine(testParam);
}

TEST_F(DistributedTest, shmem_moe_combine_bfloat16_256_5120_0_160_8_4)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    Distributed::TestShmemMoeCombine(testParam);
}

TEST_F(DistributedTest, shmem_moe_combine_bfloat16_8_5120_0_160_8_8)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    Distributed::TestShmemMoeCombine(testParam);
}

TEST_F(DistributedTest, shmem_moe_combine_bfloat16_256_5120_0_160_8_8)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    Distributed::TestShmemMoeCombine(testParam);
}
} // namespace npu::tile_fwk::Distributed
