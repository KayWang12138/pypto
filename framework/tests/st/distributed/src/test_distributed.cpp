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
 * \file test_distributed.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "test_common.h"
#include "distributed_test_framework.h"
#include "test_distributed.h"

namespace npu::tile_fwk::Distributed {

class DistributedTest : public testing::TestWithParam<OpMetaData> {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override
    {
        Distributed::TestFrameworkInit(testParam, hcomTestParam, physicalDeviceId);
        std::string folderPath = "output/output_" + getTimeStamp() + "_" + std::to_string(physicalDeviceId);
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

    // 通用测试入口
    void RunDistributedTestGeneric(const std::string& opName, const nlohmann::json& testData)
    {
        if (!testData.contains("input_tensors") || testData["input_tensors"].empty()) {
            FAIL() << "No input tensors in testData: " << testData.dump();
        }
        std::string dtype = testData["input_tensors"][0]["dtype"];
        DisTemplateOpRegistry::GetRegistry().Run(opName, testParam, dtype);
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
    int physicalDeviceId = 0;
};

// 注册模版算子
REGISTER_DIS_TEMPLATE_OP(Allgather, Distributed::TestDynAllGather);
REGISTER_DIS_TEMPLATE_OP(Reducescatter, Distributed::TestShmemReduceScatter);
REGISTER_DIS_TEMPLATE_OP(Allreduce, Distributed::TestShmemAllReduce);
REGISTER_DIS_TEMPLATE_OP(Allreduce_Add_AllreduceFunc, Distributed::TestShmemAllReduceAddAllReduce);
REGISTER_DIS_TEMPLATE_OP(MoeDistributedCombine, Distributed::TestMoeDistributedCombine);


INSTANTIATE_TEST_SUITE_P(TestAllgather, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>("Allgather")));
TEST_P(DistributedTest, TestAllgather)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    RunDistributedTestGeneric("Allgather", GetParam().testData_);
}

INSTANTIATE_TEST_SUITE_P(TestReducescatter, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>("Reducescatter")));
TEST_P(DistributedTest, TestReducescatter)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    RunDistributedTestGeneric("Reducescatter", GetParam().testData_);
}

INSTANTIATE_TEST_SUITE_P(TestAllreduce, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>("Allreduce")));
TEST_P(DistributedTest, TestAllreduce)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    RunDistributedTestGeneric("Allreduce", GetParam().testData_);
}

INSTANTIATE_TEST_SUITE_P(TestMoeDistributedCombine, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>("MoeDistributedCombine")));
TEST_P(DistributedTest, TestMoeDistributedCombine)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    RunDistributedTestGeneric("MoeDistributedCombine", GetParam().testData_);
}

INSTANTIATE_TEST_SUITE_P(TestAllreduce_Add_Allreduce, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>("Allreduce_Add_Allreduce")));
TEST_P(DistributedTest, TestAllreduce_Add_Allreduce)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    RunDistributedTestGeneric("Allreduce_Add_Allreduce", GetParam().testData_);
}

INSTANTIATE_TEST_SUITE_P(TestAllgather_AttnPost_Reducescatter, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>("Allgather_AttnPost_Reducescatter")));
TEST_P(DistributedTest, TestAllgather_AttnPost_Reducescatter)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    Distributed::TestAllGatherAttentionPostReducescatter(testParam);
}

} // namespace npu::tile_fwk::Distributed
