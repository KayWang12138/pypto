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

    static void SetUpTestCase() 
    {
        GegisterOps();
    }

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
    void RunDistributedTestGeneric(const nlohmann::json& testData)
    {
        if (!testData.contains("input_tensors") || testData["input_tensors"].empty()) {
            FAIL() << "No input tensors in testData: " << testData.dump();
        }
        std::string opName = testData["operation"];
        std::string dtype = testData["input_tensors"][0]["dtype"];
        DisOpRegister::GetRegister().Run(opName, testParam, dtype);
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

// 各模板算子的Func
struct AllGatherFunc {
    template <typename T>
    void operator()(OpTestParam &testParam) const
    {
        Distributed::TestDynAllGather<T>(testParam);
    }
};

struct ReduceScatterFunc {
    template <typename T>
    void operator()(OpTestParam &testParam) const
    {
        Distributed::TestShmemReduceScatter<T>(testParam);
    }
};

struct AllReduceFunc {
    template <typename T>
    void operator()(OpTestParam &testParam) const
    {
        Distributed::TestShmemAllReduce<T>(testParam);
    }
};

struct AllReduceAddAllReduceFunc {
    template <typename T>
    void operator()(OpTestParam &testParam) const
    {
        Distributed::TestShmemAllReduceAddAllReduce<T>(testParam);
    }
};

struct MoeDistributedCombineFunc {
    template <typename T>
    void operator()(OpTestParam& testParam) const
    {
        Distributed::TestMoeDistributedCombine<T>(testParam);
    }
};

// 注册所有算子
void GegisterOps()
{
    auto& reg = DisOpRegister::GetRegister();
    reg.RegisterOp("AllGather", AllGatherFunc{});  // 模板算子
    reg.RegisterOp("ReduceScatter", ReduceScatterFunc{});
    reg.RegisterOp("AllReduce", AllReduceFunc{});
    reg.RegisterOp("AllReduceAddAllReduce", AllReduceAddAllReduceFunc{});
    reg.RegisterOp("MoeDistributedCombine", MoeDistributedCombineFunc{});
    reg.disRegisterMap["MoeDispatch"] = [](OpTestParam &testParam, const std::string&) {
 	    Distributed::TestShmemMoeDispatch(testParam);
    };
    reg.disRegisterMap["AllGatherAttnPostReduceScatter"] = [](OpTestParam &testParam, const std::string&) {
        Distributed::TestAllGatherAttentionPostReducescatter(testParam);
    };
    // 后续按照上面格式增加算子
}


INSTANTIATE_TEST_SUITE_P(TestDistributedOps, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>()));
TEST_P(DistributedTest, TestOps)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    RunDistributedTestGeneric(GetParam().testData_);
}

} // namespace npu::tile_fwk::Distributed