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
        std::string outputDir = "output";
        bool res = CreateDir(outputDir);
        CHECK(res) << "Failed to create directory: " << outputDir;
        std::string folderPath = outputDir + "/output_" + getTimeStamp() + "_" + std::to_string(physicalDeviceId);
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
    void RunDistributedTestGeneric(const nlohmann::json& testData, const std::string& fileName)
    {
        if (!testData.contains("input_tensors") || testData["input_tensors"].empty()) {
            FAIL() << "No input tensors in testData: " << testData.dump();
        }
        std::string opName = testData["operation"].get<std::string>();
        std::string dtype = testData["input_tensors"][0]["dtype"].get<std::string>();
        std::string caseName = testData["case_name"].get<std::string>();
        std::string goldenDir = GetGoldenDirPath(testData, fileName);
        printf("[Distributed] Running: op=%s, case=%s\n",
            opName.c_str(), caseName.c_str());
        DisOpRegister::GetRegister().Run(opName, testParam, dtype, goldenDir);
        ALOG_INFO("test case finished successfully: op=%s, case=%s, json file=%s.", 
            opName.c_str(), caseName.c_str(), fileName.c_str());
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


// 注册所有算子
void GegisterOps()
{
    auto& reg = DisOpRegister::GetRegister();
    // 模板算子
    reg.RegisterOp("AllGather", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllGather<T>(testParam, goldenDir);
    });
    reg.RegisterOp("ReduceScatter", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestReduceScatter<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduce", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduceV2", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_v2<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduceV3", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_v3<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduceV4", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_v4<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduceV5", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_v5<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduceV6", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_v6<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduceV6Light", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_v6_light<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduceV7", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_v7<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduceV8", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_v8<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduceV7Light", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_v7_light<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduceV8Light", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_v8_light<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduceIRCheck", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduceIREquivalence<T>(testParam, goldenDir);
    });
    reg.RegisterOp("TwoShotAllReduceV2", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_TwoShot_v2<T>(testParam, goldenDir);
    });
    reg.RegisterOp("TwoShotAllReduceV3", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_TwoShot_v3<T>(testParam, goldenDir);
    });
    reg.RegisterOp("TwoShotAllReduceV4", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_TwoShot_v4<T>(testParam, goldenDir);
    });
    reg.RegisterOp("TwoShotAllReduceV5", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduce_TwoShot_v5<T>(testParam, goldenDir);
    });
    reg.RegisterOp("TwoShotAllReduceIRCheck", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestTwoShotAllReduceIREquivalence<T>(testParam, goldenDir);
    });
    reg.RegisterOp("AllReduceAddAllReduce", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestAllReduceAddAllReduce<T>(testParam, goldenDir);
    });
    reg.RegisterOp("MoeDistributedCombine", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestMoeDistributedCombine<T>(testParam, goldenDir);
    });
    reg.RegisterOp("MoeDispatch", []<typename T>(OpTestParam& testParam, std::string& goldenDir) {
        Distributed::TestShmemMoeDispatch<T>(testParam, goldenDir);
    });
    reg.disRegisterMap["AllGatherAttnPostReduceScatter"] = [](OpTestParam& testParam, const std::string&, std::string& goldenDir) {
        Distributed::TestAllGatherAttentionPostReducescatter(testParam, goldenDir);
    };
    // 后续按照上面格式增加算子

}


INSTANTIATE_TEST_SUITE_P(TestDistributedOps, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>()));
TEST_P(DistributedTest, TestOps)
{
    RunDistributedTestGeneric(GetParam().testData_, GetParam().fileName_);
}
} // namespace npu::tile_fwk::Distributed
