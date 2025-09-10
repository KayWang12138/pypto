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
 * \file test_dynamic_bin.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek/page_attention.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
using namespace npu::tile_fwk::machine;

static constexpr int tiling32 = 32;

class DynamicBindingTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
public:
    void SetUp() override {
        npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac::SetUp();
        config::SetHostConfig(KEY_ONLY_CODEGEN, true);
        TileShape::Current().SetVecTile(tiling32, tiling32);
        TileShape::Current().SetCubeTile({tiling32, tiling32}, {tiling32, tiling32}, {tiling32, tiling32});
    }
};

namespace {

TEST_F(DynamicBindingTest, TestDefaultCompute) {
    int n = 1 * tiling32;
    int m = 2 * tiling32;

    std::vector<int32_t> inputAData(n * m, 0);
    std::vector<int32_t> inputBData(n * m, 0);
    std::vector<int32_t> outputData(n * m, 0);
    std::vector<int32_t> outputGolden(n * m, 0);
    for (int i = 0; i < n * m; i++) {
        inputAData[i] = i;
        inputBData[i] = i * 2;
        outputGolden[i] = i * 3;
    }

    Tensor inputA(DT_INT32, {n, m}, "inputA");
    Tensor inputB(DT_INT32, {n, m}, "inputB");
    Tensor output(DT_INT32, {n, m}, "output");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<int32_t>(inputB, inputBData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(output, outputData),
    });

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {inputA, inputB}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(m / tiling32)) {
            auto tmpA = View(inputA, {tiling32, tiling32}, {0, i * tiling32});
            auto tmpB = View(inputB, {tiling32, tiling32}, {0, i * tiling32});
            auto tmpO = Add(tmpA, tmpB);
            Assemble(tmpO, {0, i * tiling32}, output);
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::DeviceRunOnce(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t *)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBindingTest, TestDeviceCompute) {
    auto agent = RuntimeAgent::GetAgent();
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());

    int n = 1 * tiling32;
    int m = 2 * tiling32;
    uint8_t *inputADevAddr = nullptr;
    uint8_t *inputBDevAddr = nullptr;
    uint8_t *outputDevAddr = nullptr;
    agent->AllocDevAddr(&inputADevAddr, n * m * sizeof(int32_t));
    agent->AllocDevAddr(&inputBDevAddr, n * m * sizeof(int32_t));
    agent->AllocDevAddr(&outputDevAddr, n * m * sizeof(int32_t));

    std::vector<int32_t> inputAData(n * m, 0);
    std::vector<int32_t> inputBData(n * m, 0);
    std::vector<int32_t> outputData(n * m, 0);
    std::vector<int32_t> outputGolden(n * m, 0);
    for (int i = 0; i < n * m; i++) {
        inputAData[i] = i;
        inputBData[i] = i * 2;
        outputGolden[i] = i * 3;
    }

    agent->CopyToDev(inputADevAddr, (uint8_t *)inputAData.data(), inputAData.size() * sizeof(int32_t));
    agent->CopyToDev(inputBDevAddr, (uint8_t *)inputBData.data(), inputBData.size() * sizeof(int32_t));

    FunctionConfig funConfig;

    Tensor inputA(DT_INT32, {n, m}, "inputA");
    Tensor inputB(DT_INT32, {n, m}, "inputB");
    Tensor output(DT_INT32, {n, m}, "output");

    FUNCTION("main", funConfig, {inputA, inputB}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(m / tiling32)) {
            auto tmpA = View(inputA, {tiling32, tiling32}, {0, i * tiling32});
            auto tmpB = View(inputB, {tiling32, tiling32}, {0, i * tiling32});
            auto tmpO = Add(tmpA, tmpB);
            Assemble(tmpO, {0, i * tiling32}, output);
        }
    }

    std::vector<DeviceTensorData> inputList = {
        DeviceTensorData((uintdevptr_t)inputADevAddr, inputA.GetShape()),
        DeviceTensorData((uintdevptr_t)inputBDevAddr, inputB.GetShape()),
    };
    std::vector<DeviceTensorData> outputList = {
        DeviceTensorData((uintdevptr_t)outputDevAddr, output.GetShape()),
    };

    auto aicpuStream = machine::GetRA()->GetStreamAICPU();
    auto aicoreStream = machine::GetRA()->GetStream();
    DeviceLauncher::DeviceRunOnceWithDeviceTensorData(Program::GetInstance().GetLastFunction(), inputList, outputList, aicpuStream, aicoreStream);

    agent->CopyFromDev((uint8_t *)outputData.data(), outputDevAddr, outputData.size() * sizeof(int32_t));

    EXPECT_TRUE(resultCmp(outputGolden, (int32_t *)outputData.data(), 0.001f));
}

}
