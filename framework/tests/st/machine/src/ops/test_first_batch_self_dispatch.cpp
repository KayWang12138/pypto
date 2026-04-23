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

#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/runtime/device_launcher.h"
#include "machine/runtime/emulation_launcher.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class FirstBatchSelfDispatchST : public testing::Test {
public:
    void SetUp() override
    {
        DeviceLauncherContext::Get().DeviceInit();
        RuntimeSetDevice(GetDeviceIdByEnvVar());
    }

    void TearDown() override { DeviceLauncherContext::Get().DeviceFini(); }
};

TEST_F(FirstBatchSelfDispatchST, EmulationRunWithFirstBatchSelfDispatchEnabled)
{
    config::SetRuntimeOption<uint8_t>(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::DEFAULT_SCH));
    config::SetRuntimeOption<bool>(ENABLE_AICORE_FIRST_BATCH_SELF_DISPATCH, true);
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_MAX_NUM, 4);

    TileShape::Current().SetVecTile(32, 32);

    Tensor inputA(DT_INT32, {128, 128}, "A");
    Tensor inputB(DT_INT32, {128, 128}, "B");
    Tensor output(DT_INT32, {128, 128}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateConstantTensor<int32_t>(inputB, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FUNCTION("first_batch_self_dispatch_st", {inputA, inputB}, {output})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(2))
        {
            (void)i;
            output = Add(inputA, inputB);
        }
    }

    DeviceLauncherConfig launcherConfig;
    launcherConfig.blockdim = 24;
    EXPECT_EQ(
        0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), nullptr, launcherConfig));
}

TEST_F(FirstBatchSelfDispatchST, DISABLED_DeviceRunBaselineWithoutFirstBatchSelfDispatch)
{
    SetInterpreterConfig();
    config::SetRuntimeOption<bool>(ENABLE_AICORE_FIRST_BATCH_SELF_DISPATCH, false);

    TileShape::Current().SetVecTile(32, 32);

    std::vector<int32_t> inputAData(128 * 128, 1);
    std::vector<int32_t> inputBData(128 * 128, 2);
    std::vector<int32_t> outputData(128 * 128, 0);
    std::vector<int32_t> outputGolden(128 * 128, 3);

    Tensor inputA(DT_INT32, {128, 128}, "A");
    Tensor inputB(DT_INT32, {128, 128}, "B");
    Tensor output(DT_INT32, {128, 128}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<int32_t>(inputB, inputBData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(output, outputData),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FUNCTION("first_batch_self_dispatch_st_device_baseline", {inputA, inputB}, {output}) { output = Add(inputA, inputB); }

#ifdef BUILD_WITH_CANN
    ASSERT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction()));
    auto outputResult = ProgramData::GetInstance().GetOutputData(0);
    ASSERT_NE(outputResult, nullptr);
    EXPECT_TRUE(resultCmp(outputGolden, reinterpret_cast<int32_t*>(outputResult->data()), 0.001f));
#endif
}

TEST_F(FirstBatchSelfDispatchST, DISABLED_DeviceRunWithFirstBatchSelfDispatchEnabled)
{
    SetInterpreterConfig();
    config::SetRuntimeOption<uint8_t>(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::DEFAULT_SCH));
    config::SetRuntimeOption<bool>(ENABLE_AICORE_FIRST_BATCH_SELF_DISPATCH, true);
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_MAX_NUM, 4);

    TileShape::Current().SetVecTile(32, 32);

    std::vector<int32_t> inputAData(128 * 128, 1);
    std::vector<int32_t> inputBData(128 * 128, 2);
    std::vector<int32_t> outputData(128 * 128, 0);
    std::vector<int32_t> outputGolden(128 * 128, 3);

    Tensor inputA(DT_INT32, {128, 128}, "A");
    Tensor inputB(DT_INT32, {128, 128}, "B");
    Tensor output(DT_INT32, {128, 128}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<int32_t>(inputB, inputBData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(output, outputData),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FUNCTION("first_batch_self_dispatch_st_device", {inputA, inputB}, {output}) { output = Add(inputA, inputB); }

#ifdef BUILD_WITH_CANN
    ASSERT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction()));
    auto outputResult = ProgramData::GetInstance().GetOutputData(0);
    ASSERT_NE(outputResult, nullptr);
    EXPECT_TRUE(resultCmp(outputGolden, reinterpret_cast<int32_t*>(outputResult->data()), 0.001f));
#endif
}

