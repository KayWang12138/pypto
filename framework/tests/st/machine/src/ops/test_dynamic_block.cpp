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
 * \file test_dynamic_bin.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek/page_attention.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/runtime/device_launcher.h"
#include "machine/runtime/emulation_launcher.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
using namespace npu::tile_fwk::machine;

static constexpr int tiling32 = 32;

class DynamicBlockTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        DeviceLauncherContext::Get().DeviceInit();
        rtSetDevice(GetDeviceIdByEnvVar());
     }

    void TearDown() override {
        DeviceLauncherContext::Get().DeviceFini();
    }
};

namespace {

TEST_F(DynamicBlockTest, VectorCube) {
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 4;
    Tensor inputA(DT_FP32, {n, n}, "A");
    Tensor inputB(DT_FP32, {n, n}, "B");
    Tensor output(DT_FP32, {n, n}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(inputA, 1),
        RawTensorData::CreateConstantTensor<float>(inputB, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0),
    });

    std::vector<float> outputGolden(n * n, 512);
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(output, outputGolden),
    });

    Tensor e;
    FUNCTION("main", {inputA, inputB}, {output}) {
        Tensor sum(DT_FP32, {n, n}, "sum");
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto a = Add(inputA, inputA);
            output = Matrix::Matmul<false, false>(DT_FP32, a, inputB);
        }
    }
    DeviceLauncherConfig config;
    config.blockdim = 24; // 24:max aicore num
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), {}, {}, config));

    DeviceLauncher::DeviceRunCacheKernelEnable(Program::GetInstance().GetLastFunction(), true);

#ifdef BUILD_WITH_CANN
    EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction(), config));
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (float *)outputResult->data(), 0.001f));
#endif
}

}
