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
#include "device_launcher.h"
#include "emulation_launcher.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
using namespace npu::tile_fwk::machine;

static constexpr int tiling32 = 32;

class DynamicControlFlowCacheTest : public testing::Test {
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

TEST_F(DynamicControlFlowCacheTest, KernelReuse) {
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, true);
    config::SetRuntimeOption<int64_t>(CFGCACHE_DEVICE_TASK_NUM, 100);
    config::SetRuntimeOption<int64_t>(CFGCACHE_ROOT_TASK_NUM, 100);
    config::SetRuntimeOption<int64_t>(CFGCACHE_LEAF_TASK_NUM, 10000);


    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);

    int n = tiling * 4;
    Tensor inputA(DT_INT32, {n, n}, "A");
    Tensor inputB(DT_INT32, {n, n}, "B");
    Tensor output(DT_INT32, {n, n}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateConstantTensor<int32_t>(inputB, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    std::vector<int32_t> outputGolden(n * n, 6);
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    Tensor e;
    FUNCTION("main", {inputA, inputB}, {output}) {
        Tensor sum(DT_INT32, {n, n}, "sum");
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling)) {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling)) {
                auto a = View(inputA, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(inputB, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, sum);
            }
        }
        LOOP("X", FunctionType::DYNAMIC_LOOP, _, LoopRange(1)) {
            (void)_;
            output = Add(sum, sum);
        }
    }

    DeviceLauncherConfig config;
    config.controlFlowCache = true;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), config));

    DeviceLauncher::DeviceRunCacheKernelEnable(Program::GetInstance().GetLastFunction(), true);

#ifdef BUILD_WITH_CANN
    for (int k = 0; k < 3; k++) {
        EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction()));
        auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
        EXPECT_TRUE(resultCmp(outputGolden, (int32_t *)outputResult->data(), 0.001f));
    }
#endif
}

TEST_F(DynamicControlFlowCacheTest, CheckShape) {
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, true);
    config::SetRuntimeOption<int64_t>(CFGCACHE_DEVICE_TASK_NUM, 100);
    config::SetRuntimeOption<int64_t>(CFGCACHE_ROOT_TASK_NUM, 100);
    config::SetRuntimeOption<int64_t>(CFGCACHE_LEAF_TASK_NUM, 10000);

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);

    int mid = tiling * 8;
    Tensor inputA(DT_INT32, {-1, -1}, "A");
    Tensor inputB(DT_INT32, {-1, -1}, "B");
    Tensor output(DT_INT32, {-1, -1}, "O");

    int n1 = tiling * 4;
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 1),
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 6),
    });

    FUNCTION("main", {inputA, inputB}, {output}) {
        Tensor sum(DT_INT32, {mid, mid}, "sum");
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling)) {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling)) {
                auto a = View(inputA, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(inputB, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, sum);
            }
        }
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling)) {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling)) {
                auto a = View(sum, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(sum, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, output);
            }
        }
    }

    DeviceLauncherConfig config;
    config.controlFlowCache = true;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), config));

    DevAscendProgram *devProg = reinterpret_cast<DevAscendProgram *>(
        const_cast<uint8_t*>(DeviceLauncher::GetDevProg(Program::GetInstance().GetLastFunction()).data()));
    EXPECT_NE(devProg->controlFlowCache.deviceTaskCount, 0);

    devProg->RelocProgram((intptr_t)devProg);
    devProg->RelocControlFlowCache(0, (intptr_t)devProg, 0, 0);

    {
        // check success
        DevTensorData devTensorList[] = {
            {0, {2, {n1, n1}}},
            {0, {2, {n1, n1}}},
            {0, {2, {n1, n1}}},
        };
        DevStartArgsBase arg = {devTensorList, 2, 1, nullptr};
        EXPECT_TRUE(devProg->controlFlowCache.MatchInputOutput(&arg));
    }
    {
        // check failed for count
        DevStartArgsBase arg = {nullptr, 0, 0, nullptr};
        EXPECT_FALSE(devProg->controlFlowCache.MatchInputOutput(&arg));
    }
    {
        // check failed for dimension
        DevTensorData devTensorList[] = {
            {0, {2, {n1, n1}}},
            {0, {2, {n1, n1}}},
            {0, {3, {n1, n1, n1}}},
        };
        DevStartArgsBase arg = {devTensorList, 2, 1, nullptr};
        EXPECT_FALSE(devProg->controlFlowCache.MatchInputOutput(&arg));
    }
    {
        // check failed for shape
        DevTensorData devTensorList[] = {
            {0, {2, {n1, n1}}},
            {0, {2, {n1, n1}}},
            {0, {2, {n1, n1 + n1}}},
        };
        DevStartArgsBase arg = {devTensorList, 2, 1, nullptr};
        EXPECT_FALSE(devProg->controlFlowCache.MatchInputOutput(&arg));
    }

    devProg->RelocControlFlowCache((intptr_t)devProg, 0, 0, 0);
    devProg->RelocProgram(-(intptr_t)devProg);

    int n2 = tiling * 2;
    ProgramData::GetInstance().GetInputDataList()[0] = RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n2, n2}), 2);
    ProgramData::GetInstance().GetInputDataList()[1] = RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n2, n2}), 3);
    ProgramData::GetInstance().GetOutputDataList()[0] = RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n2, n2}), 0);

    std::vector<int32_t> outputGolden(n2 * n2, 10);
#ifdef BUILD_WITH_CANN
    EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction()));
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t *)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicControlFlowCacheTest, CheckLackMemory) {
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, true);
    config::SetRuntimeOption<int64_t>(CFGCACHE_DEVICE_TASK_NUM, 1);
    config::SetRuntimeOption<int64_t>(CFGCACHE_ROOT_TASK_NUM, 1);
    config::SetRuntimeOption<int64_t>(CFGCACHE_LEAF_TASK_NUM, 1);
    config::SetRuntimeOption<int64_t>(FIRST_STITCH_TASK_LOOP_NUM, 128);

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);

    int mid = tiling * 8;
    Tensor inputA(DT_INT32, {-1, -1}, "A");
    Tensor inputB(DT_INT32, {-1, -1}, "B");
    Tensor output(DT_INT32, {-1, -1}, "O");

    int n1 = tiling * 4;
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 1),
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(Tensor(DT_INT32, {n1, n1}), 0),
    });

    FUNCTION("main", {inputA, inputB}, {output}) {
        Tensor sum(DT_INT32, {mid, mid}, "sum");
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling)) {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling)) {
                auto a = View(inputA, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(inputB, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, sum);
            }
        }
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling)) {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling)) {
                auto a = View(sum, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(sum, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, output);
            }
        }
    }

    DeviceLauncherConfig config;
    config.controlFlowCache = true;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), config));

    DevAscendProgram *devProg = reinterpret_cast<DevAscendProgram *>(
        const_cast<uint8_t*>(DeviceLauncher::GetDevProg(Program::GetInstance().GetLastFunction()).data()));
    EXPECT_EQ(devProg->controlFlowCache.deviceTaskCount, 0);
    EXPECT_EQ(devProg->controlFlowCache.deviceTaskSkippedCount, 1);

    std::vector<int32_t> outputGolden(n1 * n1, 6);
#ifdef BUILD_WITH_CANN
    EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction()));
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t *)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicControlFlowCacheTest, CheckGetTensorData) {
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, true);
    config::SetRuntimeOption<int64_t>(CFGCACHE_DEVICE_TASK_NUM, 10000);
    config::SetRuntimeOption<int64_t>(CFGCACHE_ROOT_TASK_NUM, 100);
    config::SetRuntimeOption<int64_t>(CFGCACHE_LEAF_TASK_NUM, 100);

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);

    int mid = tiling * 8;
    Tensor inputA(DT_INT32, {-1, -1}, "A");
    Tensor inputB(DT_INT32, {-1, -1}, "B");
    Tensor inputC(DT_INT32, {-1, -1}, "C");
    Tensor output(DT_INT32, {-1, -1}, "O");

    FUNCTION("main", {inputA, inputB, inputC}, {output}) {
        Tensor sum(DT_INT32, {mid, mid}, "sum");
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling)) {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling)) {
                auto a = View(inputA, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling + GetTensorData(inputC, {0, 0})}));
                auto b = View(inputB, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling + GetTensorData(inputC, {0, 0})}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, sum);
            }
        }
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling)) {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling)) {
                auto a = View(sum, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling + GetTensorData(sum, {0, 0})}));
                auto b = View(sum, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling + GetTensorData(sum, {0, 0})}));
                Assemble(Mul(a, b), {i * tiling, j * tiling}, output);
            }
        }
    }

    DeviceLauncherConfig config;
    config.controlFlowCache = true;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), config));

    DevAscendProgram *devProg = reinterpret_cast<DevAscendProgram *>(
        const_cast<uint8_t*>(DeviceLauncher::GetDevProg(Program::GetInstance().GetLastFunction()).data()));
    EXPECT_NE(devProg->controlFlowCache.getInputDataCount, 0);
    EXPECT_NE(devProg->controlFlowCache.getTensorDataCount, 0);
}

}
