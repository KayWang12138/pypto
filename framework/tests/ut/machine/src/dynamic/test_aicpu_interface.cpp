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
 * \file test_aicpu_interface.cpp
 * \brief
 */
#include <vector>
#include "gtest/gtest.h"
#include "machine/device/machine_interface/pypto_aicpu_interface.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/dump/kernel_dump_utils.h"
#include "machine/runtime/machine_agent.h"
#include "interface/program/program.h"
#include "interface/utils/file_utils.h"
#define private public
#include "cost_model/simulation/backend.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

extern "C" uint32_t DynPyptoKernelServerNull(void *targ);
class TestAicpuInterce : public testing::Test {
public:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {}

    void SetUp() override {
    }

    void TearDown() override {}
};

TEST_F(TestAicpuInterce, test_pypto_kernel_server_null) {
    AstKernelArgs pyptoKernelArgs;
    DeviceArgs devKernelArgs;
    devKernelArgs.aicpuSoLen = 2;
    pyptoKernelArgs.cfgdata = static_cast<int64_t *>(static_cast<void *>(&devKernelArgs));
    auto ret = DynPyptoKernelServerNull(&pyptoKernelArgs);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestAicpuInterce, test_kernel_dump) {
    const std::vector<int64_t> shape = {64, 64};
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A0");
    Tensor inputB(DT_FP32, shape, "B1");
    Tensor output(DT_FP32, shape, "C0");

    config::SetBuildStatic(true);
    FUNCTION("ADD", {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    auto function = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD0");
    auto task_1 = std::make_shared<MachineTask>(0, function);
    auto deviceMachineTask = std::make_shared<MachineTask>(task_1->GetTaskId(), task_1->GetFunction());
    auto deviceAgentTask = std::make_shared<DeviceAgentTask>(deviceMachineTask);

    KernelDumpUtils kernelDump;
    std::string jsonDir = "/tmp/pypto/";
    std::string kerneName = "pypto_add";
    kernelDump.DumpJsonFile(deviceAgentTask.get(), kerneName,jsonDir);
    std::vector<JsonInfo> binJsonPath;
    std::string jsonFilePath = jsonDir + kerneName + ".json";
    std::string binFileName = "add_bin";
    kernelDump.WriteFatbinJson(binJsonPath, jsonFilePath, binFileName);
    auto ret = IsPathExist(jsonFilePath);
    EXPECT_EQ(ret, false);
}


TEST_F(TestAicpuInterce, test_cost_mode) {
    const std::vector<int64_t> shape = {64, 64};
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A1");
    Tensor inputB(DT_FP32, shape, "B1");
    Tensor output(DT_FP32, shape, "C1");

    config::SetBuildStatic(true);
    FUNCTION("ADD", {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    config::SetPlatformConfig("ENABLE_DYN_COST_MODEL", true);
    CostModelAgent costModel;
    void *costModeData = nullptr;
    costModel.RunCostModel(costModeData);

    config::SetRuntimeOption<int>("run_mode", 1);
    std::string fileDir = config::LogTopFolder();
    std::string path = config::LogTopFolder() + "/dyn_topo.txt";
    CreateMultiLevelDir(fileDir);
    std::string topo = "seqNo, taskId, rootIndex, rootHash, opMagic, leafIndex, leafHash, coreType, psgId, successors";
    DumpFile(topo, path);
    costModel.RunDynCostModel();
    auto ret = IsPathExist(path);
    EXPECT_EQ(ret, true);
}
