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
#include "machine/device/machine_interface/pypto_aicpu_interface.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/dump/kernel_dump_utils.h"
#include "machine/runtime/machine_agent.h"
#define private public

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
    DeviceArgs devKernelArgs;
    devKernelArgs.aicpuSoLen = 2;
    auto ret = DynPyptoKernelServerNull(&devKernelArgs);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestAicpuInterce, test_kernel_dump) {
    npu::tile_fwk::MachinePipe machinePipe;
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    config::SetBuildStatic(true);
    FUNCTION("ADD", {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    auto function = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD");
    auto task_1 = std::make_shared<MachineTask>(0, function);
    auto deviceMachineTask = std::make_shared<MachineTask>(task_1->GetTaskId(), task_1->GetFunction());
    auto deviceAgentTask = std::make_shared<DeviceAgentTask>(deviceMachineTask);


    // DeviceAgentTask agentTask1(task_1);

    KernelDumpUtils kernelDump;
    std::string jsonDir = "/tmp/pypto/";
    std::string kerneName = "pypto_add";
    kernelDump.DumpJsonFile(deviceAgentTask, kerneName,jsonDir);
}

