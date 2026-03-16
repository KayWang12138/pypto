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
 * \file test_dynamic_runner.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include <cstdlib>
#include "machine/runtime/device_runner.h"
#include "machine/runtime/device_launcher.h"
#include "machine/runtime/host_prof.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "machine/device/machine_interface/pypto_aicpu_interface.h"
#include "machine/utils/machine_ws_intf.h"
#include "interface/program/program.h"
#include "interface/utils/file_utils.h"
#include "tilefwk/aicpu_common.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/runtime/dump_device_perf.h"
#define private public
using namespace npu::tile_fwk;

extern "C" uint32_t DynPyptoKernelServerNull(void *targ);
extern "C" uint32_t DynTileFwkBackendKernelServer(void *targ);
extern "C" uint32_t StaticTileFwkBackendKernelServer(void *targ);
class TestDynamicDeviceRunner : public testing::Test {
public:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}
};

TEST_F(TestDynamicDeviceRunner, TestInitArgs) {
    auto &runner = DeviceRunner::Get();
    [[maybe_unused]]DeviceArgs args;
    args.nrAic = 2;
    args.nrAiv = 2;
    args.nrValidAic = args.nrAic;
    runner.InitDynamicArgs(args);
    runner.DumpAiCoreExecutionTimeData();
    runner.DumpAiCorePmuData();
    runner.SynchronizeDeviceToHostProfData();
}

TEST_F(TestDynamicDeviceRunner, TestDynamicRun) {
    auto &runner = npu::tile_fwk::DeviceRunner::Get();
    [[maybe_unused]]DeviceArgs args;
    args.nrAic = 2;
    args.nrAiv = 2;
    runner.InitDynamicArgs(args);
    [[maybe_unused]]npu::tile_fwk::DeviceKernelArgs taskArgs;
    std::vector<uint8_t> tensorInfo(sizeof(dynamic::AiCpuArgs));
    taskArgs.inputs = reinterpret_cast<int64_t*>(tensorInfo.data());
    taskArgs.outputs = 0;
    runner.args_.nrAic = 2;
    runner.args_.nrAiv = 2;
    int ret = runner.DynamicRun(0, 0, 0, 0, &taskArgs, 2);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestDynamicDeviceRunner, TestRegisterDynamicKernel) {
    [[maybe_unused]]rtBinHandle staticHdl_;
    npu::tile_fwk::DeviceRunner runner;
    runner.RegisterKernelBin(&staticHdl_);
}

TEST_F(TestDynamicDeviceRunner, test_pypto_kernel_server_null) {
    DeviceKernelArgs pyptoKernelArgs;
    DeviceArgs devKernelArgs;
    devKernelArgs.aicpuSoLen = 2;
    pyptoKernelArgs.cfgdata = static_cast<int64_t *>(static_cast<void *>(&devKernelArgs));
    auto ret = DynPyptoKernelServerNull(&pyptoKernelArgs);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestDynamicDeviceRunner, test_pypto_kernel_server_null_args_null) {
    auto ret = DynPyptoKernelServerNull(nullptr);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestDynamicDeviceRunner, test_pypto_kernel_server_execute_func_invalid) {
    DeviceKernelArgs pyptoKernelArgs {};
    auto ret = DynPyptoKernelServer(&pyptoKernelArgs);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestDynamicDeviceRunner, test_pypto_kernel_server_init_func_invalid) {
    DeviceKernelArgs pyptoKernelArgs {};
    auto ret = DynPyptoKernelServerInit(&pyptoKernelArgs);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestDynamicDeviceRunner, test_launch_init) {
    DeviceKernelArgs pyptoKernelArgs;
    DeviceArgs devKernelArgs;
    devKernelArgs.aicpuPerfAddr = 1;
    pyptoKernelArgs.cfgdata = static_cast<int64_t *>(static_cast<void *>(&devKernelArgs));
    auto ret = DynTileFwkBackendKernelServer(&pyptoKernelArgs);
    EXPECT_EQ(ret, -1);
}

TEST_F(TestDynamicDeviceRunner, test_static) {
    EXPECT_EQ(StaticTileFwkBackendKernelServer(nullptr), 0);
}