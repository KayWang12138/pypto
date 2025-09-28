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
 * \file device_runner.cpp
 * \brief
 */

#ifdef ENABLE_BUILD_WITH_CANN
#include "emulation_launcher.h"

#include "machine/host/backend.h"

extern "C" int __attribute__((weak)) DynTileFwkBackendKernelServer(void *targ) {
    (void)targ;
    return -1;
};
extern "C" int __attribute__((weak)) DynTileFwkBackendKernelServerInit(void *targ) {
    (void)targ;
    return -1;
};

namespace npu::tile_fwk::dynamic {

int EmulationLauncher::EmulationLaunchOnceWithHostTensorData(
        Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
        const DeviceLauncherConfig &config) {
    std::cout << "!!! Emulation Launch " << "\n";

    AstKernelArgs kArgs;
    DeviceLauncher::DeviceInitTilingData(EmulationMemoryUtils(), kArgs, function, config);
    DeviceLauncher::DeviceInitKernelInOuts(EmulationMemoryUtils(), kArgs, inputList, outputList);

    constexpr int threadNum = 6;
    std::thread aicpuThreadList[threadNum];
    int aicpuResultList[threadNum] = {0};
    std::atomic<int> idx{0};
    auto *devProg = (DevAscendProgram *)(kArgs.cfgdata);
    auto rc = DynTileFwkBackendKernelServerInit(&kArgs);
    if (rc != 0) {
        return rc;
    }
    for (int i = 0; i < static_cast<int>(devProg->devArgs.nrAicpu); i++) {
        aicpuThreadList[i] = std::thread([&](int threadIndex) {
            int tidx = idx++;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tidx, &cpuset);
            char name[64];
            sprintf(name, "aicput%d", tidx);
            std::cout << "start thread: " << name << std::endl;
            pthread_setname_np(pthread_self(), name);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            aicpuResultList[threadIndex] = DynTileFwkBackendKernelServer(&kArgs);
        }, i);
    }

    for (int i = 0; i < threadNum; i++) {
        if (aicpuThreadList[i].joinable()) {
            aicpuThreadList[i].join();
        }
    }

    for (int i = 0; i < threadNum; i++) {
        if (aicpuResultList[i] != 0) {
            return aicpuResultList[i];
        }
    }
    return 0;
}

int EmulationLauncher::EmulationRunOnce(Function *function, const DeviceLauncherConfig &config) {
    auto &inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto &outputDataList = ProgramData::GetInstance().GetOutputDataList();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    std::tie(inputDeviceDataList, outputDeviceDataList) = DeviceLauncher::BuildInputOutput(EmulationMemoryUtils(), inputDataList, outputDataList);
    int rc = EmulationLaunchOnceWithHostTensorData(function, inputDeviceDataList, outputDeviceDataList, config);
    return rc;
}

}

#endif
