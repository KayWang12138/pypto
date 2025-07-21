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
 * \file test_device_machine.cpp
 * \brief
 */

#include <regex>
#include <gtest/gtest.h>
#include <iostream>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "machine/device/device_machine.h"

using namespace npu::tile_fwk;

class DeviceMachineTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

extern "C" int ASTServerKernel(void *targ);

TEST(DeviceMachineTest, DeviceMachinetest) {
    std::cout << "start to test runtime" << std::endl;
    DeviceArgs args = {};
    args.nrAicpu = 5;
    std::uint64_t tastWastTime = 0;
    args.taskWastTime = (uint64_t)&tastWastTime;

    std::thread aicpus[6];
    std::atomic<int> idx{0};
    for (int i = 0; i < 5; i++) {
        aicpus[i] = std::thread([&]() {
            int tidx = idx++;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tidx, &cpuset);
            char name[64];
            sprintf(name, "aicput%d", tidx);
            pthread_setname_np(pthread_self(), name);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            ASTServerKernel(&args);
        });
    }

    for (int i = 0; i < 5; i++) {
        aicpus[i].join();
    }
}

TEST(DeviceMachineTest, test_get_task_time) {
    DeviceArgs args = {};
    args.nrAicpu = 1;
    std::uint64_t tastWastTime = 0;
    args.taskWastTime = (uint64_t)&tastWastTime;

    std::thread aicpus[1];
    std::atomic<int> idx{0};
    for (int i = 0; i < 1; i++) {
        aicpus[i] = std::thread([&]() {
            int tidx = idx++;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tidx, &cpuset);
            char name[64];
            sprintf(name, "aicput%d", tidx);
            pthread_setname_np(pthread_self(), name);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            ASTServerKernel(&args);
        });
    }

    for (int i = 0; i < 1; i++) {
        aicpus[i].join();
    }
}