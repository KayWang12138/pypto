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
 * \file test_new_hostmachine.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <regex>
#include <fstream>
#include <chrono>
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config.h"
#include "interface/machine/host/host_machine.h"
#include "interface/function/function.h"

namespace npu::tile_fwk { //测试新的hostmachine

class HostMachineTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

TEST_F(HostMachineTest, HostMachineTest_test1) {
    std::cout.rdbuf()->pubsetbuf(NULL, std::ios::out);

    HostMachine& hostMachine = Program::GetInstance().GetHostMachine();
    hostMachine.Init(HostMachineMode::SERVER);

    Function func(Program::GetInstance(), "", "", nullptr);
    for (int i = 0; i < 100; ++i) {
        hostMachine.SubTask(&func);
    }

    std::cout << "hostmachine test1 end" << std::endl;
    hostMachine.Destroy();
}

} // namespace npu::tile_fwk
