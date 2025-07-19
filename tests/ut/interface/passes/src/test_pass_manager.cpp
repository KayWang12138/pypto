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
 * \file test_pass_manager.cpp
 * \brief Unit test for pass manager.
 */
#include <fstream>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "passes/pass_manager.h"
#include "operation/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_registry.h"
#include "ut_json/ut_json_tool.h"


namespace npu::tile_fwk {
class PassTestCast : public Pass {
public:
    PassTestCast() : Pass("PassTestCast") {}

    Status PreCheck(Function &function) override {
        (void) function;
        return FAILED;
    }
    Status PostCheck(Function &function) override {
        (void) function;
        return FAILED;
    }
    Status RunOnFunction(Function &function) override {
        (void) function;
        return FAILED;
    }
    Status CreateLogFolder(const std::string &topFolder, size_t i) const override {
        (void) topFolder;
        (void) i;
        return FAILED;
    }
    Status PrintFunction(Function& function, const std::string &logFolder, bool beforeFunction) override {
        (void) function;
        (void) logFolder;
        (void) beforeFunction;
        return FAILED;
    }
    Status DumpFunctionJson(Function& function, const std::string &logFolder, bool beforeFunction) override {
        (void) function;
        (void) logFolder;
        (void) beforeFunction;
        return FAILED;
    }
    Status PreRun(Function &function) override {
        (void) function;
        return FAILED;
    }
    Status PostRun(Function &function) override {
        (void) function;
        return FAILED;
    }
};

class PassManagerTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    }
    void TearDown() override {}
};

TEST_F(PassManagerTest, TestPassManager) {
    REG_PASS(PassTestCast);
    PassManager::Instance().RegisterStrategy("PM_TEST", {
                        {   "PassTestCast1",   "PassTestCast1",  PassType::TYPE_TENSOR_GRAPH}});
    PassManager::Instance().RegisterStrategy("PM_TEST2", {
                        {   "PassTestCast1",   "PassTestCast1",  PassType::TYPE_TENSOR_GRAPH}});
    auto errPasses = PassManager::Instance().GetStrategyPasses("PM_TEST1");
    EXPECT_TRUE(errPasses.empty());
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPassManager", "TestPassManager", nullptr);
    EXPECT_TRUE(PassManager::Instance().RunPass(Program::GetInstance(), *currFunctionPtr, "PM_TEST") == FAILED);
    EXPECT_TRUE(PassManager::Instance().RunPass(Program::GetInstance(), *currFunctionPtr, "PM_TEST2") == FAILED);
}

TEST_F(PassManagerTest, TestPassBase) {
    PassTestCast passTestCase;
    auto logFolder = passTestCase.LogFolder("output", 0);
    EXPECT_TRUE(logFolder.empty() == false);
    auto currFunctionPtr1 = std::make_shared<Function>(Program::GetInstance(), "TestPassManager1", "TestPassManager1", nullptr);
    PassConfigs configs;
    configs.printFunction = true;
    passTestCase.SetPassConfigs(configs);
    auto res = passTestCase.Run(*currFunctionPtr1, "TestPassManager1", "TestPassManager1");
    EXPECT_TRUE(res == FAILED);
    configs.printFunction = false;
    configs.dumpFunctionGraphBeforePass = true;
    passTestCase.SetPassConfigs(configs);
    res = passTestCase.Run(*currFunctionPtr1, "TestPassManager1", "TestPassManager1");
    EXPECT_TRUE(res == FAILED);
    configs.printFunction = false;
    configs.dumpFunctionGraphBeforePass = false;
    configs.preCheck = true;
    passTestCase.SetPassConfigs(configs);
    res = passTestCase.Run(*currFunctionPtr1, "TestPassManager1", "TestPassManager1");
    EXPECT_TRUE(res == FAILED);
}
}
