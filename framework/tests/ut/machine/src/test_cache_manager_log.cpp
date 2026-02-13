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
 * \file test_cache_manager_log.cpp
 * \brief Unit tests for cache_manager.cpp covering MACHINE_LOG calls at lines 56, 63, 117
 */

#include <gtest/gtest.h>
#include <fstream>
#include <cstdlib>
#include "interface/configs/config_manager.h"
#include "interface/utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "tilefwk/tilefwk_log.h"

#define private public
#include "machine/cache_manager/cache_manager.h"
#undef private

using namespace npu::tile_fwk;

namespace {
const std::string CM_TEST_TMP_DIR = "/tmp/test_cache_manager_log";
}

class TestCacheManagerLog : public testing::Test {
public:
    static void SetUpTestCase() {
        CreateMultiLevelDir(CM_TEST_TMP_DIR);
    }

    static void TearDownTestCase() {
        std::string cmd = "rm -rf " + CM_TEST_TMP_DIR;
        [[maybe_unused]] int ret = system(cmd.c_str());
    }

    void SetUp() override {
        // Save original HOME
        const char *env = std::getenv("HOME");
        origHome_ = env ? std::string(env) : "";
        hasHome_ = (env != nullptr);
    }

    void TearDown() override {
        // Restore HOME
        if (hasHome_) {
            setenv("HOME", origHome_.c_str(), 1);
        } else {
            unsetenv("HOME");
        }
        config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, false);
    }

private:
    std::string origHome_;
    bool hasHome_ = false;
};

// ===================================================================
// Covers Line 56: MACHINE_LOGE("Env[HOME] is not existed or empty.")
// HOME env is not set, binary cache is enabled -> Initialize fails
// ===================================================================
TEST_F(TestCacheManagerLog, Initialize_HomeEnvNotSet) {
    config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, true);
    unsetenv("HOME");

    CacheManager cm;
    bool result = cm.Initialize();
    EXPECT_FALSE(result);
}

// ===================================================================
// Covers Line 63: MACHINE_LOGE("Failed to create cache dir[%s].", cacheDirPath_.c_str())
// HOME is set to a path where subdirectories cannot be created (e.g. /proc)
// -> RealPath(cacheDirPath_) is empty AND CreateMultiLevelDir fails
// ===================================================================
TEST_F(TestCacheManagerLog, Initialize_CreateCacheDirFails) {
    config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, true);
    // /proc is a special filesystem where mkdir fails
    setenv("HOME", "/proc", 1);

    CacheManager cm;
    bool result = cm.Initialize();
    EXPECT_FALSE(result);
}

// ===================================================================
// Covers Line 117: MACHINE_LOGI("Bin file[%s] and [%s] already exists.", ...)
// All three cache files (bin, so, json) already exist -> SaveTaskFile returns early
// ===================================================================
TEST_F(TestCacheManagerLog, SaveTaskFile_FilesAlreadyExist) {
    // Set up CacheManager with cache enabled and a known cache dir
    CacheManager cm;
    cm.cacheMode_ = CacheMode::Enable;
    cm.isInit_ = true;
    cm.cacheDirPath_ = CM_TEST_TMP_DIR + "/cache_exist";
    CreateMultiLevelDir(cm.cacheDirPath_);

    // Set up OpInfoManager function name
    std::string opFuncName = "ut_log_test_func";
    OpInfoManager::GetInstance().GetOpFuncName() = opFuncName;

    // Compute the expected file paths (matching cache_manager.cpp logic)
    std::string cacheKey = "ut_log_test_key";
    std::string binFile = cm.cacheDirPath_ + "/ast_op_" + cacheKey + ".o";
    std::string soFile = cm.cacheDirPath_ + "/lib" + opFuncName + "_control.so";
    std::string jsonFile = cm.cacheDirPath_ + "/lib" + opFuncName + "_control.json";

    // Create all three files so they exist
    std::ofstream(binFile).close();
    std::ofstream(soFile).close();
    std::ofstream(jsonFile).close();

    // Create a MachineTask with a non-null Function pointer (never dereferenced before line 117 return)
    auto machineTask = std::make_shared<MachineTask>(1, reinterpret_cast<Function *>(0x1234));
    machineTask->SetCacheKey(cacheKey);
    auto deviceTask = std::make_unique<DeviceAgentTask>(machineTask);

    // SaveTaskFile checks: cache enabled ✓, deviceAgentTask != null ✓, GetFunction() != null ✓
    // Then checks RealPath of all three files -> all exist -> line 117 MACHINE_LOGI hit -> return
    cm.SaveTaskFile(deviceTask.get());

    // No crash means the early return at line 117 was taken (files already exist)
    SUCCEED();
}
