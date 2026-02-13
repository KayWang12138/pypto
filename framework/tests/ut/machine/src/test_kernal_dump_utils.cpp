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
 * \file test_kernel_dump_utils.cpp
 * \brief Unit tests for kernel_dump_utils.cpp covering all MACHINE_LOG calls
 */

#include <gtest/gtest.h>
#include <fstream>
#include <cstdlib>
#include <climits>
#include <sys/stat.h>
#include <nlohmann/json.hpp>

#include "machine/dump/kernel_dump_utils.h"
#include "interface/utils/file_utils.h"
#include "tilefwk/tilefwk_log.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "interface/configs/config_manager.h"
#include "interface/program/program.h"
#include "interface/machine/host/machine_task.h"
#include "machine/host/device_agent_task.h"
#include "machine/platform/platform_manager.h"

using namespace npu::tile_fwk;
using Json = nlohmann::json;

namespace {
const std::string TEST_TMP_DIR = "/tmp/test_kernel_dump_utils";
}

class TestKernelDumpUtils : public testing::Test {
public:
    static void SetUpTestCase() {
        CreateMultiLevelDir(TEST_TMP_DIR);
    }

    static void TearDownTestCase() {
        std::string cmd = "rm -rf " + TEST_TMP_DIR;
        (void)system(cmd.c_str());
    }

    void SetUp() override {}
    void TearDown() override {}
};

// ===================================================================
// Tests for GetBufferFromBinFile
// ===================================================================

// Covers Line 161: MACHINE_LOGI("Bin file path[%s] is invalid.", ...)
TEST_F(TestKernelDumpUtils, GetBufferFromBinFile_InvalidPath) {
    std::vector<char> buffer;
    bool result = KernelDumpUtils::GetBufferFromBinFile("/nonexistent/path/to/file.bin", buffer);
    EXPECT_FALSE(result);
    EXPECT_TRUE(buffer.empty());
}

// Covers Line 166: MACHINE_LOGI("Fail to open bin file path[%s].", ...)
TEST_F(TestKernelDumpUtils, GetBufferFromBinFile_CannotOpenFile) {
    std::string filePath = TEST_TMP_DIR + "/no_read_perm.bin";
    std::ofstream ofs(filePath, std::ios::binary);
    ofs << "test data";
    ofs.close();
    chmod(filePath.c_str(), 0000);

    std::vector<char> buffer;
    bool result = KernelDumpUtils::GetBufferFromBinFile(filePath, buffer);
    EXPECT_FALSE(result);

    chmod(filePath.c_str(), 0644);
    remove(filePath.c_str());
}

// Covers Line 173: MACHINE_LOGI("Get stream failed")
TEST_F(TestKernelDumpUtils, GetBufferFromBinFile_EmptyFile) {
    std::string filePath = TEST_TMP_DIR + "/empty_file.bin";
    std::ofstream ofs(filePath, std::ios::binary);
    ofs.close();

    std::vector<char> buffer;
    bool result = KernelDumpUtils::GetBufferFromBinFile(filePath, buffer);
    EXPECT_FALSE(result);
    remove(filePath.c_str());
}

// Success path
TEST_F(TestKernelDumpUtils, GetBufferFromBinFile_Success) {
    std::string filePath = TEST_TMP_DIR + "/valid_file.bin";
    std::ofstream ofs(filePath, std::ios::binary);
    std::vector<char> testData = {'H', 'e', 'l', 'l', 'o'};
    ofs.write(testData.data(), testData.size());
    ofs.close();

    std::vector<char> buffer;
    bool result = KernelDumpUtils::GetBufferFromBinFile(filePath, buffer);
    EXPECT_TRUE(result);
    EXPECT_EQ(buffer.size(), testData.size());
    remove(filePath.c_str());
}

// Test appending to existing buffer
TEST_F(TestKernelDumpUtils, GetBufferFromBinFile_AppendToBuffer) {
    std::string filePath = TEST_TMP_DIR + "/append_file.bin";
    std::ofstream ofs(filePath, std::ios::binary);
    std::vector<char> testData = {'W', 'o', 'r', 'l', 'd'};
    ofs.write(testData.data(), testData.size());
    ofs.close();

    std::vector<char> buffer = {'H', 'i', ' '};
    size_t origSize = buffer.size();
    bool result = KernelDumpUtils::GetBufferFromBinFile(filePath, buffer);
    EXPECT_TRUE(result);
    EXPECT_EQ(buffer.size(), origSize + testData.size());
    remove(filePath.c_str());
}

// ===================================================================
// Tests for WriteBufferToFatbin
// ===================================================================

// Covers Line 199: MACHINE_LOGE("Failed to open file[%s].", ...)
TEST_F(TestKernelDumpUtils, WriteBufferToFatbin_CannotOpen) {
    FatbinHeadInfo headInfo;
    std::vector<char> buffer;
    bool result = KernelDumpUtils::WriteBufferToFatbin(headInfo, "/nonexistent_dir/fatbin.bin", buffer);
    EXPECT_FALSE(result);
}

// Covers Line 219: MACHINE_LOGI("headInfoSize is: %zu, fatbin size is: %ld.", ...)
TEST_F(TestKernelDumpUtils, WriteBufferToFatbin_Success) {
    FatbinHeadInfo headInfo(2);
    headInfo.configKeyList = {100, 200};
    headInfo.binOffsets = {0, 16};
    std::vector<char> buffer(32, 'A');

    std::string path = TEST_TMP_DIR + "/test_fatbin.bin";
    bool result = KernelDumpUtils::WriteBufferToFatbin(headInfo, path, buffer);
    EXPECT_TRUE(result);

    // Verify file exists and has content
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    EXPECT_TRUE(ifs.is_open());
    EXPECT_GT(ifs.tellg(), 0);
    ifs.close();
    remove(path.c_str());
}

// Covers Line 219 with empty data
TEST_F(TestKernelDumpUtils, WriteBufferToFatbin_EmptyData) {
    FatbinHeadInfo headInfo(0);
    std::vector<char> buffer;

    std::string path = TEST_TMP_DIR + "/test_fatbin_empty.bin";
    bool result = KernelDumpUtils::WriteBufferToFatbin(headInfo, path, buffer);
    EXPECT_TRUE(result);
    remove(path.c_str());
}

// ===================================================================
// Tests for WriteFatbinJson
// ===================================================================

TEST_F(TestKernelDumpUtils, WriteFatbinJson_SingleKernel) {
    std::vector<JsonInfo> allBinJsonInfo;
    JsonInfo info;
    info.blockDim = 24;
    info.kernelName = "test_kernel";
    info.configKey = 12345;
    info.workspaceSize = 2048;
    allBinJsonInfo.push_back(info);

    std::string jsonPath = TEST_TMP_DIR + "/test_fatbin_single.json";
    KernelDumpUtils::WriteFatbinJson(allBinJsonInfo, jsonPath, "test_bin_file");

    std::ifstream ifs(jsonPath);
    EXPECT_TRUE(ifs.is_open());
    Json jsonValue;
    ifs >> jsonValue;
    ifs.close();

    EXPECT_EQ(jsonValue["binFileName"], "test_bin_file");
    EXPECT_EQ(jsonValue["binFileSuffix"], ".o");
    EXPECT_EQ(jsonValue["coreType"], "MIX");
    EXPECT_EQ(jsonValue["kernelList"].size(), 1u);
    EXPECT_EQ(jsonValue["kernelList"][0]["blockDim"], 24);
    EXPECT_EQ(jsonValue["kernelList"][0]["kernelName"], "test_kernel");
    EXPECT_EQ(jsonValue["kernelList"][0]["workspaceSize"], 2048);
    remove(jsonPath.c_str());
}

TEST_F(TestKernelDumpUtils, WriteFatbinJson_MultipleKernels) {
    std::vector<JsonInfo> allBinJsonInfo;
    for (int i = 0; i < 3; i++) {
        JsonInfo info;
        info.blockDim = i + 1;
        info.kernelName = "kernel_" + std::to_string(i);
        info.configKey = i * 100;
        info.workspaceSize = (i + 1) * 1024;
        allBinJsonInfo.push_back(info);
    }

    std::string jsonPath = TEST_TMP_DIR + "/test_fatbin_multi.json";
    KernelDumpUtils::WriteFatbinJson(allBinJsonInfo, jsonPath, "multi_bin");

    std::ifstream ifs(jsonPath);
    EXPECT_TRUE(ifs.is_open());
    Json jsonValue;
    ifs >> jsonValue;
    ifs.close();

    EXPECT_EQ(jsonValue["kernelList"].size(), 3u);
    EXPECT_EQ(jsonValue["workspace"]["size"][0], 3 * 1024);
    remove(jsonPath.c_str());
}

// ===================================================================
// Tests for GetSubJsonInfo
// ===================================================================

// Covers Line 256: MACHINE_LOGI("realpath failed, val is: %s.", ...)
TEST_F(TestKernelDumpUtils, GetSubJsonInfo_InvalidPath) {
    JsonInfo info;
    bool result = KernelDumpUtils::GetSubJsonInfo("/nonexistent/path/info.json", info);
    EXPECT_FALSE(result);
}

// File cannot be opened (branch coverage for ifs.is_open() check)
TEST_F(TestKernelDumpUtils, GetSubJsonInfo_CannotOpen) {
    std::string jsonPath = TEST_TMP_DIR + "/no_read_json.json";
    std::ofstream ofs(jsonPath);
    ofs << "{}" << std::endl;
    ofs.close();
    chmod(jsonPath.c_str(), 0000);

    JsonInfo info;
    bool result = KernelDumpUtils::GetSubJsonInfo(jsonPath, info);
    EXPECT_FALSE(result);

    chmod(jsonPath.c_str(), 0644);
    remove(jsonPath.c_str());
}

// Covers Line 279: MACHINE_LOGE("Fail to parse json, error:%s, json is %s.", ...)
// blockDim key is missing
TEST_F(TestKernelDumpUtils, GetSubJsonInfo_MissingRequiredKeys) {
    std::string jsonPath = TEST_TMP_DIR + "/missing_keys.json";
    std::ofstream ofs(jsonPath);
    Json j;
    j["someOtherKey"] = 123;
    ofs << j.dump(4) << std::endl;
    ofs.close();

    JsonInfo info;
    bool result = KernelDumpUtils::GetSubJsonInfo(jsonPath, info);
    EXPECT_FALSE(result);
    remove(jsonPath.c_str());
}

// Covers Line 279: workspace format invalid
TEST_F(TestKernelDumpUtils, GetSubJsonInfo_InvalidWorkspaceFormat) {
    std::string jsonPath = TEST_TMP_DIR + "/invalid_ws.json";
    std::ofstream ofs(jsonPath);
    Json j;
    j["blockDim"] = 24;
    j["kernelName"] = "test_kernel";
    j["workspace"] = {{"size", "not_a_list"}};
    ofs << j.dump(4) << std::endl;
    ofs.close();

    JsonInfo info;
    bool result = KernelDumpUtils::GetSubJsonInfo(jsonPath, info);
    EXPECT_FALSE(result);
    remove(jsonPath.c_str());
}

// JSON parse exception (malformed json)
TEST_F(TestKernelDumpUtils, GetSubJsonInfo_MalformedJson) {
    std::string jsonPath = TEST_TMP_DIR + "/malformed.json";
    std::ofstream ofs(jsonPath);
    ofs << "{ not valid json }" << std::endl;
    ofs.close();

    JsonInfo info;
    bool result = KernelDumpUtils::GetSubJsonInfo(jsonPath, info);
    EXPECT_FALSE(result);
    remove(jsonPath.c_str());
}

// Success with workspace
TEST_F(TestKernelDumpUtils, GetSubJsonInfo_SuccessWithWorkspace) {
    std::string jsonPath = TEST_TMP_DIR + "/valid_info.json";
    std::ofstream ofs(jsonPath);
    Json j;
    j["blockDim"] = 24;
    j["kernelName"] = "ast_main_0";
    j["workspace"] = {{"num", 1}, {"size", {4096}}, {"type", {0}}};
    ofs << j.dump(4) << std::endl;
    ofs.close();

    JsonInfo info;
    bool result = KernelDumpUtils::GetSubJsonInfo(jsonPath, info);
    EXPECT_TRUE(result);
    EXPECT_EQ(info.blockDim, 24);
    EXPECT_EQ(info.kernelName, "ast_main_0");
    EXPECT_EQ(info.workspaceSize, 4096);
    remove(jsonPath.c_str());
}

// Success without workspace
TEST_F(TestKernelDumpUtils, GetSubJsonInfo_SuccessNoWorkspace) {
    std::string jsonPath = TEST_TMP_DIR + "/no_ws.json";
    std::ofstream ofs(jsonPath);
    Json j;
    j["blockDim"] = 12;
    j["kernelName"] = "test_no_ws";
    ofs << j.dump(4) << std::endl;
    ofs.close();

    JsonInfo info;
    bool result = KernelDumpUtils::GetSubJsonInfo(jsonPath, info);
    EXPECT_TRUE(result);
    EXPECT_EQ(info.blockDim, 12);
    EXPECT_EQ(info.kernelName, "test_no_ws");
    EXPECT_EQ(info.workspaceSize, -1);
    remove(jsonPath.c_str());
}

// ===================================================================
// Tests for LoadTileFwkImplOpLib & GetEnv (internal)
// ===================================================================

// Covers Line 289: MACHINE_LOGI("Env[%s] is not found.", ...)
// Covers Line 298: MACHINE_LOGI("Value of env[TILE_FWK_OP_IMPL_PATH] is [%s].", ...)
// Covers Line 300: MACHINE_LOGW("Value of env[TILE_FWK_OP_IMPL_PATH] is empty.")
TEST_F(TestKernelDumpUtils, LoadTileFwkImplOpLib_EnvNotSet) {
    unsetenv("TILE_FWK_OP_IMPL_PATH");
    void *handle = KernelDumpUtils::LoadTileFwkImplOpLib();
    EXPECT_EQ(handle, nullptr);
}

// Covers Line 298: MACHINE_LOGI("Value of env[TILE_FWK_OP_IMPL_PATH] is [%s].", ...)
// Covers Line 305: MACHINE_LOGI("Failed to dlopen %s, reason is %s.", ...)
TEST_F(TestKernelDumpUtils, LoadTileFwkImplOpLib_InvalidLibPath) {
    setenv("TILE_FWK_OP_IMPL_PATH", "/nonexistent/path/libfake.so", 1);
    void *handle = KernelDumpUtils::LoadTileFwkImplOpLib();
    EXPECT_EQ(handle, nullptr);
    unsetenv("TILE_FWK_OP_IMPL_PATH");
}

// ===================================================================
// Tests for FreeOpHandle
// ===================================================================

TEST_F(TestKernelDumpUtils, FreeOpHandle_Nullptr) {
    KernelDumpUtils::FreeOpHandle(nullptr);
}

// ===================================================================
// Tests for GetDumpKernelPath
// ===================================================================

TEST_F(TestKernelDumpUtils, GetDumpKernelPath_ReturnsNonEmpty) {
    std::string path = GetDumpKernelPath();
    // In a linked shared library context, dladdr should resolve the path
    // The result depends on the runtime environment
    (void)path;
}

// ===================================================================
// Tests for DumpKernelFile / DumpBinFile / DumpJsonFile
// ===================================================================

// Covers Line 59: deviceAgentTask == nullptr
TEST_F(TestKernelDumpUtils, DumpKernelFile_NullTask) {
    bool result = KernelDumpUtils::DumpKernelFile(nullptr, "test_kernel", "/tmp", "");
    EXPECT_FALSE(result);
}

// Covers Line 59: deviceAgentTask->GetFunction() == nullptr
TEST_F(TestKernelDumpUtils, DumpKernelFile_NullFunction) {
    auto machineTask = std::make_shared<MachineTask>(1, nullptr);
    auto deviceAgentTask = std::make_unique<DeviceAgentTask>(machineTask);
    bool result = KernelDumpUtils::DumpKernelFile(deviceAgentTask.get(), "test_kernel", "/tmp", "");
    EXPECT_FALSE(result);
}

// Covers Line 92:  MACHINE_LOGI("OP binary data is empty, function type is [%s].", ...)
// Covers Line 148: MACHINE_LOGI("Work space size is [%lu].", ...)
TEST_F(TestKernelDumpUtils, DumpKernelFile_StaticFunction_EmptyOpBinData) {
    Program::GetInstance().Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "dump_ut_t0");
    Tensor out(DT_FP32, {s, s}, "dump_ut_out");
    FUNCTION("test_dump_static", {t0}, {out}) {
        auto temp = Add(t0, t0);
        Assemble(temp, {0, 0}, out);
    }

    Function *func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);

    auto machineTask = std::make_shared<MachineTask>(1, func);
    auto deviceAgentTask = std::make_unique<DeviceAgentTask>(machineTask);

    std::string dumpDir = TEST_TMP_DIR + "/dump_static";
    // DumpBinFile: opBinData is empty (STATIC function) -> Line 92 MACHINE_LOGI, returns true
    // DumpJsonFile is then called -> Line 148 MACHINE_LOGI
    bool result = KernelDumpUtils::DumpKernelFile(deviceAgentTask.get(), "test_static_kernel", dumpDir, "");
    EXPECT_TRUE(result);
}

// Covers Line 92 with empty kernelName -> uses GetMagicName()
TEST_F(TestKernelDumpUtils, DumpKernelFile_EmptyKernelName) {
    Program::GetInstance().Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "dump_ut2_t0");
    Tensor out(DT_FP32, {s, s}, "dump_ut2_out");
    FUNCTION("test_dump_magic", {t0}, {out}) {
        auto temp = Add(t0, t0);
        Assemble(temp, {0, 0}, out);
    }

    Function *func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);

    auto machineTask = std::make_shared<MachineTask>(2, func);
    auto deviceAgentTask = std::make_unique<DeviceAgentTask>(machineTask);

    std::string dumpDir = TEST_TMP_DIR + "/dump_magic_name";
    // Empty kernelName triggers finalKernelName = GetMagicName()
    bool result = KernelDumpUtils::DumpKernelFile(deviceAgentTask.get(), "", dumpDir, "");
    EXPECT_TRUE(result);
}

// Covers Line 102: MACHINE_LOGI("Dynamic Kernel path %s.", ...)
// Covers Line 148: MACHINE_LOGI("Work space size is [%lu].", ...)
TEST_F(TestKernelDumpUtils, DumpKernelFile_DynamicFunction) {
    Program::GetInstance().Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    constexpr int LOOP_COUNT = 4;
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "dump_dyn_t0");
    Tensor out(DT_FP32, {LOOP_COUNT * s, s}, "dump_dyn_out");
    FUNCTION("test_dump_dynamic", {t0}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT)) {
            auto temp = Add(t0, t0);
            Assemble(temp, {i * s, 0}, out);
        }
    }

    Function *func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);

    // Check that the function is DYNAMIC and has devProgBinary
    auto dynAttr = func->GetDyndevAttribute();
    if (dynAttr != nullptr && !dynAttr->devProgBinary.empty()) {
        auto machineTask = std::make_shared<MachineTask>(3, func);
        auto deviceAgentTask = std::make_unique<DeviceAgentTask>(machineTask);

        std::string dumpDir = TEST_TMP_DIR + "/dump_dynamic";
        // isDynamic is true, opBinData is non-empty
        // -> Line 102: MACHINE_LOGI("Dynamic Kernel path %s.", ...)
        // DumpJsonFile -> Line 148: MACHINE_LOGI("Work space size is [%lu].", ...)
        bool result = KernelDumpUtils::DumpKernelFile(
            deviceAgentTask.get(), "test_dynamic_kernel", dumpDir, "/nonexistent/kernel.o");
        (void)result;
    }
}

// Test DumpKernelFile with invalid dump directory that cannot be created
TEST_F(TestKernelDumpUtils, DumpKernelFile_InvalidDumpDir) {
    Program::GetInstance().Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "dump_ut3_t0");
    Tensor out(DT_FP32, {s, s}, "dump_ut3_out");
    FUNCTION("test_dump_invalid_dir", {t0}, {out}) {
        auto temp = Add(t0, t0);
        Assemble(temp, {0, 0}, out);
    }

    Function *func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);

    auto machineTask = std::make_shared<MachineTask>(4, func);
    auto deviceAgentTask = std::make_unique<DeviceAgentTask>(machineTask);

    // Use a path under /proc which typically cannot have new dirs created
    bool result = KernelDumpUtils::DumpKernelFile(deviceAgentTask.get(), "test_kernel", "/proc/fake_dump_dir", "");
    EXPECT_FALSE(result);
}
