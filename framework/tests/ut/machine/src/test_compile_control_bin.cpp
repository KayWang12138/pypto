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
 * \file test_compile_control_bin.cpp
 * \brief Unit tests for compile_control_bin.cpp covering all MACHINE_LOG calls
 */

#include <gtest/gtest.h>
#include <fstream>
#include <cstdlib>
#include <nlohmann/json.hpp>

#include "machine/compile/compile_control_bin.h"
#include "interface/utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "machine/utils/machine_utils.h"
#include "tilefwk/tilefwk_log.h"

using namespace npu::tile_fwk;
using Json = nlohmann::json;

// Forward declarations for internal (non-header) functions with external linkage
namespace npu::tile_fwk {
void GenCustomOpInfo(const std::string &funcName, const std::string &controlAicpuPath,
                     const std::string &constrolSoName);
bool GenTilingFunc(const std::string &funcName, const std::string &controlAicpuPath);
bool TieFwkAicpuPreCompile(std::string &preCompileO, std::string &controlAicpuPath);
bool SharedAicpuCompile(const std::string &funcName, const std::string &aicpuDirPath,
                        const std::string &preCompileO);
}

namespace {
const std::string TEST_TMP_DIR = "/tmp/test_compile_control_bin";
}

class TestCompileControlBin : public testing::Test {
public:
    static void SetUpTestCase() {
        CreateMultiLevelDir(TEST_TMP_DIR);
    }

    static void TearDownTestCase() {
        std::string cmd = "rm -rf " + TEST_TMP_DIR;
        [[maybe_unused]]int ret = system(cmd.c_str());
    }

    void SetUp() override {
        setenv("LC_ALL", "C", 1);
        setenv("LANG", "C", 1);
    }
    void TearDown() override {}
};

// ===================================================================
// Tests for GenCustomOpInfo
// ===================================================================

// Covers Line 65: MACHINE_LOGE("Contrust custom op json failed")
// DumpFile fails because the directory does not exist
TEST_F(TestCompileControlBin, GenCustomOpInfo_DumpFileFails) {
    GenCustomOpInfo("test_func", "/nonexistent_dir/aicpu", "libtest_control");
    // GenCustomOpInfo returns void; we just verify it doesn't crash
    // The MACHINE_LOGE at line 65 is triggered because DumpFile fails
}

// Success path for GenCustomOpInfo (no MACHINE_LOG on success, but tests the happy path)
TEST_F(TestCompileControlBin, GenCustomOpInfo_Success) {
    std::string aicpuPath = TEST_TMP_DIR + "/gen_custom_op";
    CreateMultiLevelDir(aicpuPath);

    GenCustomOpInfo("test_func", aicpuPath, "libtest_control");

    // Verify the json file was created
    std::string jsonFilePath = aicpuPath + "/libtest_control.json";
    std::ifstream ifs(jsonFilePath);
    EXPECT_TRUE(ifs.is_open());
    if (ifs.is_open()) {
        Json jsonValue;
        ifs >> jsonValue;
        ifs.close();
    }
}

// ===================================================================
// Tests for GenTilingFunc
// ===================================================================

// GenTilingFunc succeeds when directory exists
TEST_F(TestCompileControlBin, GenTilingFunc_Success) {
    std::string controlPath = TEST_TMP_DIR + "/gen_tiling_func";
    CreateMultiLevelDir(controlPath);

    bool result = GenTilingFunc("test_op", controlPath);
    EXPECT_TRUE(result);

    // Verify the generated cpp file exists
    std::string cppFilePath = controlPath + "/control_flow_kernel.cpp";
    std::ifstream ifs(cppFilePath);
    EXPECT_TRUE(ifs.is_open());
    ifs.close();
}

// GenTilingFunc fails when directory does not exist and DumpFile fails
TEST_F(TestCompileControlBin, GenTilingFunc_DumpFileFails) {
    bool result = GenTilingFunc("test_op", "/nonexistent_dir/aicpu");
    EXPECT_FALSE(result);
}

// ===================================================================
// Tests for TileFwkAiCpuCompile (public API)
// ===================================================================

// Covers Line 139: MACHINE_LOGE("Gen op[%s]  not success\n", ...)
// GenTilingFunc fails because the directory path cannot be created
TEST_F(TestCompileControlBin, TileFwkAiCpuCompile_GenTilingFuncFails) {
    // Use /proc as base path - cannot create subdirectories under /proc
    bool result = TileFwkAiCpuCompile("test_func", "/proc/fake_aicpu_dir");
    EXPECT_FALSE(result);
}

// Covers Line 104: MACHINE_LOGD("PreCompileCmd is %s, file is %s\n", ...)
// Covers Line 107: MACHINE_LOGE("Precompile %s fail\n", ...)
// Covers Line 145: MACHINE_LOGE("Op %s preCompile fail\n", ...)
// GenTilingFunc succeeds but TieFwkAicpuPreCompile fails (no ARM cross-compiler available)
TEST_F(TestCompileControlBin, TileFwkAiCpuCompile_PreCompileFails) {
    std::string dumpDir = TEST_TMP_DIR + "/aicpu_compile";
    // Create the required directory structure: dumpDir/funcName/aicpu/
    std::string funcName = "precompile_test";
    std::string controlAicpuPath = dumpDir + "/" + funcName + "/aicpu/";
    CreateMultiLevelDir(controlAicpuPath);

    // TileFwkAiCpuCompile will:
    // 1. GenTilingFunc writes control_flow_kernel.cpp to controlAicpuPath -> succeeds
    // 2. TieFwkAicpuPreCompile tries to compile the file with cross-compiler -> fails
    //    -> Line 104 MACHINE_LOGD is hit for each file
    //    -> Line 107 MACHINE_LOGE is hit when compilation fails
    // 3. Back in TileFwkAiCpuCompile -> Line 145 MACHINE_LOGE
    bool result = TileFwkAiCpuCompile(funcName, dumpDir);
    EXPECT_FALSE(result);
}

// ===================================================================
// Tests for TieFwkAicpuPreCompile (internal function, accessed via extern decl)
// ===================================================================

// Covers Line 104: MACHINE_LOGD("PreCompileCmd is %s, file is %s\n", ...)
// Covers Line 107: MACHINE_LOGE("Precompile %s fail\n", ...)
// Compile a .cpp file with a non-existent compiler
TEST_F(TestCompileControlBin, TieFwkAicpuPreCompile_CompileFails) {
    std::string compileDir = TEST_TMP_DIR + "/precompile_test/";
    CreateMultiLevelDir(compileDir);

    // Create a dummy .cpp file
    std::string cppFile = compileDir + "dummy.cpp";
    std::ofstream ofs(cppFile);
    ofs << "\"UT: force compile fail\"" << std::endl;
    ofs.close();

    std::string preCompileO;
    std::string dirPath = compileDir;
    // The compiler (DeviceMahineCompiler) is likely empty or non-existent in UT environment
    // system() call will fail -> line 107 MACHINE_LOGE
    bool result = TieFwkAicpuPreCompile(preCompileO, dirPath);
    EXPECT_FALSE(result);
}

// TieFwkAicpuPreCompile with empty directory (no files to compile) -> succeeds trivially
TEST_F(TestCompileControlBin, TieFwkAicpuPreCompile_NoFiles) {
    std::string emptyDir = TEST_TMP_DIR + "/empty_compile_dir/";
    CreateMultiLevelDir(emptyDir);

    std::string preCompileO;
    std::string dirPath = emptyDir;
    bool result = TieFwkAicpuPreCompile(preCompileO, dirPath);
    EXPECT_TRUE(result);
    EXPECT_TRUE(preCompileO.empty());
}

// ===================================================================
// Tests for SharedAicpuCompile (internal function, accessed via extern decl)
// ===================================================================

// Covers Line 125: MACHINE_LOGE("RUNDeviceMachine compile fail\n")
// Compiler not available -> system() fails
TEST_F(TestCompileControlBin, SharedAicpuCompile_CompileFails) {
    std::string aicpuDir = TEST_TMP_DIR + "/shared_compile_test";
    CreateMultiLevelDir(aicpuDir);

    std::string preCompile0 = "/nonexistent_ut_fake.o";
    bool result = SharedAicpuCompile("test_func", aicpuDir, preCompile0);
    EXPECT_FALSE(result);
}
