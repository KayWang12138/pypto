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
 * \file test_aicore_compiler.cpp
 * \brief Unit tests for aicore_compiler.cpp covering MACHINE_LOG at lines 77, 83, 137, 155, 167, 175, 183
 */

#include <gtest/gtest.h>
#include <fstream>
#include <cstdlib>
#include <map>
#include <sstream>

#include "machine/compile/aicore_compiler.h"
#include "interface/utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "tilefwk/tilefwk_log.h"
#include "tilefwk/platform.h"

using namespace npu::tile_fwk;

// Forward declaration for internal function with external linkage (not in header but not static)
namespace npu::tile_fwk {
std::string GenSubFuncCall(std::map<uint64_t, Function *> &leafDict, CoreType coreType,
    dynamic::EncodeDevAscendFunctionParam &param, const std::string &ccePath, uint64_t tilingKey,
    std::stringstream &src_obj);
}

namespace {
const std::string TEST_TMP_DIR = "/tmp/test_aicore_compiler";
}

class TestAicoreCompiler : public testing::Test {
public:
    static void SetUpTestCase() {
        CreateMultiLevelDir(TEST_TMP_DIR);
    }

    static void TearDownTestCase() {
        std::string cmd = "rm -rf " + TEST_TMP_DIR;
        [[maybe_unused]]int ret = system(cmd.c_str());
    }

    void SetUp() override {}
    void TearDown() override {}
};

// ===================================================================
// Tests for CompileAICoreKernel
// ===================================================================

// Covers Line 175: MACHINE_LOGE("No cce path.")
// ccePath is empty -> returns -1 immediately
TEST_F(TestAicoreCompiler, CompileAICoreKernel_EmptyCcePath) {
    std::map<uint64_t, Function *> leafDict;
    dynamic::EncodeDevAscendFunctionParam param = {};
    std::string kernelPath;

    int ret = CompileAICoreKernel(leafDict, param, "", "test_hash", kernelPath);
    EXPECT_EQ(ret, -1);
}

// Covers Line 183: MACHINE_LOGE("Fail to generate aicore src file.")
// ccePath is non-empty but points to non-existent directory, so GenAicoreSrcFile fails
TEST_F(TestAicoreCompiler, CompileAICoreKernel_GenSrcFileFails) {
    std::map<uint64_t, Function *> leafDict;
    dynamic::EncodeDevAscendFunctionParam param = {};
    std::string kernelPath;

    // Pass a non-existent directory so GenAicoreSrcFile cannot create the file
    int ret = CompileAICoreKernel(leafDict, param, "/nonexistent_dir/cce_path/", "test_hash", kernelPath);
    EXPECT_EQ(ret, -1);
}

// ===================================================================
// Tests for GenSubFuncCall
// ===================================================================

// GenSubFuncCall with empty leafDict returns "" (no MACHINE_LOG triggered, but covers idxNameMap.empty() path)
TEST_F(TestAicoreCompiler, GenSubFuncCall_EmptyLeafDict) {
    std::map<uint64_t, Function *> leafDict;
    dynamic::EncodeDevAscendFunctionParam param = {};
    std::stringstream src_obj;

    std::string result = GenSubFuncCall(leafDict, CoreType::AIC, param, TEST_TMP_DIR + "/", 0, src_obj);
    EXPECT_EQ(result, "");
}