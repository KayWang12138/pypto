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
 * \file test_codegen_allgather.cpp
 * \brief Unit test for codegen.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include <vector>
#include <string>
#include "codegen/cloudnpu/codegen_cloudnpu.h"

using namespace npu::tile_fwk;
using namespace Distributed;

class TestCodegenAllGather : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}
};

void TestAllGatherFunc() {
    const char *group = "hcom123";
    std::vector<int> shape = {16, 256};

    Program::GetInstance().GetTileShape().SetDistTileShapes({16, 1, 0}, {256, 1, 0}, {2, 1, 0});
    Program::GetInstance().GetTileShape().SpecifyStaticRankId(0);
    ALOG_INFO_F("before AllGather exec, group=%s, in shape=[%d, %d]", group, shape[0], shape[1]);

    Tensor in(DT_FP16, shape, "in");
    Tensor out1(DT_FP16, shape, "out1");
    Tensor out2(DT_FP16, shape, "out2");
    std::vector<Tensor> out;
    out.push_back(out1);
    out.push_back(out2);

    ConfigManager::Instance();
    std::string funcName = "AllGather";
    FUNCTION(funcName, FunctionType::STATIC, {in, out[0], out[1]}) {
        AllGather(in, out, group);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCloudNPU codeGen;
    codeGen.GenCode(*function, {});
}

// tensor graph
TEST_F(TestCodegenAllGather, TestAllGatherTensorGraph) {
    TestAllGatherFunc();
}

void TestAllGatherOutTensorFunc() {
    const char *group = "hcom123";
    std::vector<int> shape = {16, 256};
    std::vector<int> outShape = {32, 256};

    Program::GetInstance().GetTileShape().SetDistTileShapes({16, 1, 0}, {256, 1, 0}, {2, 1, 0});
    Program::GetInstance().GetTileShape().SpecifyStaticRankId(0);
    ALOG_INFO_F("before AllGather exec, group=%s, in shape=[%d, %d]", group, shape[0], shape[1]);
    ALOG_INFO_F("before AllGather exec, group=%s, out shape=[%d, %d]", group, outShape[0], outShape[1]);

    Tensor in(DT_FP16, shape, "in");
    Tensor out(DT_FP16, outShape, "out");

    ConfigManager::Instance();
    std::string funcName = "AllGather";
    FUNCTION(funcName, FunctionType::STATIC, {in, out}) {
        out = AllGather(in, group);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCloudNPU codeGen;
    codeGen.GenCode(*function, {});
}

// tensor graph
TEST_F(TestCodegenAllGather, TestAllGatherOutTensorTensorGraph) {
    TestAllGatherOutTensorFunc();
}

void TestAllGatherAndMatmul() {
    const char *group = "hcom123";
    int m = 32;
    int n = 32;
    int procSize = 2;

    DataType dType = DataType::DT_BF16;
    std::string funcName = "AllGatherAndMatmul";
    
    PROGRAM("TestAllGatherMatmul") {
        std::vector<int> inputShape = {m, n};
        std::vector<int> matmulShape = {n, n};
        std::vector<int> resShape = {m * procSize, n};

        Tensor in(dType, inputShape, "in");
        Tensor w(dType, matmulShape, "w");
        Tensor out = Tensor(dType, resShape, "out");
        ConfigManager::Instance();
        FUNCTION(funcName, FunctionType::STATIC, {in,  w, out}) {
            Program::GetInstance().GetTileShape().SetDistTileShapes({inputShape[0] / 2, 2, 0},
                {inputShape[1] / 2, 2, 0}, {1, procSize, 0});
            Program::GetInstance().GetTileShape().SpecifyStaticRankId(0);
            auto allGatherOut = Distributed::AllGather(in, group);
            
            Program::GetInstance().GetTileShape().SetCubeTileShapes(
                    {16, 16}, {16, 16}, {32, 32});
            out = Matrix::Matmul<false, false>(dType, allGatherOut, w); 
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCloudNPU codeGen;
    codeGen.GenCode(*function, {});
}

// tensor graph
TEST_F(TestCodegenAllGather, TestAllGatherMatmulTensorGraph) {
    TestAllGatherAndMatmul();
}
