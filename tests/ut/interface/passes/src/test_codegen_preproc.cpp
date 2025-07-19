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
 * \file test_codegen_preproc.cpp
 * \brief Unit test for codegen_preproc pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/execute_graph_pass/codegen_preproc.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class CodegenPreprocTest : public testing::Test {
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

TEST_F(CodegenPreprocTest, TestSaveGmTensorParamIdxInCall) {
    const std::vector<int> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    Program::GetInstance().GetTileShape().SetVecTileShapes(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    FUNCTION("ADD", FunctionType::STATIC, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    std::string jsonFilePath = "./config/pass/json/codegen_preproc_save_gm_tensor_param_idx_in_call.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);

    // Call the pass
    auto currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD");
    currentFunction->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    currentFunction->SetUnderDynamicFunction(true);
    int index{0};
    for (auto &op : currentFunction->rootFunc_->programs_[0]->Operations()) {
        if (OpcodeManager::Inst().IsCopyInOrOut(op.GetOpcode())) {
            if (IsCopyIn(op.GetOpcode()))
                op.SetIOpAttrOffset(0, index++);
            else
                op.SetOOpAttrOffset(0, index++);
        }
    }

    CodegenPreprocPass codegenPreprocPass;
    codegenPreprocPass.RunOnFunction(*currentFunction);

    // auto currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_AddFunction");
    for (const auto &op : currentFunction->rootFunc_->programs_[0]->Operations()) {
        if (OpcodeManager::Inst().IsCopyInOrOut(op.GetOpcode())) {
            ASSERT_TRUE(op.HasAttr("GmTensorParamIdxInCallFunc"));
        }
    }
}