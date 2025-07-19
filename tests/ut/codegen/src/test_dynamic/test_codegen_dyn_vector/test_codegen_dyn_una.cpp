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
 * \file test_codegen_dyn_una.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "common/data_type.h"
#include "codegen/codegen.h"
#include "codegen/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"

using namespace npu::tile_fwk;

class TestCodegenDynUna : public ::testing::Test {
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

TEST_F(TestCodegenDynUna, TestAbsDynamic) {
    int S0 = 8;
    int S1 = 4608;
    int D0 = 8;
    int D1 = 4608;

    std::vector<int> srcShape = {S0, S1};
    std::vector<int> dstShape = {D0, D1};

    Program::GetInstance().GetTileShape().SetVecTileShapes({8, 128});
    Tensor input_a(DataType::DT_FP16, srcShape, "A");
    Tensor output(DataType::DT_FP16, dstShape, "C");

    FUNCTION("ABS_T", FunctionType::STATIC, {input_a, output}) {
        output = Abs(input_a);
    }
    auto function = Program::GetInstance().GetFunctionByRawName("TENSOR_ABS_T");
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);
    ConfigManager::Instance().SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    for (auto &subFunc : function->rootFunc_->programs_) {
        for (auto &op : subFunc.second->Operations()) {
            if (OpcodeManager::Inst().IsCopyIn(op.GetOpcode()) || OpcodeManager::Inst().IsCopyOut(op.GetOpcode())) {
                if (IsCopyIn(op.GetOpcode()))
                    op.SetIOpAttrOffset(0, 0);
                else
                    op.SetOOpAttrOffset(0, 0);
                op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
            }
        }
        DynParamInfo fakeParam = {3, 0, 0, DynParamInfoType::VALID_SHAPE, 0};
        subFunc.second->dynParamTable_.emplace("sym_113_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_113_dim_1", fakeParam);
    }

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
