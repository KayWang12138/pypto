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
 * \file test_codegen_dyn_gather.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "common/data_type.h"
#include "codegen/codegen.h"
#include "codegen/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"

using namespace npu::tile_fwk;

class TestCodegenDynGather : public ::testing::Test {
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

constexpr const int GATHER_SHAPE0 = 16;
constexpr const int GATHER_SHAPE1 = 32;

TEST_F(TestCodegenDynGather, TestGather) {
    constexpr const int S2 = 32;
    constexpr const int D = 64;
    constexpr const int B = 1;
    constexpr const int S = 32;
    std::vector<int> shape0 = {S2, D};
    std::vector<int> shape1 = {B, S};
    int axis = 0;
    std::vector<int> shape2 = {B, S, D};

    Program::GetInstance().GetTileShape().SetVecTileShapes({1, GATHER_SHAPE0, GATHER_SHAPE1});

    Tensor inputSrc0(DT_FP32, shape0, "x");
    Tensor inputSrc1(DT_INT32, shape1, "indices");
    Tensor output(DT_FP32, shape2, "output");

    ConfigManager::Instance();
    std::string funcName = "GATHER_T";
    FUNCTION(funcName, FunctionType::STATIC, {inputSrc0, inputSrc1, output}) {
        output = Gather(inputSrc0, inputSrc1, axis);
    }
    auto function = Program::GetInstance().GetFunctionByRawName("TENSOR_" + funcName);

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
        subFunc.second->dynParamTable_.emplace("sym_26_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_26_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_27_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_27_dim_1", fakeParam);
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
