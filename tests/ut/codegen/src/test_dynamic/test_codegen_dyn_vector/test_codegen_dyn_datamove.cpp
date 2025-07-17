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
 * \file test_codegen_dyn_datamove.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "common/data_type.h"
#include "codegen/codegen.h"
#include "codegen/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"

using namespace npu::tile_fwk;

class TestCodegenDynDataMove : public ::testing::Test {
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

TEST_F(TestCodegenDynDataMove, TestDatamoveUnalignDim3) {
    int n = 1;
    int s = 32;
    int d = 437;
    std::vector<int> shape{n, s, d};
    std::vector<int> resShape{s, n, d};

    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 32, 512);

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, resShape, "res");
    std::string funcName = "DATAMOVE";
    FUNCTION(funcName, FunctionType::STATIC, {input, output}) {
        output = Transpose(input, {0, 1});
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
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
        subFunc.second->dynParamTable_.emplace("sym_2_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_2_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_2_dim_2", fakeParam);
    }

    npu::tile_fwk::CodeGenCloudNPU codeGen;
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynDataMove, TestDatamoveUnalignDim4) {
    int b = 4;
    int n = 1;
    int s = 32;
    int d = 437;
    std::vector<int> shape{b, n, s, d};
    std::vector<int> resShape{b, s, n, d};

    Program::GetInstance().GetTileShape().SetVecTileShapes(2, 1, 32, 512);

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, resShape, "res");
    std::string funcName = "DATAMOVE";
    FUNCTION(funcName, FunctionType::STATIC, {input, output}) {
        output = Transpose(input, {1, 2});
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
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
        DynParamInfo fakeParam = {4, 0, 0, DynParamInfoType::VALID_SHAPE, 0};
        subFunc.second->dynParamTable_.emplace("sym_13_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_13_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_13_dim_2", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_13_dim_3", fakeParam);
    }

    npu::tile_fwk::CodeGenCloudNPU codeGen;
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynDataMove, TestDatamoveAlignDim4) {
    int b = 4;
    int n = 1;
    int s = 32;
    int d = 437;
    std::vector<int> shape{b, n, s, d};
    std::vector<int> resShape{b, s, n, d};

    Program::GetInstance().GetTileShape().SetVecTileShapes(2, 1, 32, 512);

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, resShape, "res");
    std::string funcName = "DATAMOVE";
    FUNCTION(funcName, FunctionType::STATIC, {input, output}) {
        output = Transpose(input, {1, 2});
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
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
    }

    npu::tile_fwk::CodeGenCloudNPU codeGen;
    codeGen.GenCode(*function, {});
}
