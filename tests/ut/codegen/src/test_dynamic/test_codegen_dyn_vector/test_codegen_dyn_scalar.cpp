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
 * \file test_codegen_dyn_scalar.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"

namespace npu::tile_fwk {

class TestCodegenDynScalar : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenDynScalar, TestScalarAdds) {
    std::vector<int64_t> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{b * s, 35};

    TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "res");
    std::string funcName = "ScalarAddS";
    FunctionConfig funConfig(FunctionType::STATIC);
    ;
    FUNCTION(funcName, funConfig, {input, output}) {
        output = ScalarAddS(input, Element(DataType::DT_FP32, 127.0), true);
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
        DynParamInfo fakeParam = {3, 0, 0, DynParamInfoType::VALID_SHAPE, 0, SymbolicScalar()};
        subFunc.second->dynParamTable_.emplace("sym_2_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_2_dim_1", fakeParam);
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynScalar, TestScalarDivs) {
    std::vector<int64_t> vecTileShape = {128, 128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{s, b * s, 35};

    TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1], vecTileShape[2]);
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "res");
    std::string funcName = "ScalarDivS";
    FunctionConfig funConfig(FunctionType::STATIC);
    ;
    FUNCTION(funcName, funConfig, {input, output}) {
        output = ScalarDivS(input, Element(DataType::DT_FP32, 127.0), true);
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
        DynParamInfo fakeParam = {3, 0, 0, DynParamInfoType::VALID_SHAPE, 0, SymbolicScalar()};
        subFunc.second->dynParamTable_.emplace("sym_2_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_2_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_2_dim_2", fakeParam);
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
} // namespace npu::tile_fwk