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
 * \file test_codegen_dyn_binary_brc.cpp
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

class TestCodegenDynBinaryBrc : public ::testing::Test {
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

// mul (32, 512), (32, 1)
TEST_F(TestCodegenDynBinaryBrc, TestMulDynamic) {
    config::SetOperationConfig("FORCE_COMBINE_AXIS", true);
    std::vector<int64_t> shape1 = {32, 512};
    std::vector<int64_t> shape2 = {32, 1};
    Program::GetInstance().GetTileShape().SetVecTileShapes({32, 256});
    Tensor input_a(DataType::DT_FP32, shape1, "A");
    Tensor input_b(DataType::DT_FP32, shape1, "B");
    Tensor output(DataType::DT_FP32, shape1, "C");
    ConfigManager::Instance();

    FunctionConfig funConfig = {.funcType = FunctionType::STATIC};
    FUNCTION("MUL_T", funConfig, {input_a, input_b, output}) {
        // add RowSumSingle to test brc case
        auto input_c = RowSumSingle(input_b);
        output = Mul(input_a, input_c);
    }
    auto function = Program::GetInstance().GetFunctionByRawName("TENSOR_MUL_T");
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
        subFunc.second->dynParamTable_.emplace("sym_18_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_18_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_19_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_19_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_32_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_32_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_42_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_42_dim_1", fakeParam);
    }

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
} // namespace npu::tile_fwk