/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_sincos.cpp
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
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynSinCos : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        config::Reset();
        Program::GetInstance().Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenDynSinCos, SinLayout)
{
    MockFuncDynUnaryConf config;
    config.dtype = DT_INT16;
    auto function =
        GenMockFuncDynUnary("Sin", config, [](Tensor& input, Tensor& output) { output = Sin(input); });

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});

    std::string res = GetResultFromCpp(*function);
    std::string expect = R"(TMulS<LastUse2Dim<0, 0>, float>(ubTensor_4, ubTensor_2, 0.318309873);)";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynSinCos, CosLayout)
{
    MockFuncDynUnaryConf config;
    config.dtype = DT_INT16;
    auto function =
        GenMockFuncDynUnary("Cos", config, [](Tensor& input, Tensor& output) { output = Cos(input); });

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});

    std::string res = GetResultFromCpp(*function);
    std::string expect = R"(TMulS<LastUse2Dim<0, 0>, float>(ubTensor_4, ubTensor_2, 0.318309873);)";
    CheckStringExist(expect, res);
}

} // namespace npu::tile_fwk
