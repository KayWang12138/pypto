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
 * \file test_codegen_argmax_argmin.cpp
 * \brief Unit test for ArgMax and ArgMin codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/function/function.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenArgmaxArgmin : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetBuildStatic(true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenArgmaxArgmin, TestOperationArgMaxLineTileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    int shape0 = 6;
    int shape1 = 1;
    int shape2 = 8;
    int shape3 = 1024;
    std::vector<int64_t> shape = {shape0 * shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0 * shape1, 1, shape3};
    TileShape::Current().SetVecTile({2, 8, 512});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMax3dimMoe_TILERENSOR";
    FUNCTION(funcName, {input_a, output}) {
        output = ArgMax(input_a, 1, true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    std::string res = GetResultFromCpp(*function);
    std::string expect = R"!!!(TRowArgMaxLine<3>(ubTensor_3, ubTensor_1, ubTensor_4);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenArgmaxArgmin, TestOperationArgMinLineTileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    int shape0 = 6;
    int shape1 = 1;
    int shape2 = 8;
    int shape3 = 1024;
    std::vector<int64_t> shape = {shape0 * shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0 * shape1, 1, shape3};
    TileShape::Current().SetVecTile({2, 8, 512});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMin3dimMoe_TILERENSOR";
    FUNCTION(funcName, {input_a, output}) {
        output = ArgMin(input_a, 1, true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    std::string res = GetResultFromCpp(*function);
    std::string expect = R"!!!(TRowArgMinLine<3>(ubTensor_3, ubTensor_1, ubTensor_4);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenArgmaxArgmin, TestOperationArgMaxSingleTileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    int shape0 = 64;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    TileShape::Current().SetVecTile({16, 64});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMaxSingle_TILERENSOR";
    FUNCTION(funcName, {input_a, output}) {
        output = ArgMax(input_a, -1, true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    std::string res = GetResultFromCpp(*function);
    std::string expect = R"!!!(TRowArgMaxSingle<LastUse3Dim<0, 0, 1>>(ubTensor_3, ubTensor_1, ubTensor_4);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenArgmaxArgmin, TestOperationArgMinSingleTileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    int shape0 = 64;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    TileShape::Current().SetVecTile({16, 64});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMinSingle_TILERENSOR";
    FUNCTION(funcName, {input_a, output}) {
        output = ArgMin(input_a, -1, true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    std::string res = GetResultFromCpp(*function);
    std::string expect = R"!!!(TRowArgMinSingle<LastUse3Dim<0, 0, 1>>(ubTensor_3, ubTensor_1, ubTensor_4);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenArgmaxArgmin, TestOperationArgMaxFP16TileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    int shape0 = 32;
    int shape1 = 64;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    TileShape::Current().SetVecTile({16, 32});

    Tensor input_a(DataType::DT_FP16, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMaxFP16_TILERENSOR";
    FUNCTION(funcName, {input_a, output}) {
        output = ArgMax(input_a, -1, true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenArgmaxArgmin, TestOperationArgMinFP16TileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    int shape0 = 32;
    int shape1 = 64;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    TileShape::Current().SetVecTile({16, 32});

    Tensor input_a(DataType::DT_FP16, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMinFP16_TILERENSOR";
    FUNCTION(funcName, {input_a, output}) {
        output = ArgMin(input_a, -1, true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenArgmaxArgmin, TestOperationArgMaxBF16TileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    int shape0 = 32;
    int shape1 = 64;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    TileShape::Current().SetVecTile({16, 32});

    Tensor input_a(DataType::DT_BF16, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMaxBF16_TILERENSOR";
    FUNCTION(funcName, {input_a, output}) {
        output = ArgMax(input_a, -1, true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenArgmaxArgmin, TestOperationArgMinBF16TileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    int shape0 = 32;
    int shape1 = 64;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    TileShape::Current().SetVecTile({16, 32});

    Tensor input_a(DataType::DT_BF16, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMinBF16_TILERENSOR";
    FUNCTION(funcName, {input_a, output}) {
        output = ArgMin(input_a, -1, true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

} // namespace npu::tile_fwk
