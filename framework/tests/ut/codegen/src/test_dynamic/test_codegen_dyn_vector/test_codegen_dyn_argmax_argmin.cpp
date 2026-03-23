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
 * \file test_codegen_dyn_argmax_argmin.cpp
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
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynArgmaxArgmin : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMaxLine) {
    int shape0 = 6;
    int shape1 = 1;
    int shape2 = 8;
    int shape3 = 1024;
    std::vector<int64_t> shape = {shape0 * shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0 * shape1, 1, shape3};
    TileShape::Current().SetVecTile({2, 8, 512});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMax3dim";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMax(input_a, 1, true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMinLine) {
    int shape0 = 6;
    int shape1 = 1;
    int shape2 = 8;
    int shape3 = 1024;
    std::vector<int64_t> shape = {shape0 * shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0 * shape1, 1, shape3};
    TileShape::Current().SetVecTile({2, 8, 512});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMin3dim";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMin(input_a, 1, true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMaxSingleTileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);

    int shape0 = 257;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};

    TileShape::Current().SetVecTile({128, 64});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMaxSingle_TILETENSOR";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMax(input_a, -1, true);
        }
    }
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input_a, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});

    std::string res = GetResultFromCpp(*function);
    const std::string expect = R"(TRowArgMaxSingle<LastUse3Dim<0, 0, 1>>(ubTensor_20, ubTensor_17, ubTensor_21);)";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMinSingleTileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);

    int shape0 = 257;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};

    TileShape::Current().SetVecTile({128, 64});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMinSingle_TILETENSOR";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMin(input_a, -1, true);
        }
    }
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input_a, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});

    std::string res = GetResultFromCpp(*function);
    const std::string expect = R"(TRowArgMinSingle<LastUse3Dim<0, 0, 1>>(ubTensor_20, ubTensor_17, ubTensor_21);)";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMaxFP16) {
    int shape0 = 64;
    int shape1 = 32;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};

    TileShape::Current().SetVecTile({16, 16});

    Tensor input_a(DataType::DT_FP16, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMaxFP16";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMax(input_a, -1, true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMinFP16) {
    int shape0 = 64;
    int shape1 = 32;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};

    TileShape::Current().SetVecTile({16, 16});

    Tensor input_a(DataType::DT_FP16, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMinFP16";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMin(input_a, -1, true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMaxBF16) {
    int shape0 = 32;
    int shape1 = 64;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};

    TileShape::Current().SetVecTile({16, 32});

    Tensor input_a(DataType::DT_BF16, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMaxBF16";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMax(input_a, -1, true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMinBF16) {
    int shape0 = 32;
    int shape1 = 64;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};

    TileShape::Current().SetVecTile({16, 32});

    Tensor input_a(DataType::DT_BF16, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMinBF16";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMin(input_a, -1, true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMaxAxis0) {
    int shape0 = 128;
    int shape1 = 64;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {1, shape1};

    TileShape::Current().SetVecTile({32, 32});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMaxAxis0";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMax(input_a, 0, true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMinAxis0) {
    int shape0 = 128;
    int shape1 = 64;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {1, shape1};

    TileShape::Current().SetVecTile({32, 32});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMinAxis0";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMin(input_a, 0, true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMaxNoKeepDim) {
    int shape0 = 64;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0};

    TileShape::Current().SetVecTile({16, 64});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMaxNoKeepDim";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMax(input_a, -1, false);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMinNoKeepDim) {
    int shape0 = 64;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0};

    TileShape::Current().SetVecTile({16, 64});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMinNoKeepDim";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMin(input_a, -1, false);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMax3D) {
    int shape0 = 8;
    int shape1 = 16;
    int shape2 = 32;
    std::vector<int64_t> shape = {shape0, shape1, shape2};
    std::vector<int64_t> outshape = {shape0, 1, shape2};

    TileShape::Current().SetVecTile({4, 8, 16});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMax3D";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMax(input_a, 1, true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynArgmaxArgmin, TestOperationArgMin3D) {
    int shape0 = 8;
    int shape1 = 16;
    int shape2 = 32;
    std::vector<int64_t> shape = {shape0, shape1, shape2};
    std::vector<int64_t> outshape = {shape0, 1, shape2};

    TileShape::Current().SetVecTile({4, 8, 16});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_INT32, outshape, "C");

    std::string funcName = "ArgMin3D";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ArgMin(input_a, 1, true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

} // namespace npu::tile_fwk
