/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_quantize.cpp
 * \brief Unit test for quantize and dequantize codegen.
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
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynQuantize : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}
};

// Symmetric quantization: FP32 -> INT8, axis=-1 (per-row)
TEST_F(TestCodegenDynQuantize, QuantizeSymmetricToInt8)
{
    int S0 = 8;
    int S1 = 128;

    std::vector<int64_t> inputShape = {S0, S1};
    std::vector<int64_t> scaleShape = {S0}; // per-row: each row has its own scale
    std::vector<int64_t> outputShape = {S0, S1};

    TileShape::Current().SetVecTile({8, 128});
    Tensor input(DataType::DT_FP32, inputShape, "input");
    Tensor scale(DataType::DT_FP32, scaleShape, "scale");
    Tensor zeroPoints; // Empty for symmetric quantization
    Tensor output(DataType::DT_INT8, outputShape, "output");

    std::string funcName = "QUANTIZE_SYMMETRIC_INT8";
    FUNCTION(funcName, {input, scale, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Quantize(input, scale, DataType::DT_INT8, -1, zeroPoints);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// Asymmetric quantization: FP32 -> UINT8, axis=-1 (per-row)
TEST_F(TestCodegenDynQuantize, QuantizeAsymmetricToUInt8)
{
    int S0 = 16;
    int S1 = 64;

    std::vector<int64_t> inputShape = {S0, S1};
    std::vector<int64_t> scaleShape = {S0}; // per-row: each row has its own scale
    std::vector<int64_t> zeroPointShape = {S0}; // per-row: each row has its own zeroPoint
    std::vector<int64_t> outputShape = {S0, S1};

    TileShape::Current().SetVecTile({16, 64});
    Tensor input(DataType::DT_FP32, inputShape, "input");
    Tensor scale(DataType::DT_FP32, scaleShape, "scale");
    Tensor zeroPoints(DataType::DT_INT32, zeroPointShape, "zeroPoints");
    Tensor output(DataType::DT_UINT8, outputShape, "output");

    std::string funcName = "QUANTIZE_ASYMMETRIC_UINT8";
    FUNCTION(funcName, {input, scale, zeroPoints, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Quantize(input, scale, DataType::DT_UINT8, -1, zeroPoints);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// Symmetric quantization with axis=-2 (per-column)
TEST_F(TestCodegenDynQuantize, QuantizeSymmetricAxisM2)
{
    int S0 = 4;
    int S1 = 256;

    std::vector<int64_t> inputShape = {S0, S1};
    std::vector<int64_t> scaleShape = {S1}; // per-column: each column has its own scale
    std::vector<int64_t> outputShape = {S0, S1};

    TileShape::Current().SetVecTile({4, 128});
    Tensor input(DataType::DT_FP32, inputShape, "input");
    Tensor scale(DataType::DT_FP32, scaleShape, "scale");
    Tensor zeroPoints; // Empty for symmetric quantization
    Tensor output(DataType::DT_INT8, outputShape, "output");

    std::string funcName = "QUANTIZE_SYMMETRIC_AXIS_M2";
    FUNCTION(funcName, {input, scale, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Quantize(input, scale, DataType::DT_INT8, -2, zeroPoints);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// Asymmetric quantization with axis=-2 (per-column)
TEST_F(TestCodegenDynQuantize, QuantizeAsymmetricAxisM2)
{
    int S0 = 4;
    int S1 = 256;

    std::vector<int64_t> inputShape = {S0, S1};
    std::vector<int64_t> scaleShape = {S1}; // per-column: each column has its own scale
    std::vector<int64_t> zeroPointShape = {S1}; // per-column: each column has its own zeroPoint
    std::vector<int64_t> outputShape = {S0, S1};

    TileShape::Current().SetVecTile({4, 128});
    Tensor input(DataType::DT_FP32, inputShape, "input");
    Tensor scale(DataType::DT_FP32, scaleShape, "scale");
    Tensor zeroPoints(DataType::DT_INT32, zeroPointShape, "zeroPoints");
    Tensor output(DataType::DT_UINT8, outputShape, "output");

    std::string funcName = "QUANTIZE_ASYMMETRIC_AXIS_M2";
    FUNCTION(funcName, {input, scale, zeroPoints, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Quantize(input, scale, DataType::DT_UINT8, -2, zeroPoints);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// Dequantize: INT8 -> FP32, axis=-1 (per-row)
TEST_F(TestCodegenDynQuantize, DequantizeInt8ToFP32)
{
    int S0 = 8;
    int S1 = 128;

    std::vector<int64_t> inputShape = {S0, S1};
    std::vector<int64_t> scaleShape = {S0}; // per-row: each row has its own scale
    std::vector<int64_t> zeroPointShape = {S0}; // per-row: each row has its own zeroPoint
    std::vector<int64_t> outputShape = {S0, S1};

    TileShape::Current().SetVecTile({8, 128});
    Tensor input(DataType::DT_INT8, inputShape, "input");
    Tensor scale(DataType::DT_FP32, scaleShape, "scale");
    Tensor zeroPoints(DataType::DT_FP32, zeroPointShape, "zeroPoints");
    Tensor output(DataType::DT_FP32, outputShape, "output");

    std::string funcName = "DEQUANTIZE_INT8_FP32";
    FUNCTION(funcName, {input, scale, zeroPoints, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Dequantize(input, scale, DataType::DT_FP32, -1, zeroPoints);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// Dequantize: INT16 -> FP32, axis=-1 (per-row)
TEST_F(TestCodegenDynQuantize, DequantizeInt16ToFP32)
{
    int S0 = 16;
    int S1 = 64;

    std::vector<int64_t> inputShape = {S0, S1};
    std::vector<int64_t> scaleShape = {S0}; // per-row: each row has its own scale
    std::vector<int64_t> zeroPointShape = {S0}; // per-row: each row has its own zeroPoint
    std::vector<int64_t> outputShape = {S0, S1};

    TileShape::Current().SetVecTile({16, 64});
    Tensor input(DataType::DT_INT16, inputShape, "input");
    Tensor scale(DataType::DT_FP32, scaleShape, "scale");
    Tensor zeroPoints(DataType::DT_FP32, zeroPointShape, "zeroPoints");
    Tensor output(DataType::DT_FP32, outputShape, "output");

    std::string funcName = "DEQUANTIZE_INT16_FP32";
    FUNCTION(funcName, {input, scale, zeroPoints, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Dequantize(input, scale, DataType::DT_FP32, -1, zeroPoints);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// Dequantize INT8 with axis=-2 (per-column)
TEST_F(TestCodegenDynQuantize, DequantizeInt8AxisM2)
{
    int S0 = 4;
    int S1 = 256;

    std::vector<int64_t> inputShape = {S0, S1};
    std::vector<int64_t> scaleShape = {S1}; // per-column: each column has its own scale
    std::vector<int64_t> zeroPointShape = {S1}; // per-column: each column has its own zeroPoint
    std::vector<int64_t> outputShape = {S0, S1};

    TileShape::Current().SetVecTile({4, 128});
    Tensor input(DataType::DT_INT8, inputShape, "input");
    Tensor scale(DataType::DT_FP32, scaleShape, "scale");
    Tensor zeroPoints(DataType::DT_FP32, zeroPointShape, "zeroPoints");
    Tensor output(DataType::DT_FP32, outputShape, "output");

    std::string funcName = "DEQUANTIZE_INT8_AXIS_M2";
    FUNCTION(funcName, {input, scale, zeroPoints, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Dequantize(input, scale, DataType::DT_FP32, -2, zeroPoints);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// Dequantize INT16 with axis=-2 (per-column)
TEST_F(TestCodegenDynQuantize, DequantizeInt16AxisM2)
{
    int S0 = 4;
    int S1 = 256;

    std::vector<int64_t> inputShape = {S0, S1};
    std::vector<int64_t> scaleShape = {S1}; // per-column: each column has its own scale
    std::vector<int64_t> zeroPointShape = {S1}; // per-column: each column has its own zeroPoint
    std::vector<int64_t> outputShape = {S0, S1};

    TileShape::Current().SetVecTile({4, 128});
    Tensor input(DataType::DT_INT16, inputShape, "input");
    Tensor scale(DataType::DT_FP32, scaleShape, "scale");
    Tensor zeroPoints(DataType::DT_FP32, zeroPointShape, "zeroPoints");
    Tensor output(DataType::DT_FP32, outputShape, "output");

    std::string funcName = "DEQUANTIZE_INT16_AXIS_M2";
    FUNCTION(funcName, {input, scale, zeroPoints, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Dequantize(input, scale, DataType::DT_FP32, -2, zeroPoints);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// Quantize-Dequantize chain (symmetric, axis=-1 per-row)
TEST_F(TestCodegenDynQuantize, QuantizeDequantizeSymmetricChain)
{
    int S0 = 8;
    int S1 = 128;

    std::vector<int64_t> inputShape = {S0, S1};
    std::vector<int64_t> scaleShape = {S0}; // per-row: each row has its own scale
    std::vector<int64_t> int8Shape = {S0, S1};
    std::vector<int64_t> outputShape = {S0, S1};

    TileShape::Current().SetVecTile({8, 128});
    Tensor input(DataType::DT_FP32, inputShape, "input");
    Tensor scale(DataType::DT_FP32, scaleShape, "scale");
    Tensor zeroPointsQ; // Empty for symmetric quantization
    Tensor zeroPointsD(DataType::DT_FP32, scaleShape, "zeroPointsD");
    Tensor int8Tensor(DataType::DT_INT8, int8Shape, "int8Tensor");
    Tensor output(DataType::DT_FP32, outputShape, "output");

    std::string funcName = "QUANTIZE_DEQUANTIZE_SYMMETRIC";
    FUNCTION(funcName, {input, scale, zeroPointsD, int8Tensor, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            int8Tensor = Quantize(input, scale, DataType::DT_INT8, -1, zeroPointsQ);
            output = Dequantize(int8Tensor, scale, DataType::DT_FP32, -1, zeroPointsD);
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
