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
 * \file test_codegen_op_cloudnpu.cpp
 * \brief Unit test for CodeGenOpCloudNPU internal interfaces.
 */

#include <string>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodeGenOpCloudNPU : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetBuildStatic(true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}
};

TEST_F(TestCodeGenOpCloudNPU, TestQueryTileTensorTypeByIdx_Basic) {
    const std::vector<int64_t> shape = {64, 64};
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "TestQueryTypeByIdx";
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();

    TileTensor tileTensor;
    tileTensor.isConstant = false;
    tileTensor.magic = 1001;
    tileTensor.dim = 2;
    tileTensor.dtype = DataType::DT_FP32;
    tileTensor.bufType = BufferType::BUF_UB;
    tileTensor.bufVar = "UB_S0_E16384";
    tileTensor.usingType = "UBTileTensorFP32Dim2_Test";
    tileTensor.tensorName = "ubTile_test";
    tileTensor.shape = {"64", "64"};
    tileTensor.rawShape = {64, 64};
    tileTensor.shapeInLoop.loopDepth = 0;

    symbolManager->AddTileTensor(tileTensor);

    auto ubTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    ubTensor->SetMagic(1001);
    function->GetTensorMap().inverseMap_[ubTensor->GetMagic()] = ubTensor;

    Operation &op = function->AddOperation(npu::tile_fwk::Opcode::OP_ADD, {ubTensor}, {ubTensor});

    CodeGenCtx ctx;
    CodeGenOpCloudNPUCtx opCtx(symbolManager, *function, *(function->rootFunc_->programs_[0]), op);
    CodeGenOpCloudNPU cop(opCtx);

    std::string result = cop.QueryTileTensorTypeByIdx(0);

    EXPECT_EQ(result, "UBTileTensorFP32Dim2_Test");
}

TEST_F(TestCodeGenOpCloudNPU, TestQueryTileTensorNameByIdx_Basic) {
    const std::vector<int64_t> shape = {32, 32};
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "TestQueryNameByIdx";
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();

    TileTensor tileTensor;
    tileTensor.isConstant = false;
    tileTensor.magic = 2001;
    tileTensor.dim = 2;
    tileTensor.dtype = DataType::DT_FP16;
    tileTensor.bufType = BufferType::BUF_UB;
    tileTensor.bufVar = "UB_S1_E8192";
    tileTensor.usingType = "UBTileTensorFP16Dim2";
    tileTensor.tensorName = "ubTile_fp16_test";
    tileTensor.shape = {"32", "32"};
    tileTensor.rawShape = {32, 32};
    tileTensor.shapeInLoop.loopDepth = 0;

    symbolManager->AddTileTensor(tileTensor);

    auto ubTensor = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_UB, shape});
    ubTensor->SetMagic(2001);
    function->GetTensorMap().inverseMap_[ubTensor->GetMagic()] = ubTensor;

    Operation &op = function->AddOperation(npu::tile_fwk::Opcode::OP_ADD, {ubTensor}, {ubTensor});

    CodeGenCtx ctx;
    CodeGenOpCloudNPUCtx opCtx(symbolManager, *function, *(function->rootFunc_->programs_[0]), op);
    CodeGenOpCloudNPU cop(opCtx);

    cop.operandWithMagic[0] = 2001;
    cop.rawShape[0] = {32, 32};

    std::string result = cop.QueryTileTensorNameByIdx(0);

    EXPECT_EQ(result, "ubTile_fp16_test");
}

TEST_F(TestCodeGenOpCloudNPU, TestQueryTileTensorTypeByIdx_DDRBuffer) {
    const std::vector<int64_t> shape = {128, 128};
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "TestQueryTypeByIdxDDR";
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();

    TileTensor tileTensor;
    tileTensor.isConstant = false;
    tileTensor.magic = 3001;
    tileTensor.dim = 2;
    tileTensor.dtype = DataType::DT_FP32;
    tileTensor.bufType = BufferType::BUF_DDR;
    tileTensor.bufVar = "GET_PARAM_ADDR(0)";
    tileTensor.usingType = "GMTileTensorFP32Dim2_DDR";
    tileTensor.tensorName = "gmTile_ddr_test";
    tileTensor.shape = {"128", "128"};
    tileTensor.rawShape = {128, 128};
    tileTensor.shapeInLoop.loopDepth = 0;

    symbolManager->AddTileTensor(tileTensor);

    std::vector<SymbolicScalar> dynValidShape = {128, 128};
    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, dynValidShape});
    function->GetTensorMap().inverseMap_[ddrTensor->GetMagic()] = ddrTensor;

    Operation &op = function->AddOperation(npu::tile_fwk::Opcode::OP_ADD, {ddrTensor}, {ddrTensor});

    CodeGenCtx ctx;
    CodeGenOpCloudNPUCtx opCtx(symbolManager, *function, *(function->rootFunc_->programs_[0]), op);
    CodeGenOpCloudNPU cop(opCtx);

    std::string result = cop.QueryTileTensorTypeByIdx(0);

    EXPECT_EQ(result, "GMTileTensorFP32Dim2_DDR");
}

TEST_F(TestCodeGenOpCloudNPU, TestQueryTileTensorByIdx_MultipleOperands) {
    const std::vector<int64_t> shape = {64, 64};
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "TestQueryByIdxMultiple";
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();

    TileTensor tileTensor0;
    tileTensor0.isConstant = false;
    tileTensor0.magic = 4001;
    tileTensor0.dim = 2;
    tileTensor0.dtype = DataType::DT_FP32;
    tileTensor0.bufType = BufferType::BUF_UB;
    tileTensor0.bufVar = "UB_S0";
    tileTensor0.usingType = "UBTileTensorFP32Dim2_0";
    tileTensor0.tensorName = "ubTile_0_multi";
    tileTensor0.shape = {"64", "64"};
    tileTensor0.rawShape = {64, 64};
    tileTensor0.shapeInLoop.loopDepth = 0;

    TileTensor tileTensor1;
    tileTensor1.isConstant = false;
    tileTensor1.magic = 4002;
    tileTensor1.dim = 2;
    tileTensor1.dtype = DataType::DT_FP32;
    tileTensor1.bufType = BufferType::BUF_UB;
    tileTensor1.bufVar = "UB_S1";
    tileTensor1.usingType = "UBTileTensorFP32Dim2_1";
    tileTensor1.tensorName = "ubTile_1_multi";
    tileTensor1.shape = {"64", "64"};
    tileTensor1.rawShape = {64, 64};
    tileTensor1.shapeInLoop.loopDepth = 0;

    symbolManager->AddTileTensor(tileTensor0);
    symbolManager->AddTileTensor(tileTensor1);

    auto ubTensor0 = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    ubTensor0->SetMagic(4001);
    function->GetTensorMap().inverseMap_[ubTensor0->GetMagic()] = ubTensor0;

    auto ubTensor1 = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    ubTensor1->SetMagic(4002);
    function->GetTensorMap().inverseMap_[ubTensor1->GetMagic()] = ubTensor1;

    Operation &op = function->AddOperation(npu::tile_fwk::Opcode::OP_ADD, {ubTensor0, ubTensor1}, {ubTensor0});

    CodeGenCtx ctx;
    CodeGenOpCloudNPUCtx opCtx(symbolManager, *function, *(function->rootFunc_->programs_[0]), op);
    CodeGenOpCloudNPU cop(opCtx);

    std::string result0 = cop.QueryTileTensorTypeByIdx(0);
    std::string result1 = cop.QueryTileTensorTypeByIdx(1);

    EXPECT_EQ(result0, "UBTileTensorFP32Dim2_0");
    EXPECT_EQ(result1, "UBTileTensorFP32Dim2_1");
}

} // namespace npu::tile_fwk
