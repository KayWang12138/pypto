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
 * \file test_codegen_symbol_mgr.cpp
 * \brief Unit test for SymbolManager internal interfaces.
 */

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestSymbolManager : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetBuildStatic(true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }

    void TearDown() override {}
};

TEST_F(TestSymbolManager, TestQueryTileTensorTypeByBufVar_Basic) {
    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();

    TileTensor tileTensor;
    tileTensor.isConstant = false;
    tileTensor.magic = 100;
    tileTensor.dim = 2;
    tileTensor.dtype = DataType::DT_FP32;
    tileTensor.bufType = BufferType::BUF_UB;
    tileTensor.bufVar = "UB_S0_E16384";
    tileTensor.usingType = "UBTileTensorFP32Dim2";
    tileTensor.tensorName = "ubTile_0";
    tileTensor.shape = {"64", "64"};
    tileTensor.stride = {"64", "1"};
    tileTensor.rawShape = {64, 64};
    tileTensor.shapeInLoop.loopDepth = 0;

    symbolManager->AddTileTensor(tileTensor);

    std::string result = symbolManager->QueryTileTensorTypeByBufVar("UB_S0_E16384");

    EXPECT_EQ(result, "UBTileTensorFP32Dim2");
}

TEST_F(TestSymbolManager, TestQueryTileTensorTypeByBufVar_DDRBuffer) {
    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();

    TileTensor tileTensor;
    tileTensor.isConstant = false;
    tileTensor.magic = 200;
    tileTensor.dim = 2;
    tileTensor.dtype = DataType::DT_FP16;
    tileTensor.bufType = BufferType::BUF_DDR;
    tileTensor.bufVar = "GET_PARAM_ADDR(0)";
    tileTensor.usingType = "GMTileTensorFP16Dim2";
    tileTensor.tensorName = "gmTile_0";
    tileTensor.shape = {"128", "128"};
    tileTensor.stride = {"128", "1"};
    tileTensor.rawShape = {128, 128};
    tileTensor.shapeInLoop.loopDepth = 0;

    symbolManager->AddTileTensor(tileTensor);

    std::string result = symbolManager->QueryTileTensorTypeByBufVar("GET_PARAM_ADDR(0)");

    EXPECT_EQ(result, "GMTileTensorFP16Dim2");
}

TEST_F(TestSymbolManager, TestQueryTileTensorTypeByBufVar_MultipleTensors) {
    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();

    TileTensor tileTensor1;
    tileTensor1.isConstant = false;
    tileTensor1.magic = 100;
    tileTensor1.dim = 2;
    tileTensor1.dtype = DataType::DT_FP32;
    tileTensor1.bufType = BufferType::BUF_UB;
    tileTensor1.bufVar = "UB_S0_E16384";
    tileTensor1.usingType = "UBTileTensorFP32Dim2_0";
    tileTensor1.tensorName = "ubTile_0";
    tileTensor1.shape = {"64", "64"};
    tileTensor1.rawShape = {64, 64};
    tileTensor1.shapeInLoop.loopDepth = 0;

    TileTensor tileTensor2;
    tileTensor2.isConstant = false;
    tileTensor2.magic = 101;
    tileTensor2.dim = 2;
    tileTensor2.dtype = DataType::DT_FP32;
    tileTensor2.bufType = BufferType::BUF_UB;
    tileTensor2.bufVar = "UB_S1_E8192";
    tileTensor2.usingType = "UBTileTensorFP32Dim2_1";
    tileTensor2.tensorName = "ubTile_1";
    tileTensor2.shape = {"32", "32"};
    tileTensor2.rawShape = {32, 32};
    tileTensor2.shapeInLoop.loopDepth = 0;

    symbolManager->AddTileTensor(tileTensor1);
    symbolManager->AddTileTensor(tileTensor2);

    std::string result1 = symbolManager->QueryTileTensorTypeByBufVar("UB_S0_E16384");
    std::string result2 = symbolManager->QueryTileTensorTypeByBufVar("UB_S1_E8192");

    EXPECT_EQ(result1, "UBTileTensorFP32Dim2_0");
    EXPECT_EQ(result2, "UBTileTensorFP32Dim2_1");
}

TEST_F(TestSymbolManager, TestQueryTileTensorTypeByBufVar_DifferentDataTypes) {
    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();

    TileTensor tileTensorFP16;
    tileTensorFP16.isConstant = false;
    tileTensorFP16.magic = 100;
    tileTensorFP16.dim = 2;
    tileTensorFP16.dtype = DataType::DT_FP16;
    tileTensorFP16.bufType = BufferType::BUF_UB;
    tileTensorFP16.bufVar = "UB_FP16";
    tileTensorFP16.usingType = "UBTileTensorFP16Dim2";
    tileTensorFP16.tensorName = "ubTile_fp16";
    tileTensorFP16.shape = {"64", "64"};
    tileTensorFP16.rawShape = {64, 64};
    tileTensorFP16.shapeInLoop.loopDepth = 0;

    TileTensor tileTensorINT32;
    tileTensorINT32.isConstant = false;
    tileTensorINT32.magic = 101;
    tileTensorINT32.dim = 2;
    tileTensorINT32.dtype = DataType::DT_INT32;
    tileTensorINT32.bufType = BufferType::BUF_UB;
    tileTensorINT32.bufVar = "UB_INT32";
    tileTensorINT32.usingType = "UBTileTensorINT32Dim2";
    tileTensorINT32.tensorName = "ubTile_int32";
    tileTensorINT32.shape = {"32", "32"};
    tileTensorINT32.rawShape = {32, 32};
    tileTensorINT32.shapeInLoop.loopDepth = 0;

    symbolManager->AddTileTensor(tileTensorFP16);
    symbolManager->AddTileTensor(tileTensorINT32);

    std::string resultFP16 = symbolManager->QueryTileTensorTypeByBufVar("UB_FP16");
    std::string resultINT32 = symbolManager->QueryTileTensorTypeByBufVar("UB_INT32");

    EXPECT_EQ(resultFP16, "UBTileTensorFP16Dim2");
    EXPECT_EQ(resultINT32, "UBTileTensorINT32Dim2");
}

TEST_F(TestSymbolManager, TestQueryTileTensorNameByBufVar_Basic) {
    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();

    TileTensor tileTensor;
    tileTensor.isConstant = false;
    tileTensor.magic = 100;
    tileTensor.dim = 2;
    tileTensor.dtype = DataType::DT_FP32;
    tileTensor.bufType = BufferType::BUF_UB;
    tileTensor.bufVar = "UB_S0_E16384";
    tileTensor.usingType = "UBTileTensorFP32Dim2";
    tileTensor.tensorName = "ubTile_test";
    tileTensor.shape = {"64", "64"};
    tileTensor.rawShape = {64, 64};
    tileTensor.shapeInLoop.loopDepth = 0;

    symbolManager->AddTileTensor(tileTensor);

    std::string result = symbolManager->QueryTileTensorNameByBufVar("UB_S0_E16384");

    EXPECT_EQ(result, "ubTile_test");
}

TEST_F(TestSymbolManager, TestQueryTileTensorByBufVar_ReturnReference) {
    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();

    TileTensor tileTensor;
    tileTensor.isConstant = false;
    tileTensor.magic = 100;
    tileTensor.dim = 2;
    tileTensor.dtype = DataType::DT_FP32;
    tileTensor.bufType = BufferType::BUF_UB;
    tileTensor.bufVar = "UB_S0_E16384";
    tileTensor.usingType = "UBTileTensorFP32Dim2";
    tileTensor.tensorName = "ubTile_0";
    tileTensor.shape = {"64", "64"};
    tileTensor.rawShape = {64, 64};
    tileTensor.shapeInLoop.loopDepth = 0;

    symbolManager->AddTileTensor(tileTensor);

    const TileTensor &result = symbolManager->QueryTileTensorByBufVar("UB_S0_E16384");

    EXPECT_EQ(result.magic, 100);
    EXPECT_EQ(result.dim, 2);
    EXPECT_EQ(result.dtype, DataType::DT_FP32);
    EXPECT_EQ(result.bufType, BufferType::BUF_UB);
    EXPECT_EQ(result.bufVar, "UB_S0_E16384");
    EXPECT_EQ(result.usingType, "UBTileTensorFP32Dim2");
    EXPECT_EQ(result.tensorName, "ubTile_0");
}

} // namespace npu::tile_fwk
