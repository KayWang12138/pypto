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
 * \file test_codegen_scalar.cpp
 * \brief Unit test for codegen.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include <vector>
#include <string>
#include "codegen/cloudnpu/codegen_cloudnpu.h"

namespace npu::tile_fwk {
constexpr int DIM2 = 2;
constexpr int DIM3 = 3;
constexpr int DIM4 = 4;
constexpr int VALUE128 = 128;
constexpr float F_127 = 127.0;

class TestCodegenScalar : public ::testing::Test {
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

void TestQuant(std::vector<int64_t> &inputShape) {
    int shapeDim = inputShape.size();
    std::vector<int64_t> scaleShape(shapeDim, 0);
    for (int i = 0; i < shapeDim; i++) {
        scaleShape[i] = (i == shapeDim - 1) ? 1 : inputShape[i];
    }

    std::vector<int64_t> vecTileShape = {VALUE128, VALUE128};

    // depend on shapeDim
    switch (shapeDim) {
        case DIM2: TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]); break;
        case DIM3: TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[0], vecTileShape[1]); break;
        case DIM4: TileShape::Current().SetVecTile(1, 1, vecTileShape[0], vecTileShape[1]); break;
        default: ASSERT(true) << "unsupport dim " << shapeDim << " \n"; break;
    }

    Tensor input(DataType::DT_FP16, inputShape, "input");
    Tensor output(DataType::DT_INT8, inputShape, "output");
    Tensor scaleDeQuant(DataType::DT_FP32, scaleShape, "scaleDeQuant");

    std::string funcName = "Quant";
    FunctionConfig funConfig = {.funcType = FunctionType::STATIC};
    FUNCTION(funcName, funConfig, {input, output, scaleDeQuant}) {
        auto res = Quant(input);
        output = std::get<0>(res);
        scaleDeQuant = std::get<1>(res);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenScalar, TestQuant_32_1_7168) {
    std::vector<int64_t> inputShape = {32, 1, 7168};
    TestQuant(inputShape);
}

TEST_F(TestCodegenScalar, TestScalarOp) {
    std::vector<int64_t> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{b * s, 35};

    TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "res");
    std::string funcName = "ScalarAddS";
    FunctionConfig funConfig = {.funcType = FunctionType::STATIC};
    FUNCTION(funcName, funConfig, {input, output}) {
        auto output_a = ScalarAddS(input, Element(DataType::DT_FP32, F_127), true);
        auto output_b = ScalarSubS(output_a, Element(DataType::DT_FP32, F_127), true);
        auto output_c = ScalarMulS(output_b, Element(DataType::DT_FP32, F_127), true);
        auto output_d = ScalarDivS(output_c, Element(DataType::DT_FP32, F_127), true);
        output = ScalarMaxS(output_d, Element(DataType::DT_FP32, F_127), true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenScalar, TestPipeAll) {
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestAddParams", "TestAddParams", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    (void)copy_op1;
    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    (void)copy_op2;
    auto &add_op = currFunctionPtr->AddOperation(Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void)add_op;
    std::vector<std::shared_ptr<LogicalTensor>> input;
    std::vector<std::shared_ptr<LogicalTensor>> output;
    Operation &syncOp = currFunctionPtr->AddOperation(npu::tile_fwk::Opcode::OP_BAR_ALL, {input}, {output});
    syncOp.syncQueue_ = {PipeType::PIPE_ALL, PipeType::PIPE_ALL, CoreType::AIV, CoreType::AIV, -1};
    auto &copy_out_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void)copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*rootFuncPtr, {});
    EXPECT_TRUE(true);
}
} // namespace npu::tile_fwk
