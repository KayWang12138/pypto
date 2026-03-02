/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file test_function_converage.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include <iostream>
#include <memory>
#include "interface/configs/config_manager.h"
#define private public
#define protected public
#include "interface/function/function.h"
#undef private
#undef protected
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/utils/id_gen.h"

using namespace npu::tile_fwk;

class FunctionCoverageTest : public testing::Test {
public:
    static void SetUpTestCase() {
        std::cout << "FunctionCoverageTest SetUpTestCase" << std::endl;
        LoggerManager::GetManager().level = LoggerLevel::ERROR;
    }

    static void TearDownTestCase() {
        std::cout << "FunctionCoverageTest TearDownTestCase" << std::endl;
        LoggerManager::GetManager().level = LoggerLevel::ERROR;
    }

    void SetUp() override {
        std::cout << "FunctionCoverageTest SetUp" << std::endl;
        Program::GetInstance().Reset();
    }

    void TearDown() override { std::cout << "FunctionCoverageTest TearDown" << std::endl; }
};

TEST_F(FunctionCoverageTest, ConverageCase1) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("ConverageFunc1") {
        Tensor in0 = Reciprocal(input);
        Tensor in1 = Exp(in0);
        Tensor in2 = Sqrt(in1);
        Tensor in3 = Exp(in2);

        Tensor in4 = Exp(in0);
        Tensor in5 = Sqrt(in4);
        Tensor in6 = Exp(in5);
        output = Mul(in3, in6);
    }
    std::cout << "Dump program: " << Program::GetInstance().Dump() << std::endl;

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_ConverageFunc1");
    ASSERT_NE(func, nullptr);
    SubfuncInvokeInfoTy::TensorParamPackTy tensorParam;

    // GetParamIndex
    EXPECT_EQ(func->GetParamIndex(input.GetStorage()->GetRawTensor()), INVALID_IN_OUT_INDEX);
    EXPECT_EQ(func->GetParamIndex(output.GetStorage()->GetRawTensor()), INVALID_IN_OUT_INDEX);
    EXPECT_EQ(func->GetParamIndex(func->GetIncast()[0]->GetRawTensor()), INVALID_IN_OUT_INDEX);
    EXPECT_EQ(func->GetParamIndex(func->GetOutcast()[0]->GetRawTensor()), INVALID_IN_OUT_INDEX);

    std::cout << "===========TensorMagicCheck==========" << std::endl;
    func->TensorMagicCheck();
    std::cout << "===========OperationLoopCheck==========" << std::endl;
    func->OperationLoopCheck("shake it off");

    std::cout << func->DumpJson(true).dump() << std::endl;
    std::cout << func->DumpJson(false).dump() << std::endl;

    Function *currFunc = Program::GetInstance().GetCurrentFunction();
    ASSERT_NE(currFunc, nullptr);
    currFunc->ValidCheck();
}

TEST_F(FunctionCoverageTest, ConverageCase2) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    auto inputPtr = std::make_shared<uint8_t>(1);
    auto outputPtr = std::make_shared<uint8_t>(2);
    Tensor input(DT_FP32, shape, inputPtr.get(), "input");
    Tensor output(DT_FP32, shape, outputPtr.get(), "output");
    config::SetBuildStatic(true);
    FUNCTION("ConverageFunc1", {input, output}) {
        Tensor in0 = Reciprocal(input);
        Tensor in1 = Exp(in0);
        Tensor in2 = Sqrt(in1);
        Tensor in3 = Exp(in2);

        Tensor in4 = Exp(in0);
        Tensor in5 = Sqrt(in4);
        Tensor in6 = Exp(in5);
        output = Mul(in3, in6);
    }
    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_ConverageFunc1");
    ASSERT_NE(func, nullptr);

    // GetParamIndex
    EXPECT_EQ(func->GetParamIndex(input.GetStorage()->GetRawTensor()), 0);
    EXPECT_EQ(func->GetParamIndex(output.GetStorage()->GetRawTensor()), 1);
    EXPECT_EQ(func->GetParamIndex(func->GetIncast()[0]->GetRawTensor()), 0);
    EXPECT_EQ(func->GetParamIndex(func->GetOutcast()[0]->GetRawTensor()), 1);
}

TEST_F(FunctionCoverageTest, ConverageCase3) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    auto inputPtr = std::make_shared<uint8_t>(1);
    auto outputPtr = std::make_shared<uint8_t>(2);
    Tensor input(DT_FP32, shape, inputPtr.get(), "input");
    Tensor output(DT_FP32, shape, outputPtr.get(), "output");
    config::SetBuildStatic(true);
    FUNCTION("ConverageFunc") {
        Tensor in0 = Reciprocal(input);
        Tensor in1 = Exp(in0);
        Tensor in2 = Sqrt(in1);
        Tensor in3 = Exp(in2);

        Tensor in4 = Exp(in0);
        Tensor in5 = Sqrt(in4);
        Tensor in6 = Exp(in5);
        output = Mul(in3, in6);
    }
    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_ConverageFunc");
    ASSERT_NE(func, nullptr);

    // GetParamIndex
    EXPECT_EQ(func->GetParamIndex(input.GetStorage()->GetRawTensor()), -1);
    EXPECT_EQ(func->GetParamIndex(output.GetStorage()->GetRawTensor()), -1);
    EXPECT_EQ(func->GetParamIndex(func->GetIncast()[0]->GetRawTensor()), -1);
    EXPECT_EQ(func->GetParamIndex(func->GetOutcast()[0]->GetRawTensor()), -1);
}

TEST_F(FunctionCoverageTest, ConverageCase4) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{16, 64};
    std::vector<int64_t> childShape{16, 16};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c(DataType::DT_FP32, shape, "c");

    constexpr int LOOP_END = 4;
    constexpr int CHILD_SHAPE_OFFSET = 16;

    FUNCTION("main", {a, b}, {c}) {
        LOOP("D3", FunctionType::DYNAMIC_LOOP, k, LoopRange(0, LOOP_END)) {
            auto a0 = View(a, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto b0 = View(b, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto c0 = Add(a0, b0);
            Assemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
            LOOP("D4", FunctionType::DYNAMIC_LOOP, n, LoopRange(0, LOOP_END)) {
                auto d0 = View(a, childShape, {0, n * CHILD_SHAPE_OFFSET});
                auto e0 = View(b, childShape, {0, n * CHILD_SHAPE_OFFSET});
                auto f0 = Add(a0, b0);
                Assemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
            }
            auto func = Program::GetInstance().GetCurrentFunction();
            EXPECT_EQ(func->InsertLoopIdxNameList("i"), true);
            EXPECT_EQ(func->InsertLoopIdxNameList("i"), true);
            EXPECT_EQ(func->InsertLoopIdxNameList("k"), false);
            EXPECT_EQ(func->InsertLoopIdxNameList("n"), true);
        }
    }
}

TEST_F(FunctionCoverageTest, ConverageCase5) {
    config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);

    // duplicate funcname
    config::SetBuildStatic(true);
    FUNCTION("ConverageFunc") {
        IdGen<IdType::FUNCTION>::Inst().SetId(2);
        Program::GetInstance().BeginFunction(
            "TENSOR_ConverageFunc", FunctionType::DYNAMIC, GraphType::TENSOR_GRAPH, {}, false);
    }
    EXPECT_EQ(Program::GetInstance().GetFunctionMap().size(), 2);
    Program::GetInstance().GetCurrentFunction()->SetFunctionType(FunctionType::STATIC);
    auto ret = Program::GetInstance().EndFunction("TENSOR_ConverageFunc1", false);
    EXPECT_EQ(ret, std::make_tuple(nullptr, nullptr, false));
}

TEST_F(FunctionCoverageTest, TestReuseTensorCase1) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    config::SetBuildStatic(true);
    FUNCTION("R1") {
        Tensor in0 = Exp(input);
        Tensor in1 = Reciprocal(in0);
        Tensor in2 = Exp(in1);
        Tensor in3 = Exp(in2);
        Tensor in4 = Sqrt(in3);
        output = Exp(in4);
    }

    const Function *constFunc = Program::GetInstance().GetFunctionByRawName("TENSOR_R1");
    ASSERT_NE(constFunc, nullptr);
    Function *func = const_cast<Function *>(constFunc);
    ASSERT_NE(func, nullptr);
    std::cout << "=========" << std::endl;
    Operation *reOp = nullptr;
    Operation *sqrtOp = nullptr;
    for (auto &op : func->Operations()) {
        std::cout << "Op:" << op.opmagic << " " << op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor> &inputTensor : op.GetIOperands()) {
            for (const auto &itemOp : inputTensor->GetProducers()) {
                std::cout << "(" << itemOp->opmagic << ", " << itemOp->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor> &outputTensor : op.GetOOperands()) {
            for (const auto &itemOp : outputTensor->GetConsumers()) {
                std::cout << "(" << itemOp->opmagic << ", " << itemOp->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl << std::endl;
        if (op.GetOpcode() == Opcode::OP_RECIPROCAL) {
            reOp = &op;
        }
        if (op.GetOpcode() == Opcode::OP_SQRT) {
            sqrtOp = &op;
        }
    }
    std::cout << "=========" << std::endl;
    ASSERT_NE(reOp, nullptr);
    ASSERT_NE(sqrtOp, nullptr);
    LogicalTensorPtr srcTensor = reOp->GetOutputOperand(0);
    LogicalTensorPtr dstTensor = sqrtOp->GetOutputOperand(0);
    // EXPECT_EQ(func->TensorReuse(dst_tensor, src_tensor), false);
    // src_tensor->offset = {0, 0};
    // dst_tensor->offset = {0, 0};
    EXPECT_EQ(func->TensorReuse(dstTensor, srcTensor), true);
    EXPECT_EQ(srcTensor->tensor, dstTensor->tensor);
    // EXPECT_EQ(sqrt_op->GetInCtrlOperations(), re_op->GetInCtrlOperations());
    EXPECT_EQ(sqrtOp->GetInCtrlOperations().empty(), true);
}

TEST_F(FunctionCoverageTest, TestFunctionHash) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    config::SetBuildStatic(true);
    FUNCTION("R2") {
        Tensor in0 = Exp(input);
        Tensor in1 = Reciprocal(in0);
        Tensor in2 = Exp(in1);
        Tensor in3 = Exp(in2);
        Tensor in4 = Sqrt(in3);
        output = Exp(in4);
    }

    const Function *constFunc = Program::GetInstance().GetFunctionByRawName("TENSOR_R2");
    ASSERT_NE(constFunc, nullptr);
    Function *func = const_cast<Function *>(constFunc);
    ASSERT_NE(func, nullptr);
}
