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
 * \file test_expand_function.cpp
 * \brief Unit test for ExpandFunction pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "ir/value.h"
#include "ir/opcode.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "passes/pass_mgr/pass_manager.h"

#define private public
#include "passes/block_graph_pass/loopaxes_proc.h"

namespace npu {
namespace tile_fwk {
static const int kKeepOut = -1;
static const int kNum0 = 0;
static const int kNum1 = 1;
static const int kNum2 = 2;
static const int kNum4 = 4;
static const int kNum16 = 2;
static const std::vector<int64_t> expectedLoopAxis1 = {kNum2, kNum2};
static const std::vector<int64_t> expectedLoopAxis2 = {kNum4};

class TestLoopaxesProcPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}
    void TearDown() override {}
};

std::shared_ptr<ProgramModule> MakeTestBlockIr() {
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder(module);
    IRBuilderContext ctx;
    // ===== Function signature =====
    FunctionSignature sig;

    // tensor<[32, 16, 64, 128], f32>
    std::vector<ScalarValuePtr> tensorShape1 = {std::make_shared<ScalarValue>(int64_t(32)),
                                                std::make_shared<ScalarValue>(int64_t(16)),
                                                std::make_shared<ScalarValue>(int64_t(64)),
                                                std::make_shared<ScalarValue>(int64_t(128))};
    // tensor<[32, 128, 64], f32>
    std::vector<ScalarValuePtr> tensorShape2 = {std::make_shared<ScalarValue>(int64_t(32)),
                                                std::make_shared<ScalarValue>(int64_t(128)),
                                                std::make_shared<ScalarValue>(int64_t(64))};
    // tensor<[128, 64], f32>
    std::vector<ScalarValuePtr> tensorShape3 = {std::make_shared<ScalarValue>(int64_t(128)),
                                                std::make_shared<ScalarValue>(int64_t(64))};
    auto inputTensor = std::make_shared<TensorValue>(tensorShape1, DataType::FP32, "incast");
    auto outputTensor = std::make_shared<TensorValue>(tensorShape3, DataType::FP32, "output");
    
    sig.arguments = {inputTensor, outputTensor};

    // ===== Function =====
    auto func = builder.CreateFunction("test_loopaxes", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    builder.EnterFunctionBody(ctx, func);

    // ubTensor1 = view(input)
    auto ubTensor1 = builder.CreateTensor(ctx, tensorShape1, DataType::FP32, "ubTensor1");
    auto op1 = builder.CreateUnaryOp(Opcode::OP_VIEW, inputTensor, ubTensor1);
    builder.Emit(ctx, op1);

    // ubTensor2 = exp(ubTensor1)
    auto ubTensor2 = builder.CreateTensor(ctx, tensorShape1, DataType::FP32, "ubTensor2");
    auto op2 = builder.CreateUnaryOp(Opcode::OP_EXP, ubTensor1, ubTensor2);
    builder.Emit(ctx, op2);

    // ubTensor3 = reciprocal(ubTensor2)
    auto ubTensor3 = builder.CreateTensor(ctx, tensorShape1, DataType::FP32, "ubTensor3");
    auto op4 = builder.CreateUnaryOp(Opcode::OP_RECIPROCAL, ubTensor2, ubTensor3);
    builder.Emit(ctx, op4);

    // ubTensor4 = nop(ubTensor3)
    auto ubTensor4 = builder.CreateTensor(ctx, tensorShape2, DataType::FP32, "ubTensor4");
    auto op3 = builder.CreateUnaryOp(Opcode::OP_NOP, ubTensor3, ubTensor4);
    builder.Emit(ctx, op3);

    // ubTensor5 = exp(ubTensor4)
    auto ubTensor5 = builder.CreateTensor(ctx, tensorShape2, DataType::FP32, "ubTensor5");
    auto op5 = builder.CreateUnaryOp(Opcode::OP_EXP, ubTensor4, ubTensor5);
    builder.Emit(ctx, op5);

    // output = nop(ubTensor5)
    auto op6 = builder.CreateUnaryOp(Opcode::OP_NOP, ubTensor5, outputTensor);
    builder.Emit(ctx, op5);

    builder.CreateReturn(ctx, { });
    ctx.PopScope();

 	std::cout << *module << std::endl;
    return module;
}

TEST_F(TestLoopaxesProcPass, LoopaxesProcUTest1) {
    auto module = MakeTestBlockIR();
}
}
}