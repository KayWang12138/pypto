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
 * \file test_ir_builder.cpp
 * \brief
 */

#include <cstdint>
#include <memory>
#include <unordered_set>
#include "gtest/gtest.h"


#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "ir/opcode.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/value.h"
#include "ir/operation_base.h"
#include "ir/transform/ir_printer.h"

namespace pto{

TEST(IRTEST, TestBuilder) {
    // ===== Module =====
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // ===== Signature =====
    FunctionSignature sig;

    // tensor<[b, 128], fp32>
    std::vector<int64_t> tileShape = { 128, 128 };

    auto inputTensor  = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto scale1       = std::make_shared<ScalarValue>(DataType::FP32, "scale1", ScalarValueKind::Symbolic);

    auto result = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");

    sig.arguments = { inputTensor, scale1, result };

    // ===== Function =====
    auto func = builder.CreateFunction("test_value", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    // enter func scope + create an initial block as insertion point
    builder.EnterFunctionBody(ctx, func);

    // mul1_res = mul(input, scale1)
    auto mulVal1 = builder.CreateTile(ctx, tileShape, DataType::FP32, "mul1_res");
    auto mulOp1 = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTensor, scale1, mulVal1);
    builder.Emit(ctx, mulOp1);

    auto pi = builder.CreateConst(ctx, 3.14, "const_pi");

    // mul2_res = mul(mul1_res, pi)
    auto mulVal2 = builder.CreateTile(ctx, tileShape, DataType::FP32, "output");
    auto mulOp2 = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, mulVal1, pi, mulVal2);
    builder.Emit(ctx, mulOp2);

    builder.CreateReturn(ctx, { });

    ASSERT_EQ(ctx.func, func);
    ASSERT_EQ(ctx.compound, func->GetCompound());
    ASSERT_EQ(ctx.activeOpStmt, func->GetCompound()->GetStatement(0));

    ctx.PopScope();


    ASSERT_EQ(ctx.func, nullptr);
    ASSERT_EQ(ctx.compound, nullptr);
    ASSERT_EQ(ctx.activeOpStmt, nullptr);

    // ===== Program attributes =====
    module->Attributes()["arch"] = "\"PTOv2\"";
    module->Attributes()["tile_default"] = "{ M=16, N=16, K=16 }";
    module->Attributes()["enable_debug"] = "true";

    std::stringstream ss;
    IRPrinter printer(ss);
    printer.VisitProgram(module);
    std::cout << ss.str() << std::endl;
}

TEST(IRTEST, TestControlFlow) {
    // ===== Module =====
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // ===== Signature =====
    FunctionSignature sig;

    // tensor<[b, 128], fp32>
    auto batch = std::make_shared<ScalarValue>(DataType::INT32, "batch", ScalarValueKind::Symbolic);
    auto constant128 = std::make_shared<ScalarValue>(int64_t(128), "const_128");
    std::vector<ScalarValuePtr> tensorShape = { batch, constant128 };

    std::vector<int64_t> tileShape = { 128, 128 };

    auto inputX = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "inputX");
    auto inputY = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "inputY");
    auto scale1 = std::make_shared<ScalarValue>(DataType::FP32, "scale1", ScalarValueKind::Symbolic);
    auto scale2 = std::make_shared<ScalarValue>(DataType::FP32, "scale2", ScalarValueKind::Symbolic);

    auto resultX = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "outputX");
    auto resultY = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "outputY");

    sig.arguments = { inputX, inputY, scale1, scale2, resultX, resultY };

    sig.results.push_back(std::make_shared<ScalarValue>(DataType::INT32));

    // ===== Function =====
    auto func = builder.CreateFunction("test_control", FunctionKind::ControlFlow, sig, /*setAsEntry=*/false);
    module->SetProgramEntry(func);
        // 进入函数体作用域
    builder.EnterFunctionBody(ctx, func);

    // for i = 0 to batch step 1
    auto i = builder.CreateScalar(ctx, DataType::INT32, "i");
    auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
    auto constant1 = builder.CreateConst(ctx, int64_t(1), "const_1");
    auto fs = builder.CreateForStmt(ctx, i, constant0, batch, constant1);

    // test for attribute
    fs->Attributes()["unroll"] = "4";

    builder.EnterForBody(ctx, fs);

    // 目前没有 view / assemble，直接在整张 tensor 上做计算
    // outputX = add(inputX, scale1)
    auto resLoopX = builder.CreateTile(ctx, tileShape, DataType::FP32, "outputX");
    auto addOpX = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, resLoopX, scale1, resLoopX);
    builder.Emit(ctx, addOpX);

    // outputY = add(inputY, scale2)
    auto resLoopY = builder.CreateTile(ctx, tileShape, DataType::FP32, "outputY");
    auto addOpY = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, resLoopY, scale2, resLoopY);
    builder.Emit(ctx, addOpY);

    // if i then outputX = mul(outputX, scale1) else outputY = mul(outputY, scale2)
    auto ifs = builder.CreateIfStmt(ctx, i);

    builder.EnterIfThen(ctx, ifs);

    auto resIfX = builder.CreateTile(ctx, tileShape, DataType::FP32, "outputX");
    auto mulOpX = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, resLoopX, scale1, resIfX);
    builder.Emit(ctx, mulOpX);

    // test compound remove value
    ifs->GetThenCompound()->RemoveValue(resIfX);
    ASSERT_EQ(ifs->GetThenCompound()->FindValue("outputX"), resLoopX);
    ifs->GetThenCompound()->SetEnvVar("outputX", resIfX);

    ctx.PopScope();


    builder.EnterIfElse(ctx, ifs);

    auto resIfY = builder.CreateTile(ctx, tileShape, DataType::FP32, "outputY");
    auto mulOpY = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, resLoopY, scale2, resIfY);
    builder.Emit(ctx, mulOpY);

    ctx.PopScope();

    builder.ExitIfStatement(ctx, ifs);

    // check if then and else yield
    auto thenYield = std::dynamic_pointer_cast<YieldStatement>(ifs->GetThenCompound()->GetStatement(ifs->GetThenCompound()->GetStatementsNum() - 1));
    std::unordered_set<ValuePtr> thenYieldSet(thenYield->Values().begin(), thenYield->Values().end());
    std::unordered_set<ValuePtr> thenYieldSetGolden{resIfX, resLoopY};
    ASSERT_EQ(thenYieldSet, thenYieldSetGolden);

    auto elseYield = std::dynamic_pointer_cast<YieldStatement>(ifs->GetElseCompound()->GetStatement(ifs->GetElseCompound()->GetStatementsNum() - 1));
    std::unordered_set<ValuePtr> elseYieldSet(elseYield->Values().begin(), elseYield->Values().end());
    std::unordered_set<ValuePtr> elseYieldSetGolden{resLoopX, resIfY};
    ASSERT_EQ(elseYieldSet, elseYieldSetGolden);
    ASSERT_NE(elseYield, nullptr);
    ASSERT_GE(elseYield->Values().size(), 2);
    ASSERT_EQ(elseYield->Values()[0], resIfY);
    ASSERT_EQ(elseYield->Values()[1], resLoopX);

    ctx.PopScope(); // for-body

    builder.ExitForStatement(ctx, fs);

    // check for yield of for-statement: for 的结果应等于 if 的结果
    auto ifsInFor = std::dynamic_pointer_cast<IfStatement>(fs->GetCompound()->GetStatement(1));
    auto ifResults = ifsInFor->Results();
    auto forYields = fs->Yield()->Values();
    std::unordered_set<ValuePtr> ifResultSet(ifResults.begin(), ifResults.end());
    std::unordered_set<ValuePtr> forYieldSet(forYields.begin(), forYields.end());
    ASSERT_EQ(ifResultSet, forYieldSet);

    // return
    builder.CreateReturn(ctx, {constant0});

    ctx.PopScope(); // function-body

    std::stringstream ss;
    IRPrinter printer(ss);
    printer.VisitProgram(module);
    std::cout << ss.str() << std::endl;
}

TEST(IRTEST, TestStaticForLoop) {
    // ===== Module =====
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // ===== Signature =====
    FunctionSignature sig;

    // func(in1: Tile, in2: Scalar, out: Tile)
    // Tile shape: [16, 32]
    std::vector<size_t> tileShape = {16, 32};
    auto in1 = std::make_shared<TileValue>(tileShape, DataType::FP32, "in1");
    auto in2 = std::make_shared<ScalarValue>(DataType::FP32, "in2", ScalarValueKind::Symbolic);
    auto out = std::make_shared<TileValue>(tileShape, DataType::FP32, "out");

    sig.arguments = { in1, in2, out };
    sig.results.push_back(std::make_shared<ScalarValue>(DataType::INT32));

    // ===== Function =====
    auto func = builder.CreateFunction("test_static_for", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);
    builder.EnterFunctionBody(ctx, func);

    // buf = Tile() - create a new tile value for buf
    // buf = in1 - initialize buf with in1
    // We'll create buf with the same name, and use in1 as its initial value
    // The loop will make buf a loop-carried variable
    auto buf = builder.CreateTile(ctx, tileShape, DataType::FP32, "buf");
    
    // Initialize buf with in1: buf = in1
    // We'll use a simple operation to copy in1 to buf
    // Since there's no direct copy op, we'll use ADD with zero or MUL with 1
    // For simplicity, we'll use ADD: buf = in1 + 0
    auto zero = builder.CreateConst(ctx, 0.0, "zero");
    auto initBufOp = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, in1, zero, buf);
    builder.Emit(ctx, initBufOp);
    
    // Reset activeOpStmt to ensure the next operation creates a new OpStatement
    ctx.ResetInsertionPoint();

    // Static for loop: for i = 0 to 5 step 1 (all constants)
    auto i = builder.CreateScalar(ctx, DataType::INT32, "i");
    auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
    auto constantEnd = builder.CreateConst(ctx, int64_t(5), "const_5");
    auto constantStep = builder.CreateConst(ctx, int64_t(1), "const_1");
    
    // Create static for statement with all constant bounds
    auto staticFor = builder.CreateForStmt(ctx, i, constant0, constantEnd, constantStep);
    
    builder.EnterForBody(ctx, staticFor);

    // Inside static loop:
    // tmp1 = Mul(buf, in2)
    auto tmp1 = builder.CreateTile(ctx, tileShape, DataType::FP32, "tmp1");
    auto mulOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, buf, in2, tmp1);
    builder.Emit(ctx, mulOp);

    // tmp2 = Add(tmp1, in2)
    auto tmp2 = builder.CreateTile(ctx, tileShape, DataType::FP32, "tmp2");
    auto addOp = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, tmp1, in2, tmp2);
    builder.Emit(ctx, addOp);

    // buf = Exp(tmp2)
    // Use 'buf' name so it will be recognized as loop-carried variable
    auto bufUpdated = builder.CreateTile(ctx, tileShape, DataType::FP32, buf->GetName());
    auto expOp = builder.CreateUnaryOp(Opcode::OP_EXP, tmp2, bufUpdated);
    builder.Emit(ctx, expOp);

    ctx.PopScope(); // for-body
    builder.ExitForStatement(ctx, staticFor);

    // out = buf
    // After the loop, use the loop result (which is stored in Results())
    // ExitForStatement updates the environment, so buf should now point to the loop result
    // But to be safe, we can also use staticFor->Results()[0] if available
    ValuePtr finalBuf = buf;
    if (staticFor->Results().size() > 0) {
        finalBuf = staticFor->Results()[0];
    }
    // Reset activeOpStmt to ensure this operation creates a new OpStatement after the loop
    ctx.ResetInsertionPoint();
    // Use ADD with zero to copy: out = finalBuf + 0
    auto zeroOut = builder.CreateConst(ctx, 0.0, "zero_out");
    // finalBuf should be a TileValuePtr, cast it to ensure type safety
    auto finalBufTile = std::dynamic_pointer_cast<TileValue>(finalBuf);
    ASSERT_NE(finalBufTile, nullptr);
    auto copyOutOp = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, finalBufTile, zeroOut, out);
    builder.Emit(ctx, copyOutOp);

    // Verify static for loop structure
    ASSERT_NE(staticFor, nullptr);
    ASSERT_NE(staticFor->GetRange(), nullptr);
    
    // Check that all bounds are constants (immediate values)
    auto start = staticFor->GetStart();
    auto end = staticFor->GetEnd();
    auto step = staticFor->GetStep();
    
    ASSERT_NE(start, nullptr);
    ASSERT_NE(end, nullptr);
    ASSERT_NE(step, nullptr);
    
    // Verify loop body has multiple statements (operations)
    ASSERT_GT(staticFor->GetCompound()->GetStatementsNum(), 0);
    
    // Verify that loop body contains OpStatement with multiple operations
    auto opStmt = std::dynamic_pointer_cast<OpStatement>(
        staticFor->GetCompound()->GetStatement(0));
    ASSERT_NE(opStmt, nullptr);
    ASSERT_GE(opStmt->Operations().size(), 3); // At least 3 operations (Mul, Add, Exp)
    
    // Return
    auto returnVal = builder.CreateConst(ctx, int64_t(0), "return_val");
    builder.CreateReturn(ctx, {returnVal});

    ctx.PopScope(); // function-body
    
    std::stringstream ss;
    IRPrinter printer(ss);
    printer.VisitProgram(module);
    std::cout << ss.str() << std::endl;
}

} // namespace pto
