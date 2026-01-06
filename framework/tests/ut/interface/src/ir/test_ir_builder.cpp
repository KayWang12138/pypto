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
#include "gtest/gtest.h"


#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "ir/opcode.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/value.h"


namespace pto{

TEST(IRTEST, TestBuilder) {
    // ===== Module =====
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // ===== Signature =====
    FunctionSignature sig;

    // tensor<[b, 128], fp32>
    auto batch = std::make_shared<ScalarValue>(DataType::INT32, "b", ScalarValueKind::Symbolic);
    std::vector<ScalarValuePtr> tensorShape = { batch, std::make_shared<ScalarValue>(int64_t(128)) };

    auto inputTensor  = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "input");
    auto scale1       = std::make_shared<ScalarValue>(DataType::FP32, "scale1", ScalarValueKind::Symbolic);

    auto dynLen       = std::make_shared<ScalarValue>(DataType::INT32, "len", ScalarValueKind::Symbolic);

    sig.arguments = { inputTensor, scale1, dynLen };

    auto resultSig = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "output");
    sig.results.push_back(resultSig);

    // ===== Function =====
    auto func = builder.CreateFunction("test_value", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    {
        // enter func scope + create an initial block as insertion point
        builder.EnterFunctionBody(ctx, func);

        // mul1_res = mul(input, scale1)
        auto mulVal1 = builder.CreateTensor(ctx, tensorShape, DataType::FP32, "mul1_res");
        auto mulOp1 = builder.CreateBinaryOp(Opcode::OP_MUL, inputTensor, scale1, mulVal1);
        builder.Emit(ctx, mulOp1);

        auto pi = builder.CreateConst(ctx, 3.14, "const_pi");

        // mul2_res = mul(mul1_res, pi)
        auto mulVal2 = builder.CreateTensor(ctx, tensorShape, DataType::FP32, "mul2_res");
        auto mulOp2 = builder.CreateBinaryOp(Opcode::OP_MUL, mulVal1, pi, mulVal2);
        builder.Emit(ctx, mulOp2);

        builder.CreateReturn(ctx, { mulVal2 });

        ASSERT_EQ(ctx.func, func);
        ASSERT_EQ(ctx.compound, func->GetCompound());
        ASSERT_EQ(ctx.activeOpStmt, func->GetCompound()->GetStatements()[0]);

        ctx.PopScope();
    }

    ASSERT_EQ(ctx.func, nullptr);
    ASSERT_EQ(ctx.compound, nullptr);
    ASSERT_EQ(ctx.activeOpStmt, nullptr);

    // ===== Program attributes =====
    module->Attributes()["arch"] = "\"PTOv2\"";
    module->Attributes()["tile_default"] = "{ M=16, N=16, K=16 }";
    module->Attributes()["enable_debug"] = "true";

    std::cout << *module << std::endl;
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

    {
        auto funcGuard = builder.EnterFunctionBody(func);
        builder.EnterFunctionBody(ctx, func);
        auto opStmt = builder.CreateOpStmt(ctx);

        // for i = 0 to batch step 1
        auto i = builder.CreateScalar(ctx, DataType::INT32, "i");
        auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
        auto constant1 = builder.CreateConst(ctx, int64_t(1), "const_1");
        auto fs = builder.CreateForStmt(ctx, i, constant0, batch, constant1);
        {
            builder.EnterForBody(ctx, fs);

            // don't have view op now
            // loopX = view(inputX, {1, 128}, {i, 0})
            // loopY = view(inputY, {1, 128}, {i, 0})

            // outputX = mul(loopX, scale1)
            resLoopX = builder.CreateTensor({batch, constant128}, DataType::FP32, "outputX");
            auto addOpX =  builder.CreateBinaryOp(
                Opcode::OP_ADD, 
                inputX, scale1,
                resLoopX
            );
            builder.Emit(addOpX);
            // don't have assemble op now
            // outputX = add(inputX, scale1)
            auto resLoopX = builder.CreateTensor(ctx, tensorShape, DataType::FP32, "outputX");
            auto addOpX = builder.CreateBinaryOp(Opcode::OP_ADD, inputX, scale1, resLoopX);
            builder.Emit(ctx, addOpX);

            // outputY = mul(looY, scale2)
            resLoopY = builder.CreateTensor({batch, constant128}, DataType::FP32, "outputY");
            auto addOpY = builder.CreateBinaryOp(
                Opcode::OP_ADD, 
                inputY, scale2,
                resLoopY
            );
            builder.Emit(addOpY);
            // outputY = add(inputY, scale2)
            auto resLoopY = builder.CreateTensor(ctx, tensorShape, DataType::FP32, "outputY");
            auto addOpY = builder.CreateBinaryOp(Opcode::OP_ADD, inputY, scale2, resLoopY);
            builder.Emit(ctx, addOpY);

            // if i then outputX = mul(outputX, scale1) else outputY = mul(outputY, scale2)
            auto ifs = builder.CreateIfStmt(i);
            auto ifs = builder.CreateIfStmt(ctx, "i");
            ValuePtr resIfX, resIfY;
            {
                builder.EnterIfThen(ctx, ifs);

                resIfX = builder.CreateTensor({batch, constant128}, DataType::FP32, "outputX");
                auto mulOpX = builder.CreateBinaryOp(
                    Opcode::OP_MUL, 
                    resLoopX, scale1,
                    resIfX  
                );
                builder.Emit(mulOpX);
                resIfX = builder.CreateTensor(ctx, tensorShape, DataType::FP32, "outputX");
                auto mulOpX = builder.CreateBinaryOp(Opcode::OP_MUL, resLoopX, scale1, resIfX);
                builder.Emit(ctx, mulOpX);

                ctx.PopScope();
            }
            {
                auto ifElseGuard = builder.EnterIfElse(ifs);
                
                resIfY = builder.CreateTensor({batch, constant128}, DataType::FP32, "outputY");
                auto mulOpY = builder.CreateBinaryOp(
                    Opcode::OP_MUL, 
                    resLoopY, scale2,
                    resIfY    
                );
                builder.Emit(mulOpY);
                builder.EnterIfElse(ctx, ifs);

                resIfY = builder.CreateTensor(ctx, tensorShape, DataType::FP32, "outputY");
                auto mulOpY = builder.CreateBinaryOp(Opcode::OP_MUL, resLoopY, scale2, resIfY);
                builder.Emit(ctx, mulOpY);

                ctx.PopScope();
            }
            builder.ExitIfStatement(ctx, ifs);

            // check if then and else yield
            auto thenYield = std::dynamic_pointer_cast<YieldStatement>(*ifs->GetThenCompound()->GetStatements().rbegin());
            std::unordered_set<ValuePtr> thenYieldSet(thenYield->Values().begin(), thenYield->Values().end());
            std::unordered_set<ValuePtr> thenYieldSetGolden{resIfX, resLoopY};
            ASSERT_EQ(thenYieldSet, thenYieldSetGolden);
            auto elseYield = std::dynamic_pointer_cast<YieldStatement>(*ifs->GetElseCompound()->GetStatements().rbegin());
            std::unordered_set<ValuePtr> elseYieldSet(elseYield->Values().begin(), elseYield->Values().end());
            std::unordered_set<ValuePtr> elseYieldSetGolen{resLoopX, resIfY};
            ASSERT_EQ(elseYieldSet, elseYieldSetGolen);
            ASSERT_NE(elseYield, nullptr);
            ASSERT_GE(elseYield->Values().size(), 2);
            ASSERT_EQ(elseYield->Values()[0], resIfY);
            ASSERT_EQ(elseYield->Values()[1], resLoopX);

            ctx.PopScope();
        }
        builder.ExitForStatement(ctx, fs);

        // // check for yeild
        auto ifs = std::dynamic_pointer_cast<IfStatement>(fs->GetCompound()->GetStatements()[1]);
        auto ifResults = ifs->Results();
        auto forYields = fs->Yield()->Values();
        std::unordered_set<ValuePtr> ifResultSet(ifResults.begin(), ifResults.end());
        std::unordered_set<ValuePtr> forYieldSet(forYields.begin(), forYields.end());
        ASSERT_EQ(ifResultSet, forYieldSet);

        // return 
        builder.CreateReturn({constant0});
        // return outputX, outputY
        builder.CreateReturn(ctx, fs->Results());

        ctx.PopScope();
    }
    std::cout << *module << std::endl;
}

} // namespace pto