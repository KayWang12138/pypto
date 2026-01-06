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
        auto guard = builder.EnterFunctionBody(func);

        // mul1_res = mul(input, scale1)
        auto mulVal1 = builder.CreateTensor(tensorShape, DataType::FP32, "mul1_res");
        auto mulOp1 = builder.CreateBinaryOp(Opcode::OP_MUL, inputTensor, scale1, mulVal1);
        builder.Emit(mulOp1);

        auto pi = builder.CreateConst(3.14, "const_pi");

        // mul2_res = mul(mul1_res, pi)
        auto mulVal2 = builder.CreateTensor(tensorShape, DataType::FP32, "mul2_res");
        auto mulOp2 = builder.CreateBinaryOp(Opcode::OP_MUL, mulVal1, pi, mulVal2);
        builder.Emit(mulOp2);

        builder.CreateReturn({ mulVal2 });

        ASSERT_EQ(builder.GetCurrentFunction(), func);
        ASSERT_EQ(builder.GetCurrentCompound(), func->GetCompound());
        ASSERT_EQ(builder.GetCurrentOpStmt(), func->GetCompound()->GetStatements()[0]);
    }

    ASSERT_EQ(builder.GetCurrentFunction(), nullptr);
    ASSERT_EQ(builder.GetCurrentCompound(), nullptr);
    ASSERT_EQ(builder.GetCurrentOpStmt(), nullptr);

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

        // for i = 0 to batch step 1
        auto i = builder.CreateScalar(DataType::INT32, "i");
        auto constant0 = builder.CreateConst(int64_t(0), "const_0");
        auto constant1 = builder.CreateConst(int64_t(1), "const_1");
        auto fs = builder.CreateForStmt(i, constant0, batch, constant1);
        ValuePtr resLoopX, resLoopY;
        {
            auto fsGuard = builder.EnterForBody(fs);

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

            // outputY = mul(looY, scale2)
            resLoopY = builder.CreateTensor({batch, constant128}, DataType::FP32, "outputY");
            auto addOpY = builder.CreateBinaryOp(
                Opcode::OP_ADD, 
                inputY, scale2,
                resLoopY
            );
            builder.Emit(addOpY);

            // if i then outputX = mul(outputX, scale1) else outputY = mul(outputY, scale2)
            auto ifs = builder.CreateIfStmt("i");
            ValuePtr resIfX, resIfY;
            {
                auto ifThenGuard = builder.EnterIfThen(ifs);

                // ifX = view(outputX, {1, 128}, {i, 0})

                resIfX = builder.CreateTensor({batch, constant128}, DataType::FP32, "outputX");
                auto mulOpX = builder.CreateBinaryOp(
                    Opcode::OP_MUL, 
                    resLoopX, scale1,
                    resIfX  
                );
                builder.Emit(mulOpX);
            }
            {
                auto ifElseGuard = builder.EnterIfElse(ifs);
                
                // ifY = view(outputY, {1, 128}, {i, 0})

                resIfY = builder.CreateTensor({batch, constant128}, DataType::FP32, "outputY");
                auto mulOpY = builder.CreateBinaryOp(
                    Opcode::OP_MUL, 
                    resLoopY, scale2,
                    resIfY    
                );
                builder.Emit(mulOpY);
            }
            builder.ExitIfStatement(ifs);

            // check if then and else yield
            auto thenYield = std::dynamic_pointer_cast<YieldStatement>(*ifs->GetThenCompound()->GetStatements().rbegin());
            std::unordered_set<ValuePtr> thenYieldSet(thenYield->Values().begin(), thenYield->Values().end());
            std::unordered_set<ValuePtr> thenYieldSetGolden{resIfX, resLoopY};
            ASSERT_EQ(thenYieldSet, thenYieldSetGolden);
            auto elseYield = std::dynamic_pointer_cast<YieldStatement>(*ifs->GetElseCompound()->GetStatements().rbegin());
            std::unordered_set<ValuePtr> elseYieldSet(elseYield->Values().begin(), elseYield->Values().end());
            std::unordered_set<ValuePtr> elseYieldSetGolen{resLoopX, resIfY};
            ASSERT_EQ(elseYieldSet, elseYieldSetGolen);
        }
        builder.ExitForStatement(fs);

        // // check for yeild
        auto ifs = std::dynamic_pointer_cast<IfStatement>(fs->GetCompound()->GetStatements()[1]);
        auto ifResults = ifs->Results();
        auto forYields = fs->Yield()->Values();
        std::unordered_set<ValuePtr> ifResultSet(ifResults.begin(), ifResults.end());
        std::unordered_set<ValuePtr> forYieldSet(forYields.begin(), forYields.end());
        ASSERT_EQ(ifResultSet, forYieldSet);

        // return 
        builder.CreateReturn({constant0});
    }
    std::cout << *module << std::endl;
}

} // namespace pto