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
    std::vector<ScalarValuePtr> tensorShape = { batch, std::make_shared<ScalarValue>(int64_t(128)) };

    auto inputX = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "inputX");
    auto inputY = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "inputY");
    auto scale1 = std::make_shared<ScalarValue>(DataType::FP32, "scale1", ScalarValueKind::Symbolic);
    auto scale2 = std::make_shared<ScalarValue>(DataType::FP32, "scale2", ScalarValueKind::Symbolic);

    sig.arguments = { inputX, inputY, scale1, scale2};

    auto resultSigX = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "outputX");
    auto resultSigY = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "outputY");
    sig.results.push_back(resultSigX);
    sig.results.push_back(resultSigY);

    // ===== Function =====
    auto func = builder.CreateFunction("test_control", FunctionKind::ControlFlow, sig, /*setAsEntry=*/false);
    module->SetProgramEntry(func);

    {
        auto funcGuard = builder.EnterFunctionBody(func);
        auto opStmt = builder.CreateOpStmt();

        // for i = 0 to batch step 1
        auto i = builder.CreateScalar(DataType::INT32, "i");
        auto constant0 = builder.CreateConst(int64_t(0), "const_0");
        auto constant1 = builder.CreateConst(int64_t(1), "const_1");
        auto fs = builder.CreateForStmt(i, constant0, batch, constant1);
        {
            auto fsGuard = builder.EnterForBody(fs);

            // outputX = add(inputX, scale1)
            auto resLoopX = builder.CreateTensor(tensorShape, DataType::FP32, "outputX");
            auto addOpX = builder.CreateBinaryOp(Opcode::OP_ADD, inputX, scale1, resLoopX);
            builder.Emit(addOpX);

            // outputY = add(inputY, scale2)
            auto resLoopY = builder.CreateTensor(tensorShape, DataType::FP32, "outputY");
            auto addOpY = builder.CreateBinaryOp(Opcode::OP_ADD, inputY, scale2, resLoopY);
            builder.Emit(addOpY);

            // if i then outputX = mul(outputX, scale1) else outputY = mul(outputY, scale2)
            auto ifs = builder.CreateIfStmt("i");
            ValuePtr resIfX, resIfY;
            {
                auto ifThenGuard = builder.EnterIfThen(ifs);

                resIfX = builder.CreateTensor(tensorShape, DataType::FP32, "outputX");
                auto mulOpX = builder.CreateBinaryOp(Opcode::OP_MUL, resLoopX, scale1, resIfX);
                builder.Emit(mulOpX);
            }
            {
                auto ifElseGuard = builder.EnterIfElse(ifs);

                resIfY = builder.CreateTensor(tensorShape, DataType::FP32, "outputY");
                auto mulOpY = builder.CreateBinaryOp(Opcode::OP_MUL, resLoopY, scale2, resIfY);
                builder.Emit(mulOpY);
            }
            builder.ExitIfStatement(ifs);

            // check if then and else yield
            auto thenYield = std::dynamic_pointer_cast<YieldStatement>(*ifs->GetThenCompound()->GetStatements().rbegin());
            ASSERT_NE(thenYield, nullptr);
            ASSERT_GE(thenYield->Values().size(), 2);
            ASSERT_EQ(thenYield->Values()[0], resLoopY);
            ASSERT_EQ(thenYield->Values()[1], resIfX);
            auto elseYield = std::dynamic_pointer_cast<YieldStatement>(*ifs->GetElseCompound()->GetStatements().rbegin());
            ASSERT_NE(elseYield, nullptr);
            ASSERT_GE(elseYield->Values().size(), 2);
            ASSERT_EQ(elseYield->Values()[0], resIfY);
            ASSERT_EQ(elseYield->Values()[1], resLoopX);
        }
        builder.ExitForStatement(fs);

        // check for yield
        // Find the if statement in the for loop body
        IfStatementPtr ifsInFor = nullptr;
        for (const auto& stmt : fs->GetCompound()->GetStatements()) {
            ifsInFor = std::dynamic_pointer_cast<IfStatement>(stmt);
            if (ifsInFor) break;
        }
        ASSERT_NE(ifsInFor, nullptr);
        auto ifResults = ifsInFor->Results();
        auto forYield = fs->Yield();
        ASSERT_NE(forYield, nullptr);
        ASSERT_EQ(forYield->Values(), ifResults);

        // return outputX, outputY
        builder.CreateReturn(fs->Results());
    }
    std::cout << *module << std::endl;
}

} // namespace pto