#include "gtest/gtest.h"

#include <iostream>
#include <memory>
#include <vector>

#include "ir/opcode.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/value.h"
#include "ir/builder/ir_builder.h"

namespace pto{

TEST(IRTEST, TestTensorOperation){
    // ===== Program module =====
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder(module);

    // ===== Function signature =====
    FunctionSignature sig;

    // Input tensor: tensor<[B, 128], f32>
    auto B = std::make_shared<ScalarValue>(DataType::INT32, "B", ScalarValueKind::Symbolic);
    std::vector<ScalarValuePtr> tensorShape = { B, std::make_shared<ScalarValue>(int64_t(128)) };
    auto inputTensor =
        std::make_shared<TensorValue>(tensorShape, DataType::FP32, "input");

    sig.arguments.push_back(inputTensor);

    // Output tensor
    auto outputTensor =
        std::make_shared<TensorValue>(tensorShape, DataType::FP32, "output");
    sig.results.push_back(outputTensor);

    // ===== Function =====
    auto func = builder.CreateFunction("test_all_ops", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    {
        auto guard = builder.EnterFunctionBody(func);

        auto c2 = builder.CreateConst(2.0, "c2");
        auto c3 = builder.CreateConst(3.0, "c3");

        // tensorAdd = add(input, c2)
        auto tensorAdd = builder.CreateTensor(tensorShape, DataType::FP32, "tensorAdd");
        auto addOp = builder.CreateBinaryOp(Opcode::OP_ADD, inputTensor, c2, tensorAdd);
        builder.Emit(addOp);

        // tensorSub = sub(tensorAdd, c3)
        auto tensorSub = builder.CreateTensor(tensorShape, DataType::FP32, "tensorSub");
        auto subOp = builder.CreateBinaryOp(Opcode::OP_SUB, tensorAdd, c3, tensorSub);
        builder.Emit(subOp);

        // tensorMul = mul(tensorSub, c2)
        auto tensorMul = builder.CreateTensor(tensorShape, DataType::FP32, "tensorMul");
        auto mulOp = builder.CreateBinaryOp(Opcode::OP_MUL, tensorSub, c2, tensorMul);
        builder.Emit(mulOp);

        // tensorDiv = div(tensorMul, c2)
        auto tensorDiv = builder.CreateTensor(tensorShape, DataType::FP32, "tensorDiv");
        auto divOp = builder.CreateBinaryOp(Opcode::OP_DIV, tensorMul, c2, tensorDiv);
        builder.Emit(divOp);

        // return tensorDiv
        builder.CreateReturn({ tensorDiv });
    }

    ASSERT_EQ(func->GetCompound()->GetEnvVar("tensorDiv"), func->GetCompound()->GetEnvVar("tensorDiv"));

    std::cout << *module << std::endl;
}

};