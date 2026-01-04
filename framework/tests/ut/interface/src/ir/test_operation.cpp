#include "gtest/gtest.h"

#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "ir/op/op_opcode.h"
#include "ir/op/op_payload.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/value.h"
#include "ir/op/op_factory.h"

namespace pto{

TEST(IRTEST, TestTensorOperation){
    // ===== Program module =====
    ProgramModule module("main");

    // ===== Function signature =====
    FunctionSignature sig;

    // Input tensor: tensor<[B, 128], f32>
    Scalar B(DataType::INT32, "B");
    std::vector<Scalar> tensorShape = { B, Scalar(128) };
    auto inputTensor =
        std::make_shared<Tensor>(tensorShape, DataType::FP32, "input");

    sig.arguments.push_back(inputTensor);

    // Output tensor
    auto outputTensor =
        std::make_shared<Tensor>(tensorShape, DataType::FP32, "output");
    sig.results.push_back(outputTensor);

    auto func =
        std::make_shared<Function>("test_all_ops", FunctionKind::ControlFlow, sig);

    module.SetProgramEntry(func);
    module.AddFunction(func);

    auto c2 = std::make_shared<Scalar>(2.0f, "c2");
    auto c3 = std::make_shared<Scalar>(3.0f, "c3");

    func->GetCompound()->SetEnvVar(c2->GetName(), c2);
    func->GetCompound()->SetEnvVar(c3->GetName(), c3);

    auto block = std::make_shared<OpStatement>();

    TensorCreateSpec TCSpec{tensorShape, DataType::FP32};
    block->Operations().push_back(
        CreateOp(
            Opcode::OP_TENSOR_CREATE, 
            { },
            { outputTensor },
            std::make_shared<TensorCreatePayload>(TCSpec))
    );
    func->GetCompound()->SetEnvVar(outputTensor->GetName(), outputTensor);

    auto tensorA =
        std::make_shared<Tensor>(tensorShape, DataType::FP32, "tensorA");
    func->GetCompound()->SetEnvVar(tensorA->GetName(), tensorA);

    ViewSpec viewSpec;
    viewSpec.shape = {1, 128};
    viewSpec.offset = { Scalar(0), Scalar(0) };

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_VIEW,
            { inputTensor },
            { tensorA },
            std::make_shared<ViewPayload>(viewSpec)
        )
    );

    auto tensorAdd =
        std::make_shared<Tensor>(tensorShape, DataType::FP32, "tensorAdd");
    func->GetCompound()->SetEnvVar(tensorAdd->GetName(), tensorAdd);

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_ADD,
            { tensorA, c2 },
            { tensorAdd }
        )
    );

    auto tensorSub =
        std::make_shared<Tensor>(tensorShape, DataType::FP32, "tensorSub");
    func->GetCompound()->SetEnvVar(tensorSub->GetName(), tensorSub);

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_SUB,
            { tensorAdd, c3 },
            { tensorSub }
        )
    );

    auto tensorMul =
        std::make_shared<Tensor>(tensorShape, DataType::FP32, "tensorMul");
    func->GetCompound()->SetEnvVar(tensorMul->GetName(), tensorMul);

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_MUL,
            { tensorSub, c2 },
            { tensorMul }
        )
    );

    auto tensorDiv =
        std::make_shared<Tensor>(tensorShape, DataType::FP32, "tensorDiv");
    func->GetCompound()->SetEnvVar(tensorDiv->GetName(), tensorDiv);

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_DIV,
            { tensorMul, c2 },
            { tensorDiv }
        )
    );
    
    std::vector<Scalar> sumShape = { B };
    auto tensorSum =
        std::make_shared<Tensor>(sumShape, DataType::FP32, "sum");
    func->GetCompound()->SetEnvVar(tensorSum->GetName(), tensorSum);

    ReduceSpec sumSpec;
    sumSpec.kind = ReduceKind::RowSum;
    sumSpec.axis = 1;
    sumSpec.keepDim = false;

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_ROWSUM_SINGLE,
            { tensorDiv },
            { tensorSum },
            std::make_shared<ReducePayload>(sumSpec)
        )
    );

    auto maxScalar =
        std::make_shared<Scalar>(DataType::FP32, "max");
    func->GetCompound()->SetEnvVar(maxScalar->GetName(), maxScalar);

    ReduceSpec maxSpec;
    maxSpec.kind = ReduceKind::RowMax;
    maxSpec.axis = 0;
    maxSpec.keepDim = false;

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_ROWMAX_SINGLE,
            { tensorSum },
            { maxScalar },
            std::make_shared<ReducePayload>(maxSpec)
        )
    );

    auto tensorOut =
        std::make_shared<Tensor>(tensorShape, DataType::FP32, "output");
    func->GetCompound()->SetEnvVar(tensorOut->GetName(), tensorOut);

    AssembleSpec assembleSpec;
    assembleSpec.offset = { Scalar(0), Scalar(0) };

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_ASSEMBLE,
            { tensorDiv, outputTensor },
            { tensorOut },
            std::make_shared<AssemblePayload>(assembleSpec)
        )
    );

    func->AddStatement(std::move(block));

    auto ret = std::make_shared<ReturnStatement>();
    ret->Values().push_back(tensorOut);
    func->AddStatement(std::move(ret));

    ASSERT_EQ(func->GetCompound()->GetEnvVar("output"), tensorOut);

    std::cout << module << std::endl;
}
};