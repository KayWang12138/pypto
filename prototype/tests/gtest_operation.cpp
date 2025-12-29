#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/type.h"
#include "ir/op/op_factory.h"

namespace pto{

TEST(IRTEST, TestOperation){
    // ===== Program module =====
    ProgramModule module("@main");

    // ===== Function signature =====
    FunctionSignature sig;

    // Input tensor: tensor<[%B, 128], f32>
    Scalar B(DataType::INT32, "%B");
    std::vector<Scalar> tensorShape = { B, Scalar(128) };
    auto inputTensor =
        std::make_shared<Tensor>(tensorShape, DataType::FP32, "%input");

    sig.arguments.push_back(inputTensor);

    // Output tensor
    auto outputTensor =
        std::make_shared<Tensor>(tensorShape, DataType::FP32, "%output");
    sig.results.push_back(outputTensor);

    auto func =
        std::make_shared<Function>("@test_all_ops", FunctionKind::ControlFlow, sig);

    module.SetProgramEntry(func);
    module.AddFunction(func);

    auto c2 = std::make_shared<Scalar>(2.0f, "%c2");
    auto c3 = std::make_shared<Scalar>(3.0f, "%c3");

    func->GetScope().SetEnvVar(c2->GetSSAName(), c2);
    func->GetScope().SetEnvVar(c3->GetSSAName(), c3);

    auto block = std::make_shared<BlockStatement>();

    std::vector<size_t> tileShape{1, 128};
    std::vector<Scalar> validShape{ Scalar(1), B };

    auto tileA =
        std::make_shared<Tile>(tileShape, DataType::FP32, validShape, "%tileA");

    ViewSpec viewSpec;
    viewSpec.shape = {1, 128};
    viewSpec.offset = { Scalar(0), Scalar(0) };

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_VIEW,
            { inputTensor },
            { tileA },
            std::make_shared<ViewPayload>(viewSpec)
        )
    );

    auto tileAdd =
        std::make_shared<Tile>(tileShape, DataType::FP32, validShape, "%tileAdd");

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_ADD,
            { tileA, c2 },
            { tileAdd }
        )
    );

    auto tileSub =
        std::make_shared<Tile>(tileShape, DataType::FP32, validShape, "%tileSub");

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_SUB,
            { tileAdd, c3 },
            { tileSub }
        )
    );

    auto tileMul =
        std::make_shared<Tile>(tileShape, DataType::FP32, validShape, "%tileMul");

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_MUL,
            { tileSub, c2 },
            { tileMul }
        )
    );

    auto tileDiv =
        std::make_shared<Tile>(tileShape, DataType::FP32, validShape, "%tileDiv");

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_DIV,
            { tileMul, c2 },
            { tileDiv }
        )
    );

    auto sumScalar =
        std::make_shared<Scalar>(DataType::FP32, "%sum");

    ReduceSpec sumSpec;
    sumSpec.kind = ReduceKind::RowSum;
    sumSpec.axis = 1;
    sumSpec.keepDim = false;

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_ROWSUM_SINGLE,
            { tileDiv },
            { sumScalar },
            std::make_shared<ReducePayload>(sumSpec)
        )
    );

    auto maxScalar =
        std::make_shared<Scalar>(DataType::FP32, "%max");

    ReduceSpec maxSpec;
    maxSpec.kind = ReduceKind::RowMax;
    maxSpec.axis = 1;
    maxSpec.keepDim = false;

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_ROWMAX_SINGLE,
            { tileDiv },
            { maxScalar },
            std::make_shared<ReducePayload>(maxSpec)
        )
    );
    AssembleSpec assembleSpec;
    assembleSpec.offset = { Scalar(0), Scalar(0) };

    block->Operations().push_back(
        CreateOp(
            Opcode::OP_ASSEMBLE,
            { tileDiv },
            { outputTensor },
            std::make_shared<AssemblePayload>(assembleSpec)
        )
    );

    func->AddStatement(std::move(block));

    auto ret = std::make_shared<ReturnStatement>();
    ret->Values().push_back(outputTensor);
    func->AddStatement(std::move(ret));

    std::cout << module << std::endl;
}
};