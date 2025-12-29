#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>

#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/type.h"
#include "ir/builder/ir_builder.h"

namespace pto{

TEST(IRTEST, TestBuilder){
    // ===== Module =====
    ProgramModule module("@main");
    IRBuilder builder(&module);

    // ===== Signature =====
    FunctionSignature sig;

    // tensor<[%b, 128], fp32>
    Scalar batch(DataType::INT32, "%b", ScalarValueKind::Symbolic);
    std::vector<Scalar> tensorShape = { batch, Scalar(int64_t(128)) };

    auto inputTensor  = std::make_shared<Tensor>(tensorShape, DataType::FP32, "%input");
    auto scale1       = std::make_shared<Scalar>(DataType::FP32, "%scale1", ScalarValueKind::Symbolic);
    auto scale2       = std::make_shared<Scalar>(DataType::FP32, "%scale2", ScalarValueKind::Symbolic);
    auto condScalar   = std::make_shared<Scalar>(DataType::BOOL, "%condition", ScalarValueKind::Symbolic);
    auto dynLen       = std::make_shared<Scalar>(DataType::INT32, "%len", ScalarValueKind::Symbolic);

    sig.arguments = { inputTensor, scale1, scale2, condScalar, dynLen };

    auto resultTensor = std::make_shared<Tensor>(tensorShape, DataType::FP32, "%output");
    sig.results.push_back(resultTensor);

    // ===== Function =====
    auto func = builder.CreateFunction("@test_value", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    // ===== for (%i = %c0; %i < %b; %i += %c1) =====
    auto c0 = std::make_shared<Scalar>(int64_t(0), "%c0");
    auto c1 = std::make_shared<Scalar>(int64_t(1), "%c1");
    auto batchScalar = std::make_shared<Scalar>(DataType::INT32, "%b", ScalarValueKind::Symbolic);
    auto iterationVar = std::make_shared<Scalar>(DataType::INT32, "%i", ScalarValueKind::Symbolic);
    auto& forSt = builder.CreateForStmt(iterationVar, c0, batchScalar, c1);
    {
        // enter for-body scope + create an initial block as insertion point
        auto guard = builder.EnterForBody(forSt);
        ViewSpec viewSpec;
        viewSpec.shape  = { 1, 128 };
        viewSpec.offset = {
            Scalar(DataType::INT32, "%i", ScalarValueKind::Symbolic),
            Scalar(int64_t(0))
        };

        auto loopTileA = builder.CreateOp(
            Opcode::OP_VIEW,
            { inputTensor },
            std::make_shared<ViewPayload>(viewSpec),
            "%loop_tileA"
        )[0];

        auto& ifSt = builder.CreateIfStmt("%condition");
        {
            auto guard = builder.EnterIfThen(ifSt);

            auto thenVal = builder.CreateOp(
                Opcode::OP_MUL,
                { loopTileA, scale1 },
                nullptr,
                "%then_result"
            )[0];

            builder.CreateYield({ thenVal });
        }
        {
            auto guard = builder.EnterIfElse(ifSt);

            auto elseVal = builder.CreateOp(
                Opcode::OP_MUL,
                { loopTileA, scale2 },
                nullptr,
                "%else_result"
            )[0];

            builder.CreateYield({ elseVal });
        }
        ifSt.BuildResult();

        ValuePtr ifResult = nullptr;
        if (!ifSt.Results().empty()) {
            ifResult = ifSt.Results()[0];
            ifResult->SetName("%loop_tileB");
            forSt.GetScope().SetEnvVar(ifResult->GetSSAName(), ifResult);
        }

        AssembleSpec assembleSpec;
        assembleSpec.offset = {
            Scalar(DataType::INT32, "%i", ScalarValueKind::Symbolic),
            Scalar(int64_t(0))
        };

        builder.CreateOp(
            Opcode::OP_ASSEMBLE,
            { ifResult },
            { resultTensor },
            std::make_shared<AssemblePayload>(assembleSpec)
        );

    }

    // ===== return =====
    builder.SetCurrentFunction(*func);
    builder.CreateReturn({ resultTensor });

    // ===== Program attributes =====
    module.Attributes()["arch"] = "\"PTOv2\"";
    module.Attributes()["tile_default"] = "{ M=16, N=16, K=16 }";
    module.Attributes()["enable_debug"] = "true";

    std::cout << module << std::endl;
}

TEST(IRTEST, TestFunctionInputVisibleInAncestorValues) {
    FunctionSignature sig;

    std::vector<Scalar> shape = { Scalar(int64_t(2)), Scalar(int64_t(3)) };
    auto a = std::make_shared<Tensor>(shape, DataType::FP32, "%A");
    auto b = std::make_shared<Scalar>(DataType::INT32, "%B", ScalarValueKind::Symbolic);
    sig.arguments = { a, b };

    Function f("@test_inputs", FunctionKind::ControlFlow, sig);

    auto ancestor = f.GetScope().GetAncestorValues();
    EXPECT_EQ(ancestor.size(), sig.arguments.size());
    EXPECT_NE(ancestor.find(a->GetSSAName()), ancestor.end());
    EXPECT_NE(ancestor.find(b->GetSSAName()), ancestor.end());
    EXPECT_EQ(ancestor[a->GetSSAName()], a);
    EXPECT_EQ(ancestor[b->GetSSAName()], b);
}

} // namespace pto