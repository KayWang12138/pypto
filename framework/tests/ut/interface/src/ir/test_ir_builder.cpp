#include <cstdint>
#include "gtest/gtest.h"


#include "ir/builder/ir_builder.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/type.h"


namespace pto{

TEST(IRTEST, TestBuilder){
    // ===== Module =====
    ProgramModule module("main");
    IRBuilder builder(&module);

    // ===== Signature =====
    FunctionSignature sig;

    // tensor<[b, 128], fp32>
    Scalar batch(DataType::INT32, "b", ScalarValueKind::Symbolic);
    std::vector<Scalar> tensorShape = { batch, Scalar(int64_t(128)) };

    auto inputTensor  = std::make_shared<Tensor>(tensorShape, DataType::FP32, "input");
    auto scale1       = std::make_shared<Scalar>(DataType::FP32, "scale1", ScalarValueKind::Symbolic);
    
    auto dynLen       = std::make_shared<Scalar>(DataType::INT32, "len", ScalarValueKind::Symbolic);

    sig.arguments = { inputTensor, scale1, dynLen };

    auto resultTensor = std::make_shared<Tensor>(tensorShape, DataType::FP32, "output");
    sig.results.push_back(resultTensor);

    // ===== Function =====
    auto func = builder.CreateFunction("test_value", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    {
        // enter func scope + create an initial block as insertion point
        auto guard = builder.EnterFunctionBody(*func);

        auto constant0 = builder.CreateConst(int64_t(0), "const_0");

        ViewSpec viewSpec;
        viewSpec.shape  = { 1, 128 };
        viewSpec.offset = {
            *constant0,
            *constant0
        };

        auto loopTile = builder.CreateOp(
            Opcode::OP_VIEW,
            { inputTensor },
            std::make_shared<ViewPayload>(viewSpec),
            "loop_tile"
        )[0];

        auto mulVal1 = builder.CreateOp(
            Opcode::OP_MUL,
            { loopTile, scale1 },
            nullptr,
            "mul1_res"
        )[0];

        auto pi = builder.CreateConst(3.14, "const_pi");

        auto mulVal2 = builder.CreateOp(
            Opcode::OP_MUL,
            { mulVal1, pi },
            nullptr,
            "mul2_res"
        )[0];

        AssembleSpec assembleSpec;
        assembleSpec.offset = {
            *constant0,
            *constant0
        };

        auto assemOut = builder.CreateOp(
            Opcode::OP_ASSEMBLE,
            { mulVal2, resultTensor },
            std::make_shared<AssemblePayload>(assembleSpec),
            "output"
        )[0];

        builder.CreateReturn({ assemOut });

        ASSERT_EQ(builder.GetCurrentFunction(), func.get());
        ASSERT_NE(builder.GetCurrentScope(), nullptr);
        ASSERT_NE(builder.GetCurrentBlock(), nullptr);
    }

    ASSERT_EQ(builder.GetCurrentFunction(), nullptr);
    ASSERT_EQ(builder.GetCurrentScope(), nullptr);
    ASSERT_EQ(builder.GetCurrentBlock(), nullptr);

    builder.SetCurrentFunction(*func);

    // ===== Program attributes =====
    module.Attributes()["arch"] = "\"PTOv2\"";
    module.Attributes()["tile_default"] = "{ M=16, N=16, K=16 }";
    module.Attributes()["enable_debug"] = "true";

    std::cout << module << std::endl;
}


} // namespace pto