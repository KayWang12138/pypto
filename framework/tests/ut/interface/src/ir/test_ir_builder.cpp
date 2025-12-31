#include <cstdint>
#include <memory>
#include "gtest/gtest.h"


#include "ir/builder/ir_builder.h"
#include "ir/op/op_opcode.h"
#include "ir/op/op_payload.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/type.h"


namespace pto{

TEST(IRTEST, TestBuilder) {
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

    auto resultSig = std::make_shared<Tensor>(tensorShape, DataType::FP32, "output");
    sig.results.push_back(resultSig);

    // ===== Function =====
    auto func = builder.CreateFunction("test_value", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    {
        // enter func scope + create an initial block as insertion point
        auto guard = builder.EnterFunctionBody(*func);

        auto constant0 = builder.CreateConst(int64_t(0), "const_0");
        
        // output tensor create
        TensorCreateSpec TCSpec{tensorShape, DataType::FP32};
        auto resultTensor = builder.CreateOp(
            Opcode::OP_TENSOR_CREATE, 
            {}, 
            std::make_shared<TensorCreatePayload>(TCSpec), 
            "output"
        )[0];

        // loop_tile = view(input, {1, 128}, {0, 0})
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

        // mul1_res = mul(loop_tile, scale1)
        auto mulVal1 = builder.CreateOp(
            Opcode::OP_MUL,
            { loopTile, scale1 },
            nullptr,
            "mul1_res"
        )[0];

        auto pi = builder.CreateConst(3.14, "const_pi");

        // mul2_res = mul(mul1_res, pi)
        auto mulVal2 = builder.CreateOp(
            Opcode::OP_MUL,
            { mulVal1, pi },
            nullptr,
            "mul2_res"
        )[0];

        // assemble(output, mul2_res, {0, 0})
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
        ASSERT_EQ(builder.GetCurrentScope(), &func->GetScope());
        ASSERT_EQ(builder.GetCurrentBlock(), func->GetScope().GetStatements()[0].get());
    }

    ASSERT_EQ(builder.GetCurrentFunction(), nullptr);
    ASSERT_EQ(builder.GetCurrentScope(), nullptr);
    ASSERT_EQ(builder.GetCurrentBlock(), nullptr);

    // ===== Program attributes =====
    module.Attributes()["arch"] = "\"PTOv2\"";
    module.Attributes()["tile_default"] = "{ M=16, N=16, K=16 }";
    module.Attributes()["enable_debug"] = "true";

    std::cout << module << std::endl;
}

TEST(IRTEST, TestControlFlow) {
    // ===== Module =====
    ProgramModule module("main");
    IRBuilder builder(&module);

    // ===== Signature =====
    FunctionSignature sig;

    // tensor<[b, 128], fp32>
    auto batch = std::make_shared<Scalar>(DataType::INT32, "batch", ScalarValueKind::Symbolic);
    std::vector<Scalar> tensorShape = { *batch, Scalar(int64_t(128)) };

    auto inputX = std::make_shared<Tensor>(tensorShape, DataType::FP32, "inputX");
    auto inputY = std::make_shared<Tensor>(tensorShape, DataType::FP32, "inputY");
    auto scale1 = std::make_shared<Scalar>(DataType::FP32, "scale1", ScalarValueKind::Symbolic);
    auto scale2 = std::make_shared<Scalar>(DataType::FP32, "scale2", ScalarValueKind::Symbolic);

    sig.arguments = { inputX, inputY, scale1, scale2};

    auto resultSigX = std::make_shared<Tensor>(tensorShape, DataType::FP32, "outputX");
    auto resultSigY = std::make_shared<Tensor>(tensorShape, DataType::FP32, "outputY");
    sig.results.push_back(resultSigX);
    sig.results.push_back(resultSigY);

    // ===== Function =====
    auto func = builder.CreateFunction("test_control", FunctionKind::ControlFlow, sig, /*setAsEntry=*/false);
    module.SetProgramEntry(func);

    {
        auto funcGuard = builder.EnterFunctionBody(*func);
        auto& blk = builder.CreateBlockStmt();

        TensorCreateSpec TCSpec{tensorShape, DataType::FP32};
        auto resultX = builder.CreateOp(
            Opcode::OP_TENSOR_CREATE, 
            {}, 
            std::make_shared<TensorCreatePayload>(TCSpec), 
            "outputX"
        )[0];
        auto resultY = builder.CreateOp(
            Opcode::OP_TENSOR_CREATE, 
            {}, 
            std::make_shared<TensorCreatePayload>(TCSpec), 
            "outputY"
        )[0];

        // for i = 0 to batch step 1
        auto i = builder.CreateScalar(DataType::INT32, "i");
        auto constant0 = builder.CreateConst(int64_t(0), "const_0");
        auto constant1 = builder.CreateConst(int64_t(1), "const_1");
        auto& fs = builder.CreateForStmt(i, constant0, batch, constant1);
        {
            auto fsGuard = builder.EnterForBody(fs);

            ViewSpec viewSpec;
            viewSpec.shape  = { 1, 128 };
            viewSpec.offset = {
                *i,
                *constant0
            };
            // loopX = view(inputX, {1, 128}, {i, 0})
            auto loopX = builder.CreateOp(
                Opcode::OP_VIEW,
                { inputX },
                std::make_shared<ViewPayload>(viewSpec),
                "loopX"
            )[0];
            // loopY = view(inputY, {1, 128}, {i, 0})
            auto loopY = builder.CreateOp(
                Opcode::OP_VIEW,
                { inputY },
                std::make_shared<ViewPayload>(viewSpec),
                "loopY"
            )[0];

            // outputX = mul(looX, scale1)
            auto resLoopX = builder.CreateOp(
                Opcode::OP_ADD, 
                {loopX, scale1},
                nullptr,
                "outputX"
            )[0];

            // outputY = mul(looY, scale2)
            auto resLoopY = builder.CreateOp(
                Opcode::OP_ADD, 
                {loopY, scale2},
                nullptr,
                "outputY"
            )[0];

            // if i then outputX = mul(outputX, scale1) else outputY = mul(outputY, scale2)
            auto& ifs = builder.CreateIfStmt("i");
            ValuePtr resIfX, resIfY;
            {
                auto ifThenGuard = builder.EnterIfThen(ifs);

                resIfX = builder.CreateOp(
                    Opcode::OP_MUL, 
                    {resLoopX, scale1},
                    nullptr,
                    "outputX"    
                )[0];
            }
            {
                auto ifElseGuard = builder.EnterIfElse(ifs);
                
                resIfY = builder.CreateOp(
                    Opcode::OP_MUL, 
                    {resLoopY, scale2},
                    nullptr,
                    "outputY"    
                )[0];
            }
            builder.ExitIfStatement(ifs);

            // check if then and else yield
            auto thenYield = std::dynamic_pointer_cast<YieldStatement>(*ifs.GetThenScope().GetStatements().rbegin());
            ASSERT_EQ(thenYield->Values()[0], resLoopY);
            ASSERT_EQ(thenYield->Values()[1], resIfX);
            auto elseYield = std::dynamic_pointer_cast<YieldStatement>(*ifs.GetElseScope().GetStatements().rbegin());
            ASSERT_EQ(elseYield->Values()[0], resIfY);
            ASSERT_EQ(elseYield->Values()[1], resLoopX);
        }
        builder.ExitForStatement(fs);

        // check for yeild
        auto ifs = std::dynamic_pointer_cast<IfStatement>(fs.GetScope().GetStatements()[1]);
        auto ifResults = ifs->Results();
        auto forYield = fs.Yield();
        ASSERT_EQ(forYield->Values(), ifResults);

        // return outputX, outputY
        builder.CreateReturn(fs.Results());
    }
    std::cout << module << std::endl;
}

} // namespace pto