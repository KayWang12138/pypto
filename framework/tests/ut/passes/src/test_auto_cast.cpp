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
 * \file test_auto_cast.cpp
 * \brief Unit test for Auto Cast.
 */

#include <fstream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/tensor_graph_pass/auto_cast.h"

namespace npu {
namespace tile_fwk {

class AutoCastTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }
    void TearDown() override {}
};

TEST_F(AutoCastTest, AddBF16) {
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t1"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t2"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t3"), true);
    std::vector<Opcode> opCodes{Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}};
    std::vector<std::string> opNames{"ADD"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t3"}), true);
    Function *function = G.GetFunction();
    EXPECT_EQ(function->Operations().size(), 1);
    AutoCast autoCast;
    autoCast.RunOnFunction(*function);
    const int opNum3 = 3;
    EXPECT_EQ(function->Operations().size(), opNum3);
}

TEST_F(AutoCastTest, AddCascadeBF16) {
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t1"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t2"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t3"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t4"), true);
    std::vector<Opcode> opCodes{Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ADD1", "ADD2"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Function *function = G.GetFunction();
    const int opNum2 = 2;
    EXPECT_EQ(function->Operations().size(), opNum2);
    AutoCast autoCast;
    autoCast.RunOnFunction(*function);
    const int opNum5 = 5;
    EXPECT_EQ(function->Operations().size(), opNum5);
}

TEST_F(AutoCastTest, Int32ToFP16Cast) {
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensor(DataType::DT_INT32, {16, 16}, "t1"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, {16, 16}, "t2"), true);
    std::vector<Opcode> opCodes{Opcode::OP_CAST};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"Cast"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t2"}), true);
    Function *function = G.GetFunction();
    AutoCast autoCast;
    autoCast.RunOnFunction(*function);
    const int opNum2 = 2;
    EXPECT_EQ(function->Operations().size(), opNum2);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(AutoCastTest, AutoCastChecker_PostCheck_Normal) {
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t1"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t2"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t3"), true);
    std::vector<Opcode> opCodes{Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}};
    std::vector<std::string> opNames{"ADD"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = G.GetFunction();
    ASSERT_NE(function, nullptr); // 确保function有效

    AutoCast autoCast;
    EXPECT_EQ(autoCast.PostCheck(*function), FAILED); 
    autoCast.RunOnFunction(*function);
}

TEST_F(AutoCastTest, AutoCastChecker_Cast_InvalidOutputNum_BF16_Unsupported) {
    // 场景1：CAST输出数≠1（独立测试，确保图构建合法）
    ComputationalGraphBuilder G1;
    EXPECT_EQ(G1.AddTensor(DataType::DT_INT32, {16, 16}, "t1"), true);
    EXPECT_EQ(G1.AddTensor(DataType::DT_FP16, {16, 16}, "t2"), true);
    EXPECT_EQ(G1.AddTensor(DataType::DT_FP16, {16, 16}, "t3"), true);
    std::vector<Opcode> opCodes1{Opcode::OP_CAST};
    std::vector<std::vector<std::string>> ioperands1{{"t1"}};
    std::vector<std::vector<std::string>> ooperands1{{"t2", "t3"}}; // 2个输出（非法）
    std::vector<std::string> opNames1{"Cast"};
    EXPECT_EQ(G1.AddOps(opCodes1, ioperands1, ooperands1, opNames1, true), true);
    Function *function1 = G1.GetFunction();
    ASSERT_NE(function1, nullptr);

    AutoCast autoCast1;
    EXPECT_EQ(autoCast1.PreCheck(*function1), FAILED); // 预期返回FAILED（1）
    autoCast1.RunOnFunction(*function1);

    // 场景2：不支持BF16输入（独立测试，先添加输出张量t3）
    ComputationalGraphBuilder G2;
    EXPECT_EQ(G2.AddTensor(DataType::DT_BF16, {16, 16}, "t1"), true);
    EXPECT_EQ(G2.AddTensor(DataType::DT_FP32, {16, 16}, "t2"), true);
    EXPECT_EQ(G2.AddTensor(DataType::DT_FP32, {16, 16}, "t3"), true); 
    std::vector<Opcode> opCodes2{Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands2{{"t1", "t2"}};
    std::vector<std::vector<std::string>> ooperands2{{"t3"}};
    std::vector<std::string> opNames2{"MUL"};
    EXPECT_EQ(G2.AddOps(opCodes2, ioperands2, ooperands2, opNames2, true), true); 
    Function *function2 = G2.GetFunction();
    ASSERT_NE(function2, nullptr);

    AutoCast autoCast2;
    EXPECT_EQ(autoCast2.PostCheck(*function2), FAILED); 
    autoCast2.RunOnFunction(*function2);

    // 场景3：不支持BF16输出（独立测试，确保图构建合法）
    ComputationalGraphBuilder G3;
    EXPECT_EQ(G3.AddTensor(DataType::DT_FP32, {16, 16}, "t1"), true);
    EXPECT_EQ(G3.AddTensor(DataType::DT_BF16, {16, 16}, "t2"), true);
    std::vector<Opcode> opCodes3{Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands3{{"t1"}};
    std::vector<std::vector<std::string>> ooperands3{{"t2"}};
    std::vector<std::string> opNames3{"MUL"};
    EXPECT_EQ(G3.AddOps(opCodes3, ioperands3, ooperands3, opNames3, true), true);
    Function *function3 = G3.GetFunction();
    ASSERT_NE(function3, nullptr);

    AutoCast autoCast3;
    EXPECT_EQ(autoCast3.PostCheck(*function3), FAILED);
    autoCast3.RunOnFunction(*function3);
}

TEST_F(AutoCastTest, AutoCastChecker_PreCheck_Normal) {
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensor(DataType::DT_INT32, {16, 16}, "t1"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, {16, 16}, "t2"), true);
    std::vector<Opcode> opCodes{Opcode::OP_CAST};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"Cast"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = G.GetFunction();

    AutoCast autoCast;
    EXPECT_EQ(autoCast.PreCheck(*function), SUCCESS);
    autoCast.RunOnFunction(*function);
}

TEST_F(AutoCastTest, AutoCastChecker_Cast_InvalidInputNum) {
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensor(DataType::DT_INT32, {16, 16}, "t1"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_INT32, {16, 16}, "t2"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, {16, 16}, "t3"), true);
    std::vector<Opcode> opCodes{Opcode::OP_CAST};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}};
    std::vector<std::string> opNames{"Cast"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = G.GetFunction();

    AutoCast autoCast;
    EXPECT_EQ(autoCast.PreCheck(*function), FAILED);
    autoCast.RunOnFunction(*function);
}
} // namespace tile_fwk
} // namespace npu