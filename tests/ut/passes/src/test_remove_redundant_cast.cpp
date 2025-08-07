/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_remove_redundant_cast.cpp
 * \brief Unit test for Remove Redundant Cast.
 */

#include <fstream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "passes/pass_registry.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/tensor_graph_pass/remove_redundant_cast.h"

namespace npu {
namespace tile_fwk {

class RemoveRedundantCastTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    }
    void TearDown() override {}
};

TEST_F(RemoveRedundantCastTest, AddBF16) {
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
    RemoveRedundantCast RRC;
    RRC.RunOnFunction(*function);
    const int opNum3 = 3;
    EXPECT_EQ(function->Operations().size(), opNum3);
}

TEST_F(RemoveRedundantCastTest, AddCascadeBF16) {
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
    RemoveRedundantCast RRC;
    RRC.RunOnFunction(*function);
    const int opNum4 = 4;
    EXPECT_EQ(function->Operations().size(), opNum4);
}

TEST_F(RemoveRedundantCastTest, CastCascade) {
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t1"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_INT32, {16, 16}, "t2"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP16, {16, 16}, "t3"), true);
    std::vector<Opcode> opCodes{Opcode::OP_CAST, Opcode::OP_CAST};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}};
    std::vector<std::string> opNames{"CAST1", "CAST2"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t3"}), true);
    Function *function = G.GetFunction();
    const int opNum2 = 2;
    EXPECT_EQ(function->Operations().size(), opNum2);
    RemoveRedundantCast RRC;
    RRC.RunOnFunction(*function);
    const int opNum1 = 1;
    EXPECT_EQ(function->Operations().size(), opNum1);
}

TEST_F(RemoveRedundantCastTest, CastChainFromInCast) {
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t1"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t2"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t3"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t4"), true);
    std::vector<Opcode> opCodes{Opcode::OP_CAST, Opcode::OP_CAST, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"CAST1", "CAST2", "COPY_OUT"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Function *function = G.GetFunction();
    const int opNum3 = 3;
    EXPECT_EQ(function->Operations().size(), opNum3);
    RemoveRedundantCast RRC;
    RRC.RunOnFunction(*function);
    const int opNum1 = 1;
    EXPECT_EQ(function->Operations().size(), opNum1);
}

TEST_F(RemoveRedundantCastTest, CastChainToOutCast) {
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t1"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t2"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t3"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "tend"), true);
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_CAST, Opcode::OP_CAST};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"tend"}};
    std::vector<std::string> opNames{"COPY_IN", "CAST1", "CAST2"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"tend"}), true);
    Function *function = G.GetFunction();
    const int opNum3 = 3;
    EXPECT_EQ(function->Operations().size(), opNum3);
    RemoveRedundantCast RRC;
    RRC.RunOnFunction(*function);
    const int opNum1 = 1;
    EXPECT_EQ(function->Operations().size(), opNum1);
}

TEST_F(RemoveRedundantCastTest, TestDeepGraph) {
    ComputationalGraphBuilder G;
    const int graphDepth = 5000;
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t0"), true);
    for (int i = 1; i < graphDepth; i++) {
        std::string lastSid = std::to_string(i-1);
        std::string sid = std::to_string(i);
        EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t" + sid), true);
        EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"t" + lastSid}, {"t" + sid}, "cast" + sid, true), true);
    }
    EXPECT_EQ(G.SetInCast({"t0"}), true);
    EXPECT_EQ(G.SetOutCast({"t" + std::to_string(graphDepth - 1)}), true);
    Function *function = G.GetFunction();
    const int opNumBefore = graphDepth - 1;
    EXPECT_EQ(function->Operations().size(), opNumBefore);
    RemoveRedundantCast RRC;
    RRC.RunOnFunction(*function);
    const int opNumAfter = 1;
    EXPECT_EQ(function->Operations().size(), opNumAfter);
}

TEST_F(RemoveRedundantCastTest, TestWideGraph) {
    ComputationalGraphBuilder G;
    const int graphWidth = 5000;
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "tbegin"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "tend"), true);
    for (int i = 0; i < graphWidth; i++) {
        std::string lastSid = std::to_string(i-1);
        std::string sid = std::to_string(i);
        EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t" + sid), true);
        EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"tbegin"}, {"t" + sid}, "cast" + sid, true), true);
        EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"t" + sid}, {"tend"}, "copy_out" + sid, true), true);
    }
    EXPECT_EQ(G.SetInCast({"tbegin"}), true);
    EXPECT_EQ(G.SetOutCast({"tend"}), true);
    Function *function = G.GetFunction();
    const int opNumBefore = graphWidth * 2;
    EXPECT_EQ(function->Operations().size(), opNumBefore);
    RemoveRedundantCast RRC;
    RRC.RunOnFunction(*function);
    const int opNumAfter = graphWidth;
    EXPECT_EQ(function->Operations().size(), opNumAfter);
}
} // namespace tile_fwk
} // namespace npu