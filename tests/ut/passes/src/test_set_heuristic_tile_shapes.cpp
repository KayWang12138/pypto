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
 * \file test_set_heuristic_tile_shapes.cpp
 * \brief Unit test for SetHeuristicTileShapes.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tensor_graph_pass/set_heuristic_tile_shapes.h"
#include "computational_graph_builder.h"
#include "ut_json/ut_json_tool.h"

namespace npu::tile_fwk {
const std::string ACC_A_MUL_B = OP_ATTR_PREFIX + "atomic_add";
const std::string MATMUL_NZ_ATTR = OP_ATTR_PREFIX + "matmul_nz_attr";
const std::string A_MUL_B_ACT_M = OP_ATTR_PREFIX + "act_m";
const std::string A_MUL_B_ACT_K = OP_ATTR_PREFIX + "act_k";
const std::string A_MUL_B_ACT_N = OP_ATTR_PREFIX + "act_n";

class TestSetHeuristicTileShapes : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "SetHeuristicTileShapesTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}

    void SetMatMulAttr(ComputationalGraphBuilder &G, const std::string name,
        bool isAtomic = false, const int nzFormat = 0) {
        auto op = G.GetOp(name);
        if (op == nullptr) {
            return;
        }
        if (isAtomic) {
            op->SetAttribute(ACC_A_MUL_B, 1);
        } else {
            op->SetAttribute(ACC_A_MUL_B, 0);
        }
        op->SetAttribute(MATMUL_NZ_ATTR, nzFormat);
        op->SetAttribute(A_MUL_B_ACT_M, 0);
        op->SetAttribute(A_MUL_B_ACT_K, 0);
        op->SetAttribute(A_MUL_B_ACT_N, 0);
    }
};

TEST_F(TestSetHeuristicTileShapes, TestMain) {
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddParams", "TestAddParams", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    // Prepare the graph
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c");
    auto mat_c = G.GetTensor("mat_c");
    mat_c->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a");
    auto l1_a = G.GetTensor("l1_a");
    l1_a->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_a");
    auto l0_a = G.GetTensor("l0_a");
    l0_a->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b");
    auto l1_b = G.GetTensor("l1_b");
    l1_b->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l0_b");
    auto l0_b = G.GetTensor("l0_b");
    l0_b->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c");
    auto l0_c = G.GetTensor("l0_c");
    l0_c->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a"}, {"l0_a"}, "L1_To_L0A");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b"}, {"l0_b"}, "L1_To_L0B");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a", "l0_b"}, {"l0_c"}, "A_MUL_B");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c"}, {"mat_c"}, "L0C_Copy_out");
    SetMatMulAttr(G, "A_MUL_B", false, 1);
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c"});

    SetHeuristicTileShapes setHeuristicTileShapes;
    setHeuristicTileShapes.RunOnFunction(*rootFuncPtr);
    EXPECT_TRUE(true);
}

} // namespace acend
