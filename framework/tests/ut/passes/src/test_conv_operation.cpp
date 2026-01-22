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
 * \file test_conv_operation.cpp
 * \brief Unit test for Conv2D operation.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tensor_graph_pass/expand_function.h"

namespace npu::tile_fwk {

class TestConvOperation : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetHostConfig(KEY_STRATEGY, "ConvOperationTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    
    void TearDown() override {}
};

TEST_F(TestConvOperation, BasicConv2DTest) {
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ConvOperationTestStrategy", {});

    // fmap: 5D tensor [N, C, H, W, C0]
    // weight: 4D tensor [Co, Ci, Kh, Kw]
    constexpr int64_t N = 1;
    constexpr int64_t C = 64;
    constexpr int64_t H = 56;
    constexpr int64_t W = 56;
    constexpr int64_t C0 = 16;
    constexpr int64_t Co = 64;
    constexpr int64_t Ci = 64;
    constexpr int64_t Kh = 3;
    constexpr int64_t Kw = 3;

    std::vector<int64_t> fmapShape{N, C / C0, H, W, C0};
    std::vector<int64_t> weightShape{Co, Ci, Kh, Kw};
    std::vector<int64_t> outputShape{N, Co / C0, H, W, C0};

    Tensor fmap(DT_FP16, fmapShape, "fmap");
    Tensor weight(DT_FP16, weightShape, "weight");
    Tensor output(DT_FP16, outputShape, "output");

    // Set tile shape for conv operation
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    constexpr int TILE_K = 16;
    TileShape::Current().SetCubeTile({{TILE_M, TILE_M * 4}, {TILE_K, TILE_K * 4, TILE_K * 4}, {TILE_N, TILE_N * 4}});

    FUNCTION("TestConv2D") {
        output = npu::tile_fwk::Conv::Conv2D(DT_FP16, fmap, weight);
    }

    Function* currentFunction = Program::GetInstance().GetCurrentFunction();
    ASSERT_NE(currentFunction, nullptr);

    auto opListBefore = currentFunction->Operations().DuplicatedOpList();
    int convNumBefore = 0;
    for (auto &op : opListBefore) {
        if (op->GetOpcode() == Opcode::OP_CONV) {
            convNumBefore++;
        }
    }

    ExpandFunction expandFunction;
    ASSERT_EQ(expandFunction.RunOnFunction(*currentFunction), SUCCESS);

    auto opListAfter = currentFunction->Operations().DuplicatedOpList();
    int viewNumAfter = 0;
    for (auto &op : opListAfter) {
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            // Check if this VIEW operation has conv marker attribute
            if (op->HasAttribute("op_attr_is_conv_op")) {
                bool isConvOp = op->GetBoolAttribute("op_attr_is_conv_op");
                EXPECT_TRUE(isConvOp);
                viewNumAfter++;
            }
        }
    }

    EXPECT_EQ(convNumBefore, 1);
    EXPECT_GT(viewNumAfter, 0) << "Should have VIEW operations with conv marker after expansion";
}

TEST_F(TestConvOperation, ConvWithDifferentShapesTest) {
    // Test with different input shapes
    constexpr int64_t N = 2;
    constexpr int64_t C = 32;
    constexpr int64_t H = 28;
    constexpr int64_t W = 28;
    constexpr int64_t C0 = 16;
    constexpr int64_t Co = 128;
    constexpr int64_t Ci = 32;
    constexpr int64_t Kh = 5;
    constexpr int64_t Kw = 5;

    std::vector<int64_t> fmapShape{N, C / C0, H, W, C0};
    std::vector<int64_t> weightShape{Co, Ci, Kh, Kw};
    
    Tensor fmap(DT_FP32, fmapShape, "fmap");
    Tensor weight(DT_FP32, weightShape, "weight");
    Tensor output;

    constexpr int TILE_SIZE = 32;
    TileShape::Current().SetCubeTile({{TILE_SIZE, TILE_SIZE * 2}, {TILE_SIZE, TILE_SIZE * 2, TILE_SIZE * 2}, 
                                       {TILE_SIZE, TILE_SIZE * 2}});

    FUNCTION("TestConv2DShapes") {
        output = npu::tile_fwk::Conv::Conv2D(DT_FP32, fmap, weight);
    }

    Function* currentFunction = Program::GetInstance().GetCurrentFunction();
    ASSERT_NE(currentFunction, nullptr);

    ExpandFunction expandFunction;
    ASSERT_EQ(expandFunction.RunOnFunction(*currentFunction), SUCCESS);

    // Verify that operations are created
    auto opListAfter = currentFunction->Operations().DuplicatedOpList();
    EXPECT_GT(opListAfter.size(), 0);
}

} // namespace npu::tile_fwk
