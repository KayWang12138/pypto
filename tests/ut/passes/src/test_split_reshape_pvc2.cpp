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
 * \file test_split_reshape_pvc2.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class TestSplitReshapeOpPVC2 : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {
    }
};

TEST_F(TestSplitReshapeOpPVC2, test_reshape_assemble_multito1) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(8, 8, 8, 8);
    Function *currentFunction;
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", "USE_MAX_FREQ_LABEL", true);

    Tensor input(DT_FP32, {1, 384}, "a");
    Tensor res1;

    FUNCTION("test_reshape_assemble_multito1") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(1, 64);
        Tensor res = Exp(input);
        Tensor test = Reshape(res, {1, 1, 2, 192});
        Program::GetInstance().GetTileShape().SetVecTileShapes(2, 1, 2, 64);
        res1 = Exp(test);
        currentFunction = Program::GetInstance().GetCurrentFunction();
    }

    std::vector<int64_t> expiInShape = {1, 64};
    std::vector<int64_t> expOutShape = {1, 1, 1, 64};
    for (auto &op : currentFunction->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto &in : op.iOperand) {
                EXPECT_EQ(in->shape, expiInShape);
            }

            for (auto &out : op.oOperand) {
                EXPECT_EQ(out->shape, expOutShape);
            }
        }
    }
}

TEST_F(TestSplitReshapeOpPVC2, Test_Reshape_1to1) {
    Function *currentFunction;

    Program::GetInstance().GetTileShape().SetVecTileShapes(8, 8, 8, 8);
    Tensor input(DT_FP32, {8, 16, 16}, "a");
    Tensor res1;

    FUNCTION("Test_Reshape_1to1") {
        Tensor res = Exp(input);
        Tensor test = Reshape(res, {8, 16, 1, 16});
        res1 = Exp(test);
        currentFunction = Program::GetInstance().GetCurrentFunction();
    }

    std::vector<int64_t> expiInShape = {8,  8,  8};
    std::vector<int64_t> expOutShape = {8, 8, 1, 8};
    for (auto &op : currentFunction->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto &in : op.iOperand) {
                EXPECT_EQ(in->shape, expiInShape);
            }

            for (auto &out : op.oOperand) {
                EXPECT_EQ(out->shape, expOutShape);
            }
        }
    }
}

TEST_F(TestSplitReshapeOpPVC2, Test_Reshape_1toMulti) {
    Function *currentFunction;

    Program::GetInstance().GetTileShape().SetVecTileShapes(8, 8, 8, 8, 8);
    Tensor input(DT_FP32, {16, 4, 4}, "a");
    Tensor res1;

    FUNCTION("Test_Reshape_1toMulti") {
        auto res = Exp(input);
        auto test = Reshape(res, {16, 16});
        res1 = Exp(test);
        currentFunction = Program::GetInstance().GetCurrentFunction();
    }

    std::vector<int64_t> expiInShape = {8, 4, 4};
    std::vector<int64_t> expOutShape = {8, 16};
    for (auto &op : currentFunction->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto &in : op.iOperand) {
                EXPECT_EQ(in->shape, expiInShape);
            }

            for (auto &out : op.oOperand) {
                EXPECT_EQ(out->shape, expOutShape);
            }
        }
    }
}

TEST_F(TestSplitReshapeOpPVC2, Test_Reshape_multito1) {
    Function *currentFunction;

    Program::GetInstance().GetTileShape().SetVecTileShapes(8, 8, 8, 8);
    Tensor input(DT_FP32, {8, 16, 16}, "a");
    Tensor res1;

    FUNCTION("Test_Reshape_multito1") {
        auto res = Exp(input);
        auto test = Reshape(res, {8, 16, 4, 4});
        res1 = Exp(test);
        currentFunction = Program::GetInstance().GetCurrentFunction();
    }

    std::vector<int64_t> expiInShape = {8, 8, 16};
    std::vector<int64_t> expOutShape = {8, 8, 4, 4};
    for (auto &op : currentFunction->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto &in : op.iOperand) {
                EXPECT_EQ(in->shape, expiInShape);
            }

            for (auto &out : op.oOperand) {
                EXPECT_EQ(out->shape, expOutShape);
            }
        }
    }
}
