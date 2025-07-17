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
 * \file test_pad_local_buffer.cpp
 * \brief Unit test for RemoveRedundentReshape pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class TestPadLocalBuffer : public ::testing::Test {
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

TEST_F(TestPadLocalBuffer, test_add_last_axis_1) {
    std::vector<int> shape = {16, 16, 64, 65};

    Program::GetInstance().GetTileShape().SetVecTileShapes({8, 8, 16, 16});
    Tensor input_a(DT_FP32, shape, "A");
    Tensor input_b(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    FUNCTION("ADD_T", FunctionType::STATIC, {input_a, input_b, output}) {
        output = Add(input_a, input_b);
    }

    auto function= Program::GetInstance().GetFunctionByRawName("TENSOR_ADD_T");
    ASSERT_NE(function, nullptr);
    std::vector<int> expShape = {8, 8, 16, 8};
    std::vector<int> expOriShape = {8, 8, 16, 1};

    for (auto &op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ADD) {
            for (auto &in : op.iOperand) {
                if (in->oriShape == expOriShape) {
                    EXPECT_EQ(in->shape, expShape);
                    EXPECT_EQ(in->tensor->rawshape, expShape);
                    EXPECT_EQ(in->tensor->oriRawshape, expOriShape);
                }
            }

            for (auto &out : op.oOperand) {
                if (out->oriShape == expOriShape) {
                    EXPECT_EQ(out->shape, expShape);
                    EXPECT_EQ(out->tensor->rawshape, expShape);
                    EXPECT_EQ(out->tensor->oriRawshape, expOriShape);
                }
            }
        }
    }
}

TEST_F(TestPadLocalBuffer, test_matmul) {
    std::vector<int> shape = {32, 32};

    Program::GetInstance().GetTileShape().SetVecTileShapes({8, 8});
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {16, 16}, {16, 16});
    Tensor input_a(DT_FP32, shape, "A");
    Tensor input_b(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    FUNCTION("A_MUL_Bt", FunctionType::STATIC, {input_a, input_b, output}) {
        output = Matrix::Matmul<false, true>(DataType::DT_FP32, input_a, input_b);
    }

    auto function= Program::GetInstance().GetFunctionByRawName("TENSOR_A_MUL_Bt");
    ASSERT_NE(function, nullptr);
}

TEST_F(TestPadLocalBuffer, test_operation_row_max_single_4dim_softmax_unalign) {
    // softmax rowmax: [B,N,1,S2]
    int shape0 = 1;
    int shape1 = 128;
    int shape2 = 1;
    int shape3 = 247;
    std::vector<int> shape = {shape0, shape1, shape2, shape3};
    std::vector<int> outshape = {shape0, shape1, shape2, 1};

    PROGRAM("RowMaxSingle") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 64, 1, 64});

        Tensor input_a(DT_FP32, shape, "A");
        Tensor output(DT_FP32, outshape, "C");

        FUNCTION("RowMaxSingle", FunctionType::STATIC, {input_a, output}) {
            output = RowMaxSingle(input_a, -1);
        }
    }

    auto function= Program::GetInstance().GetFunctionByRawName("TENSOR_RowMaxSingle");
    ASSERT_NE(function, nullptr);
    int count = 0;
    std::vector<int> expShape = {1, 64, 1, 64};
    std::vector<int> expOriShape = {1, 64, 1, 55};

    for (auto &op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_PAIRMAX) {
            for (auto &in : op.iOperand) {
                if (in->oriShape == expOriShape) {
                    EXPECT_EQ(in->shape, expShape);
                    EXPECT_EQ(in->tensor->rawshape, expShape);
                    EXPECT_EQ(in->tensor->oriRawshape, expOriShape);
                    count++;
                }
            }
        }
    }
    EXPECT_EQ(count, 2);
}

TEST_F(TestPadLocalBuffer, test_operation_row_max_single_4dim_softmax_unalign_2) {
    // softmax rowmax: [B,N,1,S2]
    int shape0 = 1;
    int shape1 = 128;
    int shape2 = 1;
    int shape3 = 199;
    std::vector<int> shape = {shape0, shape1, shape2, shape3};
    std::vector<int> outshape = {shape0, shape1, shape2, 1};

    PROGRAM("RowMaxSingle") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 64, 1, 64});

        Tensor input_a(DT_FP32, shape, "A");
        Tensor output(DT_FP32, outshape, "C");

        FUNCTION("RowMaxSingle", FunctionType::STATIC, {input_a, output}) {
            output = RowMaxSingle(input_a, -1);
        }
    }

    auto function= Program::GetInstance().GetFunctionByRawName("TENSOR_RowMaxSingle");
    ASSERT_NE(function, nullptr);
    int count = 0;
    std::vector<int> expShape = {1, 64, 1, 8};
    std::vector<int> expOriShape = {1, 64, 1, 7};

    for (auto &op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_PAIRMAX) {
            for (auto &in : op.iOperand) {
                if (in->oriShape == expOriShape) {
                    EXPECT_EQ(in->shape, expShape);
                    EXPECT_EQ(in->tensor->rawshape, expShape);
                    EXPECT_EQ(in->tensor->oriRawshape, expOriShape);
                    count++;
                }
            }
        }
    }
    EXPECT_EQ(count, 2);
}
