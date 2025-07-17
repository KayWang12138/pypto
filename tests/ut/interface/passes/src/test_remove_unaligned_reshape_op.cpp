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
 * \file test_assign_memory_type_unalign.cpp
 * \brief Unit test for RemoveRedundentReshape pass.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"


using namespace npu::tile_fwk;

class TestRemoveUnalignedReshapeOp : public ::testing::Test {
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

TEST_F(TestRemoveUnalignedReshapeOp, test_Unalign_reshape) {
    std::vector<int> shape = {16, 16, 64, 1};
    std::vector<int> shapeOutput = {16, 16, 1, 64};

    Program::GetInstance().GetTileShape().SetVecTileShapes({8, 8, 16, 16});
    Tensor input_a(DT_FP32, shape, "A");
    Tensor input_b(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shapeOutput, "C");
    Function* currentFunction;

    FUNCTION("TENSOR_UNALIGN_RESHAPE_T", FunctionType::STATIC, {input_a, input_b, output}) {
        Tensor Tmp = Add(input_a, input_b);
        Tensor reshapeOutput = Reshape(Tmp, shapeOutput);
        output = Exp(reshapeOutput);
        currentFunction = Program::GetInstance().GetCurrentFunction();
    }

    std::vector<int> expiInShape = {8, 8, 16, 1};
    std::vector<int> expOutShape = {8, 8, 1, 16};

    for (auto &op : currentFunction->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto &in : op.iOperand) {
                if (in->oriShape == expiInShape) {
                    EXPECT_EQ(in->shape, expiInShape);
                    EXPECT_EQ(in->tensor->rawshape, expiInShape);
                    EXPECT_EQ(in->tensor->oriRawshape, expiInShape);
                }
            }

            for (auto &out : op.oOperand) {
                if (out->oriShape == expOutShape) {
                    EXPECT_EQ(out->shape, expOutShape);
                    EXPECT_EQ(out->tensor->rawshape, expOutShape);
                    EXPECT_EQ(out->tensor->oriRawshape, expOutShape);
                }
            }
        }
    }
}
