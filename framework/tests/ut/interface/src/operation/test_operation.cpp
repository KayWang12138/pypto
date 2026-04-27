/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_operation.cpp
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

class OperationOpsTest : public testing::Test {};

namespace {
void ExpectQuantMXScratchDtype(DataType inputDtype, const std::string& functionName)
{
    Tensor input(inputDtype, {8, 128});

    FUNCTION(functionName, {input})
    {
        auto res = QuantMX(input);
        (void)res;

        const Operation* quantOp = nullptr;
        auto* func = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(func, nullptr);
        for (const auto& op : func->Operations(false)) {
            if (op.GetOpcode() == Opcode::OP_QUANT_MX) {
                quantOp = &op;
                break;
            }
        }

        ASSERT_NE(quantOp, nullptr);
        ASSERT_EQ(quantOp->GetOOperands().size(), 4U);
        EXPECT_EQ(quantOp->GetOOperands()[2]->Datatype(), inputDtype);
        EXPECT_EQ(quantOp->GetOOperands()[3]->Datatype(), inputDtype);
    }
}
} // namespace

TEST_F(OperationOpsTest, CheckIndexAddParamsInvalid_FP16_Overflow)
{
    std::vector<int64_t> selfShape = {10, 10};
    std::vector<int64_t> srcShape = {5, 10};
    std::vector<int64_t> indicesShape = {5};
    int axis = 0;

    Tensor self(DT_FP16, selfShape);
    Tensor src(DT_FP16, srcShape);
    Tensor indices(DT_INT32, indicesShape);
    Element alpha(DT_FP16, 65505.0f);

    EXPECT_THROW(IndexAdd_(self, src, indices, axis, alpha), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedStartDataType)
{
    Element start(DT_INT8, 0);
    Element end(DT_INT32, 10);
    Element step(DT_INT32, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedEndDataType)
{
    Element start(DT_INT32, 0);
    Element end(DT_INT8, 10);
    Element step(DT_INT32, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedStepDataType)
{
    Element start(DT_INT32, 0);
    Element end(DT_INT32, 10);
    Element step(DT_INT8, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedOutputDataType)
{
    Element start(DT_INT64, 0);
    Element end(DT_INT64, INT64_MAX);
    Element step(DT_INT64, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, LogicalNot_UnsupportedDataType)
{
    std::vector<int64_t> shape = {10, 10};
    Tensor input(DT_INT32, shape);

    EXPECT_THROW(LogicalNot(input), std::exception);
}

TEST_F(OperationOpsTest, QuantMX_RoundDownExplicitFp8Output)
{
    Tensor explicitInput(DT_FP32, {8, 64}, "explicitInput");
    FUNCTION("QuantMXExplicitFp8", {explicitInput})
    {
        auto explicitFp8Res = QuantMX(explicitInput, DT_FP8E4M3, DequantScaleRoundingMode::ROUND_DOWN);
        EXPECT_EQ(std::get<0>(explicitFp8Res).GetDataType(), DT_FP8E4M3);
        EXPECT_EQ(std::get<1>(explicitFp8Res).GetDataType(), DT_FP8E8M0);
        EXPECT_EQ(std::get<0>(explicitFp8Res).GetShape(), std::vector<int64_t>({8, 64}));
        EXPECT_EQ(std::get<1>(explicitFp8Res).GetShape(), std::vector<int64_t>({8, 1, 2}));
    }
}

TEST_F(OperationOpsTest, QuantMX_DefaultRoundDownFp8Output)
{
    Tensor input(DT_FP32, {8, 64});

    FUNCTION("QuantMXDefaultFp8", {input})
    {
        auto defaultRes = QuantMX(input);
        EXPECT_EQ(std::get<0>(defaultRes).GetDataType(), DT_FP8E4M3);
        EXPECT_EQ(std::get<1>(defaultRes).GetDataType(), DT_FP8E8M0);
        EXPECT_EQ(std::get<1>(defaultRes).GetShape(), std::vector<int64_t>({8, 1, 2}));
    }
}

TEST_F(OperationOpsTest, QuantMX_PerformanceModeKeepsPublicScaleShape)
{
    Tensor input(DT_FP32, {2, 8, 128});

    FUNCTION("QuantMXPerformanceMode", {input})
    {
        auto res = QuantMX(input, DT_FP8E4M3, DequantScaleRoundingMode::ROUND_DOWN, -1, true);
        EXPECT_EQ(std::get<0>(res).GetShape(), std::vector<int64_t>({2, 8, 128}));
        EXPECT_EQ(std::get<1>(res).GetShape(), std::vector<int64_t>({2, 8, 2, 2}));

        const Operation* quantOp = nullptr;
        auto* func = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(func, nullptr);
        for (const auto& op : func->Operations(false)) {
            if (op.GetOpcode() == Opcode::OP_QUANT_MX) {
                quantOp = &op;
                break;
            }
        }

        ASSERT_NE(quantOp, nullptr);
        ASSERT_EQ(quantOp->GetOOperands().size(), 4U);
        EXPECT_EQ(quantOp->GetOOperands()[1]->GetShape(), std::vector<int64_t>({2, 32}));
        EXPECT_EQ(quantOp->GetOOperands()[2]->GetShape(), std::vector<int64_t>({2, 32}));
    }
}

TEST_F(OperationOpsTest, QuantMX_Fp16ScratchDtypeFollowsInput)
{
    ExpectQuantMXScratchDtype(DT_FP16, "QuantMXScratchDtypeFp16");
}

TEST_F(OperationOpsTest, QuantMX_Bf16ScratchDtypeFollowsInput)
{
    ExpectQuantMXScratchDtype(DT_BF16, "QuantMXScratchDtypeBf16");
}

TEST_F(OperationOpsTest, QuantMX_PositiveLastAxis)
{
    Tensor input(DT_FP32, {8, 64});

    FUNCTION("QuantMXPositiveLastAxis", {input})
    {
        auto res = QuantMX(input, DT_FP8E4M3, DequantScaleRoundingMode::ROUND_DOWN, 1);
        EXPECT_EQ(std::get<1>(res).GetShape(), std::vector<int64_t>({8, 1, 2}));
    }
}

TEST_F(OperationOpsTest, QuantMX_NonLastAxisUnsupported)
{
    Tensor input(DT_FP32, {8, 64});

    EXPECT_THROW(QuantMX(input, DT_FP8E4M3, DequantScaleRoundingMode::ROUND_DOWN, 0), std::exception);
}

TEST_F(OperationOpsTest, QuantMX_RoundUpUnsupported)
{
    Tensor input(DT_FP32, {8, 64});

    EXPECT_THROW(QuantMX(input, DT_FP8E4M3, DequantScaleRoundingMode::ROUND_UP), std::exception);
}

TEST_F(OperationOpsTest, QuantMX_Fp4OutputUnsupported)
{
    Tensor input(DT_FP32, {8, 64});

    EXPECT_THROW(QuantMX(input, DT_FP4_E2M1X2), std::exception);
    EXPECT_THROW(QuantMX(input, DT_FP4_E1M2X2), std::exception);
}
