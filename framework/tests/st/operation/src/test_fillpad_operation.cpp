/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full language governing permissions and
 * limitations under the License.
 */

/*!
 * \file test_fillpad_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct FillPadOpFuncArgs : public OpFuncArgs {
    FillPadOpFuncArgs(const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape, float padValue)
        : viewShape_(viewShape), tileShape_(tileShape), padValue_(padValue) {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    float padValue_;
};

struct FillPadOpMetaData {
    explicit FillPadOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void FillPadOperationExeFunc1Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        const struct FillPadOpFuncArgs *args = static_cast<const FillPadOpFuncArgs *>(opArgs);
        
        // 直接使用输入 tensor 的实际 shape，忽略  view_shape 
        auto inputShape = inputs[0].GetShape();
        auto tileTensor = View(inputs[0], inputShape);
        TileShape::Current().SetVecTile(args->tileShape_);
        auto res = FillPad(tileTensor, "constant", args->padValue_);
        Assemble(res, {0}, outputs[0]);
    }
}

static void FillPadOperationExeFunc2Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        const struct FillPadOpFuncArgs *args = static_cast<const FillPadOpFuncArgs *>(opArgs);
        
        auto inputShape = inputs[0].GetShape();
        auto tileTensor = View(inputs[0], inputShape);
        TileShape::Current().SetVecTile(args->tileShape_);
        auto res = FillPad(tileTensor, "constant", args->padValue_);
        Assemble(res, {0, 0}, outputs[0]);
    }
}

class FillPadOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<FillPadOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestFillPad, FillPadOperationTest,
    ::testing::ValuesIn(GetOpMetaData<FillPadOpMetaData>(
        {FillPadOperationExeFunc2Dims, FillPadOperationExeFunc1Dims},
        "FillPad")));

TEST_P(FillPadOperationTest, TestFillPad) {
    auto test_data = GetParam().test_data_;
    std::string padValueType = GetValueByName<std::string>(test_data, "pad_value_type");
    float padValue = 0.0f;
    if (padValueType == "min") {
        padValue = -std::numeric_limits<float>::infinity();
    } else if (padValueType == "max") {
        padValue = std::numeric_limits<float>::infinity();
    }
    auto args = FillPadOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), padValue);
    auto testCase = CreateTestCaseDesc<FillPadOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
