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
 * \file test_gcd_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct GcdOpFuncArgs : public OpFuncArgs {
    GcdOpFuncArgs(const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape) {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct GcdOpMetaData {
    explicit GcdOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};




static void GcdOperationExeFuncDoubleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    auto args = static_cast<const GcdOpFuncArgs *>(opArgs);

    FUNCTION("main", {inputs[0]}, {outputs[0]}) {

        const int bloop = CeilDiv(noAxisDims[0], noAxisViewShapes[0]);
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            std::vector<SymbolicScalar> indices = {bIdx};
            indices.insert(indices.begin() + axis, 0);
            const std::vector<int64_t> tensorViewShape = GetGcdViewShape(inputs[0], args->viewShape_, axis);

            SymbolicScalar validShape0 = std::min(
                SymbolicScalar(inputs[0].GetShape()[0]) - indices[0] * tensorViewShape[0], tensorViewShape[0]);
            SymbolicScalar validShape1 = std::min(
                SymbolicScalar(inputs[0].GetShape()[1]) - indices[1] * tensorViewShape[1], tensorViewShape[1]);
            auto tileTensor = View(inputs[0], tensorViewShape, {validShape0, validShape1},
                {indices[0] * tensorViewShape[0], indices[1] * tensorViewShape[1]});

            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = Gcd(tileTensor, );
            Assemble(res, {indices[0] * args->viewShape_[0], indices[1] * args->viewShape_[1]}, outputs[0]);
        }
    }
}



class GcdOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<GcdOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestGcd, GcdOperationTest,
    ::testing::ValuesIn(GetOpMetaData<GcdOpMetaData>(
        {GcdOperationExeFuncDoubleCut},
        "Gcd")));

TEST_P(GcdOperationTest, TestGcd) {
    auto test_data = GetParam().test_data_;
    auto args = GcdOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<TriOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace