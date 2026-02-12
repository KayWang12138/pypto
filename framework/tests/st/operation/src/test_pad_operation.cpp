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
 * \file test_pad_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct PadOpFuncArgs : public OpFuncArgs {
    PadOpFuncArgs(const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape) {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct PadOpMetaData {
    explicit PadOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void PadOperationExeFunc2Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const struct PadOpFuncArgs *args = static_cast<const PadOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {

                auto tileTensor = View(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                
                TileShape::Current().SetVecTile(args->tileShape_);
                
                auto res = Pad(tileTensor);
                
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

class PadOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<PadOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestPad, PadOperationTest,
    ::testing::ValuesIn(GetOpMetaData<PadOpMetaData>(
        {PadOperationExeFunc2Dims}, "Pad")));

TEST_P(PadOperationTest, TestPad) {
    auto test_data = GetParam().test_data_;
    auto args = PadOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<PadOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace