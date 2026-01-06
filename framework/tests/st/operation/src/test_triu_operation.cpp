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
 * \file test_Triu_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct TriuOpFuncArgs : public OpFuncArgs {
    TriuOpFuncArgs(int diagonal, const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape, bool flag)
        : diagonal_(diagonal), viewShape_(viewShape), tileShape_(tileShape), flag_(flag) {}

    int diagonal_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    bool flag_;
};

struct TriuOpMetaData {
    explicit TriuOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static std::vector<int64_t> GetTriuViewShape(Tensor tensor, const std::vector<int64_t> &viewShape) {
    std::vector<int64_t> resultViewShape = {};
    for (size_t i = 0; i < viewShape.size(); i++) {
        resultViewShape.push_back(std::min(viewShape[i], tensor.GetShape()[i]));
    }
    return resultViewShape;
}

/*
 * 统一成二维情况
 * (originXIdx, originYIdx) = (colOffset, -rowOffset)
 */
static std::int GetRealDiagonal(int diagonal, const std::vector<int64_t> &viewShape, int originXIdx, int originYIdx) {
    int realDiagonal = 0;
    int leftCornerYIdx = originYIdx - viewShape[-2];
    int leftCornerXIdx = originXIdx;
    int leftResult = 0 - leftCornerXIdx + diagonal;
    int rightCornerYIdx = originYIdx;
    int rightCornerXIdx = originXIdx + viewShape[-1];
    int rightResult = 0 - rightCornerXIdx + diagonal;

    if (leftResult <= 0 && rightResult <= 0) {
        realDiagonal = std::max(viewShape[-2], viewShape[-1]) + 1;
    } else if (leftResult >= 0 && rightResult >= 0) {
        realDiagonal = 0 - (std::max(viewShape[-2], viewShape[-1]) + 1);
    } else {
        realDiagonal = diagonal - originXIdx - originYIdx;
    }

    return realDiagonal;
}

static void TriuOperationExeFuncDoubleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    auto args = static_cast<const TriuOpFuncArgs *>(opArgs);

    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        const std::vector<int64_t> realViewShape = GetTriuViewShape(inputs[0], args->viewShape_);
        const int bloop = CeilDiv(inputs[0].GetShape()[0], realViewShape[0]);
        const int sloop = CeilDiv(inputs[0].GetShape()[1], realViewShape[1]);
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                SymbolicScalar validShape0 =
                    std::min(SymbolicScalar(inputs[0].GetShape()[0]) - bIdx * realViewShape[0], realViewShape[0]);
                SymbolicScalar validShape1 =
                    std::min(SymbolicScalar(inputs[0].GetShape()[1]) - sIdx * realViewShape[1], realViewShape[1]);
                auto tileTensor = View(inputs[0], realViewShape, {validShape0, validShape1},
                    {bIdx * realViewShape[0], sIdx * realViewShape[1]});

                int originXIdx = sIdx * realViewShape[1];
                int originYIdx = 0 - bIdx * realViewShape[0];
                int realDiagonal = GetRealDiagonal(args->diagonal_, realViewShape, originXIdx, originYIdx);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Triu(tileTensor, realDiagonal);
                Assemble(res, {bIdx * realViewShape[0], sIdx * realViewShape[1]}, outputs[0]);
            }
        }
    }
}

class TriuOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<TriuOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestTriu, TriuOperationTest,
    ::testing::ValuesIn(GetOpMetaData<TriuOpMetaData>({TriuOperationExeFuncDoubleCut}, "Triu")));

TEST_P(TriuOperationTest, TestTriu) {
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.inputTensors = GetInputTensors(test_data);
    testCase.outputTensors = GetOutputTensors(test_data);
    bool flag = true;
    int diagonal = GetValueByName<int>(test_data, "diagonal");
    auto args = TriuOpFuncArgs(diagonal, GetViewShape(test_data), GetTileShape(test_data), flag);
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0].GetStorage()->Symbol() + ".bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0].GetStorage()->Symbol() + ".bin"};
    TestExecutor::runTest(testCase);
}
} // namespace