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
 * \file test_TriU_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct TriUOpFuncArgs : public OpFuncArgs {
    TriUOpFuncArgs(
        const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape, int diagonal, bool isUpper)
        : viewShape_(viewShape), tileShape_(tileShape), diagonal_(diagonal), isUpper_(isUpper) {}
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    int diagonal_;
    bool isUpper_;
};

struct TriUOpMetaData {
    explicit TriUOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}
    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static std::vector<int64_t> GetTriUViewShape(Tensor tensor, const std::vector<int64_t> &viewShape) {
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
static int GetRealDiagonal(int diagonal, const std::vector<int64_t> &viewShape, int originXIdx, int originYIdx) {
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

static void TriUOperationExeFuncDoubleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
    SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
    auto args = static_cast<const TriUOpFuncArgs *>(opArgs);
    const std::vector<int64_t> realViewShape = GetTriUViewShape(inputs[0], args->viewShape_);
    const int bloop = CeilDiv(src_firstDim, realViewShape[0]);
    const int sloop = CeilDiv(src_secondDim, realViewShape[1]);
    int b = 0;
    int s = 0;

    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensor = View(inputs[0], realViewShape,
                    {std::min(src_firstDim - bIdx * realViewShape[0], realViewShape[0]),
                        std::min(src_secondDim - sIdx * realViewShape[1], realViewShape[1])},
                    {bIdx * realViewShape[0], sIdx * realViewShape[1]});
                int originXIdx = s * realViewShape[1];
                int originYIdx = -b * realViewShape[0];
                int realDiagonal = GetRealDiagonal(args->diagonal_, realViewShape, originXIdx, originYIdx);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = TriU(tileTensor, realDiagonal);
                Assemble(res, {bIdx * realViewShape[0], sIdx * realViewShape[1]}, outputs[0]);
                s += 1;
            }
            b += 1;
        }
    }
}

class TriUOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<TriUOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestTriU, TriUOperationTest,
    ::testing::ValuesIn(GetOpMetaData<TriUOpMetaData>({TriUOperationExeFuncDoubleCut}, "TriU")));

TEST_P(TriUOperationTest, TestTriU) {
    auto test_data = GetParam().test_data_;
    bool isUpper = true;
    int diagonal = GetValueByName<int>(test_data, "diagonal");
    auto args = TriUOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), diagonal, isUpper);
    auto testCase = CreateTestCaseDesc<TriUOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace