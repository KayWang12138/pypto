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
 * \file test_permute_operation.cpp
 * \brief Test for permute operation (arbitrary dimension reorder).
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;

namespace {

// Parameter structure for permute operation
struct PermuteOpFuncArgs : public OpFuncArgs {
    PermuteOpFuncArgs(const std::vector<int> &dims,
                      const std::vector<int64_t> &viewShape,
                      const std::vector<int64_t> &tileShape)
        : dims_(dims), viewShape_(viewShape), tileShape_(tileShape) {}
    std::vector<int> dims_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

// Meta data for parameterized test
struct PermuteOpMetaData {
    explicit PermuteOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}
    OpFunc opFunc_;
    nlohmann::json test_data_;
};

// 2D permute implementation
static void PermuteOperationExeFunc2Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    const auto *permuteInfo = static_cast<const PermuteOpFuncArgs *>(opArgs);
    const int firstViewShape = permuteInfo->viewShape_[0];
    const int secondViewShape = permuteInfo->viewShape_[1];
    const int ndims = 2;
    std::vector<int> dims = permuteInfo->dims_;
    for (int &d : dims) {
        if (d < 0) d += ndims;
    }

    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx,
             LoopRange(0, CeilDiv(firstDim, firstViewShape), 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx,
                 LoopRange(0, CeilDiv(secondDim, secondViewShape), 1)) {

                Tensor tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                     std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});

                TileShape::Current().SetVecTile(permuteInfo->tileShape_);
                auto res = Permute(tileTensor0, dims);

                std::vector<SymbolicScalar> origOffset = {
                    bIdx * firstViewShape,
                    sIdx * secondViewShape
                };
                std::vector<SymbolicScalar> outOffset(ndims);
                for (int i = 0; i < ndims; ++i) {
                    outOffset[i] = origOffset[dims[i]];
                }

                Assemble(res, outOffset, outputs[0]);
            }
        }
    }
}

// 3D permute implementation
static void PermuteOperationExeFunc3Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    const auto *permuteInfo = static_cast<const PermuteOpFuncArgs *>(opArgs);
    const int firstViewShape = permuteInfo->viewShape_[0];
    const int secondViewShape = permuteInfo->viewShape_[1];
    const int thirdViewShape = permuteInfo->viewShape_[2];
    const int ndims = 3;
    std::vector<int> dims = permuteInfo->dims_;
    for (int &d : dims) {
        if (d < 0) d += ndims;
    }

    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx,
             LoopRange(0, CeilDiv(firstDim, firstViewShape), 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx,
                 LoopRange(0, CeilDiv(secondDim, secondViewShape), 1)) {
                LOOP("LOOP_L2_tIdx", FunctionType::DYNAMIC_LOOP, tIdx,
                     LoopRange(0, CeilDiv(thirdDim, thirdViewShape), 1)) {

                    Tensor tileTensor0 = View(inputs[0],
                        {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                         std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                         std::min(thirdDim - tIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, tIdx * thirdViewShape});

                    TileShape::Current().SetVecTile(permuteInfo->tileShape_);
                    auto res = Permute(tileTensor0, dims);

                    std::vector<SymbolicScalar> origOffset = {
                        bIdx * firstViewShape,
                        sIdx * secondViewShape,
                        tIdx * thirdViewShape
                    };
                    std::vector<SymbolicScalar> outOffset(ndims);
                    for (int i = 0; i < ndims; ++i) {
                        outOffset[i] = origOffset[dims[i]];
                    }

                    Assemble(res, outOffset, outputs[0]);
                }
            }
        }
    }
}

// 4D permute implementation
static void PermuteOperationExeFunc4Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    const auto *permuteInfo = static_cast<const PermuteOpFuncArgs *>(opArgs);
    const int firstViewShape = permuteInfo->viewShape_[0];
    const int secondViewShape = permuteInfo->viewShape_[1];
    const int thirdViewShape = permuteInfo->viewShape_[2];
    const int fourthViewShape = permuteInfo->viewShape_[3];
    const int ndims = 4;
    std::vector<int> dims = permuteInfo->dims_;
    for (int &d : dims) {
        if (d < 0) d += ndims;
    }

    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx,
             LoopRange(0, CeilDiv(firstDim, firstViewShape), 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx,
                 LoopRange(0, CeilDiv(secondDim, secondViewShape), 1)) {
                LOOP("LOOP_L2_tIdx", FunctionType::DYNAMIC_LOOP, tIdx,
                     LoopRange(0, CeilDiv(thirdDim, thirdViewShape), 1)) {
                    LOOP("LOOP_L3_pIdx", FunctionType::DYNAMIC_LOOP, pIdx,
                         LoopRange(0, CeilDiv(fourthDim, fourthViewShape), 1)) {

                        Tensor tileTensor0 = View(inputs[0],
                            {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                             std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                             std::min(thirdDim - tIdx * thirdViewShape, thirdViewShape),
                             std::min(fourthDim - pIdx * fourthViewShape, fourthViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape, tIdx * thirdViewShape,
                             pIdx * fourthViewShape});

                        TileShape::Current().SetVecTile(permuteInfo->tileShape_);
                        auto res = Permute(tileTensor0, dims);

                        std::vector<SymbolicScalar> origOffset = {
                            bIdx * firstViewShape,
                            sIdx * secondViewShape,
                            tIdx * thirdViewShape,
                            pIdx * fourthViewShape
                        };
                        std::vector<SymbolicScalar> outOffset(ndims);
                        for (int i = 0; i < ndims; ++i) {
                            outOffset[i] = origOffset[dims[i]];
                        }

                        Assemble(res, outOffset, outputs[0]);
                    }
                }
            }
        }
    }
}

// 5D permute implementation
static void PermuteOperationExeFunc5Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    const auto *permuteInfo = static_cast<const PermuteOpFuncArgs *>(opArgs);
    const int firstViewShape = permuteInfo->viewShape_[0];
    const int secondViewShape = permuteInfo->viewShape_[1];
    const int thirdViewShape = permuteInfo->viewShape_[2];
    const int fourthViewShape = permuteInfo->viewShape_[3];
    const int fifthViewShape = permuteInfo->viewShape_[4];
    const int ndims = 5;
    std::vector<int> dims = permuteInfo->dims_;
    for (int &d : dims) {
        if (d < 0) d += ndims;
    }

    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        SymbolicScalar fifthDim = inputs[0].GetShape()[4];

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx,
             LoopRange(0, CeilDiv(firstDim, firstViewShape), 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx,
                 LoopRange(0, CeilDiv(secondDim, secondViewShape), 1)) {
                LOOP("LOOP_L2_tIdx", FunctionType::DYNAMIC_LOOP, tIdx,
                     LoopRange(0, CeilDiv(thirdDim, thirdViewShape), 1)) {
                    LOOP("LOOP_L3_pIdx", FunctionType::DYNAMIC_LOOP, pIdx,
                         LoopRange(0, CeilDiv(fourthDim, fourthViewShape), 1)) {
                        LOOP("LOOP_L4_qIdx", FunctionType::DYNAMIC_LOOP, qIdx,
                             LoopRange(0, CeilDiv(fifthDim, fifthViewShape), 1)) {

                            Tensor tileTensor0 = View(inputs[0],
                                {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape, fifthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                 std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                 std::min(thirdDim - tIdx * thirdViewShape, thirdViewShape),
                                 std::min(fourthDim - pIdx * fourthViewShape, fourthViewShape),
                                 std::min(fifthDim - qIdx * fifthViewShape, fifthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, tIdx * thirdViewShape,
                                 pIdx * fourthViewShape, qIdx * fifthViewShape});

                            TileShape::Current().SetVecTile(permuteInfo->tileShape_);
                            auto res = Permute(tileTensor0, dims);

                            std::vector<SymbolicScalar> origOffset = {
                                bIdx * firstViewShape,
                                sIdx * secondViewShape,
                                tIdx * thirdViewShape,
                                pIdx * fourthViewShape,
                                qIdx * fifthViewShape
                            };
                            std::vector<SymbolicScalar> outOffset(ndims);
                            for (int i = 0; i < ndims; ++i) {
                                outOffset[i] = origOffset[dims[i]];
                            }

                            Assemble(res, outOffset, outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

// Test class for permute operation
class PermuteOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<PermuteOpMetaData> {};

// Instantiate test suite with all execution functions and test data named "Permute"
INSTANTIATE_TEST_SUITE_P(TestPermute, PermuteOperationTest,
    ::testing::ValuesIn(
        GetOpMetaData<PermuteOpMetaData>({
            PermuteOperationExeFunc2Dims,
            PermuteOperationExeFunc3Dims,
            PermuteOperationExeFunc4Dims,
            PermuteOperationExeFunc5Dims
        }, "Permute")));

// Test body
TEST_P(PermuteOperationTest, TestPermute) {
    auto test_data = GetParam().test_data_;
    auto dims = GetValueByName<std::vector<int>>(test_data, "dims");
    auto args = PermuteOpFuncArgs(dims, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<PermuteOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}

} // namespace