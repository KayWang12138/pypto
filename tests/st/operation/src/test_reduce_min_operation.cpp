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
 * \file test_reduce_min_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
const unsigned IDX_DIM0 = 0;
const unsigned IDX_DIM1 = 1;
const unsigned IDX_DIM2 = 2;
const unsigned IDX_DIM3 = 3;

struct ReduceMinOpFuncArgs : public OpFuncArgs {
    ReduceMinOpFuncArgs(
        const std::vector<int64_t> dims, const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape)
        : dims_(dims), viewShape_(viewShape), tileShape_(tileShape) {}

    std::vector<int64_t> dims_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct ReduceMinOperationMetadata {
    explicit ReduceMinOperationMetadata(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

void ReduceMinOperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {inputs[0]}, {outputs[0]}) {
        auto args = static_cast<const ReduceMinOpFuncArgs *>(opArgs);
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        const int firstViewShape = args->viewShape_[0];
        int bloop = CeilDiv(firstDim, firstViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            auto viewTensor = View(inputs[0], {firstViewShape, secondDim},
                {std::min(firstDim - bIdx * firstViewShape, firstViewShape), secondDim}, {bIdx * firstViewShape, 0});
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = RowMinSingle(viewTensor, args->dims_[0]);
            Assemble(res, {bIdx * firstViewShape, 0}, outputs[0]);
        }
    }
}

void ReduceMin2DOperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {inputs[0]}, {outputs[0]}) {
        auto args = static_cast<const ReduceMinOpFuncArgs *>(opArgs);
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        int dim = args->dims_[0];
        if (dim < 0) {
            dim = static_cast<int>(inputs[0]->shape.size()) + dim;
        }
        SymbolicScalar viewShape[] = {args->viewShape_[0], args->viewShape_[1]};
        int loops[] = {
            CeilDiv(inputs[0]->shape[0], viewShape[0]),
            CeilDiv(inputs[0]->shape[1], viewShape[1])
        };
        viewShape[dim] = 0;
        loops[dim] = 1;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, loops[IDX_DIM0], 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, loops[IDX_DIM1], 1)) {
                auto viewTensor = View(inputs[0],
                    {
                        viewShape[0] == 0 ? firstDim : viewShape[0],
                        viewShape[1] == 0 ? secondDim : viewShape[1]
                    },
                    {
                        viewShape[0] == 0 ? firstDim : std::min(firstDim - bIdx * viewShape[0], viewShape[0]),
                        viewShape[1] == 0 ? secondDim : std::min(secondDim - sIdx * viewShape[1], viewShape[1])
                    },
                    {bIdx * viewShape[0], sIdx * viewShape[1]});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = RowMinSingle(viewTensor, args->dims_[0]);
                    Assemble(res, {bIdx * viewShape[0], sIdx * viewShape[1]}, outputs[0]);
            }
        }
    }
}

void ReduceMin3DOperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {inputs[0]}, {outputs[0]}) {
        auto args = static_cast<const ReduceMinOpFuncArgs *>(opArgs);
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar lastDim = inputs[0]->shape[2];
        int dim = args->dims_[0];
        if (dim < 0) {
            dim = static_cast<int>(inputs[0]->shape.size()) + dim;
        }
        SymbolicScalar viewShape[] = {args->viewShape_[0], args->viewShape_[1], args->viewShape_[2]};
        int loops[] = {
            CeilDiv(inputs[0]->shape[0], viewShape[0]),
            CeilDiv(inputs[0]->shape[1], viewShape[1]),
            CeilDiv(inputs[0]->shape[2], viewShape[2])
        };
        viewShape[dim] = 0;
        loops[dim] = 1;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, loops[IDX_DIM0], 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, loops[IDX_DIM0], 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, loops[IDX_DIM0], 1)) {
                    auto viewTensor = View(inputs[0],
                        {
                            viewShape[0] == 0 ? firstDim : viewShape[0],
                            viewShape[1] == 0 ? secondDim : viewShape[1],
                            viewShape[2] == 0 ? lastDim : viewShape[2]
                        },
                        {
                            viewShape[0] == 0 ? firstDim : std::min(firstDim - bIdx * viewShape[0], viewShape[0]),
                            viewShape[1] == 0 ? secondDim : std::min(secondDim - sIdx * viewShape[1], viewShape[1]),
                            viewShape[2] == 0 ? lastDim : std::min(lastDim - nIdx * viewShape[2], viewShape[2])
                        },
                        {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2]});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = RowMinSingle(viewTensor, args->dims_[0]);
                    Assemble(res, {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2]}, outputs[0]);
                }
            }
        }
    }
}

void ReduceMin4DOperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {inputs[0]}, {outputs[0]}) {
        auto args = static_cast<const ReduceMinOpFuncArgs *>(opArgs);
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        SymbolicScalar fourthDim = inputs[0]->shape[3];
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int mloop = CeilDiv(thirdDim, thirdViewShape);
        const int nloop = CeilDiv(fourthDim, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1)) {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                        auto viewTensor =
                            View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                    std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                    std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                    std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                    nIdx * fourthViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = RowMinSingle(viewTensor, args->dims_[0]);
                        Assemble(res,
                            {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape,
                                nIdx * fourthViewShape},
                            outputs[0]);
                    }
                }
            }
        }
    }
}

class ReduceMinOperationTest
    : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ReduceMinOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(TestReduceMin, ReduceMinOperationTest,
    ::testing::ValuesIn(
        GetOpMetaData<ReduceMinOperationMetadata>({ReduceMinOperationExeFunc, ReduceMin2DOperationExeFunc,
                                                      ReduceMin3DOperationExeFunc, ReduceMin4DOperationExeFunc},
            "ReduceMin")));

TEST_P(ReduceMinOperationTest, TestReduceMin) {
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.inputTensors = GetInputTensors(test_data);
    testCase.outputTensors = GetOutputTensors(test_data);
    auto dims = GetValueByName<std::vector<int64_t>>(test_data, "dims");
    auto args = ReduceMinOpFuncArgs(dims, GetViewShape(test_data), GetTileShape(test_data));
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0]->Symbol() + ".bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0]->Symbol() + ".bin"};
    TestExecutor::runTest(testCase);
}

} // namespace