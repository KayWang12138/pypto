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
 * \file test_reduce_sum_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
const unsigned IDX_DIM0 = 0;
const unsigned IDX_DIM1 = 1;
const unsigned IDX_DIM2 = 2;

struct ReduceSumOpFuncArgs : public OpFuncArgs {
    ReduceSumOpFuncArgs(std::vector<int64_t> viewShape, const std::vector<int64_t> tileShape, std::vector<int64_t> dims)
        : viewShape_(viewShape), tileShape_(tileShape), dims_(dims) {}
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    std::vector<int64_t> dims_;
};

struct ReduceSumOpMetadata {
    explicit ReduceSumOpMetadata(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}
    OpFunc opFunc_;
    nlohmann::json test_data_;
};

void ReduceSumOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    auto args = static_cast<const ReduceSumOpFuncArgs *>(opArgs);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        int dim = args->dims_[0];
        if (dim < 0) {
            dim = static_cast<int>(inputs[0]->shape.size()) + dim;
        }
        SymbolicScalar viewShape[] = {args->viewShape_[0], args->viewShape_[1]};
        viewShape[dim] = 0;
        const int batch = CeilDiv(inputs[0]->shape[1 - dim], viewShape[1 - dim]);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(batch)) {
            auto viewTensor = DViewPad(inputs[0],
                {
                    viewShape[0] == 0 ? firstDim : viewShape[0],
                    viewShape[1] == 0 ? secondDim : viewShape[1]
                },
                {
                    viewShape[0] == 0 ? firstDim : std::min(firstDim - bIdx * viewShape[0], viewShape[0]),
                    viewShape[1] == 0 ? secondDim : std::min(secondDim - bIdx * viewShape[1], viewShape[1])
                },
                {bIdx * viewShape[0], bIdx * viewShape[1]});
            Program::GetInstance().GetTileShape().SetVecTileShapes(args->tileShape_);
            auto res = RowSumSingle(viewTensor, args->dims_[0]);
            Assemble(res, {bIdx * viewShape[0], bIdx * viewShape[1]}, outputs[0]);
        }
    }
}

void ReduceSum3DOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                 const OpFuncArgs* opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    auto args = static_cast<const ReduceSumOpFuncArgs *>(opArgs);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
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
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loops[IDX_DIM0])) {
            LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loops[IDX_DIM1])) {
                LOOP("LOOP_L2_bIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loops[IDX_DIM2])) {
                    auto viewTensor = DViewPad(inputs[0],
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
                    Program::GetInstance().GetTileShape().SetVecTileShapes(args->tileShape_);
                    auto res = RowSumSingle(viewTensor, args->dims_[0]);
                    Assemble(res, {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2]}, outputs[0]);
                }
            }
        }
    }
}

class ReduceSumOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ReduceSumOpMetadata> {};

INSTANTIATE_TEST_SUITE_P(TestReduceSum, ReduceSumOperationTest, ::testing::ValuesIn(
    GetOpMetaData<ReduceSumOpMetadata>({ReduceSumOperationExeFunc, ReduceSum3DOperationExeFunc}, "ReduceSum")));

TEST_P(ReduceSumOperationTest, TestReduceSum) {
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.inputTensors = GetInputTensors(test_data);
    testCase.outputTensors = GetOutputTensors(test_data);
    auto args = ReduceSumOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data),
        GetValueByName<std::vector<int64_t>>(test_data, "dims"));
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0]->Symbol() + ".bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0]->Symbol() + ".bin"};
    TestExecutor::runTest(testCase);
}

} // namespace