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
    ReduceSumOpFuncArgs(std::vector<int> viewShape, const std::vector<int> tileShape, std::vector<int> dims)
        : viewShape_(viewShape), tileShape_(tileShape), dims_(dims) {}
    std::vector<int> viewShape_;
    std::vector<int> tileShape_;
    std::vector<int> dims_;
};

struct ReduceSumOpMetadata {
    explicit ReduceSumOpMetadata(const std::vector<OpFunc> &funcs) : opFuncs_(funcs) {}
    std::vector<OpFunc> opFuncs_;
};

void ReduceSumOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    auto args = static_cast<const ReduceSumOpFuncArgs *>(opArgs);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        const int dim = args->dims_[0];
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
            auto res = RowSumSingle(viewTensor, dim);
            DAssemble(res, {bIdx * viewShape[0], bIdx * viewShape[1]}, outputs[0]);
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
        const int dim = args->dims_[0];
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
                    auto res = RowSumSingle(viewTensor, dim);
                    DAssemble(res, {bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2]}, outputs[0]);
                }
            }
        }
    }
}

static const ReduceSumOpMetadata metaData =
    ReduceSumOpMetadata({ReduceSumOperationExeFunc, ReduceSum3DOperationExeFunc});

class ReduceSumOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ReduceSumOpMetadata> {};

INSTANTIATE_TEST_SUITE_P(TestReduceSum, ReduceSumOperationTest, ::testing::Values(metaData));

TEST_P(ReduceSumOperationTest, TestReduceSum) {
    TestCaseDesc testCase;
    auto config = GetGoldenDir() + "/test_case_data.json";
    testCase.inputTensors = GetInputTensors(config);
    testCase.outputTensors = GetOutputTensors(config);
    auto args = ReduceSumOpFuncArgs(GetViewShape(config), GetTileShape(config),
        GetValueByName<std::vector<int>>(config, "dims"));
    testCase.args = &args;
    auto func_id = GetFuncId(config);
    if (func_id < 0 || static_cast<size_t>(func_id) >= GetParam().opFuncs_.size()) {
        func_id = args.viewShape_.size() - 2;
    }
    testCase.opFunc = GetParam().opFuncs_[func_id];
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0]->Symbol() + ".bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0]->Symbol() + ".bin"};
    TestExecutor::runTest(testCase);
}

} // namespace