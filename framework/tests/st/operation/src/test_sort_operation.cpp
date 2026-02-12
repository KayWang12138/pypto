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
 * \file test_sort_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
const unsigned IDX_DIM0 = 0;
const unsigned IDX_DIM1 = 1;
const unsigned IDX_DIM2 = 2;
const unsigned IDX_DIM3 = 3;

struct SortOpFuncArgs : public OpFuncArgs {
    SortOpFuncArgs(std::vector<int64_t> viewShape, const std::vector<int64_t> tileShape, std::vector<int> dims) :
        viewShape_(viewShape), tileShape_(tileShape), dims_(dims){}
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    std::vector<int> dims_;
};

struct SortOpMetadata {
    SortOpMetadata(const OpFunc &opFunc, const nlohmann::json &test_data) :
        opFunc_(opFunc), test_data_(test_data) {}
    OpFunc opFunc_;
    nlohmann::json test_data_;
};

void SortOpExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    auto args = static_cast<const SortOpFuncArgs*>(opArgs);
    SymbolicScalar firstDim = inputs[0].GetShape()[0];
    SymbolicScalar secondDim = inputs[0].GetShape()[1];
    const int firstViewShape = args->viewShape_[0];
    const int secondViewShape = args->viewShape_[1];
    int loop[] = {
        CeilDiv(firstDim, firstViewShape),
        CeilDiv(secondDim, secondViewShape)
    };
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0])) {
            LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1])) {
                std::vector<SymbolicScalar> offset = { bIdx * args->viewShape_[0], sIdx * args->viewShape_[1] };
                auto viewTensor = View(inputs[0], args->viewShape_, {
                    std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                    std::min(secondDim - sIdx * secondViewShape, secondViewShape)
                }, offset);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Sort(viewTensor, args->dims_[0]);
                Assemble(res, offset, outputs[0]);
            }
        }
    }
}

void SortOpExeFunc3D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    auto args = static_cast<const SortOpFuncArgs*>(opArgs);
    SymbolicScalar firstDim = inputs[0].GetShape()[0];
    SymbolicScalar secondDim = inputs[0].GetShape()[1];
    SymbolicScalar thirdDim = inputs[0].GetShape()[2];
    const int firstViewShape = args->viewShape_[0];
    const int secondViewShape = args->viewShape_[1];
    const int thirdViewShape = args->viewShape_[2];
    int loop[] = {
        CeilDiv(firstDim, firstViewShape),
        CeilDiv(secondDim, secondViewShape),
        CeilDiv(thirdDim, thirdViewShape)
    };
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0])) {
            LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1])) {
                LOOP("LOOP_L2_bIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2])) {
                    std::vector<SymbolicScalar> offset = {
                        bIdx * args->viewShape_[0],
                        sIdx * args->viewShape_[1],
                        nIdx * args->viewShape_[2],
                    };
                    auto viewTensor = View(inputs[0], args->viewShape_, {
                        std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                        std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)
                    }, offset);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Sort(viewTensor, args->dims_[0]);
                    Assemble(res, offset, outputs[0]);
                }
            }
        }
    }
}

void SortOpExeFunc4D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    auto args = static_cast<const SortOpFuncArgs*>(opArgs);
    SymbolicScalar firstDim = inputs[0].GetShape()[0];
    SymbolicScalar secondDim = inputs[0].GetShape()[1];
    SymbolicScalar thirdDim = inputs[0].GetShape()[2];
    SymbolicScalar forthDim = inputs[0].GetShape()[3];
    const int firstViewShape = args->viewShape_[0];
    const int secondViewShape = args->viewShape_[1];
    const int thirdViewShape = args->viewShape_[2];
    const int forthViewShape = args->viewShape_[3];
    int loop[] = {
        CeilDiv(firstDim, firstViewShape),
        CeilDiv(secondDim, secondViewShape),
        CeilDiv(thirdDim, thirdViewShape),
        CeilDiv(forthDim, forthViewShape)
    };
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0])) {
            LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1])) {
                LOOP("LOOP_L2_bIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2])) {
                    LOOP("LOOP_L3_bIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(loop[IDX_DIM3])) {
                        std::vector<SymbolicScalar> offset = {
                            bIdx * args->viewShape_[0],
                            sIdx * args->viewShape_[1],
                            nIdx * args->viewShape_[2],
                            qIdx * args->viewShape_[3],
                        };
                        auto viewTensor = View(inputs[0], args->viewShape_, {
                            std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                            std::min(forthDim - qIdx * forthViewShape, forthViewShape)
                        }, offset);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Sort(viewTensor, args->dims_[0]);
                        Assemble(res, offset, outputs[0]);
                    }
                }
            }
        }
    }
}

class SortOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<SortOpMetadata> {};

INSTANTIATE_TEST_SUITE_P(TestSort, SortOperationTest, ::testing::ValuesIn(
    GetOpMetaData<SortOpMetadata>({SortOpExeFunc, SortOpExeFunc3D, SortOpExeFunc4D}, "Sort")));

TEST_P(SortOperationTest, TestSort) {
    auto test_data = GetParam().test_data_;
    auto args = SortOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data),
        GetValueByName<std::vector<int>>(test_data, "dims"));
    auto testCase = CreateTestCaseDesc<SortOpMetadata>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}

} // namespace