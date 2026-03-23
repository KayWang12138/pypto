/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_random_operation.cpp
 * \brief Test Random operation
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct RandomOpFuncArgs : public OpFuncArgs {
    RandomOpFuncArgs(uint64_t key, const std::vector<int64_t> &counter, 
                     const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape, uint16_t rounds)
        : key_(key), counter_(counter), viewShape_(viewShape), tileShape_(tileShape), rounds_(rounds) {}

    uint64_t key_;
    std::vector<int64_t> counter_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    uint16_t rounds_;
};

struct RandomOpMetaData {
    explicit RandomOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void RandomOperationExeFunc1D(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {

    FUNCTION("main", {}, {outputs[0]}) {
        auto args = static_cast<const RandomOpFuncArgs *>(opArgs);
        SymbolicScalar outputDim = args->viewShape_[0];
        const int64_t viewShape = args->viewShape_[0];

        const int loop = CeilDiv(outputDim, viewShape);

        LOOP("LOOP_L0_idx", FunctionType::DYNAMIC_LOOP, idx, LoopRange(0, loop, 1)) {
            auto offset = idx * viewShape;
            auto validShape = std::min(outputDim - offset, SymbolicScalar(viewShape));
            
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = Random(args->key_, args->counter_, args->viewShape_, 
                outputs[0].GetDataType(), args->rounds_);
            Assemble(res, {offset}, outputs[0]);
        }
    }
}

static void RandomOperationExeFunc2D(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {

    FUNCTION("main", {}, {outputs[0]}) {
        auto args = static_cast<const RandomOpFuncArgs *>(opArgs);
        SymbolicScalar firstDim = args->viewShape_[0];
        SymbolicScalar secondDim = args->viewShape_[1];
        const int64_t firstViewShape = args->viewShape_[0];
        const int64_t secondViewShape = args->viewShape_[1];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto offset0 = bIdx * firstViewShape;
                auto offset1 = sIdx * secondViewShape;
                auto validShape0 = std::min(firstDim - offset0, SymbolicScalar(firstViewShape));
                auto validShape1 = std::min(secondDim - offset1, SymbolicScalar(secondViewShape));
                
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Random(args->key_, args->counter_, args->viewShape_, 
                    outputs[0].GetDataType(), args->rounds_);
                Assemble(res, {offset0, offset1}, outputs[0]);
            }
        }
    }
}

class RandomOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<RandomOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestRandom, RandomOperationTest,
    ::testing::ValuesIn(GetOpMetaData<RandomOpMetaData>({RandomOperationExeFunc1D, RandomOperationExeFunc2D}, "Random")));

TEST_P(RandomOperationTest, TestRandom) {
    auto testCase = CreateTestCaseDesc<RandomOpMetaData>(GetParam(), nullptr);
    nlohmann::json test_data = GetParam().test_data_;
    
    auto viewShape = GetViewShape(test_data);
    auto tileShape = GetTileShape(test_data);
    uint16_t rounds = 10;
    if (test_data.find("rounds") != test_data.end()) {
        rounds = GetValueByName<int>(test_data, "rounds");
    }
    
    uint64_t key = 12345678901234;
    std::vector<int64_t> counter = {0, 0, 0, 0};
    if (test_data.find("key") != test_data.end()) {
        key = GetValueByName<uint64_t>(test_data, "key");
    }
    if (test_data.find("counter") != test_data.end()) {
        counter = GetValueByName<std::vector<int64_t>>(test_data, "counter");
    }
    
    auto args = RandomOpFuncArgs(key, counter, viewShape, tileShape, rounds);
    testCase.args = &args;
    TestExecutor::runTest(testCase);
}
} // namespace
