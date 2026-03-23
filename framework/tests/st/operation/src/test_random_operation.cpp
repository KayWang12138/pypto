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

static std::vector<uint64_t> SkipCounter(const std::vector<uint64_t> &counter, uint64_t offset) {
    std::vector<uint64_t> result = counter;
    
    uint32_t offsetLo = static_cast<uint32_t>(offset);
    uint32_t offsetHi = static_cast<uint32_t>(offset >> 32);
    
    uint32_t c0 = static_cast<uint32_t>(result[0] & 0xFFFFFFFF);
    uint32_t c1 = static_cast<uint32_t>(result[0] >> 32);
    uint32_t c2 = static_cast<uint32_t>(result[1] & 0xFFFFFFFF);
    uint32_t c3 = static_cast<uint32_t>(result[1] >> 32);
    
    c0 += offsetLo;
    if (c0 < offsetLo) {
        ++offsetHi;
    }
    c1 += offsetHi;
    if (c1 < offsetHi) {
        if (++c2 == 0) {
            ++c3;
        }
    }
    
    result[0] = static_cast<uint64_t>(c0) | (static_cast<uint64_t>(c1) << 32);
    result[1] = static_cast<uint64_t>(c2) | (static_cast<uint64_t>(c3) << 32);
    
    return result;
}

static uint64_t CalculateLinearOffset(const std::vector<int64_t> &offset, const std::vector<int64_t> &shape) {
    uint64_t linearOffset = 0;
    uint64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        linearOffset += static_cast<uint64_t>(offset[i]) * stride;
        stride *= static_cast<uint64_t>(shape[i]);
    }
    return linearOffset;
}

struct RandomOpFuncArgs : public OpFuncArgs {
    RandomOpFuncArgs(uint64_t key, const std::vector<uint64_t> &counter, 
                     const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape, uint16_t rounds)
        : key_(key), counter_(counter), viewShape_(viewShape), tileShape_(tileShape), rounds_(rounds) {}

    uint64_t key_;
    std::vector<uint64_t> counter_;
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
        SymbolicScalar outputDim = outputs[0].GetShape()[0];
        const int64_t viewShape = args->viewShape_[0];
        std::vector<int64_t> outputShape = outputs[0].GetShape();

        const int loop = CeilDiv(outputDim, viewShape);

        LOOP("LOOP_L0_idx", FunctionType::DYNAMIC_LOOP, idx, LoopRange(0, loop, 1)) {
            auto offset = idx * viewShape;
            auto validShape = std::min(outputDim - offset, SymbolicScalar(viewShape));
            
            std::vector<int64_t> tileOffset = {idx * viewShape};
            uint64_t linearOffset = CalculateLinearOffset(tileOffset, outputShape);
            std::vector<uint64_t> tileCounter = SkipCounter(args->counter_, linearOffset);
            
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = Random(args->key_, tileCounter, outputShape, args->rounds_);
            Assemble(res, {offset}, outputs[0]);
        }
    }
}

static void RandomOperationExeFunc2D(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {

    FUNCTION("main", {}, {outputs[0]}) {
        auto args = static_cast<const RandomOpFuncArgs *>(opArgs);
        SymbolicScalar firstDim = outputs[0].GetShape()[0];
        SymbolicScalar secondDim = outputs[0].GetShape()[1];
        const int64_t firstViewShape = args->viewShape_[0];
        const int64_t secondViewShape = args->viewShape_[1];
        std::vector<int64_t> outputShape = outputs[0].GetShape();

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto offset0 = bIdx * firstViewShape;
                auto offset1 = sIdx * secondViewShape;
                auto validShape0 = std::min(firstDim - offset0, SymbolicScalar(firstViewShape));
                auto validShape1 = std::min(secondDim - offset1, SymbolicScalar(secondViewShape));
                
                std::vector<int64_t> tileOffset = {bIdx * firstViewShape, sIdx * secondViewShape};
                uint64_t linearOffset = CalculateLinearOffset(tileOffset, outputShape);
                std::vector<uint64_t> tileCounter = SkipCounter(args->counter_, linearOffset);
                
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Random(args->key_, tileCounter, outputShape, args->rounds_);
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
    std::vector<uint64_t> counter = {0, 0};
    if (test_data.find("key") != test_data.end()) {
        key = GetValueByName<uint64_t>(test_data, "key");
    }
    if (test_data.find("counter") != test_data.end()) {
        auto counterAttr = GetValueByName<std::vector<int64_t>>(test_data, "counter");
        counter.clear();
        for (auto val : counterAttr) {
            counter.push_back(static_cast<uint64_t>(val));
        }
    }
    
    auto args = RandomOpFuncArgs(key, counter, viewShape, tileShape, rounds);
    testCase.args = &args;
    TestExecutor::runTest(testCase);
}
} // namespace
