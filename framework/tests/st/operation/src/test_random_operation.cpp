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
#include <iostream>

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

static void RandomOperationExeFunc(
    [[maybe_unused]] const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    
    FUNCTION("main", {}, {outputs[0]}) {
        auto args = static_cast<const RandomOpFuncArgs *>(opArgs);
        
        const int64_t outputDim = outputs[0].GetShape()[0];
        const int64_t viewShape = args->viewShape_[0];

        const int loop = (outputDim + viewShape - 1) / viewShape;
        int64_t tileIdx = 0;

        LOOP("LOOP_L0_idx", FunctionType::DYNAMIC_LOOP, idx, LoopRange(0, loop, 1)) {
            auto offset = idx * viewShape;
            int64_t validShape = std::min(outputDim - tileIdx * viewShape, viewShape);
            
            uint64_t tileOffset = static_cast<uint64_t>(tileIdx * viewShape);
            std::vector<uint64_t> tileCounter = SkipCounter(args->counter_, tileOffset);
            
            // 打印tileCounter的值
            std::cout << "tileIdx: " << tileIdx << ", tileOffset: " << tileOffset << ", key: " << args->key_ << ", tileCounter: [";
            for (size_t i = 0; i < tileCounter.size(); ++i) {
                std::cout << tileCounter[i];
                if (i < tileCounter.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
            
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = Random(args->key_, tileCounter, {validShape}, args->rounds_);
            Assemble(res, {offset}, outputs[0]);
            
            ++tileIdx;
        }
    }
}

class RandomOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<RandomOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestRandom, RandomOperationTest,
    ::testing::ValuesIn(GetOpMetaData<RandomOpMetaData>({RandomOperationExeFunc}, "Random")));

TEST_P(RandomOperationTest, TestRandom) {
    auto testCase = CreateTestCaseDesc<RandomOpMetaData>(GetParam(), nullptr);
    nlohmann::json test_data = GetParam().test_data_;
    
    auto viewShape = GetViewShape(test_data);
    
    auto tileShape = GetTileShape(test_data);
    
    uint16_t rounds = GetValueByName<int>(test_data, "rounds");
    
    uint64_t key = GetValueByName<uint64_t>(test_data, "key");
    
    uint64_t counter_0 = GetValueByName<uint64_t>(test_data, "counter_0");
    uint64_t counter_1 = GetValueByName<uint64_t>(test_data, "counter_1");
    std::vector<uint64_t> counter = {counter_0, counter_1};
    
    auto args = RandomOpFuncArgs(key, counter, viewShape, tileShape, rounds);
    testCase.args = &args;
    TestExecutor::runTest(testCase);
}
} // namespace
