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
    ALOG_ERROR_F("--------test SkipCounter start, counter size: %d, offset: %lu", counter.size(), offset);
    for (size_t i = 0; i < counter.size(); i++) {
        ALOG_ERROR_F("--------test counter[%d]: %lu", i, counter[i]);
    }
    
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
    
    ALOG_ERROR_F("--------test SkipCounter result: [%lu, %lu]", result[0], result[1]);
    
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

    ALOG_ERROR_F("--------RandomOperationExeFunc start");
    
    FUNCTION("main", {}, {outputs[0]}) {
        auto args = static_cast<const RandomOpFuncArgs *>(opArgs);
        ALOG_ERROR_F("--------args key: %lu, rounds: %d", args->key_, args->rounds_);
        ALOG_ERROR_F("--------args counter size: %d", args->counter_.size());
        for (size_t i = 0; i < args->counter_.size(); i++) {
            ALOG_ERROR_F("--------args counter[%d]: %lu", i, args->counter_[i]);
        }
        ALOG_ERROR_F("--------args viewShape size: %d", args->viewShape_.size());
        for (size_t i = 0; i < args->viewShape_.size(); i++) {
            ALOG_ERROR_F("--------args viewShape[%d]: %ld", i, args->viewShape_[i]);
        }
        ALOG_ERROR_F("--------args tileShape size: %d", args->tileShape_.size());
        for (size_t i = 0; i < args->tileShape_.size(); i++) {
            ALOG_ERROR_F("--------args tileShape[%d]: %ld", i, args->tileShape_[i]);
        }
        
        const int64_t outputDim = outputs[0].GetShape()[0];
        ALOG_ERROR_F("--------outputDim: %ld", outputDim);
        const int64_t viewShape = args->viewShape_[0];
        ALOG_ERROR_F("--------viewShape: %ld", viewShape);

        const int loop = (outputDim + viewShape - 1) / viewShape;
        ALOG_ERROR_F("--------loop count: %d", loop);
        int64_t tileIdx = 0;

        LOOP("LOOP_L0_idx", FunctionType::DYNAMIC_LOOP, idx, LoopRange(0, loop, 1)) {
            ALOG_ERROR_F("--------loop iteration tileIdx: %ld", tileIdx);
            auto offset = idx * viewShape;
            int64_t validShape = std::min(outputDim - tileIdx * viewShape, viewShape);
            ALOG_ERROR_F("--------validShape: %ld", validShape);
            
            uint64_t tileOffset = static_cast<uint64_t>(tileIdx * viewShape);
            ALOG_ERROR_F("--------tileOffset: %lu", tileOffset);
            std::vector<uint64_t> tileCounter = SkipCounter(args->counter_, tileOffset);
            
            TileShape::Current().SetVecTile(args->tileShape_);
            ALOG_ERROR_F("--------calling Random");
            auto res = Random(args->key_, tileCounter, {validShape}, args->rounds_);
            ALOG_ERROR_F("--------Random returned, calling Assemble");
            Assemble(res, {offset}, outputs[0]);
            ALOG_ERROR_F("--------Assemble done");
            
            ++tileIdx;
        }
    }
    ALOG_ERROR_F("--------RandomOperationExeFunc end");
}

class RandomOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<RandomOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestRandom, RandomOperationTest,
    ::testing::ValuesIn(GetOpMetaData<RandomOpMetaData>({RandomOperationExeFunc}, "Random")));

TEST_P(RandomOperationTest, TestRandom) {
    ALOG_ERROR_F("--------TestRandom start");
    auto testCase = CreateTestCaseDesc<RandomOpMetaData>(GetParam(), nullptr);
    nlohmann::json test_data = GetParam().test_data_;
    
    auto viewShape = GetViewShape(test_data);
    ALOG_ERROR_F("--------viewShape size: %d", viewShape.size());
    for (size_t i = 0; i < viewShape.size(); i++) {
        ALOG_ERROR_F("--------viewShape[%d]: %ld", i, viewShape[i]);
    }
    
    auto tileShape = GetTileShape(test_data);
    ALOG_ERROR_F("--------tileShape size: %d", tileShape.size());
    for (size_t i = 0; i < tileShape.size(); i++) {
        ALOG_ERROR_F("--------tileShape[%d]: %ld", i, tileShape[i]);
    }
    
    uint16_t rounds = 10;
    if (test_data.find("rounds") != test_data.end()) {
        rounds = GetValueByName<int>(test_data, "rounds");
    }
    ALOG_ERROR_F("--------rounds: %d", rounds);
    
    uint64_t key = 12345678901234;
    std::vector<uint64_t> counter = {0, 0};
    if (test_data.find("key") != test_data.end()) {
        key = GetValueByName<uint64_t>(test_data, "key");
    }
    ALOG_ERROR_F("--------key: %lu", key);
    
    if (test_data.find("counter") != test_data.end()) {
        auto counterAttr = GetValueByName<std::vector<int64_t>>(test_data, "counter");
        ALOG_ERROR_F("--------counterAttr size: %d", counterAttr.size());
        counter.clear();
        for (auto val : counterAttr) {
            counter.push_back(static_cast<uint64_t>(val));
            ALOG_ERROR_F("--------counter val: %ld -> %lu", val, static_cast<uint64_t>(val));
        }
    }
    ALOG_ERROR_F("--------final counter: [%lu, %lu]", counter[0], counter[1]);
    
    auto args = RandomOpFuncArgs(key, counter, viewShape, tileShape, rounds);
    testCase.args = &args;
    ALOG_ERROR_F("--------calling TestExecutor::runTest");
    TestExecutor::runTest(testCase);
    ALOG_ERROR_F("--------TestRandom end");
}
} // namespace
