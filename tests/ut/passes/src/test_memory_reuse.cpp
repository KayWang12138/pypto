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
 * \file test_memory_reuse.cpp
 * \brief Unit test for RemoveRedundentReshape pass.
 */

#include <gtest/gtest.h>
#include "interface/cache/function_cache.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "passes/execute_graph_pass/memory_reuse.h"
#include "interface/tensor/tensormap.h"
#include "models/deepseek/deepseek_mla.h"
#include "models/deepseek/deepseek_spec.h"
#include <nlohmann/json.hpp>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class TestMemoryReuse : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {
    }
};

TEST_F(TestMemoryReuse, test_connection_matrix) {
    int b = 2;
    int n = 2;
    int s = 1;
    int kvLoraRank = 512;
    int vHeadDim =128;
    int h = 512;
    std::vector<int> inShape = {b, n, s, kvLoraRank}; // (b, n, s, d)
    Tensor attnPostIn(DT_BF16, inShape, "attnPostIn");
    Tensor attenOutput;
    AttentionW aw;
    aw.kvBProjWV = Tensor(DT_BF16, {n, kvLoraRank, vHeadDim}, "kvBProjWV");
    aw.oProjW = Tensor(DT_BF16, {n * vHeadDim, h}, "oProjW");
    ConfigManager::Instance();
    FUNCTION("AttentionPost") {
        DeepseekAttention atten(deepseekConfig1, aw, 1);
        attenOutput = atten.AttentionPost2(attnPostIn);
    }
    attenOutput.GetDataType();
    auto function= Program::GetInstance().GetFunctionByRawName("TENSOR_AttentionPost");
    ASSERT_NE(function, nullptr);
    auto rootFunc = function->rootFunc_;
    auto callOps = rootFunc->Operations();
    size_t totalSize = 14;
    EXPECT_EQ(callOps.size(), totalSize);
    std::unordered_set<int64_t> storageSet;
    uint64_t totalLength = 0;
    for (auto &callop : callOps) {
        for (auto &in : callop.iOperand) {
            if (in->storage_ != nullptr && storageSet.count(in->storage_->id_) == 0) {
                storageSet.emplace(in->storage_->id_);
                totalLength += in->storage_->length_;
            }
        }
    }
    EXPECT_EQ(totalLength, 8192);
    Json jsonT = attenOutput.GetStorage()->DumpJson();
    std::unordered_map<int, std::shared_ptr<RawTensor>> rawTensorDict;
    auto newTensor = LogicalTensor::LoadJson(*function, rawTensorDict, jsonT);
    attenOutput.GetStorage()->DumpASM(true, true);
    attenOutput.GetStorage()->tensor->GetRawShapeSize();

    std::vector<int> resultOffset;
    std::vector<int> resultShape;
    CalcShapeAndOffsetOfGroup(function->inCasts_, resultOffset, resultShape);
    function->GetTensorMap().GetTensorByMagic(function->inCasts_[0]->magic);
    CalcOverlapSize(function->inCasts_[0], function->inCasts_[0]);
    CalcOverlap(function->inCasts_[0], function->inCasts_[0], true);
    CalcOverlap(function->inCasts_[0], function->inCasts_, true);
    Allocator allocator(rootFunc);
    allocator.Init();
    size_t nodeId0 = 0;
    size_t nodeId1 = 1;
    size_t nodeId2 = 2;
    size_t nodeId4 = 4;
    size_t nodeId5 = 5;
    size_t nodeId6 = 6;
    size_t nodeId13 = 13;
    size_t nodeId7 = 7;
    // 下面的值请勿随意修改，校验存在一定价值
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId0), callOps.at(nodeId1)), false);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId0), callOps.at(nodeId4)), true);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId0), callOps.at(nodeId5)), true);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId0), callOps.at(nodeId6)), true);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId5), callOps.at(nodeId13)), true);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId2), callOps.at(nodeId7)), true);
}