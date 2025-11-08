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
 * \file test_onboard_page_attention_cost.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "runtime.h"

using namespace npu::tile_fwk;

class OnBoardPaCostTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};


TEST_F(OnBoardPaCostTest, test_page_attention_low_latency) {
    aclInit(nullptr);
    config::SetOperationConfig("FORCE_COMBINE_AXIS", true);
    config::SetHostOption(ONLY_CODEGEN, true);

    TileFwkInit("");
    std::unordered_map<std::string, int> params = {
        {"b", 4},
        {"nq", 32},
        {"s2", 256}
    };

    IfaTileShapeConfig tileConfig {
        256, // block size
        32,  // nTile
        {256, 128}, // v0 tile for qkv-view-concat, q-S1D:(32,64), k/v-S2D:(256,64), merge 2D to copy
        {32, 32, 256, 256, 128, 128}, // c1 tile for S1D@S2D
        {32, 256}, // v1 tile for S1S2
        {32, 32, 256, 256, 128, 128}, // c2 tile for S1S2@S2D
        {32, 256}, // v2 tile for S1D
    };

    const int b = params["b"];
    const int nq = params["nq"];
    const int s2 = params["s2"];
    const int blockSize = tileConfig.blockSize;
    const int sq = 1;
    const int dn = 512;
    const int dr = 64;
    const int nkv = 1;

    std::vector<int> actSeqs(b, s2);
    const float softmaxScale = static_cast<float>(1.0 / std::sqrt(dn + dr));

    // 根据Per Batch实际的sequence构造blockNum，blockNum >= Sum(blockNumPerBatch)，此处选取相等场景
    int blockNum = 0;
    for (auto s : actSeqs) {
        blockNum += CeilDiv(s, blockSize);
    }


    Tensor qNope(DT_BF16, {b * sq * nq, dn}, "qNope");
    Tensor qRope(DT_BF16, {b * sq * nq, dr}, "qRope");

    Tensor kvNopeCache(DT_BF16, {blockNum * blockSize * nkv, dn}, "kNopeCache", TileOpFormat::TILEOP_NZ);
    Tensor kRopeCache(DT_BF16, {blockNum * blockSize * nkv, dr}, "kRope", TileOpFormat::TILEOP_NZ);

    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(actSeqs.begin(), actSeqs.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
    std::vector<std::vector<int>> blockTable(b, std::vector<int>(maxBlockNumPerBatch, 0));
    for (int i = 0 ; i < b; ++i) {
        for (int j = 0 ; j < maxBlockNumPerBatch; ++j) {
            blockTable[i][j] = i * maxBlockNumPerBatch + j;
        }
    }

    Tensor attentionOut(DT_FP32, {b * sq * nq, dn}, "attentionOut");

    // 计算流程开始
    TileFwkBeginFunction("IfaStatic",
        {qNope, kvNopeCache, qRope, kRopeCache, attentionOut});
    {
        IncreFlashAttention(qNope, kvNopeCache, kvNopeCache, qRope, kRopeCache, blockTable, actSeqs, softmaxScale,
            attentionOut, tileConfig);
    }
    TileFwkEndFunction();
    // 计算流程结束

    TileFwkCompile();
}

TEST_F(OnBoardPaCostTest, test_page_attention_hight_throughput) {
    aclInit(nullptr);
    config::SetOperationConfig("FORCE_COMBINE_AXIS", true);
    config::SetHostOption(ONLY_CODEGEN, true);
    const int cycle_lower_bound = 2048;
    const int l1_reuse = 4;
    const int copyin_threshold = 2 * 1024 * 1024;
    config::SetPassOption(SG_CYCLE_LOWER_BOUND, cycle_lower_bound);
    config::SetPassOption(L1_REUSE, l1_reuse);
    config::SetPassOption(COPYIN_THRESHOLD, copyin_threshold);

    TileFwkInit("");
    std::unordered_map<std::string, int> params = {
        {"b", 32},
        {"nq", 128},
        {"s2", 4096}
    };

    IfaTileShapeConfig tileConfig {
        512, // block size
        128,  // nTile
        {256, 128}, // v0 tile for qkv-view-concat, q-S1D:(32,64), k/v-S2D:(256,64), merge 2D to copy
        {128, 128, 256, 256, 128, 128}, // c1 tile for S1D@S2D
        {32, 256}, // v1 tile for S1S2
        {128, 128, 256, 256, 128, 128}, // c2 tile for S1S2@S2D
        {32, 256}, // v2 tile for S1D
    };

    const int b = params["b"];
    const int nq = params["nq"];
    const int s2 = params["s2"];
    const int blockSize = tileConfig.blockSize;
    const int sq = 1;
    const int dn = 512;
    const int dr = 64;
    const int nkv = 1;

    std::vector<int> actSeqs(b, s2);
    const float softmaxScale = static_cast<float>(1.0 / std::sqrt(dn + dr));

    // 根据Per Batch实际的sequence构造blockNum，blockNum >= Sum(blockNumPerBatch)，此处选取相等场景
    int blockNum = 0;
    for (auto s : actSeqs) {
        blockNum += CeilDiv(s, blockSize);
    }


    Tensor qNope(DT_BF16, {b * sq * nq, dn}, "qNope");
    Tensor qRope(DT_BF16, {b * sq * nq, dr}, "qRope");

    Tensor kvNopeCache(DT_BF16, {blockNum * blockSize * nkv, dn}, "kNopeCache", TileOpFormat::TILEOP_NZ);
    Tensor kRopeCache(DT_BF16, {blockNum * blockSize * nkv, dr}, "kRope", TileOpFormat::TILEOP_NZ);

    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(actSeqs.begin(), actSeqs.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
    std::vector<std::vector<int>> blockTable(b, std::vector<int>(maxBlockNumPerBatch, 0));
    for (int i = 0 ; i < b; ++i) {
        for (int j = 0 ; j < maxBlockNumPerBatch; ++j) {
            blockTable[i][j] = i * maxBlockNumPerBatch + j;
        }
    }

    Tensor attentionOut(DT_FP32, {b * sq * nq, dn}, "attentionOut");

    // 计算流程开始
    TileFwkBeginFunction("IfaStatic",
        {qNope, kvNopeCache, qRope, kRopeCache, attentionOut});
    {
        IncreFlashAttention(qNope, kvNopeCache, kvNopeCache, qRope, kRopeCache, blockTable, actSeqs, softmaxScale,
            attentionOut, tileConfig);
    }
    TileFwkEndFunction();
    // 计算流程结束

    TileFwkCompile();
}
