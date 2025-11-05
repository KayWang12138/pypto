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
 * \file test_dynamic_indexer_topk.cpp
 * \brief
 */
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek_v3.2_exp/lightning_indexer_topk.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"

using namespace npu::tile_fwk;

class TestLightningIndexerTopkQuantUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};


void TestLightningIndexerTopkQuant(IndexerTile &tileConfig) {
    config::SetHostOption(ONLY_CODEGEN, true);

    const int b = 4;
    const int s1 = 2;
    const int n1 = 64;
    const int d = 128;
    const int blockNum = 1127;
    const int blockSize = 128;
    const int n2 = 1;
    const int maxBlockNum = 1024;
    const int selectedCount = 2048;

    std::set<int> unrollList = {64, 32, 16, 8, 4, 2, 1};

    Tensor staticQuery(DT_INT8, {b, s1, n1, d}, "staticQuery");
    Tensor staticKey(DT_INT8, {blockNum, blockSize, n2, d}, "staticKey");
    Tensor staticQScale(DT_FP16, {b, s1, n1, 1}, "staticQScale");
    Tensor staticKScale(DT_FP16, {blockNum, blockSize, n2, 1}, "staticKScale");
    Tensor staticWeights(DT_FP16, {b, s1, n1}, "staticWeights");
    Tensor staticActSeq(DT_INT32, {b}, "staticActSeq");
    Tensor staticBlockTable(DT_INT32, {b, maxBlockNum}, "staticBlockTable");
    Tensor staticTopkRes(DT_INT32, {b, s1, n2, selectedCount}, "staticTopkRes");
    Tensor staticTmpOut(DT_FP32, {b * s1 * n2, maxBlockNum * blockSize}, "staticTmpOut");
    Tensor staticTopkValue(DT_FP32, {b, s1, n2, selectedCount}, "staticTopkValue");

    Tensor query(DT_INT8, {-1, -1, n1, d}, "query");
    Tensor key(DT_INT8, {-1, blockSize, n2, d}, "key");
    Tensor qScale(DT_FP16, {-1, -1, n1, 1}, "qScale");
    Tensor kScale(DT_FP16, {-1, blockSize, n2, 1}, "kScale");
    Tensor weights(DT_FP16, {-1, -1, n1}, "weights");
    Tensor actSeq(DT_INT32, {-1}, "actSeq");
    Tensor blockTable(DT_INT32, {-1, -1}, "blockTable");

    auto symB = b;
    auto symS1 = s1;
    auto symMaxBlock = maxBlockNum;

    Tensor topkRes(DT_INT32, {symB, symS1, n2, selectedCount}, "topkRes");
    Tensor tmpOut(DT_FP32, {symB * symS1 * n2, symMaxBlock * blockSize}, "tmpOut");
    Tensor topkValue(DT_FP32, {symB, symS1, n2, selectedCount}, "topkValue");

    Tensor topkResGolden(DT_INT8, {b, s1, n2, selectedCount}, "topkResGolden");
    Tensor tmpGolden(DT_FP32, {b * s1 * n2, maxBlockNum * blockSize}, "tmpGolden");
    Tensor topkValueGolden(DT_FP32, {b, s1, n2, selectedCount}, "topkValueGolden");

    FUNCTION("IndexerTopk", {query, key, qScale, kScale, weights, actSeq, blockTable}, {topkRes, tmpOut, topkValue}) {
        LightningIndexerTopkImpl(query, key, true, &qScale, &kScale,
            weights, actSeq, blockTable, topkRes, selectedCount, tileConfig, unrollList,
            &tmpOut, &topkValue);
    }
}

// DynamicIndexerTopk.indexer_topk_quant_4_b_1_s1_64k_s2
TEST_F(TestLightningIndexerTopkQuantUtest, indexer_topk_quant_4_b_1_s1_64k_s2) {
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);                    // 参数化
    config::SetPassOption(COPYIN_THRESHOLD, 100 * 1024 * 1024); // mistake
    config::SetPassOption(SG_CYCLE_LOWER_BOUND, 1024);
    config::SetPassOption(SG_CYCLE_UPPER_BOUND, 1024 * 1024);
    config::SetPassOption(L1_REUSE, 32);
    config::SetPassOption(SG_PARALLEL_NUM, 2);
    config::SetPassOption(NBUFFER_MERGE_MODE, 2);
    config::SetPassOption(VEC_NBUFFER_MAP, std::map<int64_t, int64_t>{
                                                                                    {-1, 16}
    });
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, true);
    config::SetRuntimeOption<uint8_t>(
        MACHINE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH) |
                            static_cast<uint8_t>(MachineScheduleConfig::MULTI_CORE_FAIR_SCH));

    config::SetRuntimeOption(WORKSPACE_RECYCLE_PERIOD, 128);
    config::SetRuntimeOption(ESTIMATED_STITCH_TASK_MAX_LOOP_NUM, 128);
    IndexerTile config;

    config.weightTile = {64, 128};
    config.c1Tile = {64, 64, 128, 128, 128, 128}; // (m, M), (k, K), (n, N)
    config.v1Tile = {64, 128};
    config.topkTile = {1, 4096};
    config.addsTile = {1, 1, 1, 4096};

    TestLightningIndexerTopkQuant(config);
}