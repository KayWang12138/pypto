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
 * \file test_dynamic_pa.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "models/deepseek/page_attention.h"
#include "passes/execute_graph_pass/memory_reuse.h"

using namespace npu::tile_fwk;

class DynamicPATest : public testing::Test {
public:
    void SetUp() override {
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
    }

    void TearDown() override { config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);}
protected:
    bool oriEnableAihacBackend = false;
};

void TestLoopDViewDAssemble(const Tensor &t0, const Tensor &t1, const Tensor &blockTable, Tensor &out, int s) {
    FUNCTION("main", FunctionType::DYNAMIC, {t0, t1, blockTable}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShapeDim(t0, 0) / s)) {
            SymbolicScalar idx = GetInputDataInt32Dim2(blockTable, i, 0);
            Tensor t0s = DView(t0, {s, s}, {idx * s, 0});

            Tensor qi(DT_FP32, {s, 2*s}, "qi");
            DAssemble(t1, {0, 0}, qi);
            DAssemble(t0s, {0, s}, qi);

            Tensor ki(DT_FP32, {s, 2*s}, "ki");
            DAssemble(t0s, {0, 0}, ki);
            DAssemble(t1, {0, s}, ki);

            Tensor t2 = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, ki);
            // conat((t0s + t1, t1)) @ concat (t0s, t1)^T
            DAssemble(t2, {idx * s, 0}, out);
        }
    }
}

TEST_F(DynamicPATest, TestDD) {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});
    std::vector<uint8_t> devProgBinary;

    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");  // [32, 32]
    Tensor blockTable{
        DT_INT32, {n, 1},
         "blockTable"
    };
    Tensor out(DT_FP32, {n * s, s}, "out");
    TestLoopDViewDAssemble(t0, t1, blockTable, out, s);

    auto funcMap = Program::GetInstance().GetFunctionMap();
    ALOG_INFO(funcMap.size());
}

TEST_F(DynamicPATest, dynamic_pa_low_lantency_unroll) {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    std::vector<uint8_t> devProgBinary;

    std::vector<int> input_param = {4, 1, 32, 1, 512, 64, 128, 32};
    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nk = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int blockSize = input_param[6];
    int nTile = input_param[7];
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    PaTileShapeConfig tileConfig;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v2TileShape = {nTile, 64};

    std::vector<int> seq(b, 256);

    int blockNum = 0;
    for (auto s : seq) {
        blockNum += ((s + (blockSize - 1)) / blockSize);
    }
    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = ((maxSeqAllBatch + (blockSize - 1)) / blockSize);

    Tensor qNope(DT_BF16, {b * nq * sq, dn}, "qNope");
    Tensor kNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "kNopeCache");
    Tensor vNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "vNopeCache");
    Tensor qRope(DT_BF16, {b * nq * sq, nk * dr}, "qRope");
    Tensor kRopeCache(DT_BF16, {int(blockNum * blockSize), nk * dr}, "kRope");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor paOut(DT_FP32, {b * nq * sq, dn}, "paOut");

    int maxUnrollTimes = 4;
    PageAttention(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize, softmaxScale, paOut,
        tileConfig, maxUnrollTimes);
}

TEST_F(DynamicPATest, dynamic_pa_low_lantency_manual_unroll) {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    config::SetPlatformConfig("PRINT_TENSOR_GRAPH", true);
    std::vector<uint8_t> devProgBinary;

    std::vector<int> input_param = {4, 1, 32, 1, 512, 64, 128, 32};
    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nk = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int blockSize = input_param[6];
    int nTile = input_param[7];
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    PaTileShapeConfig tileConfig;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v2TileShape = {nTile, 64};

    std::vector<int> seq(b, 256);

    int blockNum = 0;
    for (auto s : seq) {
        blockNum += ((s + (blockSize - 1)) / blockSize);
    }
    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = ((maxSeqAllBatch + (blockSize - 1)) / blockSize);

    Tensor qNope(DT_BF16, {b * nq * sq, dn}, "qNope");
    Tensor kNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "kNopeCache");
    Tensor vNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "vNopeCache");
    Tensor qRope(DT_BF16, {b * nq * sq, nk * dr}, "qRope");
    Tensor kRopeCache(DT_BF16, {int(blockNum * blockSize), nk * dr}, "kRope");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor paOut(DT_FP32, {b * nq * sq, dn}, "paOut");

    int maxUnrollTimes = 16;
    PageAttentionWithManualUnroll(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize, softmaxScale, paOut,
        tileConfig, maxUnrollTimes);

    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_main_2");
    EXPECT_NE(mainFunc, nullptr);
    EXPECT_EQ(mainFunc->GetCalleeFunctionList().size(), 1);
    auto loopFunc1 = mainFunc->GetCalleeFunctionList().front();
    EXPECT_NE(loopFunc1, nullptr);
    EXPECT_EQ(loopFunc1->GetFunctionType(), FunctionType::DYNAMIC_LOOP);
    EXPECT_EQ(loopFunc1->GetCalleeFunctionList().size(), 1);
    auto loopPathFunc1 = loopFunc1->GetCalleeFunctionList().front();
    EXPECT_NE(loopPathFunc1, nullptr);
    EXPECT_EQ(loopPathFunc1->GetFunctionType(), FunctionType::DYNAMIC_LOOP_PATH);
    EXPECT_EQ(loopPathFunc1->GetCalleeFunctionList().size(), 1);
    auto loopFunc2 = loopPathFunc1->GetCalleeFunctionList().front();
    EXPECT_NE(loopFunc2, nullptr);
    EXPECT_EQ(loopFunc2->GetFunctionType(), FunctionType::DYNAMIC_LOOP);
    EXPECT_EQ(loopFunc2->GetCalleeFunctionList().size(), 1);
    auto loopPathFunc2 = loopFunc2->GetCalleeFunctionList().front();
    EXPECT_NE(loopPathFunc2, nullptr);
    EXPECT_EQ(loopPathFunc2->GetFunctionType(), FunctionType::DYNAMIC_LOOP_PATH);
    EXPECT_EQ(loopPathFunc2->GetCalleeFunctionList().size(), 5);
    for (auto &loopFunc3 : loopPathFunc2->GetCalleeFunctionList()) {
        EXPECT_NE(loopFunc3, nullptr);
        EXPECT_EQ(loopFunc3->GetFunctionType(), FunctionType::DYNAMIC_LOOP);
        auto loopAttr = loopFunc3->GetDynloopAttribute();
        EXPECT_NE(loopAttr, nullptr);
        EXPECT_EQ(loopAttr->unrollTimes, maxUnrollTimes);
        maxUnrollTimes /= 2;
        ALOG_ERROR("unrollTimes: ", loopAttr->unrollTimes, " range: ",loopAttr->loopRange.Dump());
        EXPECT_EQ(loopAttr->pathList.size(), 4);
    }
}

TEST_F(DynamicPATest, dynamic_pa_high_throughput_only_batch_loop) {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    std::vector<uint8_t> devProgBinary;

    std::vector<int> input_param = {4, 1, 32, 1, 512, 64, 128, 32};
    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nk = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int blockSize = input_param[6];
    int nTile = input_param[7];
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    PaTileShapeConfig tileConfig;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v2TileShape = {nTile, 64};

    std::vector<int> seq(b, 256);

    int blockNum = 0;
    for (auto s : seq) {
        blockNum += ((s + (blockSize - 1)) / blockSize);
    }
    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = ((maxSeqAllBatch + (blockSize - 1)) / blockSize);

    Tensor qNope(DT_BF16, {b * nq * sq, dn}, "qNope");
    Tensor kNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "kNopeCache");
    Tensor vNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "vNopeCache");
    Tensor qRope(DT_BF16, {b * nq * sq, nk * dr}, "qRope");
    Tensor kRopeCache(DT_BF16, {int(blockNum * blockSize), nk * dr}, "kRope");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor paOut(DT_FP32, {b * nq * sq, dn}, "paOut");

    int maxUnrollTimes = 4;
    PageAttentionHighThroughput(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize,
        softmaxScale, paOut, tileConfig, maxUnrollTimes);
}

TEST_F(DynamicPATest, AllocatorCoverage) {
    Tensor qNope1(DT_BF16, {64, 64}, "qNope1");
    Tensor qNope2(DT_BF16, {64}, "qNope2");
    Tensor qNope3(DT_BF16, {64, 32}, "qNope3");
    Tensor qNope4(DT_BF16, {32, 64}, "qNope4");
    WorkspaceInfo outWspInfo(1, 0, 8192, qNope1.GetStorage());
    WorkspaceInfo inWspInfo1(1, 0, 128, qNope2.GetStorage());
    WorkspaceInfo inWspInfo2(1, 0, 4096, qNope3.GetStorage());
    WorkspaceInfo inWspInfo3(1, 0, 4096, qNope4.GetStorage());

    Allocator::IsRawQualified(outWspInfo, inWspInfo1);

    Allocator::IsRawQualified(outWspInfo, inWspInfo2);

    Allocator::IsRawQualified(outWspInfo, inWspInfo3);
}
