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
#include <fstream>
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "test_common.h"
#include "test_suite_stest_ops.h"
#include "runtime.h"
#include "device_runner.h"
#define private public
#include "machine/cache_manager/cache_manager.h"
#undef private

using namespace npu::tile_fwk;

class OnBoardPaCostTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

static void readBlockTableFromFile(const std::string& filename, int rows, int cols, std::vector<std::vector<int>> & blockTable) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading!" << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i) {
        inFile.read(reinterpret_cast<char*>(blockTable[i].data()), cols * sizeof(int));
    }

    inFile.close();
    return;
}

static void IfaCommonTestInner(IfaTestParam params, IfaTileShapeConfig tileConfig, bool enablePerformanceCheck = false) {
    const int time2cycleFactor = 50;
    const int b = params["b"];
    const int nq = params["nq"];
    const int s2 = params["s2"];
    const int timeThreshold = params["timethreshold"];
    const int blockSize = tileConfig.blockSize;
    if (enablePerformanceCheck) {
        std::cout<<"======================================= timeTh " << timeThreshold << std::endl;
    }
    const int sq = 1;
    const int dn = 512;
    const int dr = 64;
    const int nkv = 1;

    std::vector<int> actSeqs(b, s2);
    const float softmaxScale = static_cast<float>(1.0 / std::sqrt(dn + dr));

    // 输出size
    int outCap = b * 1 * nq * dn;
    uint64_t outputSize = outCap * sizeof(float);
    uint8_t *outPtr = allocDevAddr(outputSize);

    // 根据Per Batch实际的sequence构造blockNum，blockNum >= Sum(blockNumPerBatch)，此处选取相等场景
    int blockNum = 0;
    for (auto s : actSeqs) {
        blockNum += CeilDiv(s, blockSize);
    }

    // 输入size
    int qNopeSize = b * sq * nq * dn;
    int qRopeSize = b * sq * nq * dr;

    // B B H
    int kvNopeCacheSize = blockNum * blockSize * nkv * dn;
    int kRopeCacheSize = blockNum * blockSize * nkv * dr;


    // 读数据
    void *qNopeData = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_nope.bin", qNopeSize);
    void *qRopeData = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_rope.bin", qRopeSize);

    void *kvNopeCacheData = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/kv_cache_nope_nz.bin", kvNopeCacheSize);
    void *kRopeCacheData = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_rope_nz.bin", kRopeCacheSize);

    Tensor qNope(DT_BF16, {b * sq * nq, dn}, (uint8_t *)qNopeData, "qNope");
    Tensor qRope(DT_BF16, {b * sq * nq, dr}, (uint8_t *)qRopeData, "qRope");

    Tensor kvNopeCache(DT_BF16, {blockNum * blockSize * nkv, dn}, (uint8_t *)kvNopeCacheData, "kNopeCache", NodeType::LOCAL, TileOpFormat::TILEOP_NZ);
    Tensor kRopeCache(DT_BF16, {blockNum * blockSize * nkv, dr}, (uint8_t *)kRopeCacheData, "kRope", NodeType::LOCAL, TileOpFormat::TILEOP_NZ);

    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(actSeqs.begin(), actSeqs.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
    std::vector<std::vector<int>> blockTable(b, std::vector<int>(maxBlockNumPerBatch, 0));
    readBlockTableFromFile(GetGoldenDir() + "/block_table.bin", b, maxBlockNumPerBatch, blockTable);

    Tensor attentionOut(DT_FP32, {b * sq * nq, dn}, outPtr, "attentionOut");

    // 计算流程开始
    TileFwkBeginFunction("IfaStatic",
        {qNope, kvNopeCache, qRope, kRopeCache, attentionOut});
    {
        IncreFlashAttention(qNope, kvNopeCache, kvNopeCache, qRope, kRopeCache, blockTable, actSeqs, softmaxScale,
            attentionOut, tileConfig);
    }
    TileFwkEndFunction();
    // 计算流程结束

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    std::vector<void*> opArgsRun = {qNopeData, kvNopeCacheData, qRopeData, kRopeCacheData, outPtr};

    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    /*********  accuracy compare  *********/
    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outPtr, outputSize);
    readInput(GetGoldenDir() + "/atten_out.bin", golden);
    int ret = resultCmp(golden, res, 0.003f);

    /*********  performance compare  *********/
    if (enablePerformanceCheck) {
        uint64_t taskTime = npu::tile_fwk::DeviceRunner::Get().GetTasksTime();
        uint64_t threshold = timeThreshold * time2cycleFactor; // N us * 50 = cycle num
        ret = ret && (taskTime > 0 && taskTime < threshold * 1.05);
        std::cout<<"================= threshold : "<<threshold<<
        " cost : "<<taskTime<<
        " ================="<<std::endl;
    }

    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardPaCostTest, test_page_attention_low_latency_cost) {
    CacheMode cacheMode = CacheManager::Instance().GetCacheMode();
    CacheManager::Instance().cacheMode_ = CacheMode::Disable;
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    config::SetOperationConfig("FORCE_COMBINE_AXIS", true);
    Program::GetInstance().GetConfig().Set<std::map<int, int>>(L1_REUSE_MAP, {{1,4}});

    TileFwkInit("");
    IfaCommonTestInner(lowLatencyParams, lowLatencyTileParams, true);
    CacheManager::Instance().cacheMode_ = cacheMode;
}

TEST_F(OnBoardPaCostTest, test_page_attention_low_latency_cost_precision) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());

    TileFwkInit("");
    IfaCommonTestInner(lowLatencyParams, lowLatencyTileParams, false);
}

TEST_F(OnBoardPaCostTest, test_page_attention_hight_throughput_cost) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    config::SetOperationConfig("FORCE_COMBINE_AXIS", true);
    Program::GetInstance().GetConfig().Set<int>(CYCLES_THRESHOLD, 2048);
    Program::GetInstance().GetConfig().Set<int>(CYCLE_UPPER_BOUND, 20000);
    Program::GetInstance().GetConfig().Set<int>(L1_REUSE, 4);
    Program::GetInstance().GetConfig().Set<int>(CUBE_NBUFFER, 2);
    Program::GetInstance().GetConfig().Set<int>(COPYIN_THRESHOLD, 2*1024*1024);
    TileFwkInit("");
    IfaCommonTestInner(hightThroughputParams, hightThroughputTileParams, true);
}

TEST_F(OnBoardPaCostTest, test_page_attention_hight_throughput_cost_precision) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());

    TileFwkInit("");
    IfaCommonTestInner(hightThroughputParams, hightThroughputTileParams, false);
}
