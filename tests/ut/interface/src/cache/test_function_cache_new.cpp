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
 * \file test_function_cache_new.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include <thread>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_storage.h"
#include "interface/function/function.h"
#include "interface/cache/function_cache.h"

using namespace npu::tile_fwk;

class NewCacheTest : public testing::Test {
public:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(NewCacheTest, TestFunctionCache) {
    npu::tile_fwk::FunctionCache cache;
    EXPECT_EQ(cache.Size(), 0);
}

TEST_F(NewCacheTest, TestFunctionCacheInsert) {
    npu::tile_fwk::FunctionCache cache;
    HashKey key = 64;
    CacheValue cacheVal;
    cacheVal.header.coreFunctionNum = 0x1234;
    cache.Insert(key, cacheVal);
    EXPECT_EQ(cache.Size(), 1);
    EXPECT_EQ(cache.Get(key).value().header.coreFunctionNum, 0x1234);
}

void testInsert(npu::tile_fwk::FunctionCache& funCache, HashKey key, uint64_t val)
{
    CacheValue cacheVal;
    cacheVal.header.coreFunctionNum = val;
    funCache.Insert(key, cacheVal);
}

TEST_F(NewCacheTest, TestFunctionCacheInsertMultiThread) {
    npu::tile_fwk::FunctionCache cache;
    constexpr int32_t threadNum = 5000;
    std::vector<std::thread> threadVec;
    for (int32_t i = 0; i < threadNum; i++) {
        HashKey key = i;
        uint64_t val = i * 2;
        threadVec.emplace_back(std::thread(testInsert, std::ref(cache), key, val));
    }
    for (int32_t i = 0; i < threadNum; i++) {
        threadVec[i].join();
    }
    EXPECT_EQ(cache.Size(), threadNum);
    for (int32_t i = 0; i < threadNum; i++) {
        EXPECT_EQ(cache.Get(i).value().header.coreFunctionNum, 2 * i);
    }
}

TEST_F(NewCacheTest, TestFunctionCacheHitRate) {
    npu::tile_fwk::FunctionCache cache;
    constexpr int32_t threadNum = 500;
    std::vector<std::thread> threadVec;
    for (int32_t i = 0; i < threadNum; i++) {
        HashKey key = i;
        uint64_t val = i * 2;
        threadVec.emplace_back(std::thread(testInsert, std::ref(cache), key, val));
    }
    for (int32_t i = 0; i < threadNum; i++) {
        threadVec[i].join();
    }
    EXPECT_EQ(cache.Size(), threadNum);
    for (int32_t i = 0; i < threadNum * 2; i++) {
        cache.Get(i);
    }
    EXPECT_EQ(cache.GetHitRate(), "500/1000");
}

void GenerateFile(const uint32_t len, const uint32_t val, std::string fileName)
{
    // 生成文件
    auto* data = new uint32_t[len / sizeof(uint32_t)];
    for (uint32_t i = 0; i < len / sizeof(uint32_t); i++) {
        data[i] = val;
    }
    std::ofstream outFile(fileName);
    outFile.write((char*)data, len);
    outFile.close();
    delete []data;
}

TEST_F(NewCacheTest, TestFunctionCacheBinCache) {
    // 测试有3个二进制文件,文件大小分别为32/64/96Bytes
    // 则其数据结构应为
    // dataSize = 8 + 8 + 32 + 8 + 64 + 8 + 96 + 3 * 8 = 240 Bytes
    // 内存排布为:
    // | 240 | 32 72 144 | 8+32 Bytes 8+64B ytes 8+96 Bytes |
    Function func1(Program::GetInstance(), "", "", nullptr);
    const uint32_t binNum = 3;
    SubfuncParam subfuncPlaceHolder;
    std::vector<std::string> binFileNameVec(binNum);
    std::vector<uint32_t> binFileLenVec = {32, 64, 96};
    for (uint32_t i = 0; i < binNum; i++) {
        binFileNameVec[i] = "/tmp/" + std::to_string(i) + ".bin";
        GenerateFile(binFileLenVec[i], 100 + i, binFileNameVec[i]);
        // func1.programs_.insert({(uint64_t)i, subFuncVec[i]});
    }
    FunctionCache cacheInst;
    cacheInst.Insert(1, func1);

    // 比较真值及删除生成的临时文件
    CacheValue cache = cacheInst.Get(1).value();
    for (uint32_t i = 0; i < binNum; i++) {
        EXPECT_EQ(remove(binFileNameVec[i].c_str()), 0);
    }
    EXPECT_EQ(cache.binCache->dataSize, 240);
    EXPECT_EQ(cache.binCache->coreFunctionBinOffsets[0], 32);
    EXPECT_EQ(cache.binCache->coreFunctionBinOffsets[1], 72);
    EXPECT_EQ(cache.binCache->coreFunctionBinOffsets[2], 144);
    EXPECT_EQ(*(uint64_t*)((uint8_t*)cache.binCache + 32), 32);
    EXPECT_EQ(*(uint64_t*)((uint8_t*)cache.binCache + 72), 64);
    EXPECT_EQ(*(uint64_t*)((uint8_t*)cache.binCache + 144), 96);
    for (uint32_t i = 0; i < binFileLenVec[0] / sizeof(uint32_t); i++) {
        EXPECT_EQ(*((uint32_t*)((uint8_t*)cache.binCache + 32 + 8) + i), 100);
    }
    for (uint32_t i = 0; i < binFileLenVec[1] / sizeof(uint32_t); i++) {
        EXPECT_EQ(*((uint32_t*)((uint8_t*)cache.binCache + 72 + 8) + i), 101);
    }
    for (uint32_t i = 0; i < binFileLenVec[2] / sizeof(uint32_t); i++) {
        EXPECT_EQ(*((uint32_t*)((uint8_t*)cache.binCache + 144 + 8) + i), 102);
    }
}

TEST_F(NewCacheTest, TestFunctionCacheReadyFunction) {
    // 测试ReadyFunc包括3个AIC以及4个AIV
    // 总计有7个function， 8 + 7 * 9
    Function func1(Program::GetInstance(), "", "", nullptr);
    const uint32_t aicReadyNum = 3;
    const uint32_t aivReadyNum = 4;
    std::vector<int> aicReadyFuncVec = {121, 133, 144};
    std::vector<int> aivReadyFuncVec = {66, 77, 88, 99};
    func1.SetReadySubGraphIds(CoreType::AIC, aicReadyFuncVec);
    func1.SetReadySubGraphIds(CoreType::AIV, aivReadyFuncVec);

    FunctionCache cacheInst;
    cacheInst.Insert(1, func1);

    // 比较真值及删除生成的临时文件
    CacheValue cache = cacheInst.Get(1).value();
    EXPECT_EQ(cache.readyListCache->dataSize, (aicReadyNum + aivReadyNum) * 9);
    for (size_t i = 0; i < aicReadyFuncVec.size(); i++) {
        EXPECT_EQ(cache.readyListCache->readyCoreFunction[i].id, aicReadyFuncVec[i]);
        EXPECT_EQ(cache.readyListCache->readyCoreFunction[i].coreType, 0);
    }
    for (size_t i = 0; i < aivReadyFuncVec.size(); i++) {
        EXPECT_EQ(cache.readyListCache->readyCoreFunction[i + aicReadyNum].id, aivReadyFuncVec[i]);
        EXPECT_EQ(cache.readyListCache->readyCoreFunction[i + aicReadyNum].coreType, 1);
    }
}

TEST_F(NewCacheTest, TestFunctionCacheTopo) {
    // 测试有2个TOPO的场景
    Function func1(Program::GetInstance(), "", "", nullptr);
    SubfuncTopologyInfoTy::Entry entry1 = {1, -1, setType({1, 2, 3, 4, 5}), 0, 0, std::vector<int64_t>({})};
    SubfuncTopologyInfoTy::Entry entry2 = {100, -2, setType({10, 11, 12}), 0, 0, std::vector<int64_t>({})};
    func1.topoInfo_.topology_.push_back(entry1);
    func1.topoInfo_.topology_.push_back(entry2);

    FunctionCache cacheInst;
    cacheInst.Insert(1, func1);

    // 比较真值
    CacheValue cache = cacheInst.Get(1).value();
    EXPECT_EQ(cache.topoCache->dataSize, 130);
    EXPECT_EQ(cache.topoCache->coreFunctionTopoOffsets[0], 24);
    EXPECT_EQ(cache.topoCache->coreFunctionTopoOffsets[1], 89);
}