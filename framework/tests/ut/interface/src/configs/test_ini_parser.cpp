/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_config_manager.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/utils/file_utils.h"
#include "interface/configs/ini_parser.h"

using namespace npu::tile_fwk;

class TestINIParser : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {
        std::string src = GetCurRunningPath() +
                          "/../../../framework/tests/ut/machine/stubs/compiler/data/platform_config/Ascend910_9572.ini";
        std::string dst = RealPath(GetCurrentSharedLibPath() + "/configs") + "/Soc_version.ini";
        std::string command = "cp " + src + " " + dst;
        ASSERT(std::system(command.c_str()) == 0) << "Failed to copy config file: " << command;
    }
    void TearDown() override {}
};

void TestParse() {
    const std::string version = "version";
    const std::string socInfo = "SoCInfo";
    const std::string aiCoreSpec = "AICoreSpec";
    const std::string shortSocVer = "Short_SoC_version";

    struct SizeTest {
        const std::string section;
        const std::string key;
        const size_t expected;
    };
    const std::vector<SizeTest> coreTests = {
        {socInfo,     "ai_core_cnt", 28UL},
        {socInfo,   "cube_core_cnt", 28UL},
        {socInfo, "vector_core_cnt", 56UL},
        {socInfo,      "ai_cpu_cnt",  6UL}
    };
    const std::vector<SizeTest> memoryTests = {
        {aiCoreSpec, "l0_a_size",  65536UL},
        {aiCoreSpec, "l0_b_size",  65536UL},
        {aiCoreSpec, "l0_c_size", 262144UL},
        {aiCoreSpec,   "l1_size", 524288UL},
        {aiCoreSpec,   "ub_size", 253952UL}
    };
    INIParser parser;
    std::string iniPath = RealPath(GetCurrentSharedLibPath() + "/configs/Soc_version.ini");
    EXPECT_EQ(parser.Initialize(iniPath), SUCCESS);

    std::string socVersion;
    EXPECT_EQ(parser.GetStringVal(version, shortSocVer, socVersion), SUCCESS);
    EXPECT_EQ(socVersion, "Ascend910_95");

    std::unordered_map<std::string, std::string> ccecVersion;
    EXPECT_EQ(parser.GetCCECVersion(ccecVersion), SUCCESS);
    EXPECT_NE(ccecVersion.find("AIC"), ccecVersion.end());
    EXPECT_EQ(ccecVersion["AIC"], "dav-c310");
    EXPECT_NE(ccecVersion.find("AIV"), ccecVersion.end());
    EXPECT_EQ(ccecVersion["AIV"], "dav-c310");

    size_t coreNum;
    for (const auto &test : coreTests) {
        EXPECT_EQ(parser.GetSizeVal(test.section, test.key, coreNum), SUCCESS);
        EXPECT_EQ(coreNum, test.expected);
    }
    size_t memoryLimit;
    for (const auto &test : memoryTests) {
        EXPECT_EQ(parser.GetSizeVal(test.section, test.key, memoryLimit), SUCCESS);
        EXPECT_EQ(memoryLimit, test.expected);
    }
    std::vector<std::vector<std::string>> dataPath;
    EXPECT_EQ(parser.GetDataPath(dataPath), SUCCESS);
}

void AbnormalTest() {
    INIParser parser;
    const std::string version = "version";
    EXPECT_EQ(parser.Initialize(""), FAILED);

    std::unordered_map<std::string, std::string> ccecVersion;
    EXPECT_EQ(parser.GetCCECVersion(ccecVersion), FAILED);
    EXPECT_EQ(parser.GetCoreVersion(ccecVersion), FAILED);

    std::vector<std::vector<std::string>> dataPath;
    EXPECT_EQ(parser.GetDataPath(dataPath), FAILED);

    std::string iniPath = RealPath(GetCurrentSharedLibPath() + "/configs/Soc_version.ini");
    EXPECT_EQ(parser.Initialize(iniPath), SUCCESS);

    std::string test;
    EXPECT_EQ(parser.GetStringVal("none", "", test), FAILED);
    EXPECT_EQ(parser.GetStringVal(version, "none_other", test), SUCCESS);

    size_t testSize;
    EXPECT_EQ(parser.GetSizeVal("none", "", testSize), FAILED);
}

TEST_F(TestINIParser, TestParser) {
    // 保证两个用例同步 避免多进程共享资源出错
    TestParse();
    AbnormalTest();
}

TEST_F(TestINIParser, TestObtainPlatformInfo) {
    const std::string aic = "AIC";
    const std::string aiv = "AIV";
    const size_t expectAICoreCnt = 24UL;
    const size_t expectCubeCoreCnt = 24UL;
    const size_t expectVectorCoreCnt = 48UL;
    const size_t expectAICpuCnt = 6UL;
    const size_t expectl0aSize = 65536UL;
    const size_t expectl0bSize = 65536UL;
    const size_t expectl0cSize = 131072UL;
    const size_t expectl1Size = 524288UL;
    const size_t expectubSize = 196608UL;

    Platform::Instance().ObtainPlatformInfo();
    EXPECT_EQ(Platform::Instance().GetSoc().GetNPUArch(), NPUArch::DAV_2201);
    EXPECT_EQ(Platform::Instance().GetSoc().GetCCECVersion(aic), "dav-c220-cube");
    EXPECT_EQ(Platform::Instance().GetSoc().GetCCECVersion(aiv), "dav-c220-vec");
    EXPECT_EQ(Platform::Instance().GetSoc().GetCoreVersion(aic), "AIC-C-220");
    EXPECT_EQ(Platform::Instance().GetSoc().GetAICoreNum(), expectAICoreCnt);
    EXPECT_EQ(Platform::Instance().GetSoc().GetAICCoreNum(), expectCubeCoreCnt);
    EXPECT_EQ(Platform::Instance().GetSoc().GetAIVCoreNum(), expectVectorCoreCnt);
    EXPECT_EQ(Platform::Instance().GetSoc().GetAICPUNum(), expectAICpuCnt);

    EXPECT_EQ(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0A), expectl0aSize);
    EXPECT_EQ(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0B), expectl0bSize);
    EXPECT_EQ(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0C), expectl0cSize);
    EXPECT_EQ(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1), expectl1Size);
    EXPECT_EQ(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB), expectubSize);

    std::vector<MemoryType> paths;
    EXPECT_TRUE(Platform::Instance().GetDie().FindNearestPath(MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, paths));
    EXPECT_EQ(paths.size(), 2UL);
}