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
 * \file test_mix_subgraph_split.cpp
  * \brief Unit test for mixSubgraphSplit
  * */
#include <gtest/gtest.h>
#include "computational_graph_builder.h"
#include "passes/block_graph_pass/mix_dependency_analyzer.h"

namespace npu {
namespace tile_fwk {
static const int kNum0 = 0;
static const int kNum1 = 1;
static const int kNum2 = 2;
static const int kNum3 = 3;
static const int kNum4 = 4;

class MixDependencyAnalyzerTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(MixDependencyAnalyzerTest, UTest1) {
    MixDependencyAnalyzer analyzer;
    std::unordered_map<int, std::set<int>> dependencies;
    dependencies[kNum0].insert(kNum1);
    dependencies[kNum1].insert(kNum2);
    dependencies[kNum2].insert(kNum3);
    dependencies[kNum3].insert(kNum4);
    analyzer.ComputeDependencyClosure(dependencies);

    EXPECT_EQ(dependencies[kNum0].size(), kNum4);
    EXPECT_EQ(dependencies[kNum1].size(), kNum3);
    EXPECT_EQ(dependencies[kNum2].size(), kNum2);
    EXPECT_EQ(dependencies[kNum3].size(), kNum1);
}

TEST_F(MixDependencyAnalyzerTest, UTest2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "Test", "Test", nullptr);
    MixDependencyAnalyzer analyzer;
    std::unordered_map<int, std::set<int>> dependencies;
    std::unordered_map<int, std::vector<SimpleTensorParam>> allIncasts;
    std::unordered_map<int, std::vector<SimpleTensorParam>> allOutcasts;
    std::vector<int64_t> shape = {kNum1, kNum1};
    LogicalTensorPtr tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    LogicalTensorPtr tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    LogicalTensorPtr tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    LogicalTensorPtr tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    LogicalTensorPtr tensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    dependencies[kNum0].insert(kNum1);
    dependencies[kNum1].insert(kNum2);
    dependencies[kNum2].insert(kNum3);
    dependencies[kNum3].insert(kNum4);
    allIncasts[kNum0].emplace_back(SimpleTensorParam(tensor1, kNum0, kNum0));
    allIncasts[kNum2].emplace_back(SimpleTensorParam(tensor2, kNum2, kNum0));
    allIncasts[kNum3].emplace_back(SimpleTensorParam(tensor3, kNum3, kNum0));
    allOutcasts[kNum1].emplace_back(SimpleTensorParam(tensor4, kNum1, kNum0));
    allOutcasts[kNum4].emplace_back(SimpleTensorParam(tensor5, kNum3, kNum0));
    analyzer.ComputeDependencyClosure(dependencies);
    analyzer.PropagateExternalDependenciesWithClosure(dependencies, allIncasts, allOutcasts);

    EXPECT_EQ(allIncasts[kNum0].size(), kNum1);
    EXPECT_EQ(allOutcasts[kNum0].size(), kNum2);
    EXPECT_EQ(allIncasts[kNum1].size(), kNum1);
    EXPECT_EQ(allOutcasts[kNum1].size(), kNum2);
    EXPECT_EQ(allIncasts[kNum2].size(), kNum2);
    EXPECT_EQ(allOutcasts[kNum2].size(), kNum1);
    EXPECT_EQ(allIncasts[kNum3].size(), kNum3);
    EXPECT_EQ(allOutcasts[kNum3].size(), kNum1);
    EXPECT_EQ(allIncasts[kNum4].size(), kNum3);
    EXPECT_EQ(allOutcasts[kNum4].size(), kNum1);
}
} // namespace tile_fwk
} // namespace npu


