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
 * \file test_insert_sync.cpp
 * \brief Unit test for InsertSync.
 */
#include <gtest/gtest.h>
#include "tilefwk/platform.h"
#include "passes/block_graph_pass/copy_out_resolve.cpp"
#include "ut_json/ut_json_tool.h"
#define private public

using namespace npu::tile_fwk;
const int NUM_8 = 8;
const int NUM_16 = 16;
const int NUM_32 = 32;
const int NUM_64 = 64;
const int NUM_128 = 128;

namespace npu {
namespace tile_fwk {

class CopyOutResolveTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "CopyOutResolveStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};
//incast - COPYIN - EXP
TEST_F(CopyOutResolveTest, TestAll) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAll", "TestAll", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    // init LogicalTensor
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
}




// incast - Exp - Exp - outcast
TEST_F(CopyOutResolveTest, TestAllFunction) {

    // build graph
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("CopyOutResolveStrategy", {
        {   "RemoveRedundantReshape",   "RemoveRedundantReshape"},
        {                 "AutoCast",                 "AutoCast"},
        {      "InferMemoryConflict",      "InferMemoryConflict"},
        {       "RemoveUndrivenView",       "RemoveUndrivenView"},
        {           "ExpandFunction",           "ExpandFunction"},
        {        "MergeViewAssemble",        "MergeViewAssemble"},
        {             "SplitReshape",             "SplitReshape"},
        {           "SplitRawTensor",           "SplitRawTensor"},
        {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor"},
        {              "DuplicateOp",              "DuplicateOp"},
        {         "AssignMemoryType",         "AssignMemoryType"},
        {  "InferDiscontinuousInput",  "InferDiscontinuousInput"},
        {        "RemoveRedundantOp",        "RemoveRedundantOp"},
        {                   "SplitK",                   "SplitK"},
        {           "GraphPartition",           "GraphPartition"},
        {          "ReduceCopyMerge",          "ReduceCopyMerge"},
        {             "NBufferMerge",             "NBufferMerge"},
        {       "L1CopyInReuseMerge",       "L1CopyInReuseMerge"},
        {     "IntraSubgraphAdapter",     "IntraSubgraphAdapter"},
        {           "GenerateMoveOp",           "GenerateMoveOp"},
        { "CommonOperationEliminate", "CommonOperationEliminate"},
        {              "AxisCombine",              "AxisCombine"},
        {           "PadLocalBuffer",           "PadLocalBuffer"},
        {   "RemoveUnalignedReshape",   "RemoveUnalignedReshape"},
        {          "ReplaceTensor",              "ReplaceTensor"},
        {          "PreGraphProcess",          "PreGraphProcess"},
        {            "InferDynShape",            "InferDynShape"},
        {       "SubgraphToFunction",       "SubgraphToFunction"},
        {          "InferParamIndex",          "InferParamIndex"},
        {        "SrcDstBufferMerge",        "SrcDstBufferMerge"},
        {                 "AddAlloc",                 "AddAlloc"},
        {              "OoOSchedule",              "OoOSchedule"},
        {        "GlobalMemoryReuse",        "GlobalMemoryReuse"},
        {              "RemoveAlloc",              "RemoveAlloc"},
    });
    TileShape::Current().SetVecTile(NUM_32, NUM_32);
    Tensor incast(DT_FP32, {NUM_64, NUM_64}, "incast");
    Tensor outcast(DT_FP32, {NUM_64, NUM_64}, "incast");
    FUNCTION("TestAll") {
        // TileShape::Current().SetVecTile(NUM_32, NUM_32);
        // Tensor res = Exp(incast);
        // outcast = Exp(res);
        outcast= Exp(incast);

    }
    //execute pass copyoutresolve
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestAll");
    func->paramConfigs_.copyOutResolveCoalescing = 3;
    CopyOutResolve copyOutResolvePass;
    //check leafFuncAttr only had been inited
    // for (auto &leaf : func->rootFunc_->programs_) {
    //     std::shared_ptr<LeafFuncAttribute> leafAttr = leaf.second->GetLeafFuncAttribute();
    //     EXPECT_EQ(leafAttr->outcastCopyOutResolveCounterList.size(), 0);
    // }
    func->DumpJsonFile("/home/w00951930/open_pypto/pypto/AAAcor/cor2B.json");
    copyOutResolvePass.RunOnFunction(*func);
    //check leafFuncAttr had been valid
    // for (auto &leaf : func->rootFunc_->programs_) {
    //     std::shared_ptr<LeafFuncAttribute> leafAttr = leaf.second->GetLeafFuncAttribute();
    //     EXPECT_NE(leafAttr->outcastCopyOutResolveCounterList.size(), 0);
    // }
    func->DumpJsonFile("/home/w00951930/open_pypto/pypto/AAAcor/cor2A.json");
}
} // namespace tile_fwk
} // namespace npu

#undef private