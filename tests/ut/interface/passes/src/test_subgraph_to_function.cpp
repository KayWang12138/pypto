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
 * \file test_subgraph_to_function.cpp
 * \brief Unit test for SubgraphToFunction pass.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "interface/operation/attribute.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "passes/execute_graph_pass/subgraph_to_function.h"
#include "passes/pass_manager.h"
#include "ut_json/ut_json_tool.h"

using namespace npu::tile_fwk;

class SubgraphToFunctionTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        Program::GetInstance().GetHostMachine().cache_.Reset();
    }

    void TearDown() override {}
};

bool ArePsgHashesUnique(const Function &function) {
    std::unordered_set<size_t> hashSet;
    for (const auto &[psgId, program] : function.programs_) {
        (void)psgId;
        size_t hashValue = program->ComputeHash().GetHash();
        if (hashSet.find(hashValue) != hashSet.end()) {
            return false;
        }
        hashSet.insert(hashValue);
    }
    return true;
}

bool IsPSgToESgMapOneToOne(const std::multimap<int, int> &PSgToESgMap) {
    std::unordered_map<int, int> esgToPsgMap;
    for (const auto &[psgId, esgId] : PSgToESgMap) {
        if (esgToPsgMap.find(esgId) != esgToPsgMap.end()) {
            return false;
        }
        esgToPsgMap[esgId] = psgId;
    }
    return true;
}

std::multimap<int, int> GetPSgToESgMap(Function *rootFunc) {
    std::multimap<int, int> PSgToESgMap;

    for (size_t i = 0; i < rootFunc->Operations().size(); i++) {
        auto iter = rootFunc->Operations()[i].GetSubFuncInvokeInfo();
        int PSgId = iter.GetProgramId();
        int ESgId = rootFunc->Operations()[i].GetSubgraphID();
        PSgToESgMap.insert({PSgId, ESgId});
        ALOG_INFO_F("PSgId: %d, ESgId: %d", PSgId, ESgId);
    }
    return PSgToESgMap;
}

TEST_F(SubgraphToFunctionTest, DifferentOffset) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TILE_DifferentOffset", "TILE_DifferentOffset", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", "USE_MAX_FREQ_LABEL", true);

    Program::GetInstance().InsertFuncToFunctionMap("TILE_DifferentOffset", currFunctionPtr);

    constexpr int totalSubGraphCount = 3;
    constexpr int subGraphID0 = 0;
    constexpr int subGraphID1 = 1;
    constexpr int subGraphID2 = 2;

    constexpr int opMagic0 = 10032;
    constexpr int opMagic1 = 10039;
    constexpr int opMagic2 = 10043;
    constexpr int opMagic3 = 10021;
    constexpr int opMagic4 = 10024;
    constexpr int opMagic5 = 10023;
    constexpr int opMagic6 = 10026;
    constexpr int opMagic7 = 10029;
    constexpr int opMagic8 = 10030;

    constexpr int tensorMagic0 = 3;
    constexpr int tensorMagic1 = 15;
    constexpr int tensorMagic2 = 66;
    constexpr int tensorMagic3 = 79;
    constexpr int tensorMagic4 = 30;
    constexpr int tensorMagic5 = 35;
    constexpr int tensorMagic6 = 29;
    constexpr int tensorMagic7 = 34;
    constexpr int tensorMagic8 = 7;
    // prepare the graph
    std::vector<int> shape0 = {32, 8, 8};
    std::vector<int> shape1 = {16, 64};
    std::vector<int> shape2 = {16, 32};
    std::vector<int> shape3 = {16, 8, 8};
    auto shape3Imme = OpImmediate::Specified(shape3);
    auto shape2Imme = OpImmediate::Specified(shape2);
    auto shape1Imme = OpImmediate::Specified(shape1);
    std::shared_ptr<LogicalTensor> incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape0);
    incast->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    incast->SetMagic(tensorMagic0);
    incast->isSubGraphBoundary = true;

    std::shared_ptr<LogicalTensor> tensor0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    tensor0->SetMemoryTypeBoth(MEM_UB);
    tensor0->SetMagic(tensorMagic1);
    tensor0->subGraphID = subGraphID0;

    auto &copyopin0 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {tensor0});
    copyopin0.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({16, 0, 0}), MEM_UB, shape3Imme, shape3Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin0.UpdateSubgraphID(subGraphID0);
    copyopin0.opmagic = opMagic0;

    std::shared_ptr<LogicalTensor> tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    tensor1->SetMemoryTypeBoth(MEM_UB);
    tensor1->SetMagic(tensorMagic2);
    tensor1->subGraphID = subGraphID0;

    auto &reshapeop = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {tensor0}, {tensor1});
    reshapeop.UpdateSubgraphID(subGraphID0);
    reshapeop.opmagic = opMagic1;

    std::shared_ptr<LogicalTensor> input_tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    input_tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    input_tensor->SetMagic(tensorMagic3);
    input_tensor->isSubGraphBoundary = true;
    input_tensor->subGraphID = subGraphID0;

    auto &copyoutop0 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {tensor1}, {input_tensor});
    copyoutop0.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shape1Imme, shape1Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop0.UpdateSubgraphID(subGraphID0);
    copyoutop0.opmagic = opMagic2;

    std::shared_ptr<LogicalTensor> inner_tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    inner_tensor1->SetMemoryTypeBoth(MEM_UB);
    inner_tensor1->UpdateOffset({0, 0});
    inner_tensor1->subGraphID = subGraphID1;
    inner_tensor1->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> inner_tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    inner_tensor2->SetMemoryTypeBoth(MEM_UB);
    inner_tensor2->UpdateOffset({0, 32});
    inner_tensor2->subGraphID = subGraphID2;
    inner_tensor2->SetMagic(tensorMagic5);

    auto &copyopin1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {input_tensor}, {inner_tensor1});
    copyopin1.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin1.UpdateSubgraphID(subGraphID1);
    copyopin1.opmagic = opMagic3;

    auto &copyopin2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {input_tensor}, {inner_tensor2});
    copyopin2.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 32}), MEM_UB, shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin2.UpdateSubgraphID(subGraphID2);
    copyopin2.opmagic = opMagic4;

    std::shared_ptr<LogicalTensor> result_tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    result_tensor1->SetMemoryTypeBoth(MEM_UB);
    result_tensor1->subGraphID = subGraphID1;
    result_tensor1->SetMagic(tensorMagic6);
    std::shared_ptr<LogicalTensor> result_tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    result_tensor2->SetMemoryTypeBoth(MEM_UB);
    result_tensor2->subGraphID = subGraphID2;
    result_tensor2->SetMagic(tensorMagic7);
    auto &expopin1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inner_tensor1}, {result_tensor1});
    expopin1.UpdateSubgraphID(subGraphID1);
    expopin1.opmagic = opMagic5;

    auto &expopin2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inner_tensor2}, {result_tensor2});
    expopin2.UpdateSubgraphID(subGraphID2);
    expopin2.opmagic = opMagic6;

    std::shared_ptr<LogicalTensor> output_tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    output_tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    output_tensor->SetMagic(tensorMagic8);
    output_tensor->isSubGraphBoundary = true;

    auto &copyoutop1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {result_tensor1}, {output_tensor});
    copyoutop1.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop1.UpdateSubgraphID(subGraphID1);
    copyoutop1.opmagic = opMagic7;

    auto &copyoutop2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {result_tensor2}, {output_tensor});
    copyoutop2.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 32}), shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop2.UpdateSubgraphID(subGraphID2);
    copyoutop2.opmagic = opMagic8;

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(output_tensor);

    currFunctionPtr->SetTotalSubGraphCount(totalSubGraphCount);
    ALOG_INFO("draw program graph before pass.");

    std::stringstream ssBefore;
    ssBefore << "Before_subgraphToFunction";

    // call the pass
    SubgraphToFunction subgraphToFunction;
    subgraphToFunction.PreCheck(*currFunctionPtr);
    subgraphToFunction.RunOnFunction(*currFunctionPtr);
    subgraphToFunction.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_subgraphToFunction";

    // do the expect
    auto rootFunc = currFunctionPtr->rootFunc_;
    EXPECT_NE(rootFunc, nullptr);
    const auto &PSgToESgMap = GetPSgToESgMap(rootFunc);

    size_t originalSubgraphCount = currFunctionPtr->GetTotalSubGraphCount();
    std::unordered_set<int> uniquePSgIds;
    for (const auto &pair : PSgToESgMap) {
        uniquePSgIds.insert(pair.first);
    }
    size_t mergedSubgraphCount = uniquePSgIds.size();
    EXPECT_EQ(mergedSubgraphCount, originalSubgraphCount);
    EXPECT_TRUE(ArePsgHashesUnique(*rootFunc));
    EXPECT_TRUE(IsPSgToESgMapOneToOne(PSgToESgMap));
}

TEST_F(SubgraphToFunctionTest, SameOffset) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TILE_SameOffset", "TILE_SameOffset", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("TILE_SameOffset", currFunctionPtr);

    constexpr int totalSubGraphCount = 2;
    constexpr int subGraphID0 = 0;
    constexpr int subGraphID1 = 1;

    constexpr int opMagic3 = 10021;
    constexpr int opMagic4 = 10024;
    constexpr int opMagic5 = 10023;
    constexpr int opMagic6 = 10026;
    constexpr int opMagic7 = 10029;
    constexpr int opMagic8 = 10030;

    constexpr int tensorMagic3 = 79;
    constexpr int tensorMagic4 = 30;
    constexpr int tensorMagic5 = 35;
    constexpr int tensorMagic6 = 29;
    constexpr int tensorMagic7 = 34;
    constexpr int tensorMagic8 = 7;
    // prepare the graph
    std::vector<int> shape1 = {16, 64};
    std::vector<int> shape2 = {16, 32};
    std::vector<int> shape3 = {32, 32};
    auto shape2Imme = OpImmediate::Specified(shape2);
    std::shared_ptr<LogicalTensor> input_tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    input_tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    input_tensor->SetMagic(tensorMagic3);
    input_tensor->isSubGraphBoundary = true;

    std::shared_ptr<LogicalTensor> inner_tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    inner_tensor1->SetMemoryTypeBoth(MEM_UB);
    inner_tensor1->UpdateOffset({0, 0});
    inner_tensor1->subGraphID = subGraphID0;
    inner_tensor1->SetMagic(tensorMagic4);
    std::shared_ptr<LogicalTensor> inner_tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    inner_tensor2->SetMemoryTypeBoth(MEM_UB);
    inner_tensor2->UpdateOffset({0, 0});
    inner_tensor2->subGraphID = subGraphID1;
    inner_tensor2->SetMagic(tensorMagic5);
    auto &copyopin1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {input_tensor}, {inner_tensor1});
    copyopin1.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin1.UpdateSubgraphID(subGraphID0);
    copyopin1.opmagic = opMagic3;
    auto &copyopin2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {input_tensor}, {inner_tensor2});
    copyopin2.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin2.UpdateSubgraphID(subGraphID1);
    copyopin2.opmagic = opMagic4;

    std::shared_ptr<LogicalTensor> result_tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    result_tensor1->SetMemoryTypeBoth(MEM_UB);
    result_tensor1->subGraphID = subGraphID0;
    result_tensor1->SetMagic(tensorMagic6);
    std::shared_ptr<LogicalTensor> result_tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    result_tensor2->SetMemoryTypeBoth(MEM_UB);
    result_tensor2->subGraphID = subGraphID1;
    result_tensor2->SetMagic(tensorMagic7);
    auto &expopin1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inner_tensor1}, {result_tensor1});
    expopin1.UpdateSubgraphID(subGraphID0);
    expopin1.opmagic = opMagic5;
    auto &expopin2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inner_tensor2}, {result_tensor2});
    expopin2.UpdateSubgraphID(subGraphID1);
    expopin2.opmagic = opMagic6;

    std::shared_ptr<LogicalTensor> output_tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output_tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    output_tensor->SetMagic(tensorMagic8);
    output_tensor->isSubGraphBoundary = true;

    auto &copyoutop1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {result_tensor1}, {output_tensor});
    copyoutop1.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop1.UpdateSubgraphID(subGraphID0);
    copyoutop1.opmagic = opMagic7;
    auto &copyoutop2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {result_tensor2}, {output_tensor});
    copyoutop2.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({16, 0}), shape2Imme, shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop2.UpdateSubgraphID(subGraphID1);
    copyoutop2.opmagic = opMagic8;

    currFunctionPtr->inCasts_.push_back(input_tensor);
    currFunctionPtr->outCasts_.push_back(output_tensor);

    currFunctionPtr->SetTotalSubGraphCount(totalSubGraphCount);
    ALOG_INFO("draw program graph before pass.");

    Json progDump;
    progDump["version"] = "2.0";
    progDump["functions"].push_back(currFunctionPtr->DumpJson());
    auto filePath = "Before_subgraphIsomorphismPass.json";
    std::ofstream file(filePath);
    file << progDump.dump() << std::endl;
    file.close();

    std::stringstream ssBefore;
    ssBefore << "Before_subgraphToFunction";

    // call the pass
    SubgraphToFunction subgraphToFunction;
    subgraphToFunction.RunOnFunction(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_subgraphIsomorphismPass";

    // do the expect
    auto rootFunc = currFunctionPtr->rootFunc_;
    EXPECT_NE(rootFunc, nullptr);
    const auto &PSgToESgMap = GetPSgToESgMap(rootFunc);

    std::unordered_set<int> uniquePSgIds;
    for (const auto &pair : PSgToESgMap) {
        uniquePSgIds.insert(pair.first);
    }
    size_t mergedSubgraphCount = uniquePSgIds.size();
    EXPECT_EQ(mergedSubgraphCount, 1);
    EXPECT_TRUE(ArePsgHashesUnique(*rootFunc));
    EXPECT_TRUE(IsPSgToESgMapOneToOne(PSgToESgMap));
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load)
{
    int bs = 1;
    int m = 32;
    int k = 32;
    int n = 32;

    std::vector<int> shapeA = {bs, m, k};
    std::vector<int> shapeB = {bs, k, n};
    std::vector<int> shapeC = {bs, m, n};

    Program::GetInstance().GetConfig().Reset();
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});
    Tensor matA(DT_FP16, shapeA, "MatA", NodeType::LOCAL, TileOpFormat::TILEOP_NZ);
    Tensor matB(DT_FP16, shapeB, "MatB", NodeType::LOCAL, TileOpFormat::TILEOP_ND);
    Tensor matC(DT_FP32, shapeC, "MatC");
    FUNCTION("BATCHMATMUL", FunctionType::STATIC, {matA, matB, matC})
    {
        config::SetPassConfig("PVC2_OOO", "OoOSchedulePass", "DISABLE_PASS", true);
        matC = npu::tile_fwk::Matrix::BatchMatmul<false, false>(DT_FP32, matA, matB);
    }
    config::SetPassConfig("PVC2_OOO", "OoOSchedulePass", "DISABLE_PASS", false);
    auto programJson = Program::GetInstance().DumpJson();
    auto currentFunctionPtr = Program::GetInstance().GetCurrentFunction();
    EXPECT_EQ(Program::GetInstance().FunctionMapSize(), 6);
    ASSERT_NE(currentFunctionPtr, nullptr);
    EXPECT_EQ(currentFunctionPtr->Operations().size(), 1);
    EXPECT_EQ(currentFunctionPtr->GetRawName(), "PROGRAM_ENTRY");

    auto batchMatmulFunc = Program::GetInstance().GetFunctionByRawName("TENSOR_BATCHMATMUL");
    #ifndef PRIOR_SCHEDULING
    EXPECT_EQ(batchMatmulFunc->Operations().size(), 9);
    #endif

    ASSERT_NE(batchMatmulFunc->rootFunc_, nullptr);
    EXPECT_EQ(batchMatmulFunc->rootFunc_->Operations().size(), 4);
    EXPECT_EQ(batchMatmulFunc->rootFunc_->programs_.size(), 3);
    auto &oriPrograms = batchMatmulFunc->rootFunc_->programs_;
    EXPECT_EQ(oriPrograms[0]->Operations().size(), 1);
    #ifndef PRIOR_SCHEDULING
    EXPECT_EQ(oriPrograms[1]->Operations().size(), 8);
    #endif
    auto topoBefore = batchMatmulFunc->rootFunc_->topoInfo_;
    auto &entrysBefore = topoBefore.GetTopology();
    EXPECT_EQ(programJson["functions"].size(), 6);
    SubfuncInvokeInfoTy invokeInfo10000;
    SubfuncInvokeInfoTy invokeInfo10001;
    SubfuncInvokeInfoTy invokeInfo10002;
    SubfuncInvokeInfoTy invokeInfo10003;
    for (auto &op : batchMatmulFunc->rootFunc_->Operations()) {
        EXPECT_EQ(op.GetOpcode(), Opcode::OP_CALL);
        if (op.GetOpMagic() == 10000) {
            invokeInfo10000 = op.GetSubFuncInvokeInfo();
        }
        if (op.GetOpMagic() == 10001) {
            invokeInfo10001 = op.GetSubFuncInvokeInfo();
        }
        if (op.GetOpMagic() == 10002) {
            invokeInfo10002 = op.GetSubFuncInvokeInfo();
        }
        if (op.GetOpMagic() == 10003) {
            invokeInfo10003 = op.GetSubFuncInvokeInfo();
        }
    }

    Program::GetInstance().LoadJson(programJson);
    EXPECT_EQ(Program::GetInstance().FunctionMapSize(), 6);
    auto newCurrFuncPtr = Program::GetInstance().GetCurrentFunction();
    ASSERT_NE(newCurrFuncPtr, nullptr);
    // 校验CallOpAttribute
    ASSERT_NE(newCurrFuncPtr->rootFunc_, nullptr);
    EXPECT_EQ(newCurrFuncPtr->rootFunc_->Operations().size(), 4);

    for (auto &op : newCurrFuncPtr->rootFunc_->Operations()) {
        EXPECT_EQ(op.GetOpcode(), Opcode::OP_CALL);
        if (op.GetOpMagic() == 10000) {
            auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(callOpAttr, nullptr);
            EXPECT_EQ(*(callOpAttr->invokeInfo_), invokeInfo10000);
        }

        if (op.GetOpMagic() == 10001) {
            auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(callOpAttr, nullptr);
            EXPECT_EQ(*(callOpAttr->invokeInfo_), invokeInfo10001);
        }

        if (op.GetOpMagic() == 10002) {
            auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(callOpAttr, nullptr);
            EXPECT_EQ(*(callOpAttr->invokeInfo_), invokeInfo10002);
        }

        if (op.GetOpMagic() == 10003) {
            auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(callOpAttr, nullptr);
            EXPECT_EQ(*(callOpAttr->invokeInfo_), invokeInfo10003);
        }
    }
    batchMatmulFunc = Program::GetInstance().GetFunctionByRawName("TENSOR_BATCHMATMUL");
    #ifndef PRIOR_SCHEDULING
    EXPECT_EQ(batchMatmulFunc->Operations().size(), 9);
    #endif
    ASSERT_NE(batchMatmulFunc->rootFunc_, nullptr);
    EXPECT_EQ(batchMatmulFunc->rootFunc_->Operations().size(), 4);
    EXPECT_EQ(batchMatmulFunc->rootFunc_->programs_.size(), 3);

    // 校验Topo
    auto &topo = newCurrFuncPtr->rootFunc_->topoInfo_;
    auto &entrys = topo.GetTopology();
    EXPECT_EQ(entrysBefore.size(), entrys.size());

    for (size_t i = 0; i < entrysBefore.size(); i++) {
        EXPECT_EQ(entrys[i].esgId, entrysBefore[i].esgId);
        EXPECT_EQ(entrys[i].readyState, entrysBefore[i].readyState);
        EXPECT_EQ(entrys[i].outGraph, entrysBefore[i].outGraph);
    }

    // 校验rootFunc_->programs_
    auto &programs = newCurrFuncPtr->rootFunc_->programs_;
    ASSERT_EQ(programs.size(), 3);
    EXPECT_EQ(programs[0]->Operations().size(), 1);
    #ifndef PRIOR_SCHEDULING
    EXPECT_EQ(programs[1]->Operations().size(), 8);
    #endif
    // 校验CopyInCopyoutAttribute
    for (auto &op : programs[1]->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            auto copyInOpAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(copyInOpAttr, nullptr);

        }
        if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            auto copyOutOpAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(copyOutOpAttr, nullptr);
        }

    }
    EXPECT_EQ(programs[2]->Operations().size(), 1);

    programJson = Program::GetInstance().DumpJson();
    Program::GetInstance().LoadJson(programJson);
    Json programJsonNew = Program::GetInstance().DumpJson();
    EXPECT_EQ(programJsonNew.dump(), programJson.dump());
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load_1) {
    int32_t shape0 = 4;
    int32_t shape1 = 32;
    int32_t k = 8;
    bool isLargest = true;

    PROGRAM("TOPK") {
        std::vector<int> input_shape = {shape0, shape1};
        std::vector<int> output_shape = {shape0, k};
        Program::GetInstance().GetTileShape().SetVecTileShapes({shape0, shape1});
        Tensor input_a(DT_FP32, input_shape, (uint8_t *)nullptr, "A");
        auto output = std::make_tuple(Tensor(DT_FP32, output_shape, nullptr, "npu_val"),
                                      Tensor(DT_FP32, output_shape, nullptr, "resDics"));
        config::SetPassConfig("PVC2_OOO", "OoOSchedulePass", "DISABLE_PASS", true);
        FUNCTION("TOPK_T", FunctionType::STATIC, {input_a, std::get<0>(output), std::get<1>(output)}) {
            output = TopK(input_a, k, -1, isLargest);
        }
    }
    config::SetPassConfig("PVC2_OOO", "OoOSchedulePass", "DISABLE_PASS", false);

    Json programJson = Program::GetInstance().DumpJson();
    Program::GetInstance().LoadJson(programJson);
    Json programJsonNew = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJsonNew);
    Json programJsonNewNew = Program::GetInstance().DumpJson();

    #ifndef PRIOR_SCHEDULING
    EXPECT_EQ(programJsonNew.dump(), programJsonNewNew.dump());
    #endif
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, false);
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load_1_cov) {
    int32_t shape0 = 4;
    int32_t shape1 = 32;
    int32_t k = 8;
    bool isLargest = true;

    PROGRAM("TOPK") {
        std::vector<int> input_shape = {shape0, shape1};
        std::vector<int> output_shape = {shape0, k};
        Program::GetInstance().GetTileShape().SetVecTileShapes({shape0, shape1});
        Tensor input_a(DT_FP32, input_shape, (uint8_t *)nullptr, "A");
        auto output = std::make_tuple(Tensor(DT_FP32, output_shape, nullptr, "npu_val"),
                                      Tensor(DT_FP32, output_shape, nullptr, "resDics"));
        FUNCTION("TOPK_T", FunctionType::STATIC, {input_a, std::get<0>(output), std::get<1>(output)}) {
            output = TopK(input_a, k, -1, isLargest);
        }
    }

    Json programJson = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJson);
    Json programJsonNew = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJsonNew);
    Json programJsonNewNew = Program::GetInstance().DumpJson();

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, false);
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load_2) {
    IfaTileShapeConfig tileConfig {
        256, // block size
        32,  // nTile
        {256, 128}, // v0 tile for qkv-view-concat, q-S1D:(32,64), k/v-S2D:(256,64), merge 2D to copy
        {32, 32, 64, 64, 256, 256}, // c1 tile for S1D@S2D
        {32, 256}, // v1 tile for S1S2
        {32, 32, 64, 64, 256, 256}, // c2 tile for S1S2@S2D
        {32, 256}, // v2 tile for S1D
    };

    const int b = 4;
    const int nq = 32;
    const int s2 = 256;
    const int blockSize = tileConfig.blockSize;

    const int sq = 1;
    const int dn = 512;
    const int dr = 64;
    const int nkv = 1;

    std::vector<int> actSeqs(b, s2);
    const float softmaxScale = 0.8f;

    // 输出size
    // 根据Per Batch实际的sequence构造blockNum，blockNum >= Sum(blockNumPerBatch)，此处选取相等场景
    int blockNum = 0;
    for (auto s : actSeqs) {
        blockNum += CeilDiv(s, blockSize);
    }
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    PROGRAM("PageAttentionStatic") {

        Tensor qNope(DT_BF16, {b * sq * nq, dn}, (uint8_t *)nullptr, "qNope");
        Tensor qRope(DT_BF16, {b * sq * nq, dr}, (uint8_t *)nullptr, "qRope");
        Tensor kNopeCache(DT_BF16, {blockNum * blockSize * nkv, dn}, (uint8_t *)nullptr, "kNopeCache");
        Tensor kRopeCache(DT_BF16, {blockNum * blockSize * nkv, dr}, (uint8_t *)nullptr, "kRope");
        Tensor vNopeCache(DT_BF16, {blockNum * blockSize * nkv, dn}, (uint8_t *)nullptr, "vNopeCache");

        // blockTable: (b, maxBlockNumPerBatch)
        int maxSeqAllBatch = *(std::max_element(actSeqs.begin(), actSeqs.end()));
        int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
        std::vector<std::vector<int>> blockTable(b, std::vector<int>(maxBlockNumPerBatch, 0));

        Tensor attentionOut(DT_FP32, {b * sq * nq, dn}, nullptr, "attentionOut");

        // 计算流程开始
        FUNCTION("IfaStatic", FunctionType::STATIC,
            {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, attentionOut}) {
            IncreFlashAttention(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, softmaxScale,
                attentionOut, tileConfig);
        }
    }

    Json programJson = Program::GetInstance().DumpJson();
    Program::GetInstance().LoadJson(programJson);
    Json programJsonNew = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJsonNew);
    Json programJsonNewNew = Program::GetInstance().DumpJson();

    #ifndef PRIOR_SCHEDULING
    EXPECT_EQ(programJsonNew.dump(), programJsonNewNew.dump());
    #endif
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, false);
}

TEST_F(SubgraphToFunctionTest, VerifyPassResumeByJson) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    config::SetHostConfig(KEY_STRATEGY, "PreJsonResumeStrategy");
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PreJsonResumeStrategy", {
        {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
        {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
        {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
    });
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, 32, 32);
    std::vector<int> tshape = {2, 2, 64, 64};
    Tensor T(DT_FP32, tshape, "T");
    Tensor d;
    FUNCTION("A") {
        d = SoftmaxNew(T);
    }
    std::string jsonFilePath = "./config/pass/json/pass_resume.json";
    auto programJson = Program::GetInstance().DumpJson();
    DumpJsonFile(programJson, jsonFilePath);

    Program::GetInstance().Reset();
    Program::GetInstance().GetConfig().Reset();
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    config::SetHostConfig(KEY_STRATEGY, "JsonResumeStrategy");
    config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    PassManager &passManager1 = PassManager::Instance();
    passManager1.RegisterStrategy("JsonResumeStrategy", {
        {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
        {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
        {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
        {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
        {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
        {       "InsertConvertOp_01",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
        {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
        {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
        {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},
        {        "GenerateMoveOp_01",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
        {              "CubeProcess",              "CubeProcess",    PassType::TYPE_TILE_GRAPH},
        {        "GraphPartitionPass",        "GraphPartitionPass",    PassType::TYPE_TILE_GRAPH},
        {         "NBufferMergePass",         "NBufferMergePass",    PassType::TYPE_TILE_GRAPH},
        {          "UpdateMemoryMap",          "UpdateMemoryMap",    PassType::TYPE_TILE_GRAPH},
        {       "InsertConvertOp_02",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
        {        "GenerateMoveOp_02",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
        {   "SplitLargeLocalRawPass",   "SplitLargeLocalRawPass",    PassType::TYPE_TILE_GRAPH},
        {         "InsertCopyOpPass",         "InsertCopyOpPass",    PassType::TYPE_TILE_GRAPH},
        { "CommonOperationEliminate", "CommonOperationEliminate",    PassType::TYPE_TILE_GRAPH},
        {        "L1CopyInReusePass",        "L1CopyInReusePass",    PassType::TYPE_TILE_GRAPH},
        {             "PreGraphPass",             "PreGraphPass",    PassType::TYPE_TILE_GRAPH},
        {           "PadLocalBuffer",           "PadLocalBuffer",    PassType::TYPE_TILE_GRAPH},
        {  "RemoveUnalignedReshapeOp",  "RemoveUnalignedReshapeOp",    PassType::TYPE_TILE_GRAPH},
        {        "InferDynShapePass",        "InferDynShapePass",    PassType::TYPE_TILE_GRAPH},
        {       "SubgraphToFunction",       "SubgraphToFunction", PassType::TYPE_EXECUTE_GRAPH},
        {      "InferParamIndexPass",      "InferParamIndexPass", PassType::TYPE_EXECUTE_GRAPH},
        {    "SrcDstBufferMergePass",    "SrcDstBufferMergePass", PassType::TYPE_EXECUTE_GRAPH},
        {             "VFFusionPass",             "VFFusionPass", PassType::TYPE_EXECUTE_GRAPH},
        {             "AddAllocPass",             "AddAllocPass", PassType::TYPE_EXECUTE_GRAPH},
        {          "OoOSchedulePass",          "OoOSchedulePass", PassType::TYPE_EXECUTE_GRAPH},
        {              "MemoryReuse",              "MemoryReuse", PassType::TYPE_EXECUTE_GRAPH},
        {          "RemoveAllocPass",          "RemoveAllocPass", PassType::TYPE_EXECUTE_GRAPH},
        {          "PriorScheduling",          "PriorScheduling", PassType::TYPE_EXECUTE_GRAPH},
        {           "InsertSyncPass",           "InsertSyncPass", PassType::TYPE_EXECUTE_GRAPH},
        {       "CodegenPreprocPass",       "CodegenPreprocPass", PassType::TYPE_EXECUTE_GRAPH},
    });
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, 32, 32);
    std::vector<int> tshape1 = {2, 2, 64, 64};
    Tensor T1(DT_FP32, tshape1, "T1");
    Tensor d1;
    FUNCTION("A1") {
        d1 = SoftmaxNew(T);
    }
}
