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
 * \file test_subgraph_to_function.cpp
 * \brief Unit test for SubgraphToFunction pass.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "tilefwk/data_type.h"
#include "interface/operation/attribute.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "passes/tile_graph_pass/subgraph_to_function.h"
#include "passes/tile_graph_pass/static_subgraph_processor.h"
#include "passes/pass_mgr/pass_manager.h"
#include "passes/statistics/execute_graph_statistic.h"
#include "ut_json/ut_json_tool.h"
#include "passes/pass_utils/boundary_utils.h"

namespace npu {
namespace tile_fwk {

class SubgraphToFunctionTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }

    void TearDown() override {}
};

bool ArePsgHashesUnique(const Function& function)
{
    std::unordered_set<size_t> hashSet;
    for (const auto& [psgId, program] : function.programs_) {
        (void)psgId;
        size_t hashValue = program->ComputeHash().GetHash();
        if (hashSet.find(hashValue) != hashSet.end()) {
            return false;
        }
        hashSet.insert(hashValue);
    }
    return true;
}

bool IsPSgToESgMapOneToOne(const std::multimap<int, int>& PSgToESgMap)
{
    std::unordered_map<int, int> esgToPsgMap;
    for (const auto& [psgId, esgId] : PSgToESgMap) {
        if (esgToPsgMap.find(esgId) != esgToPsgMap.end()) {
            return false;
        }
        esgToPsgMap[esgId] = psgId;
    }
    return true;
}

std::multimap<int, int> GetPSgToESgMap(Function* rootFunc)
{
    std::multimap<int, int> PSgToESgMap;

    for (size_t i = 0; i < rootFunc->Operations().size(); i++) {
        auto iter = rootFunc->Operations()[i].GetSubFuncInvokeInfo();
        int PSgId = iter.GetProgramId();
        int ESgId = rootFunc->Operations()[i].GetSubgraphID();
        PSgToESgMap.insert({PSgId, ESgId});
    }
    return PSgToESgMap;
}

std::shared_ptr<Function> CreateAndRegisterFunction(const std::string& name)
{
    auto functionPtr = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
    EXPECT_NE(functionPtr, nullptr);
    Program::GetInstance().InsertFuncToFunctionMap(name, functionPtr);
    return functionPtr;
}

std::shared_ptr<LogicalTensor> CreateTensor(
    const std::shared_ptr<Function>& functionPtr, const std::vector<int64_t>& shape, int magic, MemoryType memoryType)
{
    auto tensor = std::make_shared<LogicalTensor>(*functionPtr, DT_FP32, shape);
    tensor->SetMemoryTypeBoth(memoryType);
    tensor->SetMagic(magic);
    return tensor;
}

std::shared_ptr<LogicalTensor> CreateSubgraphTensor(
    const std::shared_ptr<Function>& functionPtr, const std::vector<int64_t>& shape, int magic, int subGraphId,
    const std::vector<int64_t>& offset = {})
{
    auto tensor = CreateTensor(functionPtr, shape, magic, MEM_UB);
    if (!offset.empty()) {
        tensor->UpdateOffset(offset);
    }
    tensor->subGraphID = subGraphId;
    return tensor;
}

void AddCopyInOp(
    Function& function, const LogicalTensorPtr& input, const LogicalTensorPtr& output,
    const std::vector<int64_t>& offset, const OpImmediate& shapeImmediate, int subGraphId, int opMagic)
{
    auto& op = function.AddOperation(Opcode::OP_COPY_IN, {input}, {output});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(offset), MEM_UB, shapeImmediate, shapeImmediate, std::vector<OpImmediate>()));
    op.UpdateSubgraphID(subGraphId);
    op.opmagic = opMagic;
}

void AddCopyOutOp(
    Function& function, const LogicalTensorPtr& input, const LogicalTensorPtr& output,
    const std::vector<int64_t>& offset, const OpImmediate& shapeImmediate, int subGraphId, int opMagic)
{
    auto& op = function.AddOperation(Opcode::OP_COPY_OUT, {input}, {output});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        MEM_UB, OpImmediate::Specified(offset), shapeImmediate, shapeImmediate, std::vector<OpImmediate>()));
    op.UpdateSubgraphID(subGraphId);
    op.opmagic = opMagic;
}

void RunSubgraphToFunctionPass(Function& function)
{
    SubgraphToFunction subgraphToFunction;
    subgraphToFunction.PreCheck(function);
    subgraphToFunction.RunOnFunction(function);
    subgraphToFunction.PostCheck(function);
}

size_t CountMergedSubgraphCount(const std::multimap<int, int>& psgToEsgMap)
{
    std::unordered_set<int> uniquePSgIds;
    for (const auto& pair : psgToEsgMap) {
        uniquePSgIds.insert(pair.first);
    }
    return uniquePSgIds.size();
}

TEST_F(SubgraphToFunctionTest, DifferentOffset)
{
    auto currFunctionPtr = CreateAndRegisterFunction("TILE_DifferentOffset");

    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", "use_max_freq_label", true);

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
    const std::vector<int64_t> shape0 = {32, 8, 8};
    const std::vector<int64_t> shape1 = {16, 64};
    const std::vector<int64_t> shape2 = {16, 32};
    const std::vector<int64_t> shape3 = {16, 8, 8};
    auto shape3Imme = OpImmediate::Specified(shape3);
    auto shape2Imme = OpImmediate::Specified(shape2);
    auto shape1Imme = OpImmediate::Specified(shape1);
    auto incast = CreateTensor(currFunctionPtr, shape0, tensorMagic0, MEM_DEVICE_DDR);
    auto tensor0 = CreateSubgraphTensor(currFunctionPtr, shape3, tensorMagic1, subGraphID0);
    auto tensor1 = CreateSubgraphTensor(currFunctionPtr, shape1, tensorMagic2, subGraphID0);
    AddCopyInOp(*currFunctionPtr, incast, tensor0, {16, 0, 0}, shape3Imme, subGraphID0, opMagic0);

    auto& reshapeop = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {tensor0}, {tensor1});
    reshapeop.UpdateSubgraphID(subGraphID0);
    reshapeop.opmagic = opMagic1;

    auto input_tensor = CreateTensor(currFunctionPtr, shape1, tensorMagic3, MEM_DEVICE_DDR);
    input_tensor->subGraphID = subGraphID0;
    AddCopyOutOp(*currFunctionPtr, tensor1, input_tensor, {0, 0}, shape1Imme, subGraphID0, opMagic2);

    auto inner_tensor1 = CreateSubgraphTensor(currFunctionPtr, shape2, tensorMagic4, subGraphID1, {0, 0});
    auto inner_tensor2 = CreateSubgraphTensor(currFunctionPtr, shape2, tensorMagic5, subGraphID2, {0, 32});
    AddCopyInOp(*currFunctionPtr, input_tensor, inner_tensor1, {0, 0}, shape2Imme, subGraphID1, opMagic3);
    AddCopyInOp(*currFunctionPtr, input_tensor, inner_tensor2, {0, 32}, shape2Imme, subGraphID2, opMagic4);

    auto result_tensor1 = CreateSubgraphTensor(currFunctionPtr, shape2, tensorMagic6, subGraphID1);
    auto result_tensor2 = CreateSubgraphTensor(currFunctionPtr, shape2, tensorMagic7, subGraphID2);
    auto& expopin1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inner_tensor1}, {result_tensor1});
    expopin1.UpdateSubgraphID(subGraphID1);
    expopin1.opmagic = opMagic5;

    auto& expopin2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inner_tensor2}, {result_tensor2});
    expopin2.UpdateSubgraphID(subGraphID2);
    expopin2.opmagic = opMagic6;

    auto output_tensor = CreateTensor(currFunctionPtr, shape1, tensorMagic8, MEM_DEVICE_DDR);
    AddCopyOutOp(*currFunctionPtr, result_tensor1, output_tensor, {0, 0}, shape2Imme, subGraphID1, opMagic7);
    AddCopyOutOp(*currFunctionPtr, result_tensor2, output_tensor, {0, 32}, shape2Imme, subGraphID2, opMagic8);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(output_tensor);

    currFunctionPtr->SetTotalSubGraphCount(totalSubGraphCount);

    RunSubgraphToFunctionPass(*currFunctionPtr);

    auto rootFunc = currFunctionPtr->rootFunc_;
    EXPECT_NE(rootFunc, nullptr);
    const auto& PSgToESgMap = GetPSgToESgMap(rootFunc);

    EXPECT_EQ(CountMergedSubgraphCount(PSgToESgMap), currFunctionPtr->GetTotalSubGraphCount());
    EXPECT_TRUE(ArePsgHashesUnique(*rootFunc));
    EXPECT_TRUE(IsPSgToESgMapOneToOne(PSgToESgMap));
}

TEST_F(SubgraphToFunctionTest, SameOffset)
{
    auto currFunctionPtr = CreateAndRegisterFunction("TILE_SameOffset");

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
    const std::vector<int64_t> shape1 = {16, 64};
    const std::vector<int64_t> shape2 = {16, 32};
    const std::vector<int64_t> shape3 = {32, 32};
    auto shape2Imme = OpImmediate::Specified(shape2);
    auto input_tensor = CreateTensor(currFunctionPtr, shape1, tensorMagic3, MEM_DEVICE_DDR);

    auto inner_tensor1 = CreateSubgraphTensor(currFunctionPtr, shape2, tensorMagic4, subGraphID0, {0, 0});
    auto inner_tensor2 = CreateSubgraphTensor(currFunctionPtr, shape2, tensorMagic5, subGraphID1, {0, 0});
    AddCopyInOp(*currFunctionPtr, input_tensor, inner_tensor1, {0, 0}, shape2Imme, subGraphID0, opMagic3);
    AddCopyInOp(*currFunctionPtr, input_tensor, inner_tensor2, {0, 0}, shape2Imme, subGraphID1, opMagic4);

    auto result_tensor1 = CreateSubgraphTensor(currFunctionPtr, shape2, tensorMagic6, subGraphID0);
    auto result_tensor2 = CreateSubgraphTensor(currFunctionPtr, shape2, tensorMagic7, subGraphID1);
    auto& expopin1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inner_tensor1}, {result_tensor1});
    expopin1.UpdateSubgraphID(subGraphID0);
    expopin1.opmagic = opMagic5;
    auto& expopin2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inner_tensor2}, {result_tensor2});
    expopin2.UpdateSubgraphID(subGraphID1);
    expopin2.opmagic = opMagic6;

    auto output_tensor = CreateTensor(currFunctionPtr, shape3, tensorMagic8, MEM_DEVICE_DDR);
    AddCopyOutOp(*currFunctionPtr, result_tensor1, output_tensor, {0, 0}, shape2Imme, subGraphID0, opMagic7);
    AddCopyOutOp(*currFunctionPtr, result_tensor2, output_tensor, {16, 0}, shape2Imme, subGraphID1, opMagic8);

    currFunctionPtr->inCasts_.push_back(input_tensor);
    currFunctionPtr->outCasts_.push_back(output_tensor);

    currFunctionPtr->SetTotalSubGraphCount(totalSubGraphCount);

    Json progDump;
    progDump["version"] = "2.0";
    progDump["functions"].push_back(currFunctionPtr->DumpJson());
    auto filePath = "Before_subgraphIsomorphismPass.json";
    std::ofstream file(filePath);
    file << progDump.dump() << std::endl;
    file.close();

    RunSubgraphToFunctionPass(*currFunctionPtr);

    auto rootFunc = currFunctionPtr->rootFunc_;
    EXPECT_NE(rootFunc, nullptr);
    const auto& PSgToESgMap = GetPSgToESgMap(rootFunc);

    EXPECT_EQ(CountMergedSubgraphCount(PSgToESgMap), 1);
    EXPECT_TRUE(ArePsgHashesUnique(*rootFunc));
    EXPECT_TRUE(IsPSgToESgMapOneToOne(PSgToESgMap));
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load)
{
    int bs = 1;
    int m = 32;
    int k = 32;
    int n = 32;

    std::vector<int64_t> shapeA = {bs, m, k};
    std::vector<int64_t> shapeB = {bs, k, n};
    std::vector<int64_t> shapeC = {bs, m, n};

    config::Reset();
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    Tensor matA(DT_FP16, shapeA, "MatA", TileOpFormat::TILEOP_NZ);
    Tensor matB(DT_FP16, shapeB, "MatB", TileOpFormat::TILEOP_ND);
    Tensor matC(DT_FP32, shapeC, "MatC");
    config::SetBuildStatic(true);
    FUNCTION("BATCHMATMUL", {matA, matB, matC})
    {
        config::SetPassConfig("PVC2_OOO", "OoOSchedule", KEY_DISABLE_PASS, true);
        matC = npu::tile_fwk::Matrix::BatchMatmul(DT_FP32, matA, matB, false, false);
    }
    config::SetPassConfig("PVC2_OOO", "OoOSchedule", KEY_DISABLE_PASS, false);
    auto programJson = Program::GetInstance().DumpJson();
    auto currentFunctionPtr = Program::GetInstance().GetCurrentFunction();
    EXPECT_EQ(Program::GetInstance().FunctionMapSize(), 4);
    ASSERT_NE(currentFunctionPtr, nullptr);
    EXPECT_EQ(currentFunctionPtr->Operations().size(), 1);
    EXPECT_EQ(currentFunctionPtr->GetRawName(), "PROGRAM_ENTRY");

    auto batchMatmulFunc = Program::GetInstance().GetFunctionByRawName("TENSOR_BATCHMATMUL");
#ifndef PRIOR_SCHEDULING
    EXPECT_EQ(batchMatmulFunc->Operations().size(), 9);
#endif

    ASSERT_NE(batchMatmulFunc->rootFunc_, nullptr);
    EXPECT_EQ(batchMatmulFunc->rootFunc_->Operations().size(), 1);
    EXPECT_EQ(batchMatmulFunc->rootFunc_->programs_.size(), 1);
    auto& oriPrograms = batchMatmulFunc->rootFunc_->programs_;
    EXPECT_EQ(oriPrograms[0]->Operations().size(), 11);
#ifndef PRIOR_SCHEDULING
    EXPECT_EQ(oriPrograms[1]->Operations().size(), 8);
#endif
    auto topoBefore = batchMatmulFunc->rootFunc_->topoInfo_;
    auto& entrysBefore = topoBefore.GetTopology();
    EXPECT_EQ(programJson["functions"].size(), 4);
    SubfuncInvokeInfoTy invokeInfo10000;
    SubfuncInvokeInfoTy invokeInfo10001;
    SubfuncInvokeInfoTy invokeInfo10002;
    SubfuncInvokeInfoTy invokeInfo10003;
    for (auto& op : batchMatmulFunc->rootFunc_->Operations()) {
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
    EXPECT_EQ(Program::GetInstance().FunctionMapSize(), 4);
    auto newCurrFuncPtr = Program::GetInstance().GetCurrentFunction();
    ASSERT_NE(newCurrFuncPtr, nullptr);
    // 校验CallOpAttribute
    ASSERT_NE(newCurrFuncPtr->rootFunc_, nullptr);
    EXPECT_EQ(newCurrFuncPtr->rootFunc_->Operations().size(), 1);

    for (auto& op : newCurrFuncPtr->rootFunc_->Operations()) {
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
    EXPECT_EQ(batchMatmulFunc->rootFunc_->Operations().size(), 1);
    EXPECT_EQ(batchMatmulFunc->rootFunc_->programs_.size(), 1);

    // 校验Topo
    auto& topo = newCurrFuncPtr->rootFunc_->topoInfo_;
    auto& entrys = topo.GetTopology();
    EXPECT_EQ(entrysBefore.size(), entrys.size());

    for (size_t i = 0; i < entrysBefore.size(); i++) {
        EXPECT_EQ(entrys[i].esgId, entrysBefore[i].esgId);
        EXPECT_EQ(entrys[i].readyState, entrysBefore[i].readyState);
        EXPECT_EQ(entrys[i].outGraph, entrysBefore[i].outGraph);
    }

    // 校验rootFunc_->programs_
    auto& programs = newCurrFuncPtr->rootFunc_->programs_;
    ASSERT_EQ(programs.size(), 1);
    EXPECT_EQ(programs[0]->Operations().size(), 11);
#ifndef PRIOR_SCHEDULING
    EXPECT_EQ(programs[1]->Operations().size(), 8);
#endif
    // 校验CopyInCopyoutAttribute
    for (auto& op : programs[0]->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            auto copyInOpAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(copyInOpAttr, nullptr);
            auto outputDynShape = op.GetOOperands().front()->GetDynValidShape();
            EXPECT_NE(outputDynShape.size(), 0);
        }
        if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            auto copyOutOpAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(copyOutOpAttr, nullptr);
        }
    }

    programJson = Program::GetInstance().DumpJson();
    Program::GetInstance().LoadJson(programJson);
    Json programJsonNew = Program::GetInstance().DumpJson();
    EXPECT_EQ(programJsonNew.dump(), programJson.dump());
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load_1)
{
    int32_t shape0 = 4;
    int32_t shape1 = 32;
    int32_t k = 8;
    bool isLargest = true;

    PROGRAM("TOPK")
    {
        std::vector<int64_t> input_shape = {shape0, shape1};
        std::vector<int64_t> output_shape = {shape0, k};
        TileShape::Current().SetVecTile({shape0, shape1});
        Tensor input_a(DT_FP32, input_shape, (uint8_t*)nullptr, "A");
        auto output = std::make_tuple(
            Tensor(DT_FP32, output_shape, nullptr, "npu_val"), Tensor(DT_FP32, output_shape, nullptr, "resDics"));
        config::SetPassConfig("PVC2_OOO", "OoOSchedule", KEY_DISABLE_PASS, true);
        config::SetBuildStatic(true);
        FUNCTION("TOPK_T", {input_a, std::get<0>(output), std::get<1>(output)})
        {
            output = TopK(input_a, k, -1, isLargest);
        }
    }
    config::SetPassConfig("PVC2_OOO", "OoOSchedule", KEY_DISABLE_PASS, false);

    Json programJson = Program::GetInstance().DumpJson();
    Program::GetInstance().LoadJson(programJson);
    Json programJsonNew = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJsonNew);
    Json programJsonNewNew = Program::GetInstance().DumpJson();

#ifndef PRIOR_SCHEDULING
    EXPECT_EQ(programJsonNew.dump(), programJsonNewNew.dump());
#endif
    config::SetHostOption(COMPILE_STAGE, CS_ALL_COMPLETE);
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load_1_cov)
{
    int32_t shape0 = 4;
    int32_t shape1 = 32;
    int32_t k = 8;
    bool isLargest = true;

    PROGRAM("TOPK")
    {
        std::vector<int64_t> input_shape = {shape0, shape1};
        std::vector<int64_t> output_shape = {shape0, k};
        TileShape::Current().SetVecTile({shape0, shape1});
        Tensor input_a(DT_FP32, input_shape, (uint8_t*)nullptr, "A");
        auto output = std::make_tuple(
            Tensor(DT_FP32, output_shape, nullptr, "npu_val"), Tensor(DT_FP32, output_shape, nullptr, "resDics"));
        config::SetBuildStatic(true);
        FUNCTION("TOPK_T", {input_a, std::get<0>(output), std::get<1>(output)})
        {
            output = TopK(input_a, k, -1, isLargest);
        }
    }

    Json programJson = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJson);
    Json programJsonNew = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJsonNew);
    Json programJsonNewNew = Program::GetInstance().DumpJson();

    config::SetHostOption(COMPILE_STAGE, CS_ALL_COMPLETE);
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load_2)
{
    IfaTileShapeConfig tileConfig{
        256,                        // block size
        32,                         // nTile
        {256, 128},                 // v0 tile for qkv-view-concat, q-S1D:(32,64), k/v-S2D:(256,64), merge 2D to copy
        {32, 32, 64, 64, 256, 256}, // c1 tile for S1D@S2D
        {32, 256},                  // v1 tile for S1S2
        {32, 32, 64, 64, 256, 256}, // c2 tile for S1S2@S2D
        {32, 256},                  // v2 tile for S1D
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
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    PROGRAM("PageAttentionStatic")
    {
        Tensor qNope(DT_BF16, {b * sq * nq, dn}, (uint8_t*)nullptr, "qNope");
        Tensor qRope(DT_BF16, {b * sq * nq, dr}, (uint8_t*)nullptr, "qRope");
        Tensor kNopeCache(DT_BF16, {blockNum * blockSize * nkv, dn}, (uint8_t*)nullptr, "kNopeCache");
        Tensor kRopeCache(DT_BF16, {blockNum * blockSize * nkv, dr}, (uint8_t*)nullptr, "kRope");
        Tensor vNopeCache(DT_BF16, {blockNum * blockSize * nkv, dn}, (uint8_t*)nullptr, "vNopeCache");

        // blockTable: (b, maxBlockNumPerBatch)
        int maxSeqAllBatch = *(std::max_element(actSeqs.begin(), actSeqs.end()));
        int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
        std::vector<std::vector<int>> blockTable(b, std::vector<int>(maxBlockNumPerBatch, 0));

        Tensor attentionOut(DT_FP32, {b * sq * nq, dn}, nullptr, "attentionOut");

        // 计算流程开始
        config::SetBuildStatic(true);
        FUNCTION("IfaStatic", {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, attentionOut})
        {
            IncreFlashAttention(
                qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, softmaxScale, attentionOut,
                tileConfig);
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
    config::SetHostOption(COMPILE_STAGE, CS_ALL_COMPLETE);
}

/*
 * input -> view1(01) -> view1_out -/
 *                                  add(03) -> add_out -> abc(04) -> final_out
 * input -> View2(01) -> view2_out -/
 */
void InitGraphBuilder(ComputationalGraphBuilder& G, std::vector<int64_t> tileShape)
{
    // 1. 定义张量和操作
    std::vector<std::string> tensorNames = {"input", "view1_out", "view2_out", "add_out", "final_out"};
    std::vector<Opcode> opCodes = {Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_ADD, Opcode::OP_ABS};
    std::vector<std::vector<std::string>> ioperands = {
        {"input"},                  // view1
        {"input"},                  // view2 (确保与view1不同输出)
        {"view1_out", "view2_out"}, // add (两个不同输入)
        {"add_out"}                 // abs
    };
    std::vector<std::vector<std::string>> ooperands = {{"view1_out"}, {"view2_out"}, {"add_out"}, {"final_out"}};
    std::vector<std::string> opNames = {"view1", "view2", "add", "abs_final"};

    // 2. 添加张量和操作
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 3. 设置内存类型和边界张量
    auto input_tensor = G.GetTensor("input");
    auto final_out_tensor = G.GetTensor("final_out");
    input_tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    final_out_tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);

    // 4. 设置输入输出转换
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"final_out"}));

    // 5. 设置子图ID（所有操作在同一个子图）
    for (const auto& opName : opNames) {
        G.GetOp(opName)->UpdateSubgraphID(0);
    }
}

TEST_F(SubgraphToFunctionTest, TestBasicSubgraphConversion)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    // 初始化
    InitGraphBuilder(G, tileShape);

    // 获取Function并执行子图转换Pass
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(1); // 总子图数=1

    SubgraphToFunction pass;
    Status status = pass.RunOnFunction(*function);
    status = pass.PostCheck(*function);
    EXPECT_EQ(status, SUCCESS);

    // 验证结果
    Function* rootFunc = function->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);
    EXPECT_EQ(rootFunc->GetGraphType(), GraphType::EXECUTE_GRAPH);

    // 检查子图调用信息
    const auto& topoInfo = rootFunc->topoInfo_;
    EXPECT_EQ(topoInfo.topology_.size(), 1); // 应有一个子图调用

    // 检查子图内部操作（应保留VIEW+VIEW+ADD+ABS）
    auto leafFunc = rootFunc->programs_.begin()->second;
    EXPECT_EQ(leafFunc->Operations().size(), 4);
}

TEST_F(SubgraphToFunctionTest, MultiSubgraphDependencyWithMixedOps)
{
    ComputationalGraphBuilder G;
    const std::vector<std::string> tensorNames = {"input", "aic_out", "aiv_out", "final_out"};
    const std::vector<Opcode> opCodes = {Opcode::OP_A_MUL_B, Opcode::OP_ADD, Opcode::OP_EXP};
    const std::vector<std::vector<std::string>> ioperands = {{"input"}, {"aic_out"}, {"aiv_out"}};
    std::vector<std::vector<std::string>> ooperands = {{"aic_out"}, {"aiv_out"}, {"final_out"}};
    const std::vector<std::string> opNames = {"matmul_aic", "add_aiv", "exp_aicpu"};
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    auto configureSubgraphOp = [&G](const std::string& opName, int subGraphId, CoreType coreType, bool isCube = false) {
        auto* op = G.GetOp(opName);
        ASSERT_NE(op, nullptr);
        op->UpdateSubgraphID(subGraphId);
        op->SetCoreType(coreType);
        if (isCube) {
            op->SetAttribute(OpAttributeKey::isCube, true);
        }
    };
    configureSubgraphOp("matmul_aic", 0, CoreType::AIC, true);
    configureSubgraphOp("add_aiv", 1, CoreType::AIV);
    configureSubgraphOp("exp_aicpu", 2, CoreType::AICPU);

    auto input_tensor = G.GetTensor("input");
    auto final_out_tensor = G.GetTensor("final_out");
    ASSERT_NE(input_tensor, nullptr);
    ASSERT_NE(final_out_tensor, nullptr);
    input_tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    final_out_tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);

    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"final_out"}));

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(3);

    SubgraphToFunction pass;
    Status status = pass.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    Function* rootFunc = function->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);

    EXPECT_EQ(rootFunc->programs_.size(), 3);

    const auto& topoInfo = rootFunc->topoInfo_;
    EXPECT_EQ(topoInfo.topology_.size(), 3);
    EXPECT_EQ(topoInfo.topology_[0].outGraph, std::unordered_set<int>{1});
    EXPECT_EQ(topoInfo.topology_[1].outGraph, std::unordered_set<int>{2});
    EXPECT_TRUE(topoInfo.topology_[2].outGraph.empty());
    EXPECT_EQ(topoInfo.topology_[0].readyState, 0);
    EXPECT_EQ(topoInfo.topology_[1].readyState, -1);
    EXPECT_EQ(topoInfo.topology_[2].readyState, -1);

    const auto& callOps = rootFunc->Operations();
    ASSERT_EQ(callOps.size(), 3);

    auto expectCallGraphType = [&callOps](size_t idx, CoreType expected) {
        auto attr = dynamic_cast<CallOpAttribute*>(callOps[idx].GetOpAttribute().get());
        ASSERT_NE(attr, nullptr);
        EXPECT_EQ(attr->invokeInfo_->GetGraphType(), expected);
    };

    expectCallGraphType(0, CoreType::AIC);
    expectCallGraphType(1, CoreType::AIV);
    expectCallGraphType(2, CoreType::AICPU);

    EXPECT_EQ(rootFunc->GetReadySubGraphCount(CoreType::AIC), 1);
    EXPECT_EQ(rootFunc->GetReadySubGraphCount(CoreType::AIV), 0);
    EXPECT_EQ(rootFunc->GetReadySubGraphCount(CoreType::AICPU), 0);
}

TEST_F(SubgraphToFunctionTest, EliminateRedundantEdges)
{
    ComputationalGraphBuilder G;

    // 定义张量（需要更多张量来创建冗余路径）
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4", "t5", "t6"};

    // 定义操作和子图分配 - 创建两条路径到MAX_SG3
    std::vector<Opcode> opCodes{
        Opcode::OP_ADD,    // 子图0
        Opcode::OP_CONV,   // 子图1
        Opcode::OP_ABS,    // 子图2
        Opcode::OP_ADD,    // 子图3
        Opcode::OP_MAXIMUM // 子图4
    };

    std::vector<std::vector<std::string>> ioperands{
        {"t0", "t1"}, // ADD1_SG0
        {"t2"},       // CONV_SG1
        {"t2"},       // ABS_SG2
        {"t3", "t4"}, // ADD2_SG3
        {"t4", "t5"}  // MAX_SG4 (接收来自ADD和ABS的输入)
    };

    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};

    std::vector<std::string> opNames{"ADD1_SG0", "CONV_SG1", "ABS_SG2", "ADD2_SG3", "MAX_SG4"};

    // 创建图和操作
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 设置子图ID - 创建跨子图冗余
    G.GetOp("ADD1_SG0")->UpdateSubgraphID(0);
    G.GetOp("CONV_SG1")->UpdateSubgraphID(1);
    G.GetOp("ABS_SG2")->UpdateSubgraphID(2);
    G.GetOp("ADD2_SG3")->UpdateSubgraphID(3);
    G.GetOp("MAX_SG4")->UpdateSubgraphID(4);

    // 设置边界张量
    EXPECT_TRUE(G.SetInCast({"t0", "t1"}));
    EXPECT_TRUE(G.SetOutCast({"t6"}));

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(5); // 共5个子图
    // 2. 运行SubgraphToFunction pass
    SubgraphToFunction pass;
    pass.SetupStaticProcessor();
    // 构建基础图结构
    pass.staticProcessor_.BuildGraph(*function);
    pass.RecordIncastOutcast(*function);

    // 3. 验证初始边关系（通过消费者关系）
    auto* abs_op = G.GetOp("ABS_SG2");
    auto* add2_op = G.GetOp("ADD2_SG3");
    auto* max_op = G.GetOp("MAX_SG4");

    // 验证ABS_SG2的消费者包含ADD2_SG3、MAX_SG4
    auto abs_consumers1 = abs_op->ConsumerOps();
    EXPECT_TRUE(abs_consumers1.find(add2_op) != abs_consumers1.end());
    EXPECT_TRUE(abs_consumers1.find(max_op) != abs_consumers1.end());

    // 4. 构建颜色图并消除冗余边
    pass.staticProcessor_.BuildColorGraph(*function);
    pass.staticProcessor_.EraseRedundantColorEdges(*function);

    // 5. 验证冗余边已被移除
    const auto& colorOutGraph = pass.staticProcessor_.colorOutGraph;
    const int abs_sgid = G.GetOp("ABS_SG2")->GetSubgraphID();
    const int max_sgid = G.GetOp("MAX_SG4")->GetSubgraphID();
    const auto& abs_out_edges = colorOutGraph[abs_sgid];
    bool found = std::find(abs_out_edges.begin(), abs_out_edges.end(), max_sgid) != abs_out_edges.end();
    EXPECT_FALSE(found) << "Redundant edge not removed!";
}

TEST_F(SubgraphToFunctionTest, ReshapeDependencyHandling)
{
    ComputationalGraphBuilder G;

    // 1. 构建测试图：包含一个RESHAPE操作和其消费者
    std::vector<std::string> tensorNames{"t0", "t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_RESHAPE, Opcode::OP_ABS};
    std::vector<std::vector<std::string>> ioperands{
        {"t0"}, // RESHAPE_SG0 (无输入子图)
        {"t1"}  // ABS_SG1 (输入来自RESHAPE)
    };
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}};
    std::vector<std::string> opNames{"RESHAPE_SG0", "ABS_SG1"};

    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 2. 设置子图ID（确保RESHAPE是独立子图）
    auto set_subgraph_id = [&G](const std::string& op_name, int id) {
        auto* op = G.GetOp(op_name);
        ASSERT_NE(op, nullptr) << "Operation " << op_name << " not found!";
        op->UpdateSubgraphID(id);
    };
    set_subgraph_id("RESHAPE_SG0", 0); // RESHAPE单独子图且无输入子图
    set_subgraph_id("ABS_SG1", 1);

    // 3. 构建函数并运行pass
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(2); // 2个子图

    SubgraphToFunction pass;
    pass.SetupStaticProcessor(); // 初始化静态处理器
    // 通过静态处理器调用BuildGraph
    pass.staticProcessor_.BuildGraph(*function);
    pass.RecordIncastOutcast(*function);
    pass.ConstructParamMap(*function);

    // 4. 验证RESHAPE子图的特殊处理
    auto* reshape_op = G.GetOp("RESHAPE_SG0");
    ASSERT_NE(reshape_op, nullptr);
    const int reshape_sgid = reshape_op->GetSubgraphID();

    // 4.1 验证RESHAPE子图被正确标记
    EXPECT_TRUE(pass.staticProcessor_.isReshape[reshape_sgid])
        << "RESHAPE subgraph should be marked when it has no input subgraph and single reshape op";

    EXPECT_TRUE(function->topoInfo_.GetSuccs(reshape_sgid).empty())
        << "RESHAPE subgraph should have empty successors set";

    int expected_out_degree = 0; // 根据实际图结构调整这个值
    bool found = false;
    for (const auto& entry : function->topoInfo_.GetTopology()) {
        if (entry.esgId == 1) { // ABS_SG1的子图ID
            EXPECT_EQ(entry.readyState, -1 * expected_out_degree)
                << "Consumer subgraph's readyOrNot should exclude RESHAPE inputs";
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "ABS_SG1 subgraph entry not found in topology";
}

TEST_F(SubgraphToFunctionTest, TransViewToCopyIn_ViewWithTwoOOperands_Fail)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape = {16, 16};
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "v_in"));
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "v_out1"));
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "v_out2"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_VIEW, {"v_in"}, {"v_out1", "v_out2"}, "view_two_out"));
    Operation* viewOp = G.GetOp("view_two_out");
    ASSERT_NE(viewOp, nullptr);
    viewOp->SetAttribute(OpAttributeKey::inplaceIdx, 0);
    viewOp->UpdateSubgraphID(0);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(1);
    SubgraphToFunction pass;
    Status status = pass.RunOnFunction(*function);
    EXPECT_NE(status, SUCCESS);
}
} // namespace tile_fwk
} // namespace npu
