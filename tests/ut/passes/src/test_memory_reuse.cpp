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
 * \file test_memory_reuse.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include "interface/cache/function_cache.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "passes/execute_graph_pass/memory_reuse.h"
#include "interface/tensor/tensormap.h"
#include "operator/models/deepseek/deepseek_mla.h"
#include "operator/models/deepseek/deepseek_spec.h"
#include "computational_graph_builder.h"
#include <nlohmann/json.hpp>
#include <vector>
#include <string>

namespace npu {
namespace tile_fwk {

class TestMemoryReuse : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {
    }
};

TEST_F(TestMemoryReuse, test_connection_matrix) {
    int b = 2;
    int n = 2;
    int s = 1;
    int kvLoraRank = 512;
    int vHeadDim =128;
    int h = 512;
    std::vector<int> inShape = {b, n, s, kvLoraRank}; // (b, n, s, d)
    Tensor attnPostIn(DT_BF16, inShape, "attnPostIn");
    Tensor attenOutput;
    AttentionW aw;
    aw.kvBProjWV = Tensor(DT_BF16, {n, kvLoraRank, vHeadDim}, "kvBProjWV");
    aw.oProjW = Tensor(DT_BF16, {n * vHeadDim, h}, "oProjW");
    ConfigManager::Instance();
    FUNCTION("AttentionPost") {
        DeepseekAttention atten(deepseekConfig1, aw, 1);
        attenOutput = atten.AttentionPost2(attnPostIn);
    }
    attenOutput.GetDataType();
    auto function= Program::GetInstance().GetFunctionByRawName("TENSOR_AttentionPost");
    ASSERT_NE(function, nullptr);
    auto rootFunc = function->rootFunc_;
    auto callOps = rootFunc->Operations();
    size_t totalSize = 17; // transpose支持尾轴切分，operation数目有变化
    EXPECT_EQ(callOps.size(), totalSize);
    std::unordered_set<int64_t> storageSet;
    uint64_t totalLength = 0;
    for (auto &callop : callOps) {
        for (auto &in : callop.iOperand) {
            if (in->storage_ != nullptr && storageSet.count(in->storage_->id_) == 0) {
                storageSet.emplace(in->storage_->id_);
                totalLength += in->storage_->length_;
            }
        }
    }
    EXPECT_EQ(totalLength, 8192);
    Json jsonT = attenOutput.GetStorage()->DumpJson();
    std::unordered_map<int, std::shared_ptr<RawTensor>> rawTensorDict;
    auto newTensor = LogicalTensor::LoadJson(*function, rawTensorDict, jsonT);
    attenOutput.GetStorage()->DumpASM(true, true);
    attenOutput.GetStorage()->tensor->GetRawShapeSize();

    std::vector<int> resultOffset;
    std::vector<int> resultShape;
    CalcShapeAndOffsetOfGroup(function->inCasts_, resultOffset, resultShape);
    function->GetTensorMap().GetTensorByMagic(function->inCasts_[0]->magic);
    CalcOverlapSize(function->inCasts_[0], function->inCasts_[0]);
    CalcOverlap(function->inCasts_[0], function->inCasts_[0], true);
    CalcOverlap(function->inCasts_[0], function->inCasts_, true);
    Allocator allocator(rootFunc);
    allocator.Init();
    size_t nodeId0 = 0;
    size_t nodeId1 = 1;
    size_t nodeId2 = 2;
    size_t nodeId4 = 4;
    size_t nodeId5 = 5;
    size_t nodeId6 = 6;
    size_t nodeId13 = 13;
    size_t nodeId7 = 7;
    // 下面的值请勿随意修改，校验存在一定价值
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId0), callOps.at(nodeId1)), false);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId0), callOps.at(nodeId4)), false);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId0), callOps.at(nodeId5)), true);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId0), callOps.at(nodeId6)), true);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId5), callOps.at(nodeId13)), true);
    EXPECT_EQ(allocator.connectionMatrix_.IsConnected(callOps.at(nodeId2), callOps.at(nodeId7)), true);
}

TEST_F(TestMemoryReuse, CanReuseSeriesOpConn) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3", "CALL4"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto &op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    /* dump json */
    std::string jsonFilePath = "./config/pass/json/memory_reuse_can_reuse_series_op.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        function->DumpJsonFile(jsonFilePath);
    }

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(allocator.size_, 16 * 16 * 4 * 2); // shape: 16*16, size: 4, allocate 2 tensor memory
}

TEST_F(TestMemoryReuse, NotReuseParallelOpConn) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5", "t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3", "t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto &op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(allocator.size_, 16 * 16 * 4 * 4); // shape: 16*16, size: 4, allocate 4 tensor memory
}

TEST_F(TestMemoryReuse, NotReuseMultiInputOutput) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3", "t4", "t5"}, {"t6", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3", "t4", "t5"}, {"t6", "t7"}, {"t8"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t8"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto &op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(allocator.size_, 16 * 16 * 4 * 5); // shape: 16*16, size: 4, allocate 5 tensor memory
}

TEST_F(TestMemoryReuse, NotReuseViewOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_VIEW, Opcode::OP_CALL, Opcode::OP_ASSEMBLE, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "VIEW", "CALL1", "ASSEMBLE", "CALL2"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto &op : function->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            op->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int>{0, 0}));
        } else if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
           op->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int>{0, 0}));
        } else {
            op->SetOpAttribute(std::make_shared<CallOpAttribute>
                (function->ComputeHash(), list, function->GetMagicName()));
        }
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(allocator.size_, 16 * 16 * 4 * 2); // shape: 16*16, size: 4, allocate 4 tensor memory
}

TEST_F(TestMemoryReuse, NotReuseSeriesOpConnSizeDiff) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<std::string> tensorNames1{"t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3", "CALL4"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {64, 64}, tensorNames1), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto &op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(allocator.size_, 16 * 16 * 4 * 2 + 64 * 64 * 4 * 2); // allocate 2 + 2 tensor memory
}

TEST_F(TestMemoryReuse, CanReuseSeriesOpConnSizeDiff) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<std::string> tensorNames1{"t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3", "CALL4"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {64, 32}, tensorNames1), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto &op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(allocator.size_, 64 * 32 * 4 * 2); // allocate 2 big tensor memory
}

TEST_F(TestMemoryReuse, CanReuseSeriesOpConnMultiSizeDiff) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t4"};
    std::vector<std::string> tensorNames1{"t3"};
    std::vector<std::string> tensorNames2{"t5"};
    std::vector<std::string> tensorNames3{"t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3", "CALL4"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {64, 64}, tensorNames1), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {32, 16}, tensorNames2), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {32, 32}, tensorNames3), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto &op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();
    Status status = allocator.Allocate();

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(allocator.size_, 64 * 64 * 4 + 32 * 32 * 4); // allocate 2 large tensor memory
}

TEST_F(TestMemoryReuse, AbnormalNullRootFunction) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<std::string> tensorNames1{"t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3", "CALL4"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {64, 32}, tensorNames1), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto &op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }

    MemoryReuse reusePass;
    Status status = reusePass.RunOnFunction(*function);

    EXPECT_EQ(status, FAILED);
}

TEST_F(TestMemoryReuse, AbnormalNullStorage) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<std::string> tensorNames1{"t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"CALL0", "CALL1", "CALL2", "CALL3", "CALL4"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {64, 32}, tensorNames1), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t7"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    /* stub params */
    std::vector<std::vector<SymbolicScalar>> list;
    for (auto &op : function->Operations().DuplicatedOpList()) {
        op->SetOpAttribute(std::make_shared<CallOpAttribute>(function->ComputeHash(), list, function->GetMagicName()));
    }
    function->rootFunc_ = function;

    Allocator allocator(function->rootFunc_);
    allocator.Init();

    /* stub storage_ is null */
    auto &tensorsDesc = allocator.storageNeedToAllocate_.front();
    auto &tensor = *(tensorsDesc.tensors.begin());
    tensor->storage_ = nullptr;

    Status status = allocator.Allocate();

    EXPECT_EQ(status, FAILED);
}
} // namespace tile_fwk
} // namespace npu