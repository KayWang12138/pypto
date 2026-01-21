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
#include "passes/block_graph_pass/mix_subgraph_split.h"
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk {
constexpr uint64_t programId = 100;
constexpr int MS_NUM16 = 16;
constexpr int MS_NUM3 = 3;
constexpr int MS_NUM10005 = 10005;

class MixSubgraphSplitTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}

protected:
    // 在root function中创建callOp
    void CreateCallOpInRoot(Function& rootFunc, Function& mixFunc) {
        // 创建tensor作为callOp的输入输出
        std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};

        auto incast1 = std::make_shared<LogicalTensor>(rootFunc, DT_FP32, shape);
        auto incast2 = std::make_shared<LogicalTensor>(rootFunc, DT_FP32, shape);
        auto incast3 = std::make_shared<LogicalTensor>(rootFunc, DT_FP32, shape);
        auto outcast1 = std::make_shared<LogicalTensor>(rootFunc, DT_FP32, shape);
        auto outcast2 = std::make_shared<LogicalTensor>(rootFunc, DT_FP32, shape);

        // 添加callOp
        auto& callOp = rootFunc.AddRawOperation(
            Opcode::OP_CALL,
            {incast1, incast2, incast3},
            {outcast1, outcast2});

        // 创建CallOpAttribute
        std::vector<std::vector<SymbolicScalar>> argList;
        for (int i = 0; i < 5; ++i) {
            std::vector<SymbolicScalar> argBlock;
            for (int j = 0; j < 9; ++j) {  // 2维tensor的参数块长度
                argBlock.push_back(SymbolicScalar(static_cast<int64_t>(j + i * 10)));
            }
            argList.push_back(argBlock);
        }
        std::map<int, SymbolicScalar> outIndexToExpr;
        auto callAttr = mixFunc.CreateCallOpAttribute(argList, outIndexToExpr);
        callOp.SetOpAttribute(callAttr);
        // 设置programID
        callOp.UpdateSubgraphID(mixFunc.GetProgramId());
        // 设置invokeInfo
        auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(callAttr);
        if (callOpAttr) {
            callOpAttr->invokeInfo_ = std::make_shared<SubfuncInvokeInfoTy>();
            // 设置一些基本的invoke信息
            callOpAttr->invokeInfo_->UpdateProgramSubgraphId(programId);
        }
    }
};

TEST_F(MixSubgraphSplitTest, TestSingleMixSubgraphBasicSplit) {
    // 1. 创建测试场景
    auto rootFuncPtr = std::make_shared<Function>(
        Program::GetInstance(), "test_root", "test_root", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    // 2. 创建Mix子图（2个scope：1个Cube， 1个Vector）
    auto mixFuncPtr = std::make_shared<Function>(
        Program::GetInstance(), "test_mix_func", "test_mix_func", rootFuncPtr.get());
    mixFuncPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    mixFuncPtr->SetFunctionType(FunctionType::STATIC);
    // 添加到programs，使用programId=100
    const uint64_t mixProgramId = 100;
    rootFuncPtr->programs_[mixProgramId] = mixFuncPtr.get();
    std::vector<int64_t> tensorShape = {16, 16};
    // 3. 创建测试数据：Mix子图有3个incast，1个outcast
    auto incast1 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto incast2 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto incast3 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto outcast1 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    mixFuncPtr->inCasts_.push_back(incast1);
    mixFuncPtr->inCasts_.push_back(incast2);
    mixFuncPtr->inCasts_.push_back(incast3);
    mixFuncPtr->outCasts_.push_back(outcast1);
    // 4. 创建内部op（scope1：Cube）
    auto cubeTensor1 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto cubeTensor2 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto cubeTensor3 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto shapeImme = OpImmediate::Specified(tensorShape);
    std::vector<int64_t> offsetVec = {0, 0};
    auto offsetImme = OpImmediate::Specified(offsetVec);
    std::vector<OpImmediate> emptyVec;
    // Cube scope op（internalSubgraphID=0）
    auto& cubeCopyIn1 = mixFuncPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {cubeTensor1});
    cubeCopyIn1.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    cubeCopyIn1.SetIOpAttrOffset(0, 0);
    cubeCopyIn1.UpdateInternalSubgraphID(0);
    cubeCopyIn1.SetAttr(OpAttributeKey::isCube, true);

    auto& cubeCopyIn2 = mixFuncPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast2}, {cubeTensor2});
    cubeCopyIn2.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    cubeCopyIn2.SetIOpAttrOffset(0, 0);
    cubeCopyIn2.UpdateInternalSubgraphID(0);
    cubeCopyIn2.SetAttr(OpAttributeKey::isCube, true);
    auto& cubeMul = mixFuncPtr->AddRawOperation(Opcode::OP_A_MUL_B, {cubeTensor1, cubeTensor2}, {cubeTensor3});
    cubeMul.UpdateInternalSubgraphID(0);
    cubeMul.SetAttr(OpAttributeKey::isCube, true);

    // 5. 创建内部op（scope2：Vector AIV0）
    auto vectorTensor1 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto vectorTensor2 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    // Vector scope op（internalSubgraphID=1）
    auto& vectorAdd = mixFuncPtr->AddRawOperation(Opcode::OP_ADD, {cubeTensor3, incast3}, {vectorTensor1});
    vectorAdd.UpdateInternalSubgraphID(1);
    vectorAdd.SetAIVCore(AIVCore::AIV0);

    auto& vectorSqrt = mixFuncPtr->AddRawOperation(Opcode::OP_SQRT, {vectorTensor1}, {vectorTensor2});
    vectorSqrt.UpdateInternalSubgraphID(1);
    vectorSqrt.SetAIVCore(AIVCore::AIV0);

    auto& vectorCopyOut = mixFuncPtr->AddRawOperation(Opcode::OP_COPY_OUT, {vectorTensor2}, {outcast1});
    vectorCopyOut.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_UB, offsetImme, shapeImme, shapeImme, emptyVec));
    vectorCopyOut.SetOOpAttrOffset(0, 0);
    vectorCopyOut.UpdateInternalSubgraphID(1);
    vectorCopyOut.SetAIVCore(AIVCore::AIV0);

    // 6. 计算哈希并注册到缓存
    mixFuncPtr->ComputeHash();
    FunctionHash mixFuncHash = mixFuncPtr->GetFunctionHash();
    Program::GetInstance().GetFunctionCache().Insert(mixFuncHash, *mixFuncPtr);

    // 7. 创建调用该Mix子图的callOp
    // 创建callOp的输入输出tensor
    auto callInTensor1 = std::make_shared<LogicalTensor>(*rootFuncPtr, DT_FP32, tensorShape);
    auto callInTensor2 = std::make_shared<LogicalTensor>(*rootFuncPtr, DT_FP32, tensorShape);
    auto callInTensor3 = std::make_shared<LogicalTensor>(*rootFuncPtr, DT_FP32, tensorShape);
    auto callOutTensor = std::make_shared<LogicalTensor>(*rootFuncPtr, DT_FP32, tensorShape);
    auto& callOp = rootFuncPtr->AddRawOperation(Opcode::OP_CALL, 
        {callInTensor1, callInTensor2, callInTensor3}, 
        {callOutTensor});
    auto callAttr = std::make_shared<CallOpAttribute>();
    auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
    invokeInfo->UpdateProgramSubgraphId(mixProgramId);
    callAttr->SetCalleeHash(mixFuncHash);
    callAttr->invokeInfo_ = invokeInfo;
    // 创建参数列表
    std::vector<SymbolicScalar> linearArgs;
    // 为3个输入和1个输出创建参数块
     for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 9; j++) {  // 2维tensor的参数块
            linearArgs.push_back(SymbolicScalar(static_cast<int64_t>(j + i * 10)));
        }
    }
    callAttr->linearArgList_ = linearArgs;
    callOp.SetOpAttribute(callAttr);
    callOp.UpdateSubgraphID(mixProgramId);
    // 记录原始状态
    size_t originalProgramCount = rootFuncPtr->programs_.size();
    auto originalCallOps = rootFuncPtr->GetCallopList();
    size_t originalCallOpCount = originalCallOps.size();
    // 8. 执行拆分
    MixSubgraphSplit splitter;
    Status status = splitter.RunOnFunction(*rootFuncPtr);
    // 9. 验证结果
    // 9.1 状态验证
    ASSERT_EQ(status, SUCCESS) << "MixSubgraphSplit should succeed";
    // 9.2 program数量验证
    auto& programs = rootFuncPtr->programs_;
    EXPECT_EQ(programs.size(), 2) 
        << "Should have 2 programs after split (originally 1 Mix, split to 2 leaves)";
    // 9.3 ID连续性验证
    EXPECT_NE(programs.find(0), programs.end()) << "Should have program ID 0";
    EXPECT_NE(programs.find(1), programs.end()) << "Should have program ID 1";
    // 9.4 函数属性验证
    for (const auto& [progId, func] : programs) {
        ASSERT_NE(func, nullptr) << "Function should not be null";
        
        // 验证名称包含leaf后缀
        std::string funcName = func->GetRawName();
        EXPECT_NE(funcName.find("leaf"), std::string::npos)
            << "Function name should contain 'leaf' suffix: " << funcName;
        
        // 验证类型设置
        EXPECT_EQ(func->GetFunctionType(), FunctionType::STATIC);
        EXPECT_EQ(func->GetGraphType(), GraphType::BLOCK_GRAPH);
        
        // 验证programID一致性
        EXPECT_EQ(func->GetProgramId(), progId) 
            << "Function's program ID should match map key";
        
        // 验证LeafFuncAttribute
        auto leafAttr = func->GetLeafFuncAttribute();
        ASSERT_NE(leafAttr, nullptr) << "LeafFuncAttribute should be set";
        
        EXPECT_NE(leafAttr->mixId, static_cast<uint64_t>(-1)) << "mixId should be assigned";
        EXPECT_NE(leafAttr->mixResourceType, MixResourceType::UNKNOWN)
            << "mixResourceType should be set";
    }
    // 9.5 callOp数量验证
    auto newCallOps = rootFuncPtr->GetCallopList();
    EXPECT_EQ(newCallOps.size(), 2) 
        << "Should have 2 call ops after split (1 original * 2 components)";
    // 9.6 callOp属性验证
      for (auto* newCallOp : newCallOps) {
        ASSERT_NE(newCallOp, nullptr) << "CallOp should not be null";
        EXPECT_FALSE(newCallOp->IsDeleted()) << "CallOp should not be deleted";
        
        auto newCallAttr = dynamic_cast<CallOpAttribute*>(newCallOp->GetOpAttribute().get());
        ASSERT_NE(newCallAttr, nullptr) << "CallOpAttribute should exist";
        
        if (newCallAttr && newCallAttr->invokeInfo_) {
            uint64_t progId = newCallAttr->invokeInfo_->GetProgramId();
            EXPECT_TRUE(progId == 0 || progId == 1) 
                << "CallOp's program ID should be 0 or 1, got: " << progId;
            
            // 验证对应函数存在
            auto it = programs.find(progId);
            EXPECT_NE(it, programs.end()) 
                << "CallOp references non-existent program ID: " << progId;
            
            // 验证wrapId设置（重构后应该由MixCallOperationBuilder设置）
            EXPECT_NE(newCallAttr->wrapId, static_cast<uint64_t>(-1)) 
                << "wrapId should be set";
        }
    }
    // 9.7 scope类型验证
    int cubeCount = 0;
    int vectorCount = 0;
    
    for (const auto& [progId, func] : programs) {
        auto leafAttr = func->GetLeafFuncAttribute();
        if (leafAttr) {
            if (leafAttr->aivCore == AIVCore::UNSPECIFIED) {
                cubeCount++;
            } else if (leafAttr->aivCore == AIVCore::AIV0) {
                vectorCount++;
            }
        }
    }
    
    EXPECT_EQ(cubeCount, 1) << "Should have 1 cube component";
    EXPECT_EQ(vectorCount, 1) << "Should have 1 vector component";
    // 9.8 依赖关系验证
    // 获取两个新函数
    Function* cubeFunc = nullptr;
    Function* vectorFunc = nullptr;
    
    for (const auto& [progId, func] : programs) {
        auto leafAttr = func->GetLeafFuncAttribute();
        if (leafAttr) {
            if (leafAttr->aivCore == AIVCore::UNSPECIFIED) {
                cubeFunc = func;
            } else if (leafAttr->aivCore == AIVCore::AIV0) {
                vectorFunc = func;
            }
        }
    }
    
    ASSERT_NE(cubeFunc, nullptr) << "Cube function not found";
    ASSERT_NE(vectorFunc, nullptr) << "Vector function not found";
    // 检查Cube scope应该接收到incast1和incast2
    const auto& cubeIncasts = cubeFunc->GetIncast();
    bool cubeHasIncast1 = false;
    bool cubeHasIncast2 = false;
    
    for (const auto& tensor : cubeIncasts) {
        if (tensor == incast1) cubeHasIncast1 = true;
        if (tensor == incast2) cubeHasIncast2 = true;
    }
    
    EXPECT_TRUE(cubeHasIncast1) << "Cube function should have incast1";
    EXPECT_TRUE(cubeHasIncast2) << "Cube function should have incast2";
    // 检查Vector scope应该接收到incast1、incast2、incast3
    const auto& vectorIncasts = vectorFunc->GetIncast();
    bool vectorHasIncast1 = false;
    bool vectorHasIncast2 = false;
    bool vectorHasIncast3 = false;
    bool vectorHasCubeOutput = false;
    
    for (const auto& tensor : vectorIncasts) {
        if (tensor == incast1) vectorHasIncast1 = true;
        if (tensor == incast2) vectorHasIncast2 = true;
        if (tensor == incast3) vectorHasIncast3 = true;
    }
    EXPECT_TRUE(vectorHasIncast1) << "Vector function should have incast1";
    EXPECT_TRUE(vectorHasIncast2) << "Vector function should have incast2";
    EXPECT_TRUE(vectorHasIncast3) << "Vector function should have incast3";
    // 检查outcast1应该被正确传播到Vector scope、cube scope
    const auto& vectorOutcasts = vectorFunc->GetOutcast();
    bool vectorHasOutcast1 = false;
    
    for (const auto& tensor : vectorOutcasts) {
        if (tensor == outcast1) vectorHasOutcast1 = true;
    }
    
    EXPECT_TRUE(vectorHasOutcast1) << "Vector function should have outcast1";
     const auto& cubeOutcasts = cubeFunc->GetOutcast();
    bool cubeHasOutcast1 = false;
    
    for (const auto& tensor : cubeOutcasts) {
        if (tensor == outcast1) cubeHasOutcast1 = true;
    }
    
    EXPECT_TRUE(cubeHasOutcast1) << "Vector function should have outcast1";
    // 9.9 原始Mix子图验证
    bool originalMixFuncStillExists = false;
    for (const auto& [progId, func] : programs) {
        if (func == mixFuncPtr.get()) {
            originalMixFuncStillExists = true;
            break;
        }
    }
    EXPECT_FALSE(originalMixFuncStillExists) << "Original mix function should be removed";
    // 9.10 原始callOp清理验证
    bool originalCallOpStillExists = false;
    for (auto* callOpPtr : newCallOps) {
        if (callOpPtr == &callOp) {
            originalCallOpStillExists = true;
            break;
        }
    }
    EXPECT_FALSE(originalCallOpStillExists) << "Original callOp should be deleted";
    // 9.11 验证programs映射正确性
    // 确保所有program都有对应的callOp
    std::set<uint64_t> programIdsFromCallOps;
    for (auto* callOpPtr : newCallOps) {
        auto newCallOpAttr = dynamic_cast<CallOpAttribute*>(callOpPtr->GetOpAttribute().get());
        if (newCallOpAttr && newCallOpAttr->invokeInfo_) {
            programIdsFromCallOps.insert(newCallOpAttr->invokeInfo_->GetProgramId());
        }
    }
    for (const auto& [progId, func] : programs) {
        EXPECT_NE(programIdsFromCallOps.find(progId), programIdsFromCallOps.end())
            << "Program ID " << progId << " should have corresponding callOp";
    }
}

TEST_F(MixSubgraphSplitTest, TestDependencyRebuilding) {
    // 创建root function
    auto rootFuncPtr = std::make_shared<Function>(
        Program::GetInstance(), "test_root", "test_root", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    // ==================== 创建2个非Mix子图 ====================
    // 非Mix子图1：纯CUBE子图
    auto nonMixFunc1Ptr = std::make_shared<Function>(
        Program::GetInstance(), "test_non_mix_func1", "test_non_mix_func1", rootFuncPtr.get());
    nonMixFunc1Ptr->SetGraphType(GraphType::BLOCK_GRAPH);
    nonMixFunc1Ptr->SetFunctionType(FunctionType::STATIC);    
    // 非Mix子图2：纯VECTOR子图
    auto nonMixFunc2Ptr = std::make_shared<Function>(
        Program::GetInstance(), "test_non_mix_func2", "test_non_mix_func2", rootFuncPtr.get());
    nonMixFunc2Ptr->SetGraphType(GraphType::BLOCK_GRAPH);
    nonMixFunc2Ptr->SetFunctionType(FunctionType::STATIC);
    // ==================== 创建Mix子图3 ====================
    auto mixFunc3Ptr = std::make_shared<Function>(
        Program::GetInstance(), "test_mix_func3", "test_mix_func3", rootFuncPtr.get());
    mixFunc3Ptr->SetGraphType(GraphType::BLOCK_GRAPH);
    mixFunc3Ptr->SetFunctionType(FunctionType::STATIC);

    // 添加到programs 
    uint64_t nonMixProgramId1 = 0;
    uint64_t nonMixProgramId2 = 1;
    uint64_t mixProgramId = 2;
    rootFuncPtr->programs_[nonMixProgramId1] = nonMixFunc1Ptr.get();
    rootFuncPtr->programs_[nonMixProgramId2] = nonMixFunc2Ptr.get();
    rootFuncPtr->programs_[mixProgramId] = mixFunc3Ptr.get();

    std::vector<int64_t> tensorShape = {MS_NUM16, MS_NUM16};

    // 为非Mix子图创建输入tensors
    auto nonMixInput1 = std::make_shared<LogicalTensor>(*nonMixFunc1Ptr, DT_FP32, tensorShape);
    auto nonMixInput2 = std::make_shared<LogicalTensor>(*nonMixFunc2Ptr, DT_FP32, tensorShape);

    // 创建Mix子图外部输入tensor（来自非Mix子图1和2）
    auto logicalTensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto logicalTensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);

    // 设置incast/outcast
    nonMixFunc1Ptr->inCasts_.push_back(nonMixInput1);
    nonMixFunc1Ptr->outCasts_.push_back(logicalTensor1); // 输出到Mix子图C1
    nonMixFunc2Ptr->inCasts_.push_back(nonMixInput2);   
    nonMixFunc2Ptr->outCasts_.push_back(logicalTensor2); // 输出到Mix子图C4

    // 创建Mix子图内部tensors
    // Scope C1的tensor
    auto c1_tensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto c1_tensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    
    // Scope V2的tensor
    auto v2_tensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto v2_tensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);

    // Scope C3的tensor
    auto c3_tensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto c3_tensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    
    // Scope C4的tensor
    auto c4_tensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto c4_tensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    
    // Scope V5的tensor
    auto v5_tensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto v5_tensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    
    // Scope V6的tensor
    auto v6_tensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto v6_tensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);

    // 全局输出tensor
    auto globalOutputTensor = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);

    // 设置incast/outcast
    mixFunc3Ptr->inCasts_.push_back(logicalTensor1);  // 来自非Mix子图1
    mixFunc3Ptr->inCasts_.push_back(logicalTensor2);  // 来自非Mix子图2
    mixFunc3Ptr->outCasts_.push_back(globalOutputTensor);  // 全局输出

    // 创建OpImmediate
    auto shapeImme = OpImmediate::Specified(tensorShape);
    std::vector<int64_t> offsetVec = {0, 0};
    auto offsetImme = OpImmediate::Specified(offsetVec);
    std::vector<OpImmediate> emptyVec;

    // 构建非Mix子图1的内部结构（纯CUBE）
    auto& copyin_nonmix1 = nonMixFunc1Ptr->AddRawOperation(Opcode::OP_COPY_IN, {nonMixInput1}, {logicalTensor1});
    copyin_nonmix1.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    copyin_nonmix1.SetIOpAttrOffset(0, 0);
    copyin_nonmix1.SetAttr(OpAttributeKey::isCube, true);

    // 构建非Mix子图2的内部结构（纯VECTOR）
    auto& copyin_nonmix2 = nonMixFunc2Ptr->AddRawOperation(Opcode::OP_COPY_IN, {nonMixInput2}, {logicalTensor2});
    copyin_nonmix2.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    copyin_nonmix2.SetIOpAttrOffset(0, 0);

    // ==================== 构建Mix子图内部结构 ====================
    // 路径1: C1 -> V2 -> C3 -> V6
    // Scope C1: 处理来自非Mix子图1的输入
    auto& copyin_c1 = mixFunc3Ptr->AddRawOperation(Opcode::OP_COPY_IN, {logicalTensor1}, {c1_tensor1});
    copyin_c1.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    copyin_c1.SetIOpAttrOffset(0, 0);
    copyin_c1.UpdateInternalSubgraphID(0);
    copyin_c1.SetAttr(OpAttributeKey::isCube, true);

    auto& sqrt_c1= mixFunc3Ptr->AddRawOperation(Opcode::OP_SQRT, {c1_tensor1}, {c1_tensor2});
    sqrt_c1.UpdateInternalSubgraphID(0);
    sqrt_c1.SetAttr(OpAttributeKey::isCube, true);

    // Scope V2: 处理C1的输出
    auto& abs_v2 = mixFunc3Ptr->AddRawOperation(Opcode::OP_ABS, {c1_tensor2}, {v2_tensor1});
    abs_v2.UpdateInternalSubgraphID(1);  // V2 scope
    abs_v2.SetAIVCore(AIVCore::AIV0);

    auto& abs1_v2 = mixFunc3Ptr->AddRawOperation(Opcode::OP_ABS, {v2_tensor1}, {v2_tensor2});
    abs1_v2.UpdateInternalSubgraphID(1);
    abs1_v2.SetAIVCore(AIVCore::AIV0);

    // Scope C3: 处理V2的输出和V5的输出
    // 创建一个辅助tensor用于融合两个输入
    auto c3_aux_tensor = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto& add_c3_inputs = mixFunc3Ptr->AddRawOperation(Opcode::OP_ADD, {v2_tensor2, v5_tensor2}, {c3_aux_tensor});
    add_c3_inputs.UpdateInternalSubgraphID(2);  // C3 scope
    add_c3_inputs.SetAttr(OpAttributeKey::isCube, true);

    auto& exp_c3 = mixFunc3Ptr->AddRawOperation(Opcode::OP_EXP, {c3_aux_tensor}, {c3_tensor1});
    exp_c3.UpdateInternalSubgraphID(2);
    exp_c3.SetAttr(OpAttributeKey::isCube, true);

    auto& log_c3 = mixFunc3Ptr->AddRawOperation(Opcode::OP_LN, {c3_tensor1}, {c3_tensor2});
    log_c3.UpdateInternalSubgraphID(2);
    log_c3.SetAttr(OpAttributeKey::isCube, true);

    // 路径2: C4 -> V5 -> C3 -> V6
    // Scope C4: 处理来自非Mix子图2的输入
    auto& copyin_c4 = mixFunc3Ptr->AddRawOperation(Opcode::OP_COPY_IN, {logicalTensor2}, {c4_tensor1});
    copyin_c4.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    copyin_c4.SetIOpAttrOffset(0, 0);
    copyin_c4.UpdateInternalSubgraphID(3);  // C4 scope
    copyin_c4.SetAttr(OpAttributeKey::isCube, true);

    auto& exp_c4 = mixFunc3Ptr->AddRawOperation(Opcode::OP_EXP, {c4_tensor1}, {c4_tensor2});
    exp_c4.UpdateInternalSubgraphID(3);
    exp_c4.SetAttr(OpAttributeKey::isCube, true);

    // Scope V5: 处理C4的输出
    auto& neg_v5 = mixFunc3Ptr->AddRawOperation(Opcode::OP_NEG, {c4_tensor2}, {v5_tensor1});
    neg_v5.UpdateInternalSubgraphID(4);  // V5 scope
    neg_v5.SetAIVCore(AIVCore::AIV1);

    auto& abs_v5 = mixFunc3Ptr->AddRawOperation(Opcode::OP_ABS, {v5_tensor1}, {v5_tensor2});
    abs_v5.UpdateInternalSubgraphID(4);
    abs_v5.SetAIVCore(AIVCore::AIV1);

    // Scope V6: 处理C3的输出，生成全局输出
    auto& sqrt_v6 = mixFunc3Ptr->AddRawOperation(Opcode::OP_SQRT, {c3_tensor2}, {v6_tensor1});
    sqrt_v6.UpdateInternalSubgraphID(5);
    sqrt_v6.SetAIVCore(AIVCore::AIV0);

    auto& copyout_v6 = mixFunc3Ptr->AddRawOperation(Opcode::OP_COPY_OUT, {v6_tensor1}, {globalOutputTensor});
    copyout_v6.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_UB, offsetImme, shapeImme, shapeImme, emptyVec));
    copyout_v6.SetOOpAttrOffset(0, 0);
    copyout_v6.UpdateInternalSubgraphID(5);
    copyout_v6.SetAIVCore(AIVCore::AIV0);

    // ==================== 第二部分：在root function中创建CallOp ====================
    // 创建一个函数来构建线性参数列表
    auto createLinearArgListForTensor = [&](const std::shared_ptr<LogicalTensor>& tensor) {
        (void)tensor;
        std::vector<SymbolicScalar> args;
        args.push_back(SymbolicScalar(1)); 
        // 第一维
        args.push_back(SymbolicScalar(0));  
        args.push_back(SymbolicScalar(1));   
        args.push_back(SymbolicScalar(16)); 
        args.push_back(SymbolicScalar(0));  
        // 第二维
        args.push_back(SymbolicScalar(0));  
        args.push_back(SymbolicScalar(16)); 
        args.push_back(SymbolicScalar(16)); 
        args.push_back(SymbolicScalar(0)); 
        return args;
    };

    nonMixFunc1Ptr->ComputeHash();
    FunctionHash nonMixHash1 = nonMixFunc1Ptr->GetFunctionHash();
    Program::GetInstance().GetFunctionCache().Insert(nonMixHash1, *nonMixFunc1Ptr);
    nonMixFunc2Ptr->ComputeHash();
    FunctionHash nonMixHash2 = nonMixFunc2Ptr->GetFunctionHash();
    Program::GetInstance().GetFunctionCache().Insert(nonMixHash2, *nonMixFunc2Ptr);   
    mixFunc3Ptr->ComputeHash();
    FunctionHash mixFuncHash = mixFunc3Ptr->GetFunctionHash();
    Program::GetInstance().GetFunctionCache().Insert(mixFuncHash, *mixFunc3Ptr);
    // CallOp1: 指向非Mix子图1
    auto& callOp1 = rootFuncPtr->AddRawOperation(Opcode::OP_CALL, {}, {});
    auto callAttr1 = std::make_shared<CallOpAttribute>();
    auto invokeInfo1 = std::make_shared<SubfuncInvokeInfoTy>();
    invokeInfo1->UpdateProgramSubgraphId(nonMixProgramId1);
    callAttr1->SetCalleeHash(nonMixHash1); 
    callAttr1->invokeInfo_ = invokeInfo1;
    callOp1.SetOpAttribute(callAttr1);
    // CallOp2: 指向非Mix子图2
    auto& callOp2 = rootFuncPtr->AddRawOperation(Opcode::OP_CALL, {}, {});
    auto callAttr2 = std::make_shared<CallOpAttribute>();
    auto invokeInfo2 = std::make_shared<SubfuncInvokeInfoTy>();
    invokeInfo2->UpdateProgramSubgraphId(nonMixProgramId2);
    callAttr2->SetCalleeHash(nonMixHash2);
    callAttr2->invokeInfo_ = invokeInfo2;
    callOp2.SetOpAttribute(callAttr2);
    // 创建指向Mix子图3的CallOp
    auto& callOp = rootFuncPtr->AddRawOperation(Opcode::OP_CALL, {}, {});
    auto callAttr = std::make_shared<CallOpAttribute>();
    auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
    invokeInfo->UpdateProgramSubgraphId(mixProgramId);    
    // 设置linearArgList_（Mix子图有2个输入，1个输出）
    std::vector<SymbolicScalar> linearArgs;
    // 输入1
    auto inputArgs1_mix = createLinearArgListForTensor(logicalTensor1);
    linearArgs.insert(linearArgs.end(), inputArgs1_mix.begin(), inputArgs1_mix.end());
    // 输入2
    auto inputArgs2_mix = createLinearArgListForTensor(logicalTensor2);
    linearArgs.insert(linearArgs.end(), inputArgs2_mix.begin(), inputArgs2_mix.end());
    // 输出
    auto outputArgs_mix = createLinearArgListForTensor(globalOutputTensor);
    linearArgs.insert(linearArgs.end(), outputArgs_mix.begin(), outputArgs_mix.end());
    callAttr->linearArgList_ = linearArgs;
    callAttr->SetCalleeHash(mixFuncHash);
    callAttr->invokeInfo_ = invokeInfo;
    callOp.SetOpAttribute(callAttr);

    // ==================== 第三部分：执行MixSubgraphSplit ====================
    MixSubgraphSplit splitter;

    // 执行拆分
    Status status = splitter.RunOnFunction(*rootFuncPtr);
    EXPECT_EQ(status, SUCCESS) << "MixSubgraphSplit should succeed";

    // ==================== 第四部分：验证依赖重建结果 ====================
    auto& programs = rootFuncPtr->programs_;
    // 1. 验证Mix子图已被删除
    bool mixFuncStillExists = false;
    for (const auto& [progId, func] : programs) {
        if (func == mixFunc3Ptr.get()) {
            mixFuncStillExists = true;
            break;
        }
    }
    EXPECT_FALSE(mixFuncStillExists) << "Original mix function should be deleted";

    // 2. 验证总Function数量
    EXPECT_EQ(programs.size(), 2 + 6) << "Should have 2 non-mix + 6 mix leaf functions";

    std::vector<Function*> mixSplitFunctions;
    for (const auto& [progId, func] : programs) {
        auto leafAttr = func->GetLeafFuncAttribute();
        if (!leafAttr) continue;
        
        // 如果设置了mixId，说明是Mix子图拆分出来的
        if (leafAttr->mixId != -1) {
            mixSplitFunctions.push_back(func);
        }
    }
    
    EXPECT_EQ(mixSplitFunctions.size(), 6) << "Should have 6 functions from mix subgraph split";
    // 从programs中查找各个LeafFunction
    Function* c1Func = programs[2];
    Function* v2Func = programs[3];
    Function* c3Func = programs[4];
    Function* c4Func = programs[5];
    Function* v5Func = programs[6];
    Function* v6Func = programs[7];
    ASSERT_NE(c1Func, nullptr) << "C1 function not found at programId=2";
    ASSERT_NE(v2Func, nullptr) << "V2 function not found at programId=3";
    ASSERT_NE(c3Func, nullptr) << "C3 function not found at programId=4";
    ASSERT_NE(c4Func, nullptr) << "C4 function not found at programId=5";
    ASSERT_NE(v5Func, nullptr) << "V5 function not found at programId=6";
    ASSERT_NE(v6Func, nullptr) << "V6 function not found at programId=7";

    // 3. 验证外部依赖正确传播
    // 检查C1 scope应该接收到logicalTensor1（来自非Mix子图1）
    bool hasLogicalTensor1 = false;
    for (const auto& tensor : c1Func->GetIncast()) {
        if (tensor == logicalTensor1) {
            hasLogicalTensor1 = true;
            break;
        }
    }
    EXPECT_TRUE(hasLogicalTensor1) << "C1 should have logicalTensor1 as incast";

    // 检查V2组件也应该接收到logicalTensor1（通过C1传播）
    bool v2HasLogicalTensor1 = false;
    for (const auto& tensor : v2Func->GetIncast()) {
        if (tensor == logicalTensor1) {
            v2HasLogicalTensor1 = true;
            break;
        }
    }
    EXPECT_TRUE(v2HasLogicalTensor1) << "V2 should have logicalTensor1 as incast (propagated from C1)";
    // 检查C4 scope应该接收到logicalTensor2（来自非Mix子图2）
    bool hasLogicalTensor2 = false;
    for (const auto& tensor : c4Func->GetIncast()) {
        if (tensor == logicalTensor2) {
            hasLogicalTensor2 = true;
            break;
        }
    }
    EXPECT_TRUE(hasLogicalTensor2) << "C4 should have logicalTensor2 as incast";

    // 检查V5组件也应该接收到logicalTensor2（通过C4传播）
    bool v5HasLogicalTensor2 = false;
    for (const auto& tensor : v5Func->GetIncast()) {
        if (tensor == logicalTensor2) {
            v5HasLogicalTensor2 = true;
            break;
        }
    }
    EXPECT_TRUE(v5HasLogicalTensor2) << "V5 should have logicalTensor2 as incast (propagated from C4)";

    // 检查V6组件应该输出globalOutputTensor
    bool hasGlobalOutput = false;
    for (const auto& tensor : v6Func->GetOutcast()) {
        if (tensor == globalOutputTensor) {
            hasGlobalOutput = true;
            break;
        }
    }
    EXPECT_TRUE(hasGlobalOutput) << "V6 should have globalOutputTensor as outcast";

    // 验证冗余依赖消除：检查C3和V6不应该直接接收logicalTensor1或logicalTensor2（应通过V2和V5间接获取）
    bool c3HasDirectTensor1 = false;
    bool c3HasDirectTensor2 = false;
    for (const auto& tensor : c3Func->GetIncast()) {
        if (tensor == logicalTensor1) c3HasDirectTensor1 = true;
        if (tensor == logicalTensor2) c3HasDirectTensor2 = true;
    }
    
    EXPECT_FALSE(c3HasDirectTensor1) << "C3 should not have direct incast of logicalTensor1 (should get via V2)";
    EXPECT_FALSE(c3HasDirectTensor2) << "C3 should not have direct incast of logicalTensor2 (should get via V5)";

    bool v6HasDirectTensor1 = false, v6HasDirectTensor2 = false;
    for (const auto& tensor : v6Func->GetIncast()) {
        if (tensor == logicalTensor1) v6HasDirectTensor1 = true;
        if (tensor == logicalTensor2) v6HasDirectTensor2 = true;
    }
    EXPECT_FALSE(v6HasDirectTensor1) << "V6 should not have direct incast of logicalTensor1 (redundant)";
    EXPECT_FALSE(v6HasDirectTensor2) << "V6 should not have direct incast of logicalTensor2 (redundant)";

    // 获取所有CallOps并构建Function到CallOp的映射
    auto newCallOps = rootFuncPtr->GetCallopList();
    std::unordered_map<Function*, Operation*> funcToCallOp;
    for (auto* callOpPtr : newCallOps) {
        auto loopCallAttr = dynamic_cast<CallOpAttribute*>(callOpPtr->GetOpAttribute().get());
        if (loopCallAttr && loopCallAttr->invokeInfo_) {
            uint64_t progId = loopCallAttr->invokeInfo_->GetProgramId();
            auto it = programs.find(progId);
            if (it != programs.end()) {
                funcToCallOp[it->second] = callOpPtr;
            }
        }
    }
    Operation* c1CallOp = funcToCallOp[c1Func];
    Operation* v2CallOp = funcToCallOp[v2Func];
    Operation* c3CallOp = funcToCallOp[c3Func];
    Operation* c4CallOp = funcToCallOp[c4Func];
    Operation* v5CallOp = funcToCallOp[v5Func];
    Operation* v6CallOp = funcToCallOp[v6Func];
    ASSERT_NE(c1CallOp, nullptr) << "C1 callOp not found";
    ASSERT_NE(v2CallOp, nullptr) << "V2 callOp not found";
    ASSERT_NE(c3CallOp, nullptr) << "C3 callOp not found";
    ASSERT_NE(c4CallOp, nullptr) << "C4 callOp not found";
    ASSERT_NE(v5CallOp, nullptr) << "V5 callOp not found";
    ASSERT_NE(v6CallOp, nullptr) << "V6 callOp not found";
    // 验证内部依赖Dummy Tensors
    auto checkDependency = [](Operation* consumer, Operation* producer) -> bool {
        auto dependOperands = consumer->GetDependOperands();
        for (const auto& tensor : dependOperands) {
            if (!tensor) continue;
            bool hasProducer = false, hasConsumer = false;
            for (auto* p : tensor->GetProducers()) if (p == producer) hasProducer = true;
            for (auto* c : tensor->GetConsumers()) if (c == consumer) hasConsumer = true;
            if (hasProducer && hasConsumer) return true;
        }
        return false;
    };
    // C-C依赖：C1 -> C3, C4 -> C3
    EXPECT_TRUE(checkDependency(c3CallOp, c1CallOp)) << "Should have dependency C1 -> C3 (C-C dependency)";
    EXPECT_TRUE(checkDependency(c3CallOp, c4CallOp)) << "Should have dependency C4 -> C3 (C-C dependency)";

    // V-V依赖：V2 -> V6, V5 -> V6
    EXPECT_TRUE(checkDependency(v6CallOp, v2CallOp)) << "Should have dependency V2 -> V6 (V-V dependency)";
    EXPECT_TRUE(checkDependency(v6CallOp, v5CallOp)) << "Should have dependency V5 -> V6 (V-V dependency)";

    // 验证不应该有跨类型的依赖
    EXPECT_FALSE(checkDependency(v2CallOp, c1CallOp)) << "Should NOT have dependency C1 -> V2 (cross-type)";
    EXPECT_FALSE(checkDependency(v5CallOp, c4CallOp)) << "Should NOT have dependency C4 -> V5 (cross-type)";
    EXPECT_FALSE(checkDependency(c3CallOp, v2CallOp)) << "Should NOT have dependency V2 -> C3 (cross-type)";
    EXPECT_FALSE(checkDependency(c3CallOp, v5CallOp)) << "Should NOT have dependency V5 -> C3 (cross-type)";
}

TEST_F(MixSubgraphSplitTest, TestDependOperand) {
    // Build Graph
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Copyin2", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();

    // Add and check depend operand
    Operation *copyin2 = subGraph.GetOp("Copyin2");
    std::shared_ptr<LogicalTensor> tensor4 = subGraph.GetTensor("t4");
    copyin2->AddDependOperand(tensor4);
    Operation *add1 = subGraph.GetOp("Add1");
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t3");
    add1->AddDependOperand(tensor3);
    EXPECT_EQ(copyin2->GetDependOperands().front()->GetMagic(), MS_NUM3);
    EXPECT_EQ(copyin2->GetDependOperandSize(), 1);

    // Check depend
    tensor4->AddDependOp(copyin2);
    tensor4->AddDependOp(copyin2);
    EXPECT_EQ(tensor4->GetDependOps().size(), 1);
    tensor3->AddDependOp(add1);
    auto dependOp = *(tensor4->GetDependOps().begin());
    EXPECT_EQ(dependOp->GetOpMagic(), MS_NUM10005);
    EXPECT_EQ(tensor4->HasDependOp(copyin2), true);

    // Sort Operations
    function->SortOperations();
    Operation *alloc2 = subGraph.GetOp("Alloc2");
    auto sortedOpList = function->Operations().DuplicatedOpList();
    auto alloc2Iter = std::find(sortedOpList.begin(), sortedOpList.end(), alloc2);
    auto copyin2Iter = std::find(sortedOpList.begin(), sortedOpList.end(), copyin2);
    EXPECT_EQ(alloc2Iter - sortedOpList.begin() < copyin2Iter - sortedOpList.begin(), true);

    // Erase operands and depend Ops
    copyin2->EraseDependTensor(tensor4);
    add1->EraseDependTensor(tensor3);
    tensor4->RemoveDependOp(copyin2);
    tensor3->RemoveDependOp(add1);

    //Sort Operations
    function->SortOperations();
    auto sortedOpList2 = function->Operations().DuplicatedOpList();
    auto alloc2Iter2 = std::find(sortedOpList2.begin(), sortedOpList2.end(), alloc2);
    auto copyin2Iter2 = std::find(sortedOpList2.begin(), sortedOpList2.end(), copyin2);
    EXPECT_EQ(alloc2Iter2 - sortedOpList2.begin() > copyin2Iter2 - sortedOpList2.begin(), true);

    // Erase Operations
    copyin2->AddDependOperand(tensor4);
    tensor4->AddDependOp(copyin2);
    copyin2->SetAsDeleted();
    function->EraseOperations();
    EXPECT_EQ(tensor4->GetDependOps().size(), 0);
}
} // namespace tile_fwk
} // namespace npu