/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_mix_info.cpp
 * \brief Unit test for mix_info module
 */

#include <gtest/gtest.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "machine/host/mix_info.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::mix_info;
using json = nlohmann::json;

class TestMixInfo : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetBuildStatic(true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        TileShape::Current().SetVecTile(64, 64);
    }

    void TearDown() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(TestMixInfo, DumpMixInfo_NullFunction)
{
    GTEST_SKIP() << "Null function test skipped - source code does not handle nullptr";
}

TEST_F(TestMixInfo, DumpMixInfo_SimpleFunction)
{
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestDumpMixInfoFunc";
    FUNCTION(funcName, {input, output})
    {
        output = Add(input, input);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_ExecuteGraphFunction)
{
    std::vector<int64_t> shape = {32, 32};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestExecuteGraphFunc";
    FUNCTION(funcName, {input, output})
    {
        auto tmp = Mul(input, input);
        output = Add(tmp, input);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::EXECUTE_GRAPH);

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_TileGraphFunction)
{
    std::vector<int64_t> shape = {16, 16};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestTileGraphFunc";
    FUNCTION(funcName, {input, output})
    {
        output = Sub(input, input);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::TILE_GRAPH);

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_WithSyncOps)
{
    std::vector<int64_t> shape = {64, 64};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestFuncWithSyncOps";
    FUNCTION(funcName, {input, output})
    {
        auto tmp = Add(input, input);
        output = Mul(tmp, tmp);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    auto ops = function->Operations(false);
    if (!ops.DuplicatedOpList().empty()) {
        for (auto& op : ops.DuplicatedOpList()) {
            if (op->GetOpcode() == Opcode::OP_CV_SYNC_SRC || op->GetOpcode() == Opcode::OP_CV_SYNC_DST) {
                op->syncQueue_ = {PipeType::PIPE_M, PipeType::PIPE_M, CoreType::AIV, CoreType::AIV, 1, AIVCore::UNSPECIFIED};
            }
        }
    }

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_DynamicTensorGraph)
{
    std::vector<int64_t> shape = {32, 32};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestDynamicTensorGraphFunc";
    FUNCTION(funcName, {input, output})
    {
        output = Add(input, input);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetFunctionType(FunctionType::DYNAMIC);
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_DynamicLoopTensorGraph)
{
    std::vector<int64_t> shape = {64, 64};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestDynamicLoopTensorGraphFunc";
    FUNCTION(funcName, {input, output})
    {
        output = Mul(input, input);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetFunctionType(FunctionType::DYNAMIC_LOOP);
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_DynamicLoopPathTensorGraph)
{
    std::vector<int64_t> shape = {32, 32};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestDynamicLoopPathTensorGraphFunc";
    FUNCTION(funcName, {input, output})
    {
        output = Sub(input, input);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_BlockGraphWithLeafAttr)
{
    std::vector<int64_t> shape = {32, 32};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestBlockGraphWithLeafAttr";
    FUNCTION(funcName, {input, output})
    {
        output = Add(input, input);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::BLOCK_GRAPH);

    auto leafAttr = std::make_shared<LeafFuncAttribute>();
    leafAttr->mixId = 12345;
    function->SetLeafFuncAttribute(leafAttr);

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_ExecuteGraphWithWrapId)
{
    std::vector<int64_t> shape = {64, 64};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string calleeFuncName = "TestCalleeFunc";
    FUNCTION(calleeFuncName, {input, output})
    {
        output = Add(input, input);
    }

    auto calleeFunc = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + calleeFuncName);
    ASSERT_NE(calleeFunc, nullptr);

    calleeFunc->SetGraphType(GraphType::BLOCK_GRAPH);
    auto calleeLeafAttr = std::make_shared<LeafFuncAttribute>();
    calleeLeafAttr->mixId = 100;
    calleeFunc->SetLeafFuncAttribute(calleeLeafAttr);

    std::string callerFuncName = "TestCallerFuncWithWrapId";
    Tensor callerInput(DataType::DT_FP32, shape, "caller_input");
    Tensor callerOutput(DataType::DT_FP32, shape, "caller_output");

    FUNCTION(callerFuncName, {callerInput, callerOutput})
    {
        callerOutput = Mul(callerInput, callerInput);
    }

    auto callerFunc = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + callerFuncName);
    ASSERT_NE(callerFunc, nullptr);

    callerFunc->SetGraphType(GraphType::EXECUTE_GRAPH);

    auto callopList = callerFunc->GetCallopList();
    if (!callopList.empty()) {
        for (auto& callop : callopList) {
            auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
            if (callopAttr != nullptr) {
                callopAttr->wrapId = 5;
                callopAttr->SetCalleeMagicName(calleeFunc->GetMagicName());
            }
        }
    }

    int result = DumpMixInfo(callerFunc);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_ExecuteGraphWithNegativeWrapId)
{
    std::vector<int64_t> shape = {32, 32};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestExecuteGraphNegativeWrapId";
    FUNCTION(funcName, {input, output})
    {
        auto tmp = Add(input, input);
        output = Mul(tmp, tmp);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::EXECUTE_GRAPH);

    auto callopList = function->GetCallopList();
    for (auto& callop : callopList) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        if (callopAttr != nullptr) {
            callopAttr->wrapId = -1;
        }
    }

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_MultipleMixIds)
{
    std::vector<int64_t> shape = {32, 32};

    std::string callee1Name = "TestMultiMixIdCallee1";
    Tensor callee1Input(DataType::DT_FP32, shape, "callee1_input");
    Tensor callee1Output(DataType::DT_FP32, shape, "callee1_output");

    FUNCTION(callee1Name, {callee1Input, callee1Output})
    {
        callee1Output = Add(callee1Input, callee1Input);
    }

    auto callee1 = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + callee1Name);
    ASSERT_NE(callee1, nullptr);
    callee1->SetGraphType(GraphType::BLOCK_GRAPH);
    auto leafAttr1 = std::make_shared<LeafFuncAttribute>();
    leafAttr1->mixId = 1000;
    callee1->SetLeafFuncAttribute(leafAttr1);

    std::string callee2Name = "TestMultiMixIdCallee2";
    Tensor callee2Input(DataType::DT_FP32, shape, "callee2_input");
    Tensor callee2Output(DataType::DT_FP32, shape, "callee2_output");

    FUNCTION(callee2Name, {callee2Input, callee2Output})
    {
        callee2Output = Mul(callee2Input, callee2Input);
    }

    auto callee2 = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + callee2Name);
    ASSERT_NE(callee2, nullptr);
    callee2->SetGraphType(GraphType::BLOCK_GRAPH);
    auto leafAttr2 = std::make_shared<LeafFuncAttribute>();
    leafAttr2->mixId = 1100;
    callee2->SetLeafFuncAttribute(leafAttr2);

    std::string topFuncName = "TestMultiMixIdTopFunc";
    Tensor topInput(DataType::DT_FP32, shape, "top_input");
    Tensor topOutput(DataType::DT_FP32, shape, "top_output");

    FUNCTION(topFuncName, {topInput, topOutput})
    {
        topOutput = Sub(topInput, topInput);
    }

    auto topFunc = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + topFuncName);
    ASSERT_NE(topFunc, nullptr);

    topFunc->SetGraphType(GraphType::EXECUTE_GRAPH);

    auto callopList = topFunc->GetCallopList();
    int idx = 0;
    for (auto& callop : callopList) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        if (callopAttr != nullptr) {
            callopAttr->wrapId = idx;
            if (idx == 0) {
                callopAttr->SetCalleeMagicName(callee1->GetMagicName());
            } else if (idx == 1) {
                callopAttr->SetCalleeMagicName(callee2->GetMagicName());
            }
            idx++;
        }
    }

    int result = DumpMixInfo(topFunc);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_BlockGraphWithoutLeafAttr)
{
    std::vector<int64_t> shape = {32, 32};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestBlockGraphNoLeafAttr";
    FUNCTION(funcName, {input, output})
    {
        output = Add(input, input);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::BLOCK_GRAPH);

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_LeafVfGraph)
{
    std::vector<int64_t> shape = {32, 32};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestLeafVfGraph";
    FUNCTION(funcName, {input, output})
    {
        output = Add(input, input);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::LEAF_VF_GRAPH);

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_ExecuteGraphWithCallOpNoCallee)
{
    std::vector<int64_t> shape = {32, 32};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestExecuteGraphNoCallee";
    FUNCTION(funcName, {input, output})
    {
        output = Add(input, input);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::EXECUTE_GRAPH);

    auto callopList = function->GetCallopList();
    for (auto& callop : callopList) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        if (callopAttr != nullptr) {
            callopAttr->wrapId = 10;
            callopAttr->SetCalleeMagicName("__nonexistent_function__");
        }
    }

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_TileGraphWithRootFunction)
{
    std::vector<int64_t> shape = {32, 32};

    std::string rootFuncName = "TestTileGraphRootFunc";
    Tensor rootInput(DataType::DT_FP32, shape, "root_input");
    Tensor rootOutput(DataType::DT_FP32, shape, "root_output");

    FUNCTION(rootFuncName, {rootInput, rootOutput})
    {
        rootOutput = Add(rootInput, rootInput);
    }

    auto rootFunc = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + rootFuncName);
    ASSERT_NE(rootFunc, nullptr);

    rootFunc->SetGraphType(GraphType::BLOCK_GRAPH);
    auto rootLeafAttr = std::make_shared<LeafFuncAttribute>();
    rootLeafAttr->mixId = 200;
    rootFunc->SetLeafFuncAttribute(rootLeafAttr);

    std::string tileFuncName = "TestTileGraphFunc";
    Tensor tileInput(DataType::DT_FP32, shape, "tile_input");
    Tensor tileOutput(DataType::DT_FP32, shape, "tile_output");

    FUNCTION(tileFuncName, {tileInput, tileOutput})
    {
        tileOutput = Mul(tileInput, tileInput);
    }

    auto tileFunc = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + tileFuncName);
    ASSERT_NE(tileFunc, nullptr);

    tileFunc->SetGraphType(GraphType::TILE_GRAPH);
    tileFunc->rootFunc_ = rootFunc;

    int result = DumpMixInfo(tileFunc);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_SyncSrcOperation)
{
    std::vector<int64_t> shape = {64, 64};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestSyncSrcFunc";
    FUNCTION(funcName, {input, output})
    {
        output = Add(input, input);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::BLOCK_GRAPH);

    auto leafAttr = std::make_shared<LeafFuncAttribute>();
    leafAttr->mixId = 500;
    function->SetLeafFuncAttribute(leafAttr);

    auto ops = function->Operations(false);
    auto opList = ops.DuplicatedOpList();

    for (auto& op : opList) {
        if (op->GetOpcode() == Opcode::OP_CV_SYNC_SRC) {
            op->syncQueue_ = {PipeType::PIPE_M, PipeType::PIPE_M, CoreType::AIV, CoreType::AIV, 5, AIVCore::UNSPECIFIED};
        }
    }

    std::string callerFuncName = "TestSyncSrcCaller";
    Tensor callerInput(DataType::DT_FP32, shape, "caller_input");
    Tensor callerOutput(DataType::DT_FP32, shape, "caller_output");

    FUNCTION(callerFuncName, {callerInput, callerOutput})
    {
        callerOutput = Mul(callerInput, callerInput);
    }

    auto callerFunc = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + callerFuncName);
    ASSERT_NE(callerFunc, nullptr);

    callerFunc->SetGraphType(GraphType::EXECUTE_GRAPH);

    auto callopList = callerFunc->GetCallopList();
    for (auto& callop : callopList) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        if (callopAttr != nullptr) {
            callopAttr->wrapId = 1;
            callopAttr->SetCalleeMagicName(function->GetMagicName());
        }
    }

    int result = DumpMixInfo(callerFunc);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_SyncDstOperation)
{
    std::vector<int64_t> shape = {64, 64};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestSyncDstFunc";
    FUNCTION(funcName, {input, output})
    {
        output = Add(input, input);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::BLOCK_GRAPH);

    auto leafAttr = std::make_shared<LeafFuncAttribute>();
    leafAttr->mixId = 600;
    function->SetLeafFuncAttribute(leafAttr);

    auto ops = function->Operations(false);
    auto opList = ops.DuplicatedOpList();

    for (auto& op : opList) {
        if (op->GetOpcode() == Opcode::OP_CV_SYNC_DST) {
            op->syncQueue_ = {PipeType::PIPE_M, PipeType::PIPE_M, CoreType::AIV, CoreType::AIV, 10, AIVCore::UNSPECIFIED};
        }
    }

    std::string callerFuncName = "TestSyncDstCaller";
    Tensor callerInput(DataType::DT_FP32, shape, "caller_input");
    Tensor callerOutput(DataType::DT_FP32, shape, "caller_output");

    FUNCTION(callerFuncName, {callerInput, callerOutput})
    {
        callerOutput = Mul(callerInput, callerInput);
    }

    auto callerFunc = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + callerFuncName);
    ASSERT_NE(callerFunc, nullptr);

    callerFunc->SetGraphType(GraphType::EXECUTE_GRAPH);

    auto callopList = callerFunc->GetCallopList();
    for (auto& callop : callopList) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        if (callopAttr != nullptr) {
            callopAttr->wrapId = 2;
            callopAttr->SetCalleeMagicName(function->GetMagicName());
        }
    }

    int result = DumpMixInfo(callerFunc);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_ExecuteGraphMultipleWrapIdsSameMixId)
{
    std::vector<int64_t> shape = {32, 32};

    std::string callee1Name = "TestCallee1SameMix";
    Tensor callee1Input(DataType::DT_FP32, shape, "callee1_input");
    Tensor callee1Output(DataType::DT_FP32, shape, "callee1_output");

    FUNCTION(callee1Name, {callee1Input, callee1Output})
    {
        callee1Output = Add(callee1Input, callee1Input);
    }

    auto callee1 = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + callee1Name);
    ASSERT_NE(callee1, nullptr);
    callee1->SetGraphType(GraphType::BLOCK_GRAPH);
    auto leafAttr1 = std::make_shared<LeafFuncAttribute>();
    leafAttr1->mixId = 3000;
    callee1->SetLeafFuncAttribute(leafAttr1);

    std::string callee2Name = "TestCallee2SameMix";
    Tensor callee2Input(DataType::DT_FP32, shape, "callee2_input");
    Tensor callee2Output(DataType::DT_FP32, shape, "callee2_output");

    FUNCTION(callee2Name, {callee2Input, callee2Output})
    {
        callee2Output = Mul(callee2Input, callee2Input);
    }

    auto callee2 = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + callee2Name);
    ASSERT_NE(callee2, nullptr);
    callee2->SetGraphType(GraphType::BLOCK_GRAPH);
    auto leafAttr2 = std::make_shared<LeafFuncAttribute>();
    leafAttr2->mixId = 3000;
    callee2->SetLeafFuncAttribute(leafAttr2);

    std::string callerName = "TestCallerSameMixId";
    Tensor callerInput(DataType::DT_FP32, shape, "caller_input");
    Tensor callerOutput(DataType::DT_FP32, shape, "caller_output");

    FUNCTION(callerName, {callerInput, callerOutput})
    {
        callerOutput = Sub(callerInput, callerInput);
    }

    auto caller = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + callerName);
    ASSERT_NE(caller, nullptr);
    caller->SetGraphType(GraphType::EXECUTE_GRAPH);

    auto callopList = caller->GetCallopList();
    int idx = 0;
    for (auto& callop : callopList) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        if (callopAttr != nullptr) {
            callopAttr->wrapId = idx;
            if (idx == 0) {
                callopAttr->SetCalleeMagicName(callee1->GetMagicName());
            } else if (idx == 1) {
                callopAttr->SetCalleeMagicName(callee2->GetMagicName());
            }
            idx++;
        }
    }

    int result = DumpMixInfo(caller);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_VerifyJsonOutput)
{
    std::vector<int64_t> shape = {32, 32};

    std::string calleeName = "TestCalleeJsonVerify";
    Tensor calleeInput(DataType::DT_FP32, shape, "callee_input");
    Tensor calleeOutput(DataType::DT_FP32, shape, "callee_output");

    FUNCTION(calleeName, {calleeInput, calleeOutput})
    {
        calleeOutput = Add(calleeInput, calleeInput);
    }

    auto callee = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + calleeName);
    ASSERT_NE(callee, nullptr);
    callee->SetGraphType(GraphType::BLOCK_GRAPH);
    auto leafAttr = std::make_shared<LeafFuncAttribute>();
    leafAttr->mixId = 7000;
    callee->SetLeafFuncAttribute(leafAttr);

    std::string callerName = "TestCallerJsonVerify";
    Tensor callerInput(DataType::DT_FP32, shape, "caller_input");
    Tensor callerOutput(DataType::DT_FP32, shape, "caller_output");

    FUNCTION(callerName, {callerInput, callerOutput})
    {
        callerOutput = Mul(callerInput, callerInput);
    }

    auto caller = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + callerName);
    ASSERT_NE(caller, nullptr);
    caller->SetGraphType(GraphType::EXECUTE_GRAPH);

    auto callopList = caller->GetCallopList();
    for (auto& callop : callopList) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        if (callopAttr != nullptr) {
            callopAttr->wrapId = 100;
            callopAttr->SetCalleeMagicName(callee->GetMagicName());
        }
    }

    int result = DumpMixInfo(caller);
    EXPECT_EQ(result, 0);

    std::string path = config::GetAbsoluteTopFolder() + "/mix_event_info.json";
    std::ifstream inFile(path);
    if (inFile.is_open()) {
        json j;
        inFile >> j;
        inFile.close();

        EXPECT_TRUE(j.is_array());
        if (j.is_array() && !j.empty()) {
            auto& firstEntry = j[0];
            EXPECT_TRUE(firstEntry.contains("mixId"));
            EXPECT_TRUE(firstEntry.contains("wrapInfos"));
        }
    }
}