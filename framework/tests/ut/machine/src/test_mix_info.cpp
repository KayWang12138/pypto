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
#include "interface/function/function.h"
#include "interface/operation/attribute.h"
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
    int result = DumpMixInfo(nullptr);
    EXPECT_EQ(result, 0);
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

TEST_F(TestMixInfo, DumpMixInfo_TensorGraphDynamicType)
{
    std::vector<int64_t> shape = {32, 32};
    
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");
    
    std::string funcName = "TestTensorGraphDynamicType";
    FUNCTION(funcName, {input, output})
    {
        output = Add(input, input);
    }
    
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);
    
    function->SetGraphType(GraphType::TENSOR_GRAPH);
    function->SetFunctionType(FunctionType::DYNAMIC);
    
    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_ExecuteGraphWithNullCallFunc)
{
    std::vector<int64_t> shape = {32, 32};
    
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");
    
    std::string funcName = "TestExecuteGraphWithNullCallFunc";
    FUNCTION(funcName, {input, output})
    {
        output = Add(input, input);
    }
    
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);
    
    function->SetGraphType(GraphType::EXECUTE_GRAPH);
    
    auto inCast = std::make_shared<LogicalTensor>(*function, DataType::DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*function, DataType::DT_FP32, shape);
    
    auto& callOp = function->AddOperation(Opcode::OP_CALL, {inCast}, {outCast});
    auto callAttr = std::make_shared<CallOpAttribute>();
    callAttr->SetCalleeMagicName("NONEXIST_FUNC");
    callOp.SetOpAttribute(callAttr);
    
    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_ExecuteGraphWithWrapIdNegative)
{
    std::vector<int64_t> shape = {32, 32};
    
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");
    
    std::string funcName = "TestExecuteGraphWithWrapIdNegative";
    FUNCTION(funcName, {input, output})
    {
        output = Add(input, input);
    }
    
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);
    
    function->SetGraphType(GraphType::EXECUTE_GRAPH);
    
    auto inCast = std::make_shared<LogicalTensor>(*function, DataType::DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*function, DataType::DT_FP32, shape);
    
    auto& callOp = function->AddOperation(Opcode::OP_CALL, {inCast}, {outCast});
    auto callAttr = std::make_shared<CallOpAttribute>();
    callAttr->SetCalleeMagicName("NONEXIST_FUNC");
    callAttr->wrapId = -1;
    callOp.SetOpAttribute(callAttr);
    
    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_LeafFuncWithNullAttr)
{
    std::vector<int64_t> shape = {32, 32};
    
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");
    
    std::string funcName = "TestLeafFuncWithNullAttr";
    FUNCTION(funcName, {input, output})
    {
        output = Add(input, input);
    }
    
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);
    
    function->SetGraphType(GraphType::EXECUTE_GRAPH);
    
    auto leafFunc = std::make_shared<Function>(Program::GetInstance(), "LeafFunc", "LeafFunc", function);
    ASSERT_NE(leafFunc, nullptr);
    
    leafFunc->SetGraphType(GraphType::EXECUTE_GRAPH);
    leafFunc->SetLeafFuncAttribute(nullptr);
    
    Program::GetInstance().InsertFuncToFunctionMap(leafFunc->GetMagicName(), leafFunc);
    
    auto inCast = std::make_shared<LogicalTensor>(*function, DataType::DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*function, DataType::DT_FP32, shape);
    
    auto& callOp = function->AddOperation(Opcode::OP_CALL, {inCast}, {outCast});
    auto callAttr = std::make_shared<CallOpAttribute>();
    callAttr->SetCalleeMagicName(leafFunc->GetMagicName());
    callAttr->wrapId = 0;
    callOp.SetOpAttribute(callAttr);
    
    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}