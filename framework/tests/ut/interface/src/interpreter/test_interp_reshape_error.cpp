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
 * \file test_interp_reshape_error.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <sstream>
#include <vector>
#include <memory>

#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/calc.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/interpreter/operation.h"
#include "interface/utils/log.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/tilefwk_op.h"

namespace npu::tile_fwk {

// 辅助函数：用于测试 LogTensorList
static std::string DumpShapeVec(const std::vector<int64_t> &shape) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            ss << ", ";
        }
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

static std::string DumpSymbolicVec(const std::vector<SymbolicScalar> &symbols) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < symbols.size(); ++i) {
        if (i != 0) {
            ss << ", ";
        }
        ss << symbols[i].Dump();
    }
    ss << "]";
    return ss.str();
}

// LogTensorList 函数实现（用于测试）
static void LogTensorList(const char *role, Operation *op,
    const std::vector<std::shared_ptr<LogicalTensor>> &tensors) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }
        auto shapeStr = DumpShapeVec(tensor->shape);
        auto offsetStr = DumpShapeVec(tensor->offset);
        auto dynValidShapeStr = DumpSymbolicVec(tensor->GetDynValidShape());
        auto dynOffsetStr = DumpSymbolicVec(tensor->GetDynOffset());
        ALOG_ERROR_F(
            "ExecuteOperation error: op %s (magic=%d) %s[%zu] tensorMagic=%d, "
            "shape=%s, offset=%s, dynValidShape=%s, dynOffset=%s",
            op->GetOpcodeStr().c_str(),
            op->GetOpMagic(),
            role,
            i,
            tensor->magic,
            shapeStr.c_str(),
            offsetStr.c_str(),
            dynValidShapeStr.c_str(),
            dynOffsetStr.c_str());
    }
}

class ReshapeErrorTest : public testing::Test {
public:
    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        ProgramData::GetInstance().Reset();
        config::SetHostOption(ONLY_CODEGEN, true);
        if (!calc::IsVerifyEnabled()) {
            GTEST_SKIP() << "Verify not supported skip the verify test";
        }
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    }

    void TearDown() override {
        config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
        config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
    }
};

TEST_F(ReshapeErrorTest, TestLogTensorList) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    // 創建一個簡單的函數來獲取 Operation 對象
    Tensor input(DT_FP32, {2, 3}, "input");
    Tensor output(DT_FP32, {3, 2}, "output");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input, 1.0f),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });

    // 構建一個函數來獲取 Operation 對象
    FUNCTION("main", {input}, {output}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Reshape(output.GetDataType(), input, {3, 2});
        }
    }

    // 獲取函數中的操作
    auto func = Program::GetInstance().GetFunction("main");
    ASSERT_NE(func, nullptr);
    
    // 獲取函數中的操作列表
    auto ops = func->Operations();
    bool foundOp = false;
    for (auto &opPtr : ops) {
        if (opPtr->GetOpcode() == Opcode::OP_RESHAPE) {
            foundOp = true;
            Operation *op = opPtr.get();
            
            // 直接調用 LogTensorList 來測試
            LogTensorList("input", op, opPtr->GetIOperands());
            LogTensorList("output", op, opPtr->GetOOperands());
            
            // 驗證函數被調用（通過檢查操作是否有輸入輸出）
            EXPECT_GT(opPtr->GetIOperands().size(), 0);
            EXPECT_GT(opPtr->GetOOperands().size(), 0);
            break;
        }
    }
    EXPECT_TRUE(foundOp) << "Should find Reshape operation";
}

} // namespace npu::tile_fwk


