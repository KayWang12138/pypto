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
#include <vector>
#include <memory>

#include "interface/inner/tilefwk.h"
#include "interface/inner/pre_def.h"
#include "interface/configs/config_manager.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/interpreter/operation.h"

namespace npu::tile_fwk {

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

TEST_F(ReshapeErrorTest, TestLogTensorListDirectOperation) {
    // 直接構造一個 Program 和 Function，不通過 FUNCTION 宏
    Program program;
    Function func(program, "test_magic", "test_raw", nullptr);

    // 構造輸入 / 輸出 LogicalTensor 列表
    LogicalTensors iOperands;
    LogicalTensors oOperands;

    auto inTensor = std::make_shared<LogicalTensor>(func, DT_FP32, std::vector<int64_t>{2, 3},
        TileOpFormat::TILEOP_ND, "input");
    auto outTensor = std::make_shared<LogicalTensor>(func, DT_FP32, std::vector<int64_t>{3, 2},
        TileOpFormat::TILEOP_ND, "output");

    iOperands.push_back(inTensor);
    oOperands.push_back(outTensor);

    // 直接構造一個 Operation
    Operation op(func, Opcode::OP_RESHAPE, iOperands, oOperands, false);

    // 調用 LogTensorList 測試日誌打印，不依賴於完整的函數構建
    LogTensorList("input", &op, op.GetIOperands());
    LogTensorList("output", &op, op.GetOOperands());

    // 驗證 operation 內部確實有輸入 / 輸出
    EXPECT_EQ(op.GetIOperands().size(), 1);
    EXPECT_EQ(op.GetOOperands().size(), 1);
}

} // namespace npu::tile_fwk


