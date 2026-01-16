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

#include "gtest/gtest.h"

#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/calc.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/tilefwk_op.h"

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

TEST_F(ReshapeErrorTest, ReshapeMismatchElementCount) {
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
    config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

    // input has 2 * 3 = 6 elements, output has 4 * 2 = 8 elements
    Tensor input(DT_FP32, {2, 3}, "input");
    Tensor output(DT_FP32, {4, 2}, "output");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input, 1.0f),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });

    // 在圖中構建一個元素個數不一致的 Reshape，用於觸發報錯機制
    FUNCTION("main", {input}, {output}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Reshape(output.GetDataType(), input, {4, 2});
        }
    }
}

} // namespace npu::tile_fwk


