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
 * \file test_codegen_dyn_ffn.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "codegen/codegen.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "operator/models/deepseek/deepseek_moeinfer.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

class TestCodegenDynFFN : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}
};

void testffn() {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 128);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {128, 128}, {128, 128});

    constexpr int BATCH_SIZE = 64;
    constexpr int SEQUENCE = 1;
    constexpr int H = 7168;
    constexpr int ExpertDim = 2048;
    constexpr int BS = BATCH_SIZE * SEQUENCE;
    constexpr int BASIC_BATCH = 32;

    std::vector<int64_t> hiddenStatesShape{BS, H};
    std::vector<int64_t> weightShape{H, ExpertDim};
    std::vector<int64_t> OutShape{BS, H};

    Tensor hiddenStates(DT_FP32, hiddenStatesShape, "hiddenStates");
    Tensor ffnweight1(DT_FP16, weightShape, "weightShape1");
    Tensor ffnweight2(DT_FP16, weightShape, "weightShape2");
    Tensor ffnweight3(DT_FP16, weightShape, "weightShape3");
    Tensor ffnout(DT_FP32, OutShape, "ffnout");

    DynamicFFN(hiddenStates, ffnweight1, ffnweight2, ffnweight3, ffnout, BASIC_BATCH);
}

void testffnquant() {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 128);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {128, 128}, {128, 128});

    constexpr int BATCH_SIZE = 32;
    constexpr int SEQUENCE = 1;
    constexpr int H = 7168;
    constexpr int ExpertDim = 2048;
    constexpr int BS = BATCH_SIZE * SEQUENCE;
    constexpr int BASIC_BATCH = 32;

    std::vector<int64_t> hiddenStatesShape{BS, H};
    std::vector<int64_t> weightShape{H, ExpertDim};
    std::vector<int64_t> OutShape{BS, H};

    Tensor hiddenStates(DT_INT8, hiddenStatesShape, "hiddenStates");
    Tensor hiddenStatesScale(DT_FP32, {BS, 1}, "hiddenStatesScale");
    Tensor ffnWeight1(DT_INT8, weightShape, "ffnWeight1", TileOpFormat::TILEOP_NZ);
    Tensor ffnWeight2(DT_INT8, weightShape, "ffnWeight2", TileOpFormat::TILEOP_NZ);
    Tensor ffnWeight3(DT_INT8, weightShape, "ffnWeight3", TileOpFormat::TILEOP_NZ);
    Tensor ffnScale1(DT_FP32, {1, ExpertDim}, "ffnScale1");
    Tensor ffnScale2(DT_FP32, {1, ExpertDim}, "ffnScale2");
    Tensor ffnScale3(DT_FP32, {1, H}, "ffnScale3");
    Tensor ffnout(DT_FP32, OutShape, "ffnout");

    DynamicFFNQuant(hiddenStates, hiddenStatesScale, ffnWeight1, ffnWeight2, ffnWeight3, ffnScale1, ffnScale2,
        ffnScale3, ffnout, BASIC_BATCH);
}

TEST_F(TestCodegenDynFFN, FFNDynamicTest) {
    testffn();
}

TEST_F(TestCodegenDynFFN, FFNQuantDynamicTest) {
    testffnquant();
}
} // namespace npu::tile_fwk