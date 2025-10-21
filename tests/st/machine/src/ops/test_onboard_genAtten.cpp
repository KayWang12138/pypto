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
 * \file test_onboard_genAtten.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "machine/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/gen_Attention.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class TestGenAtten : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

constexpr int NUM_2 = 2;
constexpr int NUM_3 = 3;
constexpr int NUM_8 = 8;
constexpr int NUM_16 = 16;
constexpr int NUM_32 = 32;
constexpr int NUM_128 = 128;
constexpr int NUM_512 = 512;

template<typename T = npu::tile_fwk::float16>
void genAtten(GenAttenTileShapeConfig &tileConfig) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetRuntimeOption(MACHINE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));

    int paramsSize = 4;
    std::vector<int> inputParam(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param.bin", inputParam);

    int64_t B = inputParam[0];
    int64_t N = inputParam[1];
    int64_t S1 = inputParam[2];
    int64_t D = inputParam[3];
    DataType dType;
    if (std::is_same<T, float>::value) {
        dType = DT_FP32;
    } else {
        dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    }

    std::vector<int64_t> shape_cmpAtten = {B, S1, N, D};
    std::vector<int64_t> shape_selAtten = {B, S1, N, D};
    std::vector<int64_t> shape_winAtten = {B, S1, N, D};
    std::vector<int64_t> shape_gatingScore = {B, S1, N, NUM_3};
    std::vector<int64_t> shape_attentionOut = {B, S1, N, D};

    Tensor cmpAtten(dType, shape_cmpAtten, "cmpAtten");
    Tensor selAtten(dType, shape_selAtten, "selAtten");
    Tensor winAtten(dType, shape_winAtten, "winAtten");
    Tensor gatingScore(dType, shape_gatingScore, "gatingScore");
    Tensor out_npu(dType, shape_attentionOut, "out_npu");

    std::vector<T>cmpAttenData(B * S1 * N * D);
    std::vector<T>selAttenData(B * S1 * N * D);
    std::vector<T>winAttenData(B * S1 * N * D);
    std::vector<T>gatingScoreData(B * S1 * N * NUM_3);
    std::vector<T>out_goldenData(B * S1 * N * D);

    readInput(GetGoldenDir() + "/cmp_atten.bin", cmpAttenData);
    readInput(GetGoldenDir() + "/sel_atten.bin", selAttenData);
    readInput(GetGoldenDir() + "/win_atten.bin", winAttenData);
    readInput(GetGoldenDir() + "/gating_score.bin", gatingScoreData);
    readInput(GetGoldenDir() + "/attention_out.bin", out_goldenData);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(cmpAtten, cmpAttenData),
        RawTensorData::CreateTensor<T>(selAtten, selAttenData),
        RawTensorData::CreateTensor<T>(winAtten, winAttenData),
        RawTensorData::CreateTensor<T>(gatingScore, gatingScoreData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<T>(out_npu, 0),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<T>(out_npu, out_goldenData),
    });

    GenAttention(cmpAtten, selAtten, winAtten, gatingScore, out_npu, tileConfig);

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(out_goldenData, (T *)outs->data(), 0.001f));
#endif
}

TEST_F(TestGenAtten, TestDynamicGenAttenTest_B_16_S1_1_FP16) {
    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};
    genAtten<npu::tile_fwk::float16>(tileConfig);
}

TEST_F(TestGenAtten, TestDynamicGenAttenTest_B_16_S1_1_FP32) {
    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};
    genAtten<float>(tileConfig);
}

TEST_F(TestGenAtten, TestDynamicGenAttenTest_B_16_S1_1_BF16) {
    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};
    genAtten<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(TestGenAtten, TestDynamicGenAttenTest_B_16_S1_2_FP16) {
    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};
    genAtten<npu::tile_fwk::float16>(tileConfig);
}

TEST_F(TestGenAtten, TestDynamicGenAttenTest_B_16_S1_2_FP32) {
    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};
    genAtten<float>(tileConfig);
}

TEST_F(TestGenAtten, TestDynamicGenAttenTest_B_16_S1_2_BF16) {
    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};
    genAtten<npu::tile_fwk::bfloat16>(tileConfig);
}