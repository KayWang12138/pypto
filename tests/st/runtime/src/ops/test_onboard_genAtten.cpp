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
#include "common/data_type.h"
#include "interface/function/function.h"
#include "operation/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "runtime/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "models/nsa/gen_Attention.h"
#include "test_dynamic.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class TestGenAtten : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

struct GenAttenConfig {
    int batchSize = 0;
    int headNumSize = 0;
    int s1Size = 0;
    int dimSize = 0;
};

template<typename T = npu::tile_fwk::float16>
void genAtten(GenAttenConfig &inputConfig) {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    Program::GetInstance().GetConfig().Set<uint8_t>(MACHINE_CONFIG, static_cast<uint8_t>(MachineSchduleConfig::L2CACHE_AFFINITY_SCH));

    int B = inputConfig.batchSize;
    int N = inputConfig.headNumSize;
    int S = inputConfig.s1Size;
    int D = inputConfig.dimSize;
    DataType dType;
    if (std::is_same<T, float>::value) {
        dType = DT_FP32;
    } else {
        dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    }

    std::vector<int> shape_cmpAtten = {B, S, N, D};
    std::vector<int> shape_selAtten = {B, S, N, D};
    std::vector<int> shape_winAtten = {B, S, N, D};
    std::vector<int> shape_gatingScore = {B, S, N, NUM_3};
    std::vector<int> shape_attentionOut = {B, S, N, D};

    Tensor cmpAtten(dType, shape_cmpAtten, "cmpAtten");
    Tensor selAtten(dType, shape_selAtten, "selAtten");
    Tensor winAtten(dType, shape_winAtten, "winAtten");
    Tensor gatingScore(dType, shape_gatingScore, "gatingScore");
    Tensor out_npu(dType, shape_attentionOut, "out_npu");

    std::vector<T>cmpAttenData(B * S * N * D);
    std::vector<T>selAttenData(B * S * N * D);
    std::vector<T>winAttenData(B * S * N * D);
    std::vector<T>gatingScoreData(B * S * N * NUM_3);
    std::vector<T>out_goldenData(B * S * N * D);

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

    GenAttention(cmpAtten, selAtten, winAtten, gatingScore, out_npu);

    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    DynFuncRunner::Run(funcop);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(out_goldenData, (T *)outs->data(), 0.001f));
#endif
}

TEST_F(TestGenAtten, TestOnboardGenAttenTest_FP16_S1) {
    GenAttenConfig config;
    config.batchSize = NUM_16;
    config.headNumSize = NUM_128;
    config.s1Size = 1;
    config.dimSize = NUM_512;
    genAtten<npu::tile_fwk::float16>(config);
}

TEST_F(TestGenAtten, TestOnboardGenAttenTest_FP32_S1) {
    GenAttenConfig config;
    config.batchSize = NUM_16;
    config.headNumSize = NUM_128;
    config.s1Size = 1;
    config.dimSize = NUM_512;
    genAtten<float>(config);
}

TEST_F(TestGenAtten, TestOnboardGenAttenTest_BF16_S1) {
    GenAttenConfig config;
    config.batchSize = NUM_16;
    config.headNumSize = NUM_128;
    config.s1Size = 1;
    config.dimSize = NUM_512;
    genAtten<npu::tile_fwk::bfloat16>(config);
}

TEST_F(TestGenAtten, TestOnboardGenAttenTest_FP16_S2) {
    GenAttenConfig config;
    config.batchSize = NUM_16;
    config.headNumSize = NUM_128;
    config.s1Size = NUM_2;
    config.dimSize = NUM_512;
    genAtten<npu::tile_fwk::float16>(config);
}

TEST_F(TestGenAtten, TestOnboardGenAttenTest_FP32_S2) {
    GenAttenConfig config;
    config.batchSize = NUM_16;
    config.headNumSize = NUM_128;
    config.s1Size = NUM_2;
    config.dimSize = NUM_512;
    genAtten<float>(config);
}

TEST_F(TestGenAtten, TestOnboardGenAttenTest_BF16_S2) {
    GenAttenConfig config;
    config.batchSize = NUM_16;
    config.headNumSize = NUM_128;
    config.s1Size = NUM_2;
    config.dimSize = NUM_512;
    genAtten<npu::tile_fwk::bfloat16>(config);
}