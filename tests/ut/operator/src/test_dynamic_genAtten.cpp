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
 * \file test_dynamic_genAtten.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "operator/models/nsa/gen_Attention.h"
#include "passes/execute_graph_pass/global_memory_reuse.h"

using namespace npu::tile_fwk;

class DynamicTestGenAtten : public testing::Test {
public:
    void SetUp() override {
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
    }

    void TearDown() override { config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);}
protected:
    bool oriEnableAihacBackend = false;
};

struct GenAttenConfig {
    int batchSize = 0;
    int headNumSize = 0;
    int s1Size = 0;
    int dimSize = 0;
};

template<typename T = npu::tile_fwk::float16>
void genAtten(GenAttenConfig &inputConfig) {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    Program::GetInstance().GetConfig().Set<uint8_t>(MACHINE_CONFIG, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));
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

    std::vector<int64_t> shape_cmpAtten = {B, S, N, D};
    std::vector<int64_t> shape_selAtten = {B, S, N, D};
    std::vector<int64_t> shape_winAtten = {B, S, N, D};
    std::vector<int64_t> shape_gatingScore = {B, S, N, NUM_3};
    std::vector<int64_t> shape_attentionOut = {B, S, N, D};

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

    GenAttention(cmpAtten, selAtten, winAtten, gatingScore, out_npu);
}

TEST_F(DynamicTestGenAtten, TestOnboardGenAttenTest_FP16) {
    GenAttenConfig config;
    config.batchSize = NUM_16;
    config.headNumSize = NUM_128;
    config.s1Size = 1;
    config.dimSize = NUM_512;
    genAtten<npu::tile_fwk::float16>(config);
}