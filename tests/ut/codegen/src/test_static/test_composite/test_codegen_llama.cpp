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
 * \file test_codegen_llama.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "codegen/codegen.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/machine/host/host_machine.h"
#include "operator/models/llama/llama_def.h"
#include "interface/configs/config_manager.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"

namespace npu::tile_fwk {

class TestCodegenLlama : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}
};

void CgLLamaLayer(const AttentionDims &dimsCfg, float threadhold = 0.001f) {
    (void)threadhold;
    int b = dimsCfg.b;
    int n = dimsCfg.n;
    int s = dimsCfg.s;
    int d = dimsCfg.d;
    Tensor H(DataType::DT_FP32, {b * s, n * d}, "H");
    Tensor AW(DataType::DT_FP16, {n * d, n * d * 3}, "AW");
    Tensor DW(DataType::DT_FP16, {n * d, n * d}, "DW");
    Tensor FW(DataType::DT_FP16, {n * d, n * d * 3}, "FW");
    Tensor Res(DT_FP32, {b * s, n * d}, "Res");
    ConfigManager::Instance();
    std::string funcName = "LLAMA";
    config::SetBuildStatic(true);
    FUNCTION(funcName, {H, AW, DW, FW, Res}) {
        Res = LlamaLayer(H, AW, DW, FW, dimsCfg, SMALL_DFS_VEC_CFG, DFS_CUBE_CFG);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenLlama, TestCodegenLlama) {
    AttentionDims dimsCfg = {1, 1, 128, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    CgLLamaLayer(dimsCfg, 0.005f);
}
} // namespace npu::tile_fwk