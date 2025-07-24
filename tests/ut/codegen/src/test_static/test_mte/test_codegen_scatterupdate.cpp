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
 * \file test_codegen_scatterupdate.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"
#include "codegen/codegen.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_manager.h"
#include "operation/tilefwk_op.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"

namespace npu::tile_fwk {

class TestCodegenScatterUpdate : public ::testing::Test {
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

// ScatterUpdate
void TestScatterUpdate(std::vector<int> tileShape) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy",
        {
            {"RemoveRedundentReshape", "RemoveRedundentReshape", PassType::TYPE_TENSOR_GRAPH},
            {        "ExpandFunction",         "ExpandFunction", PassType::TYPE_TENSOR_GRAPH},
            {         "DuplicateView",          "DuplicateView",   PassType::TYPE_TILE_GRAPH},
            {     "MergeViewAssemble",      "MergeViewAssemble",   PassType::TYPE_TILE_GRAPH},
            {      "AssignMemoryType",       "AssignMemoryType",   PassType::TYPE_TILE_GRAPH},
            {       "InsertConvertOp",        "InsertConvertOp",   PassType::TYPE_TILE_GRAPH},
            {"SplitLargeFanoutTensor", "SplitLargeFanoutTensor",   PassType::TYPE_TILE_GRAPH},
            {    "SplitReshapeOpPVC2",     "SplitReshapeOpPVC2",   PassType::TYPE_TILE_GRAPH},
            {     "RemoveRedundentOp",      "RemoveRedundentOp",   PassType::TYPE_TILE_GRAPH},
            {        "GenerateMoveOp",         "GenerateMoveOp",   PassType::TYPE_TILE_GRAPH},
    });

    int h = 128, minusTwo = -2;
    Tensor output(DT_INT32, {h, h}, "output");
    Tensor idxs(DT_INT32, {h, h}, "idxs");
    Tensor keyStates(DT_INT32, {h, h}, "keyStates");

    std::string funcName = "ScatterUpdate";
    FUNCTION(funcName) {
        output = ScatterUpdate(output, idxs, keyStates, minusTwo);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenScatterUpdate, TestScatterupdateDim2) {
    TestScatterUpdate({16, 32});
}

TEST_F(TestCodegenScatterUpdate, TestBatchMatmul) {
    int bs = 1;
    int m = 32;
    int k = 32;
    int n = 32;

    std::vector<int> shapeA = {bs, m, k};
    std::vector<int> shapeB = {bs, k, n};
    std::vector<int> shapeC = {bs, m, n};

    Program::GetInstance().GetConfig().Reset();
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});
    Tensor matA(DT_FP16, shapeA, "MatA", NodeType::LOCAL, TileOpFormat::TILEOP_NZ);
    Tensor matB(DT_FP16, shapeB, "MatB", NodeType::LOCAL, TileOpFormat::TILEOP_ND);
    Tensor matC(DT_FP32, shapeC, "MatC");
    std::string funcName = "BATCHMATMUL";
    FUNCTION(funcName, FunctionType::STATIC, {matA, matB, matC}) {
        matC = npu::tile_fwk::Matrix::BatchMatmul<false, false>(DT_FP32, matA, matB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenScatterUpdate, TestScatterUpdate) {
    int S = 1;
    int S2 = 16;
    int kvLoraRank = 8;
    int qkRopeHeadDim = 8;

    std::vector<int> shape0 = {S2, kvLoraRank + qkRopeHeadDim}; // [16, 16]
    std::vector<int> shape1 = {1, S};
    std::vector<int> shape2 = {S, kvLoraRank + qkRopeHeadDim}; // [1, 16]

    Program::GetInstance().GetTileShape().SetVecTileShapes(16, 16);

    Tensor kv_len(DT_INT64, shape1, "kv_len");
    Tensor past_key_states(DT_FP32, shape0, "past_key_states");
    Tensor key_states(DT_FP32, shape2, "key_states"); // [16,16]

    // Tensor past_key_states_new(DT_FP32, shape0, "past_key_states_new");

    /* torch capture */
    std::string funcName = "ScatterUpdate";
    FUNCTION(funcName) {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
} // namespace npu::tile_fwk