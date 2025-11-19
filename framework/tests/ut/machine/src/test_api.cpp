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
 * \file test_api.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "operator/models/llama/llama_def.h"
#include "machine/runtime/runtime.h"
#include "machine/runtime/tilefwk_runtime_api.h"
#include "interface/utils/file_utils.h"
#include "tilefwk/op_registry.h"

using namespace npu::tile_fwk;

class TestAstApi : public testing::Test {
public:
    static void SetUpTestCase() {
        config::Reset();
    }

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
    }

    void TearDown() override {}
};

/* test api mode, simu torch scene */
TEST_F(TestAstApi, test_ast) {
    rtSetDevice(0);
    TileFwkInit("test_");
    AttentionDims atDims = {1, 1, 128, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    int b = atDims.b;
    int n = atDims.n;
    int s = atDims.s;
    int d = atDims.d;
    int dim0 = b * n * s; // 1024
    int dim1 = d;         // 128
    int capacity = dim0 * dim1;
    int capacity_reduce = dim0 * 1;
    std::vector<int64_t> shape = {dim0, dim1};
    std::vector<int64_t> shape_reduce = {dim0, 1};
    std::vector<uint16_t> q_data(capacity);
    std::vector<uint16_t> k_data(capacity);
    std::vector<uint16_t> v_data(capacity);
    std::vector<float> res_golden_data(capacity);
    std::vector<float> max_golden_data(capacity_reduce);
    std::vector<float> sum_golden_data(capacity_reduce);
    Tensor Q(DataType::DT_FP16, shape, "Q");
    Tensor K(DataType::DT_FP16, shape, "K");
    Tensor V(DataType::DT_FP16, shape, "V");
    Tensor M(DataType::DT_FP32, shape_reduce, "M");
    Tensor L(DataType::DT_FP32, shape_reduce, "L");
    Tensor Res(DT_FP32, shape, "Res");
    std::vector<std::reference_wrapper<Tensor>> opArgs = {Q, K, V, M, L, Res};
    aclrtStream aicpuStream = nullptr;
    rtStreamCreate(&aicpuStream, RT_STREAM_PRIORITY_DEFAULT);
    ASSERT(aicpuStream != nullptr);

    /* torch capture */
    TileFwkBeginFunction("FA", opArgs);
    {
        Res = FlashAttention(Q, K, V, M, L, atDims, DFS_VEC_CFG, DFS_CUBE_CFG);
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    /* torch prepare args device memory and run  */
    uint8_t* outTensorAddr = nullptr;
    machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));
    std::vector<void*> opArgsRun = {nullptr, nullptr, nullptr, nullptr, nullptr, outTensorAddr};
    std::vector<size_t> opSizes = {0, 0, 0, 0, 0, 100};
    TileFwkRunAsync(handle, workspaceAddr, aicpuStream,  opArgsRun, opSizes);
    int rc = rtStreamSynchronize(aicpuStream);
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }
    TileFwkFreeHandle(handle);
    TileFwkFinalize();
    std::vector<int64_t> tile_shape = {1, 1, 128, 128};
    TileFwkSetVecTileShapes(tile_shape);
    TileFwkAssign(Q, K);
    rtStreamDestroy(aicpuStream);
}
