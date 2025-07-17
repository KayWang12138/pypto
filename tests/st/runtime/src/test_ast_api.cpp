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
 * \file test_ast_api.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <functional>
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "common/data_type.h"
#include "test_common.h"
#include "models/llama/llama_def.h"
#include "runtime/runtime.h"

namespace npu::tile_fwk {

class OnBoardTestAstApi : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};


/* test api mode, simu torch scene */
TEST(OnBoardTestAstApi, test_fa_all2all_ast_api_mode) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit();

    AttentionDims atDims = {1, 1, 128, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    int b = atDims.b;
    int n = atDims.n;
    int s = atDims.s;
    int d = atDims.d;
    int dim0 = b * n * s; // 1024
    int dim1 = d;         // 128
    int capacity = dim0 * dim1;
    int capacity_reduce = dim0 * 1;
    std::vector<int> shape = {dim0, dim1};
    std::vector<int> shape_reduce = {dim0, 1};
    std::vector<uint16_t> q_data(capacity);
    std::vector<uint16_t> k_data(capacity);
    std::vector<uint16_t> v_data(capacity);
    std::vector<float> res_golden_data(capacity);
    std::vector<float> max_golden_data(capacity_reduce);
    std::vector<float> sum_golden_data(capacity_reduce);
    void *q_ptr = readToDev<uint16_t>(GetGoldenDir() + "/q.bin", capacity);
    void *k_ptr = readToDev<uint16_t>(GetGoldenDir() + "/k.bin", capacity);
    void *v_ptr = readToDev<uint16_t>(GetGoldenDir() + "/v.bin", capacity);
    readInput(GetGoldenDir() + "/res_golden.bin", res_golden_data);
    readInput(GetGoldenDir() + "/max_golden.bin", max_golden_data);
    readInput(GetGoldenDir() + "/sum_golden.bin", sum_golden_data);

    Program::GetInstance().GetConfig().Reset();
    Program::GetInstance().GetConfig().Set<int>(DB_TYPE, 1);
    Program::GetInstance().GetConfig().Set<int>(NBUFFER_NUM, 1);
    Tensor Q(DataType::DT_FP16, shape, (uint8_t *)q_ptr, "Q");
    Tensor K(DataType::DT_FP16, shape, (uint8_t *)k_ptr, "K");
    Tensor V(DataType::DT_FP16, shape, (uint8_t *)v_ptr, "V");
    Tensor M(DataType::DT_FP32, shape_reduce, "M");
    Tensor L(DataType::DT_FP32, shape_reduce, "L");
    Tensor Res(DataType::DT_FP32, shape, "Res");
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
    runtime::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    /* torch prepare args device memory and run  */
    uint8_t* outTensorAddr = nullptr;
    runtime::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));
    std::vector<void*> opArgsRun = {q_ptr, k_ptr, v_ptr, nullptr, nullptr, outTensorAddr};
    TileFwkRunAsync(handle, workspaceAddr, aicpuStream,  opArgsRun);
    int rc = rtStreamSynchronize(aicpuStream);
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    /* compare result */
    std::vector<float> res(capacity);
    runtime::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outTensorAddr, capacity * sizeof(float));
    int ret = resultCmp(res_golden_data, res, 0.001f);
    EXPECT_EQ(ret, true);
    rtStreamDestroy(aicpuStream);
}

TEST(OnBoardTestAstApi, test_add_sub_all2all_torchapi_multi_function) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit();
    int row = 64;
    int col = 64;
    const int capacity = row * col;
    std::vector<int> shape = {row, col};
    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity);
    TileFwkSetVecTileShapes({64, 64});
    Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
    Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
    Tensor output(DataType::DT_FP32, shape, "C");
    aclrtStream aicpuStream = nullptr;
    rtStreamCreate(&aicpuStream, RT_STREAM_PRIORITY_DEFAULT);
    ASSERT(aicpuStream != nullptr);

    TileFwkBeginFunction("ADD_T", {input_a, input_b, output});
    {
        Tensor temp(DataType::DT_FP32, shape, "temp");
        temp = Add(input_a, input_b);
        TileFwkAssign(output, temp);
    }
    TileFwkEndFunction();

    void* handle = TileFwkCompile();

    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    runtime::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    uint8_t* outTensorAddr = nullptr;
    runtime::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));

    std::vector<void*> opArgsRun = {x_ptr, y_ptr, outTensorAddr};
    TileFwkRunAsync(handle, workspaceAddr, aicpuStream, opArgsRun);
    int rc = rtStreamSynchronize(aicpuStream);
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("ADD_T function aicpu stream sync failed");
    }

    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);
    runtime::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outTensorAddr, capacity * sizeof(float));
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
    std::cout << "add test ret = " << ret << std::endl;

    TileFwkBeginFunction("SUB_T", {input_a, input_b, output});
    {
        Tensor temp1(DataType::DT_FP32, shape, "temp");
        temp1 = Sub(input_a, input_b);
        TileFwkAssign(output, temp1);
    }
    TileFwkEndFunction();

    void* handle_1 = TileFwkCompile();

    uint8_t* workspaceAddr_1= nullptr;
    uint64_t workspaceSize_1= 0;
    TileFwkGetWorkspaceSize(handle_1, &workspaceSize_1);
    runtime::GetRA()->AllocDevAddr(&workspaceAddr_1, workspaceSize_1);

    uint8_t* outTensorAddr_1 = nullptr;
    runtime::GetRA()->AllocDevAddr(&outTensorAddr_1, capacity * sizeof(float));

    std::vector<void*> opArgsRun_1 = {x_ptr, y_ptr, outTensorAddr_1};
    TileFwkRunAsync(handle_1, workspaceAddr_1, aicpuStream, opArgsRun_1);
    rc = rtStreamSynchronize(aicpuStream);
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("SUB_T function aicpu stream sync failed");
    }

    std::vector<float> golden_1(capacity);
    std::vector<float> res_1(capacity);
    runtime::GetRA()->CopyFromTensor((uint8_t *)res_1.data(), (uint8_t *)outTensorAddr_1, capacity * sizeof(float));
    readInput(GetGoldenDir() + "/res_sub.bin", golden_1);
    int ret1= resultCmp(golden_1, res_1, 0.001f);
    EXPECT_EQ(ret1, true);

    rtStreamDestroy(aicpuStream);
}

}
