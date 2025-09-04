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
 * \file test_ast_in_graph.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "test_common.h"
#include "machine/dump/machine_dump.h"
#include "operator/models/llama/llama_def.h"
#include "runtime.h"
#include "tilefwk_runtime_api.h"

namespace npu::tile_fwk {

class OnBoardTestAstInGraph : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

TEST(OnBoardTestAstInGraph, test_fa_all2all_128) {
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
    std::vector<int64_t> shape = {dim0, dim1};
    std::vector<int64_t> shape_reduce = {dim0, 1};
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

    std::string bin_path = MachineDump::PrepareBinPath();

    Program::GetInstance().GetConfig().Reset();
    config::SetHostConfig(KEY_DUMP_BIN_AND_JSON, true);
    config::SetHostConfig(KEY_DUMP_BIN_AND_JSON_PATH, bin_path);

    Program::GetInstance().GetConfig().Set<int>(NBUFFER_MERGE_MODE, 1);
    Tensor Q(DataType::DT_FP16, shape, (uint8_t *)q_ptr, "Q");
    Tensor K(DataType::DT_FP16, shape, (uint8_t *)k_ptr, "K");
    Tensor V(DataType::DT_FP16, shape, (uint8_t *)v_ptr, "V");
    Tensor M(DataType::DT_FP32, shape_reduce, "M");
    Tensor L(DataType::DT_FP32, shape_reduce, "L");
    Tensor Res(DataType::DT_FP32, shape, "Res");

    /* torch capture */
    TileFwkBeginFunction("FA", {Q, K, V, Res});
    {
        Res = FlashAttention(Q, K, V, M, L, atDims, DFS_VEC_CFG, DFS_CUBE_CFG);
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    auto filePath = bin_path + "/" + "FA_1.json";
    struct stat st = {};
    bool outPathExist = stat(filePath.c_str(), &st) == 0;
    EXPECT_EQ(outPathExist, true);

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);
    printf("====TEST==== WOK SIZE %lu.\n", workspaceSize);

    /* torch prepare args device memory and run  */
    uint8_t* outTensorAddr = nullptr;
    machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));
    std::vector<void*> opArgsRun = {q_ptr, k_ptr, v_ptr, outTensorAddr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStream(),  opArgsRun);

    /* compare result */
    std::vector<float> res(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outTensorAddr, capacity * sizeof(float));
    int ret = resultCmp(res_golden_data, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST(OnBoardTestAstInGraph, test_add) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit();
    int row = 64;
    int col = 64;
    const int capacity = row * col;
    std::vector<int64_t> shape = {row, col};
    void *x_ptr = readToDev("../tests/AddSub/x.bin", capacity);
    void *y_ptr = readToDev("../tests/AddSub/y.bin", capacity);
    std::string bin_path = MachineDump::PrepareBinPath();

    config::SetHostConfig(KEY_DUMP_BIN_AND_JSON, true);
    config::SetHostConfig(KEY_DUMP_BIN_AND_JSON_PATH, bin_path);
    TileFwkSetVecTileShapes({64, 64});
    Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
    Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
    Tensor output(DataType::DT_FP32, shape, "C");

    TileFwkBeginFunction("ADD_T", {input_a, input_b, output});
    {
        Tensor temp(DataType::DT_FP32, shape, "temp");
        temp = Add(input_a, input_b);
        TileFwkAssign(output, temp);
    }
    TileFwkEndFunction();

    void* handle = TileFwkCompile();

    auto filePath = bin_path + "/" + "ADD_T_1.json";
    struct stat st = {};
    bool outPathExist = stat(filePath.c_str(), &st) == 0;
    EXPECT_EQ(outPathExist, true);

    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    if (workspaceSize != 0) {
        machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);
    }

    uint8_t* outTensorAddr = nullptr;
    machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));

    std::vector<void*> opArgsRun = {x_ptr, y_ptr, outTensorAddr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStream(), opArgsRun);

    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outTensorAddr, capacity * sizeof(float));
    readInput("../tests/AddSub/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
    std::cout << "add test ret = " << ret << std::endl;
}

TEST(OnBoardTestAstInGraph, test_add1) {
    aclInit(nullptr);
    rtSetDevice(0);
    TileFwkInit();
    int row = 64;
    int col = 64;
    int capacity = row * col;
    std::vector<int64_t> shape = {row, col};
    void *x_ptr = readToDev(GetGoldenDir() + "/add_x.bin", capacity);
    void *y_ptr = readToDev(GetGoldenDir() + "/add_y.bin", capacity);
    Program::GetInstance().GetConfig().Reset();
    config::SetHostConfig(KEY_DUMP_BIN_AND_JSON, true);
    TileShape::Current().SetVecTile({8, 8});
    Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
    Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
    Tensor output(DataType::DT_FP32, shape, "C");
    TileFwkBeginFunction("ADD", {input_a, input_b, output});
    {
        output = Add(input_a, input_b);
    }
    TileFwkEndFunction();

    void* handle = TileFwkCompile();
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    printf("workspace size:%zu.\n", workspaceSize);
    if (workspaceSize != 0) {
        machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);
    }

    uint8_t* outTensorAddr = nullptr;
    machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));

    std::vector<void*> opArgsRun = {x_ptr, y_ptr, outTensorAddr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStream(),  opArgsRun);

    std::vector<float> res(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outTensorAddr, capacity * sizeof(float));
    std::vector<float> golden(capacity);
    readInput(GetGoldenDir() + "/add_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
} // namespace npu::tile_fwk
