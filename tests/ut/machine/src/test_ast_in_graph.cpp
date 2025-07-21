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
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "interface/utils/file_utils.h"
#include "models/llama/llama_def.h"
#include "machine/dump/task_dump_utils.h"
#define private public
#include "machine/dump/machine_dump.h"
#include "machine/cache_manager/cache_manager.h"
#undef private

namespace npu::tile_fwk {
namespace {
int row = 64;
int col = 64;
const int capacity = row * col;
}

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
    std::vector<int> shape = {dim0, dim1};
    std::vector<int> shape_reduce = {dim0, 1};

    std::string bin_path = MachineDump::PrepareBinPath();

    Program::GetInstance().GetConfig().Reset();
    config::SetHostConfig(KEY_DUMP_BIN_AND_JSON, true);
    config::SetHostConfig(KEY_DUMP_BIN_AND_JSON_PATH, bin_path);

    Program::GetInstance().GetConfig().Set<int>(DB_TYPE, 1);
    Program::GetInstance().GetConfig().Set<int>(NBUFFER_NUM, 1);

    Tensor Q(DataType::DT_FP16, shape, "Q");
    Tensor K(DataType::DT_FP16, shape, "K");
    Tensor V(DataType::DT_FP16, shape, "V");
    Tensor M(DataType::DT_FP32, shape_reduce, "M");
    Tensor L(DataType::DT_FP32, shape_reduce, "L");
    Tensor Res(DT_FP32, shape, "Res");

    /* torch capture */
    TileFwkBeginFunction("FA", {Q, K, V, Res});
    {
        Res = FlashAttention(Q, K, V, M, L, atDims, DFS_VEC_CFG, DFS_CUBE_CFG);
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();
    (void)handle;
    auto filePath = bin_path + "/" + "FA_1.json";
    struct stat st = {};
    bool outPathExist = stat(filePath.c_str(), &st) == 0;
    EXPECT_EQ(outPathExist, true);
}

TEST(OnBoardTestAstInGraph, test_fa_all2all_128_2) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit();
    AttentionDims atDims = {1, 2, 128, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    int b = atDims.b;
    int n = atDims.n;
    int s = atDims.s;
    int d = atDims.d;
    int dim0 = b * n * s; // 1024
    int dim1 = d;         // 128
    std::vector<int> shape = {dim0, dim1};
    std::vector<int> shape_reduce = {dim0, 1};

    Program::GetInstance().GetConfig().Reset();
    Program::GetInstance().GetConfig().Set<int>(DB_TYPE, 1);
    Program::GetInstance().GetConfig().Set<int>(NBUFFER_NUM, 1);

    Tensor Q(DataType::DT_FP16, shape, "Q");
    Tensor K(DataType::DT_FP16, shape, "K");
    Tensor V(DataType::DT_FP16, shape, "V");
    Tensor M(DataType::DT_FP32, shape_reduce, "M");
    Tensor L(DataType::DT_FP32, shape_reduce, "L");
    Tensor Res(DT_FP32, shape, "Res");

    /* torch capture */
    TileFwkBeginFunction("FA", {Q, K, V, Res});
    {
        Res = FlashAttention(Q, K, V, M, L, atDims, DFS_VEC_CFG, DFS_CUBE_CFG);
        std::string symbol = Program::GetInstance().GetCurrentFunction()->GetDistTilingManager()->CreateTilingStorage("hello_world", 10);
        std::vector<int> tilingData = {1,2,3,4};
        Program::GetInstance().GetCurrentFunction()->GetDistTilingManager()->Save(symbol, tilingData);
    }
    TileFwkEndFunction();

    DeviceAgentTask *deviceAgentTask = reinterpret_cast<DeviceAgentTask *>(TileFwkCompile());

    auto binFilePath = CacheManager::Instance().cacheDirPath_ + "/ast_op_" + Program::GetInstance().GetCurrentFunction()->GetFunctionHash().Data() + ".o";
    EXPECT_EQ(RealPath(binFilePath).empty(), false);

    EXPECT_EQ(TaskDumpUtils::RecoverTaskFromBinFile(binFilePath, deviceAgentTask), true);
    delete deviceAgentTask;
}
}
