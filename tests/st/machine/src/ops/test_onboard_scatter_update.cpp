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
 * \file test_onboard_scatter_update.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "test_common.h"
#include "test_suite_stest_ops.h"
#include "runtime.h"
#include "tilefwk_runtime_api.h"

using namespace npu::tile_fwk;

class ScatterupdateOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(ScatterupdateOnBoardTest, test_scatter_update_1_1_16_16) {
    int S = 1;
    int S2 = 16;
    int kvLoraRank = 8;
    int qkRopeHeadDim = 8;

    std::vector<int64_t> shape0 =  {S2, kvLoraRank + qkRopeHeadDim}; // [16, 16]
    std::vector<int64_t> shape1 = {1, S};
    std::vector<int64_t> shape2 = {S, kvLoraRank + qkRopeHeadDim}; // [1, 16]

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(16, 16);

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity2);
    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor key_states(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "key_states"); // [16,16]

    // Tensor past_key_states_new(DataType::DT_FP32, shape0, "past_key_states_new");
    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE", {past_key_states, kv_len, key_states});
    {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);
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
    // uint8_t* outTensorAddr = nullptr;
    // machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));
    std::vector<void*> opArgsRun = {x_ptr, indices_ptr, y_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<float> golden(capacity0);
    std::vector<float> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ScatterupdateOnBoardTest, test_scatter_update_1_1_20_20) {
    int S = 1;
    int S2 = 16;
    int kvLoraRank = 10;
    int qkRopeHeadDim = 10;

    std::vector<int64_t> shape0 =  {S2, kvLoraRank + qkRopeHeadDim}; // [16, 20]
    std::vector<int64_t> shape1 = {1, S};
    std::vector<int64_t> shape2 = {S, kvLoraRank + qkRopeHeadDim}; // [1, 20]

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(16, 16);

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity2);
    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor key_states(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "key_states"); // [16,16]

    // Tensor past_key_states_new(DataType::DT_FP32, shape0, "past_key_states_new");

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE", {past_key_states, kv_len, key_states});
    {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);
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
    // uint8_t* outTensorAddr = nullptr;
    // machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));
    std::vector<void*> opArgsRun = {x_ptr, indices_ptr, y_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<float> golden(capacity0);
    std::vector<float> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ScatterupdateOnBoardTest, test_scatter_update_1_1_16_16_bf16) {
    int S = 1;
    int S2 = 16;
    int kvLoraRank = 8;
    int qkRopeHeadDim = 8;

    std::vector<int64_t> shape0 =  {S2, kvLoraRank + qkRopeHeadDim}; // [16, 16]
    std::vector<int64_t> shape1 = {1, S};
    std::vector<int64_t> shape2 = {S, kvLoraRank + qkRopeHeadDim}; // [1, 16]

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(16, 16);

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity2);
    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_BF16, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor key_states(DataType::DT_BF16, shape2, (uint8_t *)y_ptr, "key_states"); // [16,16]

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE", {past_key_states, kv_len, key_states});
    {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);
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
    // uint8_t* outTensorAddr = nullptr;
    // machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));
    std::vector<void*> opArgsRun = {x_ptr, indices_ptr, y_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<npu::tile_fwk::bfloat16> golden(capacity0);
    std::vector<npu::tile_fwk::bfloat16> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(npu::tile_fwk::bfloat16));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp<npu::tile_fwk::bfloat16>(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(ScatterupdateOnBoardTest, test_scatter_update_1_16_16_16) {
    int S = 16;
    int S2 = 16;
    int kvLoraRank = 8;
    int qkRopeHeadDim = 8;

    std::vector<int64_t> shape0 =  {S2, kvLoraRank + qkRopeHeadDim}; // [16, 16]
    std::vector<int64_t> shape1 = {1, S};
    std::vector<int64_t> shape2 = {S, kvLoraRank + qkRopeHeadDim}; // [16, 16]

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(16, 16);

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity2);
    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor key_states(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "key_states"); // [16,16]

    // Tensor past_key_states_new(DataType::DT_FP32, shape0, "past_key_states_new");

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE", {past_key_states, key_states, kv_len});
    {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);
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
    // uint8_t* outTensorAddr = nullptr;
    // machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));
    std::vector<void*> opArgsRun = {x_ptr, y_ptr, indices_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<float> golden(capacity0);
    std::vector<float> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(ScatterupdateOnBoardTest, test_scatter_update_1_48_4096_512) {
    int B = 32;
    int S = 1;
    int S2 = 48;
    int kvLoraRank = 256;
    int qkRopeHeadDim = 256;

    std::vector<int64_t> shape0 =  {S2, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> shape1 = {B, S};
    std::vector<int64_t> shape2 = {B * S, kvLoraRank + qkRopeHeadDim};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(16, 512);

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity2);
    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor key_states(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "key_states"); // [16,16]

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE", {past_key_states, key_states, kv_len});
    {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2, "PA_BSND");
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
    // uint8_t* outTensorAddr = nullptr;
    // machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));
    std::vector<void*> opArgsRun = {x_ptr, y_ptr, indices_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<float> golden(capacity0);
    std::vector<float> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(ScatterupdateOnBoardTest, test_scatter_update_1_1_16_16_exp) {
    int S = 1;
    int S2 = 16;
    int kvLoraRank = 8;
    int qkRopeHeadDim = 8;

    std::vector<int64_t> shape0 =  {S2, kvLoraRank + qkRopeHeadDim}; // [16, 16]
    std::vector<int64_t> shape1 = {1, S};
    std::vector<int64_t> shape2 = {S, kvLoraRank + qkRopeHeadDim}; // [1, 16]

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(16, 16);

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity2);
    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor key_states(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "key_states"); // [1,16]
    Tensor res(DataType::DT_FP32, shape0, "res");
    // Tensor past_key_states_new(DataType::DT_FP32, shape0, "past_key_states_new");

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE_EXP", {past_key_states,kv_len,key_states, res});
    {
        TileShape::Current().SetVecTile(1, 8);
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);
        TileShape::Current().SetVecTile(16, 16);
        res = Exp(past_key_states);
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
    machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity0 * sizeof(float));
    std::vector<void*> opArgsRun = {x_ptr, indices_ptr, y_ptr, outTensorAddr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<float> golden(capacity0);
    std::vector<float> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)outTensorAddr, capacity0 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ScatterupdateOnBoardTest, test_scatter_update_2_1_512_576) {
   int B = 2;
    int S = 1;
    int S2 = 512;
    int kvLoraRank = 512;
    int qkRopeHeadDim = 64;

    std::vector<int64_t> shape0 =  {B, 1, S2, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> shape1 = {B, S};
    std::vector<int64_t> shape2 = {B, 1, S, kvLoraRank + qkRopeHeadDim};

    int capacity0 = shape0[0] * shape0[1] * shape0[2] * shape0[3];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1] * shape2[2] * shape2[3];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(1, 1, 1, 128);

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity2);

    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor key_states(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "key_states");

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE", {past_key_states, kv_len, key_states});
    {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);
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
    std::vector<void*> opArgsRun = {x_ptr, indices_ptr, y_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<float> golden(capacity0);
    std::vector<float> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ScatterupdateOnBoardTest, test_scatter_update_2_1_512_576_multi_row3) {
   int B = 2;
    int S = 3;
    int S2 = 512;
    int kvLoraRank = 512;
    int qkRopeHeadDim = 64;

    std::vector<int64_t> shape0 =  {B, 1, S2, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> shape1 = {B, S};
    std::vector<int64_t> shape2 = {B, 1, S, kvLoraRank + qkRopeHeadDim};

    int capacity0 = shape0[0] * shape0[1] * shape0[2] * shape0[3];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1] * shape2[2] * shape2[3];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(1, 1, 1, 128);

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity2);

    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor key_states(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "key_states");

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE_MULTI_ROW", {past_key_states, kv_len, key_states});
    {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);
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
    std::vector<void*> opArgsRun = {x_ptr, indices_ptr, y_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<float> golden(capacity0);
    std::vector<float> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f, 100000);
    EXPECT_EQ(ret, true);
}


TEST_F(ScatterupdateOnBoardTest, test_scatter_update_2_1_512_576_graphD) {
    int B = 2;
    int S = 1;
    int S2 = 512;
    int kvLoraRank = 512;
    int qkRopeHeadDim = 64;

    std::vector<int64_t> shape0 =  {B, 1, S2, kvLoraRank + qkRopeHeadDim}; // [2,1,512,576]
    std::vector<int64_t> shape1 = {B, S};
    std::vector<int64_t> shape2 = {B, 1, S, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> shape_compress_kv = {B, S, kvLoraRank};
    std::vector<int64_t> shape_k_pe_rope = {B, 1, S, qkRopeHeadDim};

    int capacity0 = shape0[0] * shape0[1] * shape0[2] * shape0[3];
    int capacity1 = shape1[0] * shape1[1];
    int capacity_compress_kv = shape_compress_kv[0] * shape_compress_kv[1] * shape_compress_kv[2];
    int capacity_k_pe_rope = shape_k_pe_rope[0] * shape_k_pe_rope[1] * shape_k_pe_rope[2] * shape_k_pe_rope[3];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *compress_kv_ptr = readToDev(GetGoldenDir() + "/compressed_kv.bin", capacity_compress_kv);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_k_pe_rope);

    TileShape::Current().SetVecTile(1, 1, 128, 128);

    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_FP32, {B, 1, S2, kvLoraRank + qkRopeHeadDim}, (uint8_t *)x_ptr, "past_key_states");
    // Tensor key_states(DataType::DT_FP32, {B, 1, S, kvLoraRank + qkRopeHeadDim}, "key_states"); // [2,1,1,576]
    Tensor compressed_kv(DataType::DT_FP32, {B, S, kvLoraRank}, (uint8_t *)compress_kv_ptr, "compressed_kv");
    Tensor k_pe_rope(DataType::DT_FP32, {B, 1, S, qkRopeHeadDim}, (uint8_t *)y_ptr, "k_pe_rope"); // (b,1,s,qkRopeHeadDim)

    // Tensor k_nope_new(DataType::DT_FP32, {B, 1, S, kvLoraRank}, "k_nope_new"); // (B,1,S,kvLoraRank)

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE_GRAPHD", {past_key_states, compressed_kv, kv_len, k_pe_rope});
    {
        Tensor k_nope = RmsNorm(compressed_kv); // (B, S, kvLoraRank)
        Tensor k_nope_new = Reshape(k_nope, {B, 1, S, kvLoraRank}); // (B,1,S,kvLoraRank)

        Tensor key_states = Concat({k_nope_new, k_pe_rope}, -1); // (B,1,S, kvLoraRank + qkRopeHeadDim)
        TileShape::Current().SetVecTile(1, 1, 512, 64);
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);

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
    std::vector<void*> opArgsRun = {x_ptr, compress_kv_ptr, indices_ptr, y_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<float> golden(capacity0);
    std::vector<float> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(ScatterupdateOnBoardTest, test_scatter_update_2_1_512_576_graphD_bf16) {
    int B = 2;
    int S = 1;
    int S2 = 512;
    int kvLoraRank = 512;
    int qkRopeHeadDim = 64;

    std::vector<int64_t> shape0 =  {B, 1, S2, kvLoraRank + qkRopeHeadDim}; // [2,1,512,576]
    std::vector<int64_t> shape1 = {B, S};
    std::vector<int64_t> shape2 = {B, 1, S, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> shape_compress_kv = {B, S, kvLoraRank};
    std::vector<int64_t> shape_k_pe_rope = {B, 1, S, qkRopeHeadDim};

    int capacity0 = shape0[0] * shape0[1] * shape0[2] * shape0[3];
    int capacity1 = shape1[0] * shape1[1];
    int capacity_compress_kv = shape_compress_kv[0] * shape_compress_kv[1] * shape_compress_kv[2];
    int capacity_k_pe_rope = shape_k_pe_rope[0] * shape_k_pe_rope[1] * shape_k_pe_rope[2] * shape_k_pe_rope[3];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *compress_kv_ptr = readToDev(GetGoldenDir() + "/compressed_kv.bin", capacity_compress_kv);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_k_pe_rope);

    TileShape::Current().SetVecTile(1, 1, 128, 128);

    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_BF16, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor compressed_kv(DataType::DT_BF16, {B, S, kvLoraRank}, (uint8_t *)compress_kv_ptr, "compressed_kv");
    Tensor k_pe_rope(DataType::DT_BF16, {B, 1, S, qkRopeHeadDim}, (uint8_t *)y_ptr, "k_pe_rope"); // (b,1,s,qkRopeHeadDim)

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE_GRAPHD", {past_key_states, compressed_kv, kv_len, k_pe_rope});
    {
        Tensor k_nope = RmsNorm(compressed_kv); // (B, S, kvLoraRank)
        Tensor k_nope_new = Reshape(k_nope, {B, 1, S, kvLoraRank}); // (B,1,S,kvLoraRank)

        Tensor key_states = Concat({k_nope_new, k_pe_rope}, -1); // (B,1,S, kvLoraRank + qkRopeHeadDim)
        TileShape::Current().SetVecTile(1, 1, 1, 128);
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);

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
    std::vector<void*> opArgsRun = {x_ptr, compress_kv_ptr, indices_ptr, y_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<npu::tile_fwk::bfloat16> golden(capacity0);
    std::vector<npu::tile_fwk::bfloat16> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(npu::tile_fwk::bfloat16));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp<npu::tile_fwk::bfloat16>(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(ScatterupdateOnBoardTest, test_scatter_update_32_1_512_576_graphD_bf16) {
    int B = 32;
    int S = 1;
    int S2 = 512;
    int kvLoraRank = 512;
    int qkRopeHeadDim = 64;

    std::vector<int64_t> shape0 =  {B, 1, S2, kvLoraRank + qkRopeHeadDim}; // [2,1,512,576]
    std::vector<int64_t> shape1 = {B, S};
    std::vector<int64_t> shape2 = {B, 1, S, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> shape_compress_kv = {B, S, kvLoraRank};
    std::vector<int64_t> shape_k_pe_rope = {B, 1, S, qkRopeHeadDim};

    int capacity0 = shape0[0] * shape0[1] * shape0[2] * shape0[3];
    int capacity1 = shape1[0] * shape1[1];
    int capacity_compress_kv = shape_compress_kv[0] * shape_compress_kv[1] * shape_compress_kv[2];
    int capacity_k_pe_rope = shape_k_pe_rope[0] * shape_k_pe_rope[1] * shape_k_pe_rope[2] * shape_k_pe_rope[3];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *compress_kv_ptr = readToDev(GetGoldenDir() + "/compressed_kv.bin", capacity_compress_kv);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_k_pe_rope);

    TileShape::Current().SetVecTile(1, 1, 128, 128);

    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_BF16, {B, 1, S2, kvLoraRank + qkRopeHeadDim}, (uint8_t *)x_ptr, "past_key_states");
    Tensor compressed_kv(DataType::DT_BF16, {B, S, kvLoraRank}, (uint8_t *)compress_kv_ptr, "compressed_kv");
    Tensor k_pe_rope(DataType::DT_BF16, {B, 1, S, qkRopeHeadDim}, (uint8_t *)y_ptr, "k_pe_rope"); // (b,1,s,qkRopeHeadDim)

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE_GRAPHD", {past_key_states, compressed_kv, kv_len, k_pe_rope});
    {
        Tensor k_nope = RmsNorm(compressed_kv); // (B, S, kvLoraRank)
        Tensor k_nope_new = Reshape(k_nope, {B, 1, S, kvLoraRank}); // (B,1,S,kvLoraRank)

        Tensor key_states = Concat({k_nope_new, k_pe_rope}, -1); // (B,1,S, kvLoraRank + qkRopeHeadDim)
        TileShape::Current().SetVecTile(1, 1, 1, 64);
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);

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
    std::vector<void*> opArgsRun = {x_ptr, compress_kv_ptr, indices_ptr, y_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<npu::tile_fwk::bfloat16> golden(capacity0);
    std::vector<npu::tile_fwk::bfloat16> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(npu::tile_fwk::bfloat16));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp<npu::tile_fwk::bfloat16>(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(ScatterupdateOnBoardTest, test_scatter_update_32_1_512_576_graphD) {
    int B = 32;
    int S = 1;
    int S2 = 512;
    int kvLoraRank = 512;
    int qkRopeHeadDim = 64;

    std::vector<int64_t> shape0 =  {B, 1, S2, kvLoraRank + qkRopeHeadDim}; // [32,1,512,576]
    std::vector<int64_t> shape1 = {B, S};
    std::vector<int64_t> shape2 = {B, 1, S, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> shape_compress_kv = {B, S, kvLoraRank};
    std::vector<int64_t> shape_k_pe_rope = {B, 1, S, qkRopeHeadDim};

    int capacity0 = shape0[0] * shape0[1] * shape0[2] * shape0[3];
    int capacity1 = shape1[0] * shape1[1];
    int capacity_compress_kv = shape_compress_kv[0] * shape_compress_kv[1] * shape_compress_kv[2];
    int capacity_k_pe_rope = shape_k_pe_rope[0] * shape_k_pe_rope[1] * shape_k_pe_rope[2] * shape_k_pe_rope[3];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *compress_kv_ptr = readToDev(GetGoldenDir() + "/compressed_kv.bin", capacity_compress_kv);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_k_pe_rope);

    TileShape::Current().SetVecTile(1, 1, 128, 128);

    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_FP32, {B, 1, S2, kvLoraRank + qkRopeHeadDim}, (uint8_t *)x_ptr, "past_key_states");
    Tensor compressed_kv(DataType::DT_FP32, {B, S, kvLoraRank}, (uint8_t *)compress_kv_ptr, "compressed_kv");
    Tensor k_pe_rope(DataType::DT_FP32, {B, 1, S, qkRopeHeadDim}, (uint8_t *)y_ptr, "k_pe_rope"); // (b,1,s,qkRopeHeadDim)

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE_GRAPHD", {past_key_states, compressed_kv, kv_len, k_pe_rope});
    {
        Tensor k_nope = RmsNorm(compressed_kv); // (B, S, kvLoraRank)
        Tensor k_nope_new = Reshape(k_nope, {B, 1, S, kvLoraRank}); // (B,1,S,kvLoraRank)
        Tensor key_states = Concat({k_nope_new, k_pe_rope}, -1); // (B,1,S, kvLoraRank + qkRopeHeadDim)
        TileShape::Current().SetVecTile(1, 1, 512, 64);
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);

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
    std::vector<void*> opArgsRun = {x_ptr, compress_kv_ptr, indices_ptr, y_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<float> golden(capacity0);
    std::vector<float> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(ScatterupdateOnBoardTest, test_scatter_update_64_7168_64_moe) {
    int S = 64;
    int S2 = 64;
    int kvLoraRank = 7160;
    int qkRopeHeadDim = 8;

    std::vector<int64_t> shape0 =  {S2, kvLoraRank + qkRopeHeadDim}; // [16, 16]
    std::vector<int64_t> shape1 = {1, S};
    std::vector<int64_t> shape2 = {S, kvLoraRank + qkRopeHeadDim}; // [1, 16]

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(16, 16);

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity2);
    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor key_states(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "key_states"); // [16,16]

    // Tensor past_key_states_new(DataType::DT_FP32, shape0, "past_key_states_new");

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE", {past_key_states, kv_len, key_states});
    {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);
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
    // uint8_t* outTensorAddr = nullptr;
    // machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));
    std::vector<void*> opArgsRun = {x_ptr, indices_ptr, y_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<float> golden(capacity0);
    std::vector<float> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(ScatterupdateOnBoardTest, test_scatter_update_1_48_4096_512_BSNZ) {
    int B = 32;
    int S = 1;
    int S2 = 48;
    int kvLoraRank = 256;
    int qkRopeHeadDim = 256;

    std::vector<int64_t> shape0 =  {S2, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> shape1 = {B, S};
    std::vector<int64_t> shape2 = {B * S, kvLoraRank + qkRopeHeadDim};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(16, 512);

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity2);
    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor key_states(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "key_states"); // [16,16]

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE", {past_key_states, key_states, kv_len});
    {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2, "PA_NZ", 16);
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
    // uint8_t* outTensorAddr = nullptr;
    // machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));
    std::vector<void*> opArgsRun = {x_ptr, y_ptr, indices_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<float> golden(capacity0);
    std::vector<float> npu_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)npu_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, npu_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ScatterupdateOnBoardTest, test_scatter_update_1_48_4096_512_BSNZ_bf16) {
    int B = 32;
    int S = 1;
    int S2 = 1920*128;
    int kvLoraRank = 256;
    int qkRopeHeadDim = 256;

    std::vector<int64_t> shape0 =  {S2, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> shape1 = {B, S};
    std::vector<int64_t> shape2 = {B * S, kvLoraRank + qkRopeHeadDim};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(16, 512);

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity2);
    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(DataType::DT_BF16, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor key_states(DataType::DT_BF16, shape2, (uint8_t *)y_ptr, "key_states");

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE", {past_key_states, key_states, kv_len});
    {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2, "PA_NZ", 128);
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
    // uint8_t* outTensorAddr = nullptr;
    // machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));
    std::vector<void*> opArgsRun = {x_ptr, y_ptr, indices_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<npu::tile_fwk::bfloat16> golden(capacity0);
    std::vector<npu::tile_fwk::bfloat16> npu_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)npu_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(npu::tile_fwk::bfloat16));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp<npu::tile_fwk::bfloat16>(golden, npu_res, 0.001f);
    EXPECT_EQ(ret, true);
}

// 4维bsnd
TEST_F(ScatterupdateOnBoardTest, test_scatter_update_1_1_1_64_BSND_4dims) {
    // 4, 1, 1, 64, 10, 128
    int64_t b = 20;
    int64_t s = 2;
    int64_t n = 1;
    int64_t d = 32;
    int64_t blockNum = 20;
    int64_t blockSize = 20;

    DataType dateType = DataType::DT_FP32;

    std::vector<int64_t> shape0 = {blockNum, blockSize, n, d};
    std::vector<int64_t> shape1 = {b, s};
    std::vector<int64_t> shape2 = {b, s, n, d};

    int capacity0 = shape0[0] * shape0[1] * shape0[2] * shape0[3];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1] * shape2[2] * shape2[3];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(b, s, n, d);

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity2);
    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(dateType, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor key_states(dateType, shape2, (uint8_t *)y_ptr, "key_states");

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE", {past_key_states, key_states, kv_len});
    {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2, "PA_BSND", blockSize);
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
    // uint8_t* outTensorAddr = nullptr;
    // machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));
    std::vector<void*> opArgsRun = {x_ptr, y_ptr, indices_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<float> golden(capacity0);
    std::vector<float> npu_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)npu_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp<float>(golden, npu_res, 0.000f);
    EXPECT_EQ(ret, true);
}

// 2维bsnd
TEST_F(ScatterupdateOnBoardTest, test_scatter_update_1_1_1_64_BSND_2dims) {
    int64_t b = 20;
    int64_t s = 2;
    int64_t n = 1;
    int64_t d = 32;
    int64_t blockNum = 20;
    int64_t blockSize = 20;

    DataType dateType = DataType::DT_FP32;

    std::vector<int64_t> shape0 = {blockNum * blockSize * n, d};
    std::vector<int64_t> shape1 = {b, s};
    std::vector<int64_t> shape2 = {b * s * n, d};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");
    TileShape::Current().SetVecTile(b, d);

    void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
    void *indices_ptr = readToDev<int64_t>(GetGoldenDir() + "/indices.bin", capacity1);
    void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity2);
    Tensor kv_len(DataType::DT_INT64, shape1, (uint8_t *)indices_ptr, "kv_len");
    Tensor past_key_states(dateType, shape0, (uint8_t *)x_ptr, "past_key_states");
    Tensor key_states(dateType, shape2, (uint8_t *)y_ptr, "key_states");

    /* torch capture */
    TileFwkBeginFunction("SCATTERUPDATE", {past_key_states, key_states, kv_len});
    {
        past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2, "PA_BSND", 1);
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
    // uint8_t* outTensorAddr = nullptr;
    // machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity * sizeof(float));
    std::vector<void*> opArgsRun = {x_ptr, y_ptr, indices_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity0 size:" << capacity0 << std::endl;
    std::vector<float> golden(capacity0);
    std::vector<float> npu_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)npu_res.data(), (uint8_t *)x_ptr, capacity0 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp<float>(golden, npu_res, 0.000f);
    EXPECT_EQ(ret, true);
}