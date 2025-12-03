/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_onboard_scatter_element.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "test_common.h"
#include "test_suite_stest_ops.h"
#include "machine/runtime/runtime.h"
#include "machine/runtime/tilefwk_runtime_api.h"

using namespace npu::tile_fwk;

class Scatter_OnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(Scatter_OnBoardTest, test_scatter_element_float_16_64_8_32_1) {
    int S0 = 16;
    int S1 = 64;
    int D0 = 8;
    int D1 = 32;
    std::vector<int64_t> shape0 = {S0, S1};
    std::vector<int64_t> shape1 = {D0, D1};
    int axis = 1;
    std::vector<int64_t> shape2 = {S0, S1};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(0);
    TileFwkInit();

    TileShape::Current().SetVecTile({8, S1});

    void *src_ptr = readToDev(GetGoldenDir() + "/src.bin", capacity0);
    void *indices_ptr = readToDev(GetGoldenDir() + "/indices.bin", capacity1);
    Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)src_ptr, "src");
    Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
    Tensor output(DataType::DT_FP32, shape2, "output");

    /* torch capture */
    TileFwkBeginFunction("SCATTERELEMENT", {input_src0, input_src1});
    {
        input_src0 = Scatter_(input_src0, input_src1, Element(DataType::DT_FP32, 1.0), axis);
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    std::vector<void*> opArgsRun = {src_ptr, indices_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStream(),  opArgsRun);

    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)src_ptr, capacity2 * sizeof(float));
    readInput(GetGoldenDir() + "/res_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(Scatter_OnBoardTest, test_scatter_element_float_16_70_16_40_1) {
    int S0 = 16;
    int S1 = 70;
    int D0 = 16;
    int D1 = 40;
    std::vector<int64_t> shape0 = {S0, S1};
    std::vector<int64_t> shape1 = {D0, D1};
    int axis = 1;
    std::vector<int64_t> shape2 = {S0, S1};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(0);
    TileFwkInit();

    TileShape::Current().SetVecTile({8, S1});

    void *src_ptr = readToDev(GetGoldenDir() + "/src.bin", capacity0);
    void *indices_ptr = readToDev(GetGoldenDir() + "/indices.bin", capacity1);
    Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)src_ptr, "src");
    Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
    Tensor output(DataType::DT_FP32, shape2, "output");

    /* torch capture */
    TileFwkBeginFunction("SCATTERELEMENT", {input_src0, input_src1});
    {
        input_src0 = Scatter_(input_src0, input_src1, Element(DataType::DT_FP32, 1.0), axis);
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    std::vector<void*> opArgsRun = {src_ptr, indices_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetScheStream(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetScheStream());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)src_ptr, capacity2 * sizeof(float));
    readInput(GetGoldenDir() + "/res_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(Scatter_OnBoardTest, test_scatter_element_float_16_64_16_32_1) {
    int S0 = 16;
    int S1 = 64;
    int D0 = 16;
    int D1 = 32;
    std::vector<int64_t> shape0 = {S0, S1};
    std::vector<int64_t> shape1 = {D0, D1};
    int axis = 1;
    std::vector<int64_t> shape2 = {S0, S1};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(0);
    TileFwkInit();

    TileShape::Current().SetVecTile({8, S1});

    void *src_ptr = readToDev(GetGoldenDir() + "/src.bin", capacity0);
    void *indices_ptr = readToDev(GetGoldenDir() + "/indices.bin", capacity1);
    Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)src_ptr, "src");
    Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
    Tensor output(DataType::DT_FP32, shape2, "output");

    /* torch capture */
    TileFwkBeginFunction("SCATTERELEMENT", {input_src0, input_src1});
    {
        input_src0 = Scatter_(input_src0, input_src1, Element(DataType::DT_FP32, 1.0), axis);
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    std::vector<void*> opArgsRun = {src_ptr, indices_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetScheStream(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetScheStream());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)src_ptr, capacity2 * sizeof(float));
    readInput(GetGoldenDir() + "/res_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(Scatter_OnBoardTest, test_scatter_element_float_8_256_8_8_1_moe) {
    int S0 = 8;
    int S1 = 256;
    int D0 = 8;
    int D1 = 8;
    std::vector<int64_t> shape0 = {S0, S1};
    std::vector<int64_t> shape1 = {D0, D1};
    int axis = 1;
    std::vector<int64_t> shape2 = {S0, S1};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(0);
    TileFwkInit();

    TileShape::Current().SetVecTile({8, S1});

    void *src_ptr = readToDev(GetGoldenDir() + "/src.bin", capacity0);
    void *indices_ptr = readToDev(GetGoldenDir() + "/indices.bin", capacity1);
    Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)src_ptr, "src");
    Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
    Tensor output(DataType::DT_FP32, shape2, "output");

    /* torch capture */
    TileFwkBeginFunction("SCATTERELEMENT", {input_src0, input_src1});
    {
        input_src0 = Scatter_(input_src0, input_src1, Element(DataType::DT_FP32, 1.0), axis);
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    std::vector<void*> opArgsRun = {src_ptr, indices_ptr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetScheStream(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetScheStream());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)src_ptr, capacity2 * sizeof(float));
    readInput(GetGoldenDir() + "/res_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}
