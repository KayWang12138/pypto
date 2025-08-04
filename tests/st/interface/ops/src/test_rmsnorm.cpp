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
 * \file test_rmsnorm.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_static.h"

using namespace npu::tile_fwk;

class RmsNormTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(RmsNormTest, test_32_32_tileop_rmsnorm) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());

    int shape0 = 32;
    int shape1 = 32;
    std::vector<int> shape = {shape0, shape1};
    std::vector<int> outshape = {shape0, shape1};
    int inputCapacity = shape0 * shape1;
    int outputCapacity = shape0 * shape1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("RMSNORM") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({16, 16});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        FUNCTION("RMSNORM_T", FunctionType::STATIC, {input_a, output}) {
            output = RmsNorm(input_a);
        }
    }
    RunStatic();

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.003f);
    EXPECT_EQ(ret, true);
}

TEST_F(RmsNormTest, test_2_32_32_tileop_rmsnorm) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());

    int shape0 = 2;
    int shape1 = 32;
    int shape2 = 32;
    std::vector<int> shape = {shape0, shape1, shape2};
    std::vector<int> outshape = {shape0, shape1, shape2};
    int inputCapacity = shape0 * shape1 * shape2;
    int outputCapacity = shape0 * shape1 * shape2;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("RMSNORM") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 16, 16});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        FUNCTION("RMSNORM_T", FunctionType::STATIC, {input_a, output}) {
            output = RmsNorm(input_a);
        }
    }
    RunStatic();

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RmsNormTest, test_2_2_32_32_tileop_rmsnorm) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());

    int shape0 = 2;
    int shape1 = 2;
    int shape2 = 32;
    int shape3 = 32;
    std::vector<int> shape = {shape0, shape1, shape2, shape3};
    std::vector<int> outshape = {shape0, shape1, shape2, shape3};
    int inputCapacity = shape0 * shape1 * shape2 * shape3;
    int outputCapacity = shape0 * shape1 * shape2 * shape3;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("RMSNORM") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, 16, 16});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        FUNCTION("RMSNORM_T", FunctionType::STATIC, {input_a, output}) {
            output = RmsNorm(input_a);
        }
    }
    RunStatic();

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RmsNormTest, test_32_256_tileop_rmsnorm_fp16) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());

    int shape0 = 32;
    int shape1 = 256;
    std::vector<int> shape = {shape0, shape1};
    std::vector<int> outshape = {shape0, shape1};
    int inputCapacity = shape0 * shape1;
    int outputCapacity = shape0 * shape1;
    uint64_t outputSize = outputCapacity * sizeof(npu::tile_fwk::float16);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("RMSNORM") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({32, 32});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP16, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP16, outshape, out_ptr, "C");

        FUNCTION("RMSNORM_T", FunctionType::STATIC, {input_a, output}) {
            output = RmsNorm(input_a);
        }
    }
    RunStatic();

    std::vector<npu::tile_fwk::float16 > golden(outputCapacity);
    std::vector<npu::tile_fwk::float16 > res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RmsNormTest, test_32_1536_tileop_rmsnorm_fp16_realCase) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());

    int shape0 = 32;
    int shape1 = 1536;
    std::vector<int> shape = {shape0, shape1};
    std::vector<int> outshape = {shape0, shape1};
    int inputCapacity = shape0 * shape1;
    int outputCapacity = shape0 * shape1;
    uint64_t outputSize = outputCapacity * sizeof(npu::tile_fwk::float16);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("RMSNORM") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({32, 128});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP16, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP16, outshape, out_ptr, "C");

        FUNCTION("RMSNORM_T", FunctionType::STATIC, {input_a, output}) {
            output = RmsNorm(input_a);
        }
    }
    RunStatic();

    std::vector<npu::tile_fwk::float16 > golden(outputCapacity);
    std::vector<npu::tile_fwk::float16 > res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RmsNormTest, test_2_32_256_tileop_rmsnorm_fp16) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());

    int shape0 = 2;
    int shape1 = 32;
    int shape2 = 256;
    std::vector<int> shape = {shape0, shape1, shape2};
    std::vector<int> outshape = {shape0, shape1, shape2};
    int inputCapacity = shape0 * shape1 * shape2;
    int outputCapacity = shape0 * shape1 * shape2;
    uint64_t outputSize = outputCapacity * sizeof(npu::tile_fwk::float16);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("RMSNORM") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 8, 32});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP16, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP16, outshape, out_ptr, "C");

        FUNCTION("RMSNORM_T", FunctionType::STATIC, {input_a, output}) {
            output = RmsNorm(input_a);
        }
    }
    RunStatic();

    std::vector<npu::tile_fwk::float16 > golden(outputCapacity);
    std::vector<npu::tile_fwk::float16 > res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RmsNormTest, test_32_1536_tileop_rmsnorm_gamma_fp16_realCase) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());

    int shape0 = 32;
    int shape1 = 1536;
    std::vector<int> shape = {shape0, shape1};
    std::vector<int> gammaShape = {shape1};
    std::vector<int> outshape = {shape0, shape1};
    int inputCapacity = shape0 * shape1;
    int inputGammaCapacity = shape1;
    int outputCapacity = shape0 * shape1;
    uint64_t outputSize = outputCapacity * sizeof(npu::tile_fwk::float16);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("RMSNORM") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({32, 128});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        void *gamma_ptr = readToDev(GetGoldenDir() + "/gamma.bin", inputGammaCapacity);
        Tensor input_a(DataType::DT_FP16, shape, (uint8_t *)x_ptr, "A");
        Tensor input_gamma(DataType::DT_FP16, gammaShape, (uint8_t *)gamma_ptr, "B");
        Tensor output(DataType::DT_FP16, outshape, out_ptr, "C");

        FUNCTION("RMSNORM_T", FunctionType::STATIC, {input_a, input_gamma, output}) {
            output = RmsNorm(input_a, input_gamma, 1e-5f);
        }
    }
    RunStatic();

    std::vector<npu::tile_fwk::float16 > golden(outputCapacity);
    std::vector<npu::tile_fwk::float16 > res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RmsNormTest, test_2_1_512_tileop_rmsnorm_gamma_fp16_realCase) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());

    int shape0 = 32;
    int shape1 = 1;
    int shape2 = 512;
    std::vector<int> shape = {shape0, shape1, shape2};
    std::vector<int> gammaShape = {shape2};
    std::vector<int> outshape = {shape0, shape1, shape2};
    int inputCapacity = shape0 * shape1 * shape2;
    int inputGammaCapacity = shape2;
    int outputCapacity = shape0 * shape1 * shape2;
    uint64_t outputSize = outputCapacity * sizeof(npu::tile_fwk::float16);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("RMSNORM") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({2, 1, 512});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        void *gamma_ptr = readToDev(GetGoldenDir() + "/gamma.bin", inputGammaCapacity);
        Tensor input_a(DataType::DT_FP16, shape, (uint8_t *)x_ptr, "A");
        Tensor input_gamma(DataType::DT_FP16, gammaShape, (uint8_t *)gamma_ptr, "B");
        Tensor output(DataType::DT_FP16, outshape, out_ptr, "C");

        FUNCTION("RMSNORM_T", FunctionType::STATIC, {input_a, input_gamma, output}) {
            output = RmsNorm(input_a, input_gamma, 1e-5f);
        }
    }
    RunStatic();

    std::vector<npu::tile_fwk::float16 > golden(outputCapacity);
    std::vector<npu::tile_fwk::float16 > res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
