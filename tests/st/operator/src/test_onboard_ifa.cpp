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
 * \file test_onboard_ifa.cpp
 * \brief
 */

#include <functional>
#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class OnBoardIFATest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

// Sub (32, 128), (32, 1)
TEST_F(OnBoardIFATest, test_32_128_sub_32_1) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    int outCap = 32 * 128;
    uint64_t outputSize = outCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SUB") {
        std::vector<int> shape1 = {32, 128};
        std::vector<int> shape2 = {32, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outCap);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", 32 * 1);
        Program::GetInstance().GetTileShape().SetVecTileShapes({16, 64});
        Tensor input_a(DataType::DT_FP32, shape1, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape1, out_ptr, "C");
        ConfigManager::Instance();

        FUNCTION("SUB_T", FunctionType::STATIC, {input_a, input_b, output}) {
            output = Sub(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

// Sub (32, 1), (32, 1)
TEST_F(OnBoardIFATest, test_32_1_sub_32_1) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    int outCap = 32 * 1;
    uint64_t outputSize = outCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SUB") {
        std::vector<int> shape1 = {32, 1};
        std::vector<int> shape2 = {32, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outCap);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", outCap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({16, 1});
        Tensor input_a(DataType::DT_FP32, shape1, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape1, out_ptr, "C");
        ConfigManager::Instance();

        FUNCTION("SUB_T", FunctionType::STATIC, {input_a, input_b, output}) {
            output = Sub(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

// add (32, 512), (32, 1)
TEST_F(OnBoardIFATest, test_32_512_add_32_1) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    int outCap = 32 * 512;
    uint64_t outputSize = outCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int> shape1 = {32, 512};
        std::vector<int> shape2 = {32, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outCap);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", 32 * 1);
        Program::GetInstance().GetTileShape().SetVecTileShapes({16, 64});
        Tensor input_a(DataType::DT_FP32, shape1, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape1, out_ptr, "C");
        ConfigManager::Instance();

        FUNCTION("ADD_T", FunctionType::STATIC, {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}


// mul (32, 1), (32, 1)
TEST_F(OnBoardIFATest, test_32_1_mul_32_1) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    int outCap = 32 * 1;
    uint64_t outputSize = outCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int> shape1 = {32, 1};
        std::vector<int> shape2 = {32, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outCap);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", outCap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({16, 1});
        Tensor input_a(DataType::DT_FP32, shape1, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape1, out_ptr, "C");
        ConfigManager::Instance();

        FUNCTION("MUL_T", FunctionType::STATIC, {input_a, input_b, output}) {
            output = Mul(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

// mul (32, 512), (32, 1)
TEST_F(OnBoardIFATest, test_32_512_mul_32_1) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    int outCap = 32 * 512;
    uint64_t outputSize = outCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int> shape1 = {32, 512};
        std::vector<int> shape2 = {32, 512};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outCap);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", outCap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({32, 256});
        Tensor input_a(DataType::DT_FP32, shape1, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape1, out_ptr, "C");
        ConfigManager::Instance();

        FUNCTION("MUL_T", FunctionType::STATIC, {input_a, input_b, output}) {
            // add RowSumSingle to test brc case
            auto input_c = RowSumSingle(input_b);
            output = Mul(input_a, input_c);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

// exp (32, 128)
TEST_F(OnBoardIFATest, test_32_128_tileop_exp) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    int outCap = 32 * 128;
    uint64_t outputSize = outCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("EXP") {
        std::vector<int> shape = {32, 128};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outCap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({8, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        FUNCTION("EXP_T", FunctionType::STATIC, {input_a, output}) {
            output = Exp(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

// exp (32, 1)
TEST_F(OnBoardIFATest, test_32_1_tileop_exp) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    int outCap = 32 * 1;
    uint64_t outputSize = outCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("EXP") {
        std::vector<int> shape = {32, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outCap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({8, 1});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        FUNCTION("EXP_T", FunctionType::STATIC, {input_a, output}) {
            output = Exp(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

// MAX (32, 1)
TEST_F(OnBoardIFATest, test_32_1_maximum) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    int outCap = 32 * 1;
    uint64_t outputSize = outCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Max") {
        std::vector<int> shape1 = {32, 1};
        std::vector<int> shape2 = {32, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outCap);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", outCap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({8, 1});
        Tensor input_a(DataType::DT_FP32, shape1, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape1, out_ptr, "C");
        ConfigManager::Instance();

        FUNCTION("Max_T", FunctionType::STATIC, {input_a, input_b, output}) {
            output = Maximum(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

// RECIP (32, 1)
TEST_F(OnBoardIFATest, test_32_1_reciprocal) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    int outCap = 32 * 1;
    uint64_t outputSize = outCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Max") {
        std::vector<int> shape1 = {32, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outCap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({8, 1});
        Tensor input_a(DataType::DT_FP32, shape1, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, shape1, out_ptr, "C");
        ConfigManager::Instance();

        FUNCTION("Max_T", FunctionType::STATIC, {input_a, output}) {
            output = Reciprocal(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.003f);
    EXPECT_EQ(ret, true);
}

// rowmaxsingle (32, 128)
TEST_F(OnBoardIFATest, test_operation_32_128_row_max_single) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());

    int shape0 = 32;
    int shape1 = 128;
    std::vector<int> shape = {shape0, shape1};
    std::vector<int> outshape = {shape0, 1};
    int inputCapacity = shape0 * shape1;
    int outputCapacity = shape0 * 1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowMaxSingle") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({8, 32});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        FUNCTION("RowMaxSingle", FunctionType::STATIC, {input_a, output}) {
            output = RowMaxSingle(input_a, -1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardIFATest, test_operation_32_128_row_sum_single) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());

    int shape0 = 32;
    int shape1 = 128;
    std::vector<int> shape = {shape0, shape1};
    std::vector<int> outshape = {shape0, 1};
    int inputCapacity = shape0 * shape1;
    int outputCapacity = shape0 * 1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowSumSingle") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({8, 32});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        FUNCTION("RowSumSingle", FunctionType::STATIC, {input_a, output}) {
            output = RowSumSingle(input_a, -1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

// concat ((32, 512), (32, 64))
TEST_F(OnBoardIFATest, test_concat_32_512_32_64) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    int outCap = 32 * (512 + 64);
    int shape1Cap = 32 * 512;
    int shape2Cap = 32 * 64;

    uint64_t outputSize = outCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    std::vector<int> shape1 = {32, 512};
    std::vector<int> shape2 = {32, 64};
    std::vector<int> outShape = {32, 576};
    PROGRAM("CONCAT") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", shape1Cap);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", shape2Cap);

        Program::GetInstance().GetTileShape().SetVecTileShapes({64, 32});
        Tensor input_a(DataType::DT_FP32, shape1, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, outShape, (uint8_t *)out_ptr, "C");

        FUNCTION("CONCAT_T",  FunctionType::STATIC, {input_a, input_b, output}) {
            output = Concat(std::vector<Tensor>{input_a, input_b}, -1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardIFATest, test_concat_32_tensor) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    int outCap = 32 * 32 * 512;
    int shapeCap = 32 * 512;
    int tensorNum = 32;

    uint64_t outputSize = outCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    std::vector<int> shape = {32, 512};
    std::vector<int> outShape = {32 * 32, 512};
    PROGRAM("CONCAT") {
        void* x_ptr[32];
        std::vector<Tensor> inputTensors(32);
        for (int i = 0; i < tensorNum; ++i) {
            x_ptr[i] = readToDev(GetGoldenDir() + "/x" + std::to_string(i) + ".bin", shapeCap);
            // Tensor tmp(DataType::DT_FP32, shape, (uint8_t *)x_ptr[i], "A" + std::to_string(i));
            std::string varName = "A" + std::to_string(i);
            inputTensors[i] = Tensor(DataType::DT_FP32, shape, (uint8_t *)x_ptr[i], varName);
        }

        Tensor output(DataType::DT_FP32, outShape, (uint8_t *)out_ptr, "C");

        std::vector<std::reference_wrapper<Tensor>> iOTensors(inputTensors.begin(), inputTensors.end());
        iOTensors.push_back(output);

        Program::GetInstance().GetTileShape().SetVecTileShapes({32, 64});

        FUNCTION("CONCAT_T", FunctionType::STATIC, iOTensors) {
            output = Concat(inputTensors, 0);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
