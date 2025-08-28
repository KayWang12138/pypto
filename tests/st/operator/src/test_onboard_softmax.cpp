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
 * \file test_onboard_softmax.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class SoftmaxOnBoard : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(SoftmaxOnBoard, test_softmax_cast_in) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    std::vector<int64_t> shape = {2, 2, 1, 128};
    DataType iType = DataType::DT_FP16;
    DataType oType = DataType::DT_FP32;
    int cap = shape[0] * shape[1] * shape[2] * shape[3];

    uint64_t outputSize = cap * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_Cast_In") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_softmax_cast_in.bin", cap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 2, 1, 16});
        Tensor i_x(iType, shape, (uint8_t *)x_ptr, "x");
        Tensor o_x(oType, shape, out_ptr, "cast_out");

        FUNCTION("Softmax_Cast_In", FunctionType::STATIC, {i_x, o_x}) {
            o_x = Cast(i_x, oType);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::float16> x(cap);
    std::vector<float> golden(cap);
    std::vector<float> res(cap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_softmax_cast_in.bin", x);
    readInput(GetGoldenDir() + "/softmax_cast_in.bin", golden);
    int ret = resultCmpCast<npu::tile_fwk::float16, float>(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_cast_out) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    std::vector<int64_t> shape = {2, 2, 1, 128};
    DataType iType = DataType::DT_FP32;
    DataType oType = DataType::DT_FP16;
    int cap = shape[0] * shape[1] * shape[2] * shape[3];

    uint64_t outputSize = cap * sizeof(uint16_t);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_Cast_Out") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_softmax_cast_out.bin", cap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 2, 1, 16});
        Tensor i_x(iType, shape, (uint8_t *)x_ptr, "x");
        Tensor o_x(oType, shape, out_ptr, "cast_out");

        FUNCTION("Softmax_Cast_Out", FunctionType::STATIC, {i_x, o_x}) {
            o_x = Cast(i_x, oType);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> x(cap);
    std::vector<npu::tile_fwk::float16> golden(cap);
    std::vector<npu::tile_fwk::float16> res(cap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_softmax_cast_out.bin", x);
    readInput(GetGoldenDir() + "/softmax_cast_out.bin", golden);
    int ret = resultCmpCast<float, npu::tile_fwk::float16>(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_sum_single) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    std::vector<int64_t> ishape = {2, 2, 1, 128};
    std::vector<int64_t> oshape = {ishape[0], ishape[1], ishape[2], 1};
    DataType dtype = DataType::DT_FP32;
    int icap = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    int oCap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    uint64_t outputSize = oCap * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_SumSingle") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_sum.bin", icap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 2, 1, 8});
        Tensor i_x(dtype, ishape, (uint8_t *)x_ptr, "x");
        Tensor o_x(dtype, oshape, out_ptr, "softmax_sum");

        FUNCTION("SOFTMAX_SUM_T", FunctionType::STATIC, {i_x, o_x}) {
            o_x = RowSumSingle(i_x);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> x(icap);
    std::vector<float> golden(oCap);
    std::vector<float> res(oCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_sum.bin", x);
    readInput(GetGoldenDir() + "/softmax_sum.bin", golden);
    int ret = resultCmpUnary<float>(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_max_single) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    std::vector<int64_t> ishape = {2, 2, 1, 128};
    std::vector<int64_t> oshape = {ishape[0], ishape[1], ishape[2], 1};
    DataType dtype = DataType::DT_FP32;
    int icap = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    int oCap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    uint64_t outputSize = oCap * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_MaxSingle") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_max.bin", icap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 2, 1, 8});
        Tensor i_x(dtype, ishape, (uint8_t *)x_ptr, "x");
        Tensor o_x(dtype, oshape, out_ptr, "softmax_max");

        FUNCTION("SOFTMAX_MAX_T", FunctionType::STATIC, {i_x, o_x}) {
            o_x = RowMaxSingle(i_x);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> x(icap);
    std::vector<float> golden(oCap);
    std::vector<float> res(oCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_max.bin", x);
    readInput(GetGoldenDir() + "/softmax_max.bin", golden);
    int ret = resultCmpUnary<float>(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_exp) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    std::vector<int64_t> shape = {2, 2, 1, 128};
    DataType dtype = DataType::DT_FP32;
    int cap = shape[0] * shape[1] * shape[2] * shape[3];
    uint64_t outputSize = cap * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_Exp") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_exp.bin", cap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 2, 1, 8});
        Tensor input_x(dtype, shape, (uint8_t *)x_ptr, "x");
        Tensor output(dtype, shape, out_ptr, "Softmax_Exp");

        FUNCTION("SOFTMAX_EXP_T", FunctionType::STATIC, {input_x, output}) {
            output = Exp(input_x);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> x(cap);
    std::vector<float> golden(cap);
    std::vector<float> res(cap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_exp.bin", x);
    readInput(GetGoldenDir() + "/softmax_exp.bin", golden);
    int ret = resultCmpUnary(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_div) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    std::vector<int64_t> lshape = {2, 2, 1, 128};
    std::vector<int64_t> rshape = {2, 2, 1, 1};
    std::vector<int64_t> oshape = {2, 2, 1, 128};
    DataType dtype = DataType::DT_FP32;
    int lcap = lshape[0] * lshape[1] * lshape[2] * lshape[3];
    int rcap = rshape[0] * rshape[1] * rshape[2] * rshape[3];
    int ocap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    uint64_t outputSize = ocap * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_Div") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_div.bin", lcap);
        void *y_ptr = readToDev(GetGoldenDir() + "/y_div.bin", rcap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 2, 1, 8});
        Tensor input_x(dtype, lshape, (uint8_t *)x_ptr, "x");
        Tensor input_y(dtype, rshape, (uint8_t *)y_ptr, "y");
        Tensor output(dtype, oshape, out_ptr, "Softmax_Div");

        FUNCTION("SOFTMAX_SUB_T", FunctionType::STATIC, {input_x, input_y, output}) {
            output = Div(input_x, input_y);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(ocap);
    std::vector<float> res(ocap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/softmax_div.bin", golden);
    int ret = resultCmp<float>(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_sum_all) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    std::vector<int64_t> ishape = {2, 2, 1, 128};
    std::vector<int64_t> oshape = {ishape[0], ishape[1], ishape[2], ishape[3]};
    DataType dtype = DataType::DT_FP16;
    int icap = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    int oCap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    uint64_t outputSize = oCap * sizeof(uint16_t);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_Sum_All") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_sum_all.bin", icap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 2, 1, 64});
        Tensor i_x(dtype, ishape, (uint8_t *)x_ptr, "x");
        Tensor o_x(dtype, oshape, out_ptr, "softmax_sum_all");

        FUNCTION("SOFTMAX_SUM_T", FunctionType::STATIC, {i_x, o_x}) {
            o_x = SoftmaxNew(i_x);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::float16> x(icap);
    std::vector<npu::tile_fwk::float16> golden(oCap);
    std::vector<npu::tile_fwk::float16> res(oCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_sum_all.bin", x);
    readInput(GetGoldenDir() + "/softmax_sum_all.bin", golden);
    int ret = resultCmpUnary<npu::tile_fwk::float16>(x, golden, res, 0.001f, 10);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_full_inference) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    std::vector<int64_t> ishape = {2, 2, 32, 256};
    std::vector<int64_t> oshape = {ishape[0], ishape[1], ishape[2], ishape[3]};
    DataType dtype = DataType::DT_FP16;
    int icap = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    int oCap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    uint64_t outputSize = oCap * sizeof(uint16_t);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_full_inference") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_full.bin", icap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, 32, 256});
        Tensor i_x(dtype, ishape, (uint8_t *)x_ptr, "x");
        Tensor o_x(dtype, oshape, out_ptr, "Softmax_full_inference");

        FUNCTION("SOFTMAX_FULL_INFERENCE_T", FunctionType::STATIC, {i_x, o_x}) {
            o_x = SoftmaxNew(i_x);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::float16> x(icap);
    std::vector<npu::tile_fwk::float16> golden(oCap);
    std::vector<npu::tile_fwk::float16> res(oCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_full.bin", x);
    readInput(GetGoldenDir() + "/softmax_full_inference.bin", golden);
    int ret = resultCmpUnary<npu::tile_fwk::float16>(x, golden, res, 0.001f, 10);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_deepseek) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    std::vector<int64_t> ishape = {4, 8, 1, 512};
    std::vector<int64_t> oshape = {ishape[0], ishape[1], ishape[2], ishape[3]};
    DataType dtype = DataType::DT_FP16;
    int icap = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    int oCap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    uint64_t outputSize = oCap * sizeof(uint16_t);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("softmax_deepseek") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_deepseek.bin", icap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 2, 1, 256});
        Tensor i_x(dtype, ishape, (uint8_t *)x_ptr, "x");
        Tensor o_x(dtype, oshape, out_ptr, "softmax_deepseek");

        FUNCTION("SOFTMAX_DEEPSEEK", FunctionType::STATIC, {i_x, o_x}) {
            o_x = SoftmaxNew(i_x);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::float16> x(icap);
    std::vector<npu::tile_fwk::float16> golden(oCap);
    std::vector<npu::tile_fwk::float16> res(oCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_deepseek.bin", x);
    readInput(GetGoldenDir() + "/softmax_deepseek.bin", golden);
    int ret = resultCmpUnary<npu::tile_fwk::float16>(x, golden, res, 0.001f, 10);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_flash_attention) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    std::vector<int64_t> ishape = {32, 32, 1, 256};
    std::vector<int64_t> oshape = {ishape[0], ishape[1], ishape[2], ishape[3]};
    DataType dtype = DataType::DT_FP32;
    int icap = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    int oCap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    uint64_t outputSize = oCap * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("softmax_fa") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", icap);
        Program::GetInstance().GetTileShape().SetVecTileShapes({4, 4, 1, 64});
        Tensor i_x(dtype, ishape, (uint8_t *)x_ptr, "x");
        Tensor o_x(dtype, oshape, out_ptr, "softmax_deepseek");

        FUNCTION("SOFTMAX_FA", FunctionType::STATIC, {i_x, o_x}) {
            o_x = SoftmaxNew(i_x);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> x(icap);
    std::vector<float> golden(oCap);
    std::vector<float> res(oCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x.bin", x);
    readInput(GetGoldenDir() + "/softmax.bin", golden);
    int ret = resultCmpUnary<float>(x, golden, res, 0.001f, 10);
    EXPECT_EQ(ret, true);
}
