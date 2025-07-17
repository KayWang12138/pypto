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
 * \file test_quant.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "models/llama/llama_def.h"
#include "models/deepseek/deepseek_spec.h"

namespace {
int capacity;
int INT8_MAX_VALUE = 127;
constexpr float F_127 = 127.0;

void QuantPre(uint8_t **out_ptr, uint64_t *outsize) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    *outsize = capacity * sizeof(float);
    *out_ptr = allocDevAddr(*outsize);
}

void QuantPost(uint8_t *outputGmAddr, uint64_t outputSize) {
    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);
    runtime::GetRA()->CopyFromTensor((uint8_t *) res.data(), (uint8_t *) outputGmAddr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f, 64);
    EXPECT_EQ(ret, true);
}
} // namespace

using namespace npu::tile_fwk;

class QuantTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(QuantTest, Test_quant) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    std::vector<int> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    int h = std::get<int>(deepseekConfig1["hiddenSize"]);
    std::cout << "Test_deepseekAttention  b,s,h: " << b << ", " << s << ", " << h << std::endl;

    Tensor input = Tensor(DataType::DT_FP16, {b * s, h}, "input");
    Tensor res;

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(128, s), std::min(128, s)}, {256, 256}, {64, 64});
    Program::GetInstance().GetTileShape().SetVecTileShapes(vecTileShape[0], vecTileShape[1]); // for Assemble

    FUNCTION("A") {
        res = std::get<0>(Quant(input));
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(QuantTest, Test_ScalarDivS) {
    std::vector<int> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int> shape{b * s, 1};
    capacity = b * s;
    uint8_t *out_ptr = nullptr;
    uint64_t outSize = 0;
    QuantPre(&out_ptr, &outSize);
    PROGRAM("Quant") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(vecTileShape[0], vecTileShape[1]);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *) input_ptr, "input");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "res");
        FUNCTION("ScalarDivS", FunctionType::STATIC, {input, output}) {
            output = ScalarDivS(input, Element(DataType::DT_FP32, static_cast<double>(INT8_MAX_VALUE)),
                                true);
        }
    }
    QuantPost(out_ptr, outSize);
}

TEST_F(QuantTest, Test_ScalarAddS) {
    std::vector<int> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int> shape{b * s, 1};
    capacity = b * s;
    uint8_t *out_ptr = nullptr;
    uint64_t outSize = 0;
    QuantPre(&out_ptr, &outSize);
    PROGRAM("Quant") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(vecTileShape[0], vecTileShape[1]);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *) input_ptr, "input");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "res");
        FUNCTION("ScalarAddS", FunctionType::STATIC, {input, output}) {
            output = ScalarAddS(input, Element(DataType::DT_FP32, static_cast<double>(INT8_MAX_VALUE)),
                                true);
        }
    }
    QuantPost(out_ptr, outSize);
}

TEST_F(QuantTest, Test_ScalarSubS) {
    std::vector<int> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int> shape{b * s, 1};
    capacity = b * s;
    uint8_t *out_ptr = nullptr;
    uint64_t outSize = 0;
    QuantPre(&out_ptr, &outSize);
    PROGRAM("Quant") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(vecTileShape[0], vecTileShape[1]);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *) input_ptr, "input");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "res");
        FUNCTION("ScalarSubS", FunctionType::STATIC, {input, output}) {
            output = ScalarSubS(input, Element(DataType::DT_FP32, static_cast<double>(INT8_MAX_VALUE)),
                                true);
        }
    }
    QuantPost(out_ptr, outSize);
}

TEST_F(QuantTest, Test_ScalarMulS) {
    std::vector<int> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int> shape{b * s, 1};
    capacity = b * s;
    uint8_t *out_ptr = nullptr;
    uint64_t outSize = 0;
    QuantPre(&out_ptr, &outSize);
    PROGRAM("Quant") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(vecTileShape[0], vecTileShape[1]);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *) input_ptr, "input");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "res");
        FUNCTION("ScalarMulS", FunctionType::STATIC, {input, output}) {
            output = ScalarMulS(input, Element(DataType::DT_FP32, static_cast<double>(INT8_MAX_VALUE)),
                                true);
        }
    }
    QuantPost(out_ptr, outSize);
}

TEST_F(QuantTest, Test_ScalarMaxS) {
    std::vector<int> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int> shape{b * s, 1};
    capacity = b * s;
    uint8_t *out_ptr = nullptr;
    uint64_t outSize = 0;
    QuantPre(&out_ptr, &outSize);
    PROGRAM("Quant") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(vecTileShape[0], vecTileShape[1]);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *) input_ptr, "input");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "res");
        FUNCTION("ScalarMaxS", FunctionType::STATIC, {input, output}) {
            output = ScalarMaxS(input, Element(DataType::DT_FP32, static_cast<double>(INT8_MAX_VALUE-1)),
                                true);
        }
    }
    QuantPost(out_ptr, outSize);
}

TEST_F(QuantTest, Test_ScalarOp) {
    std::vector<int> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int> shape{b * s, 35};
    capacity = b * s;
    uint8_t *out_ptr = nullptr;
    uint64_t outSize = 0;
    QuantPre(&out_ptr, &outSize);
    PROGRAM("Quant") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(vecTileShape[0], vecTileShape[1]);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *) input_ptr, "input");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "res");
        FUNCTION("ScalarAddS", FunctionType::STATIC, {input, output}) {
            auto output_a = ScalarAddS(input, Element(DataType::DT_FP32, F_127), true);
            auto output_b = ScalarSubS(output_a, Element(DataType::DT_FP32, F_127),
                true);
            auto output_c = ScalarMulS(output_b, Element(DataType::DT_FP32, F_127),
                true);
            auto output_d = ScalarDivS(output_c, Element(DataType::DT_FP32, F_127),
                true);
            output = ScalarMaxS(output_d, Element(DataType::DT_FP32, F_127), true);
        }
    }
    QuantPost(out_ptr, outSize);
}
