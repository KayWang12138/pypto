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
 * \file test_onboard_rope_subgraph.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class RoPESubGraphOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

int test_init (int B, int S , int N, int qkRopeHeadDim, std::string shape_dir_path)
{
    std::vector<int64_t> qPeShape{B, S, N, qkRopeHeadDim};
    std::vector<int64_t> kPeShape{B, S, qkRopeHeadDim};
    std::vector<int64_t> idsShape{B, S};
    std::vector<int64_t> cosSinShape{S, qkRopeHeadDim};
    std::vector<int64_t> qEmbedShape{B, N, S, qkRopeHeadDim};
    std::vector<int64_t> kEmbedShape{B, 1, S, qkRopeHeadDim};

    int qSize = std::accumulate(qPeShape.begin(), qPeShape.end(), 1, std::multiplies<>());
    int kSize = std::accumulate(kPeShape.begin(), kPeShape.end(), 1, std::multiplies<>());
    int cosSinSize = std::accumulate(cosSinShape.begin(), cosSinShape.end(), 1, std::multiplies<>());
    int posIdsSize = std::accumulate(idsShape.begin(), idsShape.end(), 1, std::multiplies<>());
    int qEmbedSize = std::accumulate(qEmbedShape.begin(), qEmbedShape.end(), 1, std::multiplies<>());
    int kEmbedSize = std::accumulate(kEmbedShape.begin(), kEmbedShape.end(), 1, std::multiplies<>());

    uint8_t* qEmbed_ptr = allocDevAddr(qEmbedSize * sizeof(npu::tile_fwk::bfloat16)); // Byte
    uint8_t* kEmbed_ptr = allocDevAddr(kEmbedSize * sizeof(npu::tile_fwk::bfloat16));

    PROGRAM("RoPEMla") {

        void *q_ptr = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + shape_dir_path + "qPe.bin", qSize);
        void *k_ptr = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + shape_dir_path + "kPe.bin", kSize);
        void *cos_ptr = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + shape_dir_path + "cos.bin", cosSinSize);
        void *sin_ptr = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + shape_dir_path + "sin.bin", cosSinSize);
        void *pos_ids_ptr = readToDev<int32_t>(GetGoldenDir() + shape_dir_path + "pos_ids.bin", posIdsSize);
        Tensor qPe(DataType::DT_BF16, qPeShape, (uint8_t *)q_ptr, "qPe");
        Tensor kPe(DataType::DT_BF16, kPeShape, (uint8_t *)k_ptr, "kPe");
        Tensor cos(DataType::DT_BF16, cosSinShape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(DataType::DT_BF16, cosSinShape, (uint8_t *)sin_ptr, "sin");
        Tensor positionIds(DataType::DT_INT32, idsShape, (uint8_t *)pos_ids_ptr, "position_indices");
        Tensor qEmbed(DataType::DT_BF16, qEmbedShape, qEmbed_ptr, "qEmbed");
        Tensor kEmbed(DataType::DT_BF16, kEmbedShape, kEmbed_ptr, "kEmbed");

        RoPETileShapeConfig ropeTileConfig{
            {32, 64}, // for cos/sin->cast, [s,d]
            {1, 32, 64}, // for gather,unsqueeze, [b,s,d]
            {1, 32, 1, 64}, // [b,n,s,d]
            {1, 32, 1, 64, 64} // for transpose, [b,n,s,d/2,2]
        };

        config::SetBuildStatic(true);
        FUNCTION("RoPE", {qPe, kPe, cos, sin, positionIds, qEmbed, kEmbed}) {
            TileShape::Current().SetVecTile({1, 1, 32, 64});
            auto qPeTrans = Transpose(qPe, {1, 2}); // [b,s,n,d]->[b,n,s,d]

            int b = kPe.GetShape()[0];
            int s = kPe.GetShape()[1];
            int d = kPe.GetShape()[2];
            // 以下两步Reshape+Transpose可以优化成1个reshape：auto kPeReshape = Reshape(kPe, {b, 1, s, d}); //
            // [b,s,d]->[b,1,s,d]
            auto kPeReshape = Reshape(kPe, {b, 1, s, d}); // [b,s,d]->[b,1,s,d]
            ApplyRotaryPosEmb(qPeTrans, kPeReshape, cos, sin, positionIds, qEmbed, kEmbed, 1, ropeTileConfig);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    // qEmbed
    std::vector<npu::tile_fwk::bfloat16> qEmbedRes(qEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)qEmbedRes.data(), (uint8_t *)qEmbed_ptr, qEmbedSize * sizeof(npu::tile_fwk::bfloat16)); // Byte
    std::vector<npu::tile_fwk::bfloat16> qEmbedGolden(qEmbedSize);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + shape_dir_path + "qEmbed_res.bin", qEmbedGolden);
    int ret = resultCmp<npu::tile_fwk::bfloat16>(qEmbedGolden, qEmbedRes, 0.001f);

    // kEmbed
    std::vector<npu::tile_fwk::bfloat16> kEmbedRes(kEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)kEmbedRes.data(), (uint8_t *)kEmbed_ptr, kEmbedSize * sizeof(npu::tile_fwk::bfloat16));
    std::vector<npu::tile_fwk::bfloat16> kEmbedGolden(kEmbedSize);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + shape_dir_path + "kEmbed_res.bin", kEmbedGolden);
    ret &= resultCmp<npu::tile_fwk::bfloat16>(kEmbedGolden, kEmbedRes, 0.001f);
    return ret;
}

TEST_F(RoPESubGraphOnBoardTest, test_operation_rope_subgraph_deepseekv3) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 1;
    int S = 1;                 // IFA S=1 S=1024
    int N = 32;                // N=32
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::string shape_dir_path = "/1_1_32_64/";

    std::vector<int64_t> qPeShape{B, S, N, qkRopeHeadDim};
    std::vector<int64_t> kPeShape{B, S, qkRopeHeadDim};
    std::vector<int64_t> idsShape{B, S};
    std::vector<int64_t> cosSinShape{S, qkRopeHeadDim};
    std::vector<int64_t> qEmbedShape{B, N, S, qkRopeHeadDim};
    std::vector<int64_t> kEmbedShape{B, 1, S, qkRopeHeadDim};

    int qSize = std::accumulate(qPeShape.begin(), qPeShape.end(), 1, std::multiplies<>());
    int kSize = std::accumulate(kPeShape.begin(), kPeShape.end(), 1, std::multiplies<>());
    int cosSinSize = std::accumulate(cosSinShape.begin(), cosSinShape.end(), 1, std::multiplies<>());
    int posIdsSize = std::accumulate(idsShape.begin(), idsShape.end(), 1, std::multiplies<>());
    int qEmbedSize = std::accumulate(qEmbedShape.begin(), qEmbedShape.end(), 1, std::multiplies<>());
    int kEmbedSize = std::accumulate(kEmbedShape.begin(), kEmbedShape.end(), 1, std::multiplies<>());

    uint8_t* qEmbed_ptr = allocDevAddr(qEmbedSize * sizeof(float)); // Byte
    uint8_t* kEmbed_ptr = allocDevAddr(kEmbedSize * sizeof(float));

    PROGRAM("RoPEMla") {

        void *q_ptr = readToDev(GetGoldenDir() + shape_dir_path + "qPe.bin", qSize);
        void *k_ptr = readToDev(GetGoldenDir() + shape_dir_path + "kPe.bin", kSize);
        void *cos_ptr = readToDev(GetGoldenDir() + shape_dir_path + "cos.bin", cosSinSize);
        void *sin_ptr = readToDev(GetGoldenDir() + shape_dir_path + "sin.bin", cosSinSize);
        void *pos_ids_ptr = readToDev<int32_t>(GetGoldenDir() + shape_dir_path + "pos_ids.bin", posIdsSize);
        Tensor qPe(DataType::DT_FP32, qPeShape, (uint8_t *)q_ptr, "qPe");
        Tensor kPe(DataType::DT_FP32, kPeShape, (uint8_t *)k_ptr, "kPe");
        Tensor cos(DataType::DT_FP32, cosSinShape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(DataType::DT_FP32, cosSinShape, (uint8_t *)sin_ptr, "sin");
        Tensor positionIds(DataType::DT_INT32, idsShape, (uint8_t *)pos_ids_ptr, "position_indices");
        Tensor qEmbed(DataType::DT_FP32, qEmbedShape, qEmbed_ptr, "qEmbed");
        Tensor kEmbed(DataType::DT_FP32, kEmbedShape, kEmbed_ptr, "kEmbed");

        RoPETileShapeConfig ropeTileConfig{
            {32, 64}, // for cos/sin->cast
            {1, 32, 64}, // for gather,unsqueeze
            {1, 32, 1, 64},
            {1, 32, 1, 64, 64} // for transpose
        };

        config::SetBuildStatic(true);
        FUNCTION("RoPE", {qPe, kPe, cos, sin, positionIds, qEmbed, kEmbed}) {
            TileShape::Current().SetVecTile({1, 1, 32, 64});
            auto qPeTrans = Transpose(qPe, {1, 2}); // [b,s,n,d]->[b,n,s,d]

            int b = kPe.GetShape()[0];
            int s = kPe.GetShape()[1];
            int d = kPe.GetShape()[2];
            // 以下两步Reshape+Transpose可以优化成1个reshape：auto kPeReshape = Reshape(kPe, {b, 1, s, d}); //
            // [b,s,d]->[b,1,s,d]
            auto kPeReshape = Reshape(kPe, {b, 1, s, d}); // [b,s,d]->[b,1,s,d]
            // auto kPeTrans = Transpose(Reshape(kPe, {b, s, 1, d}), {1, 2}); // [b,s,d]->[b,s,1,d]->[b,1,s,d]
            ApplyRotaryPosEmb(qPeTrans, kPeReshape, cos, sin, positionIds, qEmbed, kEmbed, 1, ropeTileConfig);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    // qEmbed
    std::vector<float> qEmbedRes(qEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)qEmbedRes.data(), (uint8_t *)qEmbed_ptr, qEmbedSize * sizeof(float)); // Byte
    std::vector<float> qEmbedGolden(qEmbedSize);
    readInput(GetGoldenDir() + shape_dir_path + "qEmbed_res.bin", qEmbedGolden);
    int ret = resultCmp(qEmbedGolden, qEmbedRes, 0.001f);

    // for (int i = 0; i < qEmbedSize; i++) {
    //     std::cout << "index["<< i << "]: golden:" << qEmbedGolden[i] << " AST res:" << qEmbedRes[i] << std::endl;
    //     if (i > 64) {
    //         std::cout << "[WARNNING] =======qEmbedSize is " << qEmbedSize << ", not print!!!" << std::endl;
    //         break;
    //     }
    // }

    // kEmbed
    std::vector<float> kEmbedRes(kEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)kEmbedRes.data(), (uint8_t *)kEmbed_ptr, kEmbedSize * sizeof(float));
    std::vector<float> kEmbedGolden(kEmbedSize);
    // readInput(GetGoldenDir() + shape_dir_path + "k_RTR_rotate_half.bin", kEmbedGolden);
    readInput(GetGoldenDir() + shape_dir_path + "kEmbed_res.bin", kEmbedGolden);
    ret &= resultCmp(kEmbedGolden, kEmbedRes, 0.001f);

    // for (int i = 0; i < kEmbedSize; i++) {
    //     std::cout << "index["<< i << "]: golden:" << kEmbedGolden[i] << " AST res:" << kEmbedRes[i] << std::endl;
    //     if (i > 64) {
    //         std::cout << "[WARNNING] =======kEmbedSize is " << kEmbedSize << ", not print!!!" << std::endl;
    //         break;
    //     }
    // }

    EXPECT_EQ(ret, true);
}

TEST_F(RoPESubGraphOnBoardTest, test_operation_rope_subgraph_deepseekv3_fp16) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 1;
    int S = 1;                 // IFA S=1 S=1024
    int N = 32;                // N=32
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::string shape_dir_path = "/1_1_32_64/";

    std::vector<int64_t> qPeShape{B, S, N, qkRopeHeadDim};
    std::vector<int64_t> kPeShape{B, S, qkRopeHeadDim};
    std::vector<int64_t> idsShape{B, S};
    std::vector<int64_t> cosSinShape{S, qkRopeHeadDim};
    std::vector<int64_t> qEmbedShape{B, N, S, qkRopeHeadDim};
    std::vector<int64_t> kEmbedShape{B, 1, S, qkRopeHeadDim};

    int qSize = std::accumulate(qPeShape.begin(), qPeShape.end(), 1, std::multiplies<>());
    int kSize = std::accumulate(kPeShape.begin(), kPeShape.end(), 1, std::multiplies<>());
    int cosSinSize = std::accumulate(cosSinShape.begin(), cosSinShape.end(), 1, std::multiplies<>());
    int posIdsSize = std::accumulate(idsShape.begin(), idsShape.end(), 1, std::multiplies<>());
    int qEmbedSize = std::accumulate(qEmbedShape.begin(), qEmbedShape.end(), 1, std::multiplies<>());
    int kEmbedSize = std::accumulate(kEmbedShape.begin(), kEmbedShape.end(), 1, std::multiplies<>());

    uint8_t* qEmbed_ptr = allocDevAddr(qEmbedSize * sizeof(npu::tile_fwk::float16)); // Byte
    uint8_t* kEmbed_ptr = allocDevAddr(kEmbedSize * sizeof(npu::tile_fwk::float16));

    PROGRAM("RoPEMla") {

        void *q_ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + shape_dir_path + "qPe.bin", qSize);
        void *k_ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + shape_dir_path + "kPe.bin", kSize);
        void *cos_ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + shape_dir_path + "cos.bin", cosSinSize);
        void *sin_ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + shape_dir_path + "sin.bin", cosSinSize);
        void *pos_ids_ptr = readToDev<int32_t>(GetGoldenDir() + shape_dir_path + "pos_ids.bin", posIdsSize);
        Tensor qPe(DataType::DT_FP16, qPeShape, (uint8_t *)q_ptr, "qPe");
        Tensor kPe(DataType::DT_FP16, kPeShape, (uint8_t *)k_ptr, "kPe");
        Tensor cos(DataType::DT_FP16, cosSinShape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(DataType::DT_FP16, cosSinShape, (uint8_t *)sin_ptr, "sin");
        Tensor positionIds(DataType::DT_INT32, idsShape, (uint8_t *)pos_ids_ptr, "position_indices");
        Tensor qEmbed(DataType::DT_FP16, qEmbedShape, qEmbed_ptr, "qEmbed");
        Tensor kEmbed(DataType::DT_FP16, kEmbedShape, kEmbed_ptr, "kEmbed");

        RoPETileShapeConfig ropeTileConfig{
            {32, 64}, // for cos/sin->cast
            {1, 32, 64}, // for gather,unsqueeze
            {1, 32, 1, 64},
            {1, 32, 1, 64, 64} // for transpose
        };

        config::SetBuildStatic(true);
        FUNCTION("RoPE", {qPe, kPe, cos, sin, positionIds, qEmbed, kEmbed}) {
            TileShape::Current().SetVecTile({1, 1, 32, 64});
            auto qPeTrans = Transpose(qPe, {1, 2}); // [b,s,n,d]->[b,n,s,d]

            int b = kPe.GetShape()[0];
            int s = kPe.GetShape()[1];
            int d = kPe.GetShape()[2];
            // 以下两步Reshape+Transpose可以优化成1个reshape：auto kPeReshape = Reshape(kPe, {b, 1, s, d}); //
            // [b,s,d]->[b,1,s,d]
            auto kPeReshape = Reshape(kPe, {b, 1, s, d}); // [b,s,d]->[b,1,s,d]
            // auto kPeTrans = Transpose(Reshape(kPe, {b, s, 1, d}), {1, 2}); // [b,s,d]->[b,s,1,d]->[b,1,s,d]
            ApplyRotaryPosEmb(qPeTrans, kPeReshape, cos, sin, positionIds, qEmbed, kEmbed, 1, ropeTileConfig);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    // qEmbed
    std::vector<npu::tile_fwk::float16> qEmbedRes(qEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)qEmbedRes.data(), (uint8_t *)qEmbed_ptr, qEmbedSize * sizeof(npu::tile_fwk::float16)); // Byte
    std::vector<npu::tile_fwk::float16> qEmbedGolden(qEmbedSize);
    readInput<npu::tile_fwk::float16>(GetGoldenDir() + shape_dir_path + "qEmbed_res.bin", qEmbedGolden);
    int ret = resultCmp<npu::tile_fwk::float16>(qEmbedGolden, qEmbedRes, 0.01f);

    // kEmbed
    std::vector<npu::tile_fwk::float16> kEmbedRes(kEmbedSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)kEmbedRes.data(), (uint8_t *)kEmbed_ptr, kEmbedSize * sizeof(npu::tile_fwk::float16));
    std::vector<npu::tile_fwk::float16> kEmbedGolden(kEmbedSize);
    readInput<npu::tile_fwk::float16>(GetGoldenDir() + shape_dir_path + "kEmbed_res.bin", kEmbedGolden);
    ret &= resultCmp<npu::tile_fwk::float16>(kEmbedGolden, kEmbedRes, 0.01f);

    EXPECT_EQ(ret, true);
}

TEST_F(RoPESubGraphOnBoardTest, test_operation_rope_subgraph_deepseekv3_fp16_2batch) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 2;
    int S = 1;                 // IFA S=1 S=1024
    int N = 32;                // N=32
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::string shape_dir_path = "/2_1_32_64/";

    int ret = test_init (B, S, N, qkRopeHeadDim, shape_dir_path);

    EXPECT_EQ(ret, true);
}

TEST_F(RoPESubGraphOnBoardTest, test_operation_rope_subgraph_deepseekv3_bf16) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 1;
    int S = 1;                 // IFA S=1 S=1024
    int N = 32;                // N=32
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::string shape_dir_path = "/1_1_32_64/";

    int ret = test_init (B, S, N, qkRopeHeadDim, shape_dir_path);

    EXPECT_EQ(ret, true);
}

TEST_F(RoPESubGraphOnBoardTest, test_operation_rope_subgraph_deepseekv3_bf16_32batch) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 32;
    int S = 1;                 // IFA S=1 S=1024
    int N = 32;                // N=32
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::string shape_dir_path = "/32_1_32_64/";

    int ret = test_init (B, S, N, qkRopeHeadDim, shape_dir_path);

    EXPECT_EQ(ret, true);
}

TEST_F(RoPESubGraphOnBoardTest, test_operation_rope_subgraph_deepseekv3_bf16_2batch) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 2;
    int S = 1;                 // IFA S=1 S=1024
    int N = 32;                // N=32
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::string shape_dir_path = "/2_1_32_64/";

    int ret = test_init (B, S, N, qkRopeHeadDim, shape_dir_path);

    EXPECT_EQ(ret, true);
}