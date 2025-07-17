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
 * \file test_onboard_mm.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"

using namespace npu::tile_fwk;

class MatmulOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template<typename InputT, typename OnputT>
void TestMatmul(int m, int k, int n, string dataPath) {
    std::vector<int> shape_a = {m, k};
    std::vector<int> shape_b = {k, n};
    std::vector<int> shape_c = {m, n};
    const int capacity_a = m * k;
    const int capacity_b = k * n;
    const int capacity_c = m * n;

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    uint64_t outputSize = capacity_c * sizeof(OnputT);
    uint8_t* c_ptr = allocDevAddr(outputSize);
    auto InputAstDtype = GetAstDtype<InputT>();
    auto OutputAstDtype = GetAstDtype<OnputT>();


    PROGRAM("Matmul") {
        void *a_ptr = readToDev<InputT>(dataPath + "/a.bin", capacity_a);
        void *b_ptr = readToDev<InputT>(dataPath + "/b.bin", capacity_b);

        Tensor mat_a(InputAstDtype, shape_a, (uint8_t *)a_ptr, "mat_a");
        Tensor mat_b(InputAstDtype, shape_b, (uint8_t *)b_ptr, "mat_b");
        Tensor mat_c(OutputAstDtype, shape_c, c_ptr, "mat_c");

        FUNCTION("Matmul_T", FunctionType::STATIC, {mat_a, mat_b, mat_c}) {
            mat_c = npu::tile_fwk::Matrix::Matmul<false, false>(OutputAstDtype, mat_a, mat_b);  // result dtype
        }
    }
    std::vector<OnputT> dev_res(capacity_c);
    std::vector<OnputT> golden(capacity_c);
    runtime::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), c_ptr, outputSize);
    readInput(dataPath + "/c_golden.bin", golden);
    std::cout << "====== output size:" << capacity_c << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

template<typename InputT, typename OnputT>
void TestMatmulTrans(int m, int k, int n, string dataPath) {
    std::vector<int> shape_a = {m, k};
    std::vector<int> shape_b = {n, k};
    std::vector<int> shape_c = {m, n};
    const int capacity_a = m * k;
    const int capacity_b = k * n;
    const int capacity_c = m * n;

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    uint64_t outputSize = capacity_c * sizeof(OnputT);
    uint8_t* c_ptr = allocDevAddr(outputSize);
    auto InputAstDtype = GetAstDtype<InputT>();
    auto OutputAstDtype = GetAstDtype<OnputT>();


    PROGRAM("Matmul") {
        void *a_ptr = readToDev<InputT>(dataPath + "/a.bin", capacity_a);
        void *b_ptr = readToDev<InputT>(dataPath + "/b.bin", capacity_b);

        Tensor mat_a(InputAstDtype, shape_a, (uint8_t *)a_ptr, "mat_a");
        Tensor mat_b(InputAstDtype, shape_b, (uint8_t *)b_ptr, "mat_b");
        Tensor mat_c(OutputAstDtype, shape_c, c_ptr, "mat_c");

        FUNCTION("Matmul_T", FunctionType::STATIC, {mat_a, mat_b, mat_c}) {
            mat_c = npu::tile_fwk::Matrix::Matmul<false, true>(OutputAstDtype, mat_a, mat_b);  // result dtype
        }
    }
    std::vector<OnputT> dev_res(capacity_c);
    std::vector<OnputT> golden(capacity_c);
    runtime::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), c_ptr, outputSize);
    readInput(dataPath + "/c_golden.bin", golden);
    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

template<typename InputT, typename OnputT>
void TestMatmulACC(int m, int k, int n, string dataPath) {
    std::vector<int> shape_a = {m, k};
    std::vector<int> shape_b = {k, n};
    std::vector<int> shape_c = {m, n};
    const int capacity_a = m * k;
    const int capacity_b = k * n;
    const int capacity_c = m * n;

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    uint64_t outputSize = capacity_c * sizeof(OnputT);
    uint8_t* c_ptr = allocDevAddr(outputSize);
    // uint8_t* c_ptr2 = allocDevAddr(outputSize);
    auto InputAstDtype = GetAstDtype<InputT>();
    auto OutputAstDtype = GetAstDtype<OnputT>();


    PROGRAM("Matmul") {
        void *a_ptr = readToDev<InputT>(dataPath + "/a.bin", capacity_a);
        void *b_ptr = readToDev<InputT>(dataPath + "/b.bin", capacity_b);

        Tensor mat_a(InputAstDtype, shape_a, (uint8_t *)a_ptr, "mat_a");
        Tensor mat_b(InputAstDtype, shape_b, (uint8_t *)b_ptr, "mat_b");
        Tensor final_out(OutputAstDtype, shape_c, c_ptr, "final_out");
        auto kSplit = 4;
        auto kSplitSize = k / kSplit;
        FUNCTION("Matmul_T", FunctionType::STATIC, {mat_a, mat_b, final_out}) {
            Program::GetInstance().GetTileShape().SetVecTileShapes(64, 64);
            Tensor tmpC(OutputAstDtype, shape_c, "tmp_c");
            tmpC = MulS(tmpC, Element(DataType::DT_FP32, 0.0f));
            std::vector<Tensor> matmulResult;
            for (int ki = 0; ki < kSplit; ki++) {
                auto input_mk = View(mat_a, {m, kSplitSize}, {0, ki * kSplitSize});
                auto input_kn = View(mat_b, {kSplitSize, n}, {ki * kSplitSize, 0});
                auto tmpC1 = Matrix::Matmul<false, false>(OutputAstDtype, input_mk, input_kn, tmpC);
                matmulResult.emplace_back(tmpC1);
            }
            Program::GetInstance().GetTileShape().SetVecTileShapes(32, 256);
            tmpC = npu::tile_fwk::Reduce(matmulResult, ReduceMode::ATOMIC_ADD);
            Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
            final_out = AddS(tmpC, Element(DataType::DT_FP32, 0.0));
        }
    }
    std::vector<OnputT> dev_res(capacity_c);
    std::vector<OnputT> golden(capacity_c);
    runtime::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), c_ptr, outputSize);
    readInput(dataPath + "/c_golden.bin", golden);
    int ret = resultCmp(golden, dev_res, 0.001f);
    std::cout <<"golden = "<< golden[0] << " result = " << dev_res[0] << std::endl;
    EXPECT_EQ(ret, true);
}

TEST_F(MatmulOnBoardTest, test_mm_float32_64_64_64_acc) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});
    TestMatmulACC<npu::tile_fwk::float16, float>(64, 64, 64, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float32_32_7168_1536_acc) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {128, 128}, {64, 64});
    TestMatmulACC<npu::tile_fwk::float16, float>(32, 7168, 1536, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float32_32_512_128_acc) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {128, 128}, {64, 64});
    TestMatmulACC<npu::tile_fwk::float16, float>(32, 512, 128, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float32_32_1024_512_acc) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {128, 128}, {256, 256});
    TestMatmulACC<npu::tile_fwk::float16, float>(32, 1024, 512, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float32_64_64_64) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {16, 16}, {32, 32});
    TestMatmul<npu::tile_fwk::float16, float>(64, 64, 64, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float32_64_128_128) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({64, 64}, {128, 128}, {128, 128});
    TestMatmul<npu::tile_fwk::float16, float>(64, 128, 128, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float32_128_128_128) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
    TestMatmul<npu::tile_fwk::float16, float>(128, 128, 128, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float32_32_128_128) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {64, 64}, {64, 64});
    TestMatmul<npu::tile_fwk::float16, float>(32, 128, 128, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float32_32_128_64) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 32}, {32, 64}, {32, 64}, true);
    TestMatmul<npu::tile_fwk::float16, float>(32, 128, 64, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_int8_32_128_64) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 32}, {64, 128}, {32, 64}, true);
    TestMatmul<int8_t, int32_t>(32, 128, 64, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_int8_32_128_64_bt) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 32}, {64, 128}, {32, 64}, true);
    TestMatmulTrans<int8_t, int32_t>(32, 128, 64, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float_32_128_128) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 32}, {64, 128}, {64, 128}, true);
    TestMatmul<float, float>(32, 128, 128, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float_32_128_128_bt) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 32}, {64, 128}, {64, 128}, true);
    TestMatmulTrans<float, float>(32, 128, 128, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float32_32_192_64) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 32}, {32, 64, 96}, {32, 64}, true);
    TestMatmul<npu::tile_fwk::float16, float>(32, 192, 64, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float32_32_128_192) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {64, 64}, {64, 64});
    TestMatmul<npu::tile_fwk::float16, float>(32, 128, 192, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float32_256_256_256) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({64, 64}, {64, 64}, {128, 128});
    TestMatmul<npu::tile_fwk::float16, float>(256, 256, 256, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float32_32_512_576) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {128, 512}, {64, 64});
    TestMatmul<npu::tile_fwk::float16, float>(32, 512, 576, GetGoldenDir());
}

// [32*1, 7168] * [7168,1536] = [32*1, 1536]
TEST_F(MatmulOnBoardTest, test_mm_float32_32_7168_1536) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {512, 512}, {64, 64});
    TestMatmul<npu::tile_fwk::float16, float>(32, 7168, 1536, GetGoldenDir());
}

// [32*1, 1536] * [1536,32*192] = [32*1, 32*192]
TEST_F(MatmulOnBoardTest, test_mm_float32_32_1536_6144) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {256, 256}, {128, 128});
    TestMatmul<npu::tile_fwk::float16, float>(32, 1536, 6144, GetGoldenDir());
}

// [32*1, 7168] * [7168,576] = [32*1, 576]
TEST_F(MatmulOnBoardTest, test_mm_float32_32_7168_576) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {512, 512}, {64, 64});
    TestMatmul<npu::tile_fwk::float16, float>(32, 7168, 576, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float16_64_128_128) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 64}, {64, 128}, {64, 64});
    TestMatmul<npu::tile_fwk::float16, npu::tile_fwk::float16>(64, 128, 128, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float16_16_7168_2048) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {1024, 1024}, {64, 64});
    TestMatmul<npu::tile_fwk::float16, npu::tile_fwk::float16>(16, 7168, 2048, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float16_16_7168_1024) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {1024, 1024}, {64, 64});
    TestMatmul<npu::tile_fwk::float16, npu::tile_fwk::float16>(16, 7168, 1024, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_float16_64_256_128) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 64}, {64, 256}, {64, 128});
    TestMatmul<npu::tile_fwk::float16, npu::tile_fwk::float16>(64, 256, 128, GetGoldenDir());
}

// [32*1, 7168] * [7168,1536] = [32*1, 1536]
TEST_F(MatmulOnBoardTest, test_mm_float16_32_7168_1536) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {256, 256}, {128, 128});
    TestMatmul<npu::tile_fwk::float16, npu::tile_fwk::float16>(32, 7168, 1536, GetGoldenDir());
}

// [32*1, 1536] * [1536,32*192] = [32*1, 32*192]
TEST_F(MatmulOnBoardTest, test_mm_float16_32_1536_6144) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {256, 256}, {64, 64});
    TestMatmul<npu::tile_fwk::float16, npu::tile_fwk::float16>(32, 1536, 6144, GetGoldenDir());
}

// [32*1, 7168] * [7168,576] = [32*1, 576]
TEST_F(MatmulOnBoardTest, test_mm_float16_32_7168_576) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {256, 256}, {64, 64});
    TestMatmul<npu::tile_fwk::float16, npu::tile_fwk::float16>(32, 7168, 576, GetGoldenDir());
}

// [4*1, 7168] * [7168,1536] = [4*1, 1536]
TEST_F(MatmulOnBoardTest, test_mm_float16_4_7168_1536) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {256, 256}, {64, 64});
    TestMatmul<npu::tile_fwk::float16, npu::tile_fwk::float16>(4, 7168, 1536, GetGoldenDir());
}

// [4*1, 1536] * [1536,32*192] = [4*1, 32*192]
TEST_F(MatmulOnBoardTest, test_mm_float16_4_1536_6144) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {256, 256}, {64, 64});
    TestMatmul<npu::tile_fwk::float16, npu::tile_fwk::float16>(4, 1536, 6144, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_bfloat16_64_128_128) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({64, 64}, {64, 64}, {64, 64});
    TestMatmul<npu::tile_fwk::bfloat16, npu::tile_fwk::bfloat16>(64, 128, 128, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_bfloat16_f32_64_128_128) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 64}, {64, 128}, {64, 128});
    TestMatmul<npu::tile_fwk::bfloat16, float>(64, 128, 128, GetGoldenDir());
}

// m unalign
TEST_F(MatmulOnBoardTest, test_mm_unalign_float32_2_128_128) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {128, 128}, {128, 128});
    TestMatmul<npu::tile_fwk::float16, float>(2, 128, 128, GetGoldenDir());
}

// k unalign
TEST_F(MatmulOnBoardTest, test_mm_unalign_float32_16_35_32) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {32, 32}, {32, 32});
    TestMatmul<npu::tile_fwk::float16, float>(16, 35, 32, GetGoldenDir());
}

// n unalign precision failed
TEST_F(MatmulOnBoardTest, test_mm_unalign_float32_16_32_35) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {32, 32}, {32, 32});
    TestMatmul<npu::tile_fwk::float16, float>(16, 32, 35, GetGoldenDir());
}

// n unalign precision failed
TEST_F(MatmulOnBoardTest, test_mm_float32_64_64_64_bt) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});
    TestMatmulTrans<npu::tile_fwk::float16, float>(64, 64, 64, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_unalign_float32_8_576_256_bt) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {64, 64}, {64, 64});
    TestMatmulTrans<npu::tile_fwk::float16, float>(8, 576, 256, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_unalign_float32_8_64_64_bt) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});
    TestMatmulTrans<npu::tile_fwk::float16, float>(8, 64, 64, GetGoldenDir());
}

TEST_F(MatmulOnBoardTest, test_mm_int8_32_16384_7168) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {128, 128}, {128, 128});
    Program::GetInstance().GetConfig().SetL1Reuse(4);
    Program::GetInstance().GetConfig().SetCopyInThreshold(32*1024*1024);
    TestMatmul<int8_t, int32_t>(32, 16384, 7168, GetGoldenDir());
}
