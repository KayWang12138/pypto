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
 * \file test_operation_impl.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"


using namespace npu::tile_fwk;

class OperationImplTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(OperationImplTest, TestTranspose_BNSD_BSND) {
    std::vector<int64_t> shape{3, 32, 64, 16};
    Tensor a(DT_FP32, shape, "a");

    TileShape::Current().SetVecTile(1, 16, 16, 16);
    FUNCTION("BNSD_BSND") {
        a = Transpose(a, {1, 2});
    }
    a.GetStorage()->Dump();
    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(OperationImplTest, TestTranspose_BNSD2_BNS2D_small) {
    std::vector<int64_t> shape{1, 2, 64, 64, 2};
    Tensor a(DT_FP32, shape, "a");
    TileShape::Current().SetVecTile(1, 1, 64, 64, 2);

    FUNCTION("BNSD2_BNS2D") {
        a = Transpose(a, {3, 4});
    }
    a.GetStorage()->Dump();
    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(OperationImplTest, TestTranspose_BNSD2_BNS2D) {
    std::vector<int64_t> shape{1, 2, 1280, 128, 2};
    Tensor a(DT_FP32, shape, "a");
    TileShape::Current().SetVecTile(1, 1, 128, 128, 2);

    FUNCTION("BNSD2_BNS2D") {
        a = Transpose(a, {3, 4});
    }
    a.GetStorage()->Dump();
    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(OperationImplTest, TestTranspose_ABC_BAC) {
    std::vector<int64_t> shape{128, 2, 128};
    Tensor a(DT_FP32, shape, "a");

    TileShape::Current().SetVecTile(32, 1, 128);
    FUNCTION("ABC_BAC") {
        a = Transpose(a, {0, 1});
    }
    a.GetStorage()->Dump();
    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(OperationImplTest, TestTranspose_BNDS_BNSD) {
    std::vector<int64_t> shape{1, 32, 64, 2};
    Tensor a(DT_FP32, shape, "a");

    TileShape::Current().SetVecTile(1, 2, 64, 2);
    FUNCTION("BNDS_BNSD") {
        a = Transpose(a, {3, 2});
    }
    a.GetStorage()->Dump();
    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(OperationImplTest, Test_multiReshape) {
    TileShape::Current().SetVecTile(16, 16, 16, 16);
    Tensor input(DT_FP32, {8, 16, 16}, "a");
    Tensor res1;
    Tensor res2;
    FUNCTION("TestAssign") {
        auto tmp = Reshape(input, {8, 1, 16, 16});
        auto tmp1 = Reshape(tmp, {8, 4, 4, 16});
        auto tmp2 = Reshape(tmp1, {8, 16, 16});
        auto res = Exp(tmp2);
        res2 = Sqrt(tmp2);
        auto test = Reshape(res, {8, 16, 4, 4});
        res1 = Exp(test);
    }
    std::cout << Program::GetInstance().Dump() << std::endl;
    Program::GetInstance().GraphCheck();
}

TEST_F(OperationImplTest, Test_Reshape_reshape_assemble_multito1) {
    TileShape::Current().SetVecTile(8, 8, 8, 8);
    Tensor input(DT_FP32, {1, 384}, "a");
    Tensor res1;
    FUNCTION("TestAssign") {
        TileShape::Current().SetVecTile(1, 64);
        Tensor res = Exp(input);
        Tensor test = Reshape(res, {1, 1, 2, 192});
        TileShape::Current().SetVecTile(2, 1, 2, 64);
        res1 = Exp(test);
    }
    // Program::GetInstance().GraphCheck();
    // std::cout << Program::GetInstance().dump() << std::endl;
}

TEST_F(OperationImplTest, Test_Reshape_1to1) {
    TileShape::Current().SetVecTile(8, 8, 8, 8);
    Tensor input(DT_FP32, {8, 16, 16}, "a");
    Tensor res1;
    FUNCTION("TestAssign") {
        Tensor res = Exp(input);
        Tensor test = Reshape(res, {8, 16, 1, 16});
        res1 = Exp(test);
    }
    // Program::GetInstance().GraphCheck();
    // std::cout << Program::GetInstance().dump() << std::endl;
}

TEST_F(OperationImplTest, Test_Reshape_1toMulti) {
    TileShape::Current().SetVecTile(8, 8, 8, 8, 8);
    // Tensor input(DT_FP32, {2, 4, 16, 4, 4}, "a");
    Tensor input(DT_FP32, {16, 4, 4}, "a");
    Tensor res1;
    FUNCTION("TestAssign") {
        auto res = Exp(input);
        // auto test = Reshape(res, {8, 16, 16});
        auto test = Reshape(res, {16, 16});
        res1 = Exp(test);
    }
    // std::cout << Program::GetInstance().dump() << std::endl;
    // Program::GetInstance().GraphCheck();
}

TEST_F(OperationImplTest, Test_Reshape_multito1) {
    TileShape::Current().SetVecTile(8, 8, 8, 8);
    Tensor input(DT_FP32, {8, 16, 16}, "a");
    Tensor res1;
    FUNCTION("TestAssign") {
        auto res = Exp(input);
        auto test = Reshape(res, {8, 16, 4, 4});
        res1 = Exp(test);
    }
    // std::cout << Program::GetInstance().dump() << std::endl;
    // Program::GetInstance().GraphCheck();
}

TEST_F(OperationImplTest, Test_Unsqueeze) {
    TileShape::Current().SetVecTile(8, 8, 8, 8);
    Tensor input(DT_FP32, {8, 16, 16}, "a");
    Tensor res1;
    FUNCTION("TestUnsqueeze") {
        Tensor res = Exp(input);
        Tensor test = Unsqueeze(res, 1);
        res1 = Exp(test);
    }
    // Program::GetInstance().GraphCheck();
    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(OperationImplTest, TestBasicOperationMixBroadcast) {
    std::vector<int64_t> shape{32, 32};

    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, {32, 1}, "b");
    Tensor c(DT_FP32, {32, 32, 32}, "c");
    FUNCTION("A") {
        TileShape::Current().SetVecTile(8, 8);
        auto d = Add(a, b);

        TileShape::Current().SetVecTile(8, 8, 8);
        auto e = Mul(d, c);
    }
    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(OperationImplTest, Test_TopK) {
    PROGRAM("TOPK") {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "A");
        auto output = std::make_tuple(Tensor(DT_FP32, shape, "res"), Tensor(DT_FP32, shape, "resDics"));
        config::SetBuildStatic(true);
        FUNCTION("TOPK_T") {
            output = TopK(input_a, 16, -1);
        }
    }
}

TEST_F(OperationImplTest, Test_ArgSort) {
    PROGRAM("ARGSORT") {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "A");
        auto output = Tensor(DT_FP32, shape, "res"); // std::make_tuple(Tensor(DT_FP32, shape, "res"), Tensor(DT_FP32, shape, "resDics"));
        config::SetBuildStatic(true);
        FUNCTION("ARGSORT_T") {
            output = ArgSort(input_a, -1);
        }
    }
}

TEST_F(OperationImplTest, Test_MatmulWithSplitK) {
    PROGRAM("ARGSORT") {
        std::vector<int64_t> shape = {128, 128};
        auto m = 128, k = 64, n = 32;
        auto kSplit = 2;
        Tensor matA(DT_FP16, {m, k}, "mat_a");
        Tensor matB(DT_FP16, {k, n}, "mat_b");
        Tensor matC(DT_FP32, {m, n}, "mat_c");
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
        config::SetBuildStatic(true);
        FUNCTION("Matmul_T", {matA, matB, matC}) {
            Tensor tmpC(DT_FP32, {m, n}, "tmp_c");
            tmpC = Mul(tmpC, Element(DataType::DT_FP32, 0.0f));
            std::vector<Tensor> matmulResult;
            auto kSplitSize = k / kSplit;
            for (int ki = 0; ki < kSplit; ki++) {
                auto input_mk = View(matA, {m, kSplitSize}, {0, ki * kSplitSize});
                auto input_kn = View(matB, {kSplitSize, n}, {ki * kSplitSize, 0});
                auto tmpC1 = Matrix::Matmul<false, false>(DataType::DT_FP32, input_mk, input_kn, tmpC);
                matmulResult.emplace_back(tmpC1);
            }
            tmpC = npu::tile_fwk::Reduce(matmulResult, ReduceMode::ATOMIC_ADD);
            matC = Add(tmpC, Element(DataType::DT_FP32, 0.0));
        }
    }
}

TEST_F(OperationImplTest, Test_MatmulWithSplitKWithTrans) {
    PROGRAM("ARGSORT") {
        std::vector<int64_t> shape = {128, 128};
        auto m = 128, k = 64, n = 32;
        auto kSplit = 2;
        Tensor matA(DT_FP16, {m, k}, "mat_a");
        Tensor matB(DT_FP16, {k, n}, "mat_b");
        Tensor matC(DT_FP32, {m, n}, "mat_c");
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
        config::SetBuildStatic(true);
        FUNCTION("Matmul_T", {matA, matB, matC}) {
            Tensor tmpC(DT_FP32, {m, n}, "tmp_c");
            tmpC = Mul(tmpC, Element(DataType::DT_FP32, 0.0f));
            std::vector<Tensor> matmulResult;
            auto kSplitSize = k / kSplit;
            for (int ki = 0; ki < kSplit; ki++) {
                auto input_mk = View(matA, {m, kSplitSize}, {0, ki * kSplitSize});
                auto input_kn = View(matB, {kSplitSize, n}, {ki * kSplitSize, 0});
                auto tmpC1 = Matrix::Matmul<false, true>(DataType::DT_FP32, input_mk, input_kn, tmpC);
                matmulResult.emplace_back(tmpC1);
            }
            tmpC = npu::tile_fwk::Reduce(matmulResult, ReduceMode::ATOMIC_ADD);
            matC = Add(tmpC, Element(DataType::DT_FP32, 0.0));
        }
    }
}

template <DataType inputType, DataType outputType, bool IsANZ = false, bool IsBNZ = false, bool isTransB = false>
void TestNZFormatBatch(int bs, int m, int k, int n) {
    std::vector<int64_t> batch_shape_a = {bs*m, k};
    auto nLen = isTransB ? bs * n : bs * k;
    auto kLen = isTransB ? k : n;
    std::vector<int64_t> batch_shape_b = {nLen, kLen};
    std::vector<int64_t> batch_shape_c = {bs*m, n};
    PROGRAM("BATCHMATMUL") {
        config::Reset();
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
        auto afmt = IsANZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        auto bfmt = IsBNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        Tensor matA(inputType, batch_shape_a, "MatA", afmt);
        Tensor matB(inputType, batch_shape_b, "MatB", bfmt);
        Tensor matC(outputType, batch_shape_c, "MatC");
        std::vector<Tensor> matrixVec;
        config::SetBuildStatic(true);
        FUNCTION("BATCHMATMUL", {matA, matB, matC}) {
            std::vector<std::pair<Tensor, std::vector<int64_t>>> assembleVec;
            for (size_t index = 0; index < (size_t)bs; ++index) {
                auto inputA = View(matA, {m, k}, {(int)index*m, 0});
                auto inputB = isTransB ? View(matB, {n, k}, {(int)index*n, 0}) : View(matB, {k, n}, {(int)index*k, 0});
                TileShape::Current().SetMatrixSize({m, k, n});
                auto outTensor = npu::tile_fwk::Matrix::Matmul<false, isTransB>(outputType, inputA, inputB);
                std::vector<int64_t> pairSecond = {(int)index * m, 0};
                auto pair = std::make_pair(outTensor, pairSecond);
                assembleVec.emplace_back(pair);
            }
            matC = Assemble(assembleVec);
        }
    }
}

TEST_F(OperationImplTest, test_BMMT_NZ_1_128_256_128_Batch) {
    TestNZFormatBatch<DataType::DT_FP16, DataType::DT_FP32, false, true, true>(2, 128, 128, 256);
}

TEST_F(OperationImplTest, test_MaxS_FP16) {
    float scalar = 127.0;
    TileShape::Current().SetVecTile(8, 8, 8, 8);
    Tensor operand1(DT_FP16, {8, 16, 16}, "operand1");
    Element operand2(DT_FP16, scalar);
    Tensor result;
    FUNCTION("TestMaxS") {
        result = Maximum(operand1, operand2);
    }
}

TEST_F(OperationImplTest, test_MaxS_FP32) {
    float scalar = 127.0;
    TileShape::Current().SetVecTile(8, 8, 8, 8);
    Tensor operand1(DT_FP32, {8, 16, 16}, "operand1");
    Element operand2(DT_FP32, scalar);
    Tensor result;
    FUNCTION("TestMaxS") {
        result = Maximum(operand1, operand2);
    }
}

TEST_F(OperationImplTest, test_MaxS_INT8) {
    int scalar = 127;
    TileShape::Current().SetVecTile(8, 8, 8, 8);
    Tensor operand1(DT_INT8, {8, 16, 16}, "operand1");
    Element operand2(DT_INT8, scalar);
    Tensor result;
    FUNCTION("TestMaxS") {
        result = Maximum(operand1, operand2);
    }
}


TEST_F(OperationImplTest, test_MaxS_INT16) {
    int scalar = 127;
    TileShape::Current().SetVecTile(8, 8, 8, 8);
    Tensor operand1(DT_INT16, {8, 16, 16}, "operand1");
    Element operand2(DT_INT16, scalar);
    Tensor result;
    FUNCTION("TestMaxS") {
        result = Maximum(operand1, operand2);
    }
}

TEST_F(OperationImplTest, test_MaxS_INT32) {
    int scalar = 127;
    TileShape::Current().SetVecTile(8, 8, 8, 8);
    Tensor operand1(DT_INT32, {8, 16, 16}, "operand1");
    Element operand2(DT_INT32, scalar);
    Tensor result;
    FUNCTION("TestMaxS") {
        result = Maximum(operand1, operand2);
    }
}

TEST_F(OperationImplTest, test_MinS_FP16) {
    float scalar = 127.0;
    TileShape::Current().SetVecTile(8, 8, 8, 8);
    Tensor operand1(DT_FP16, {8, 16, 16}, "operand1");
    Element operand2(DT_FP16, scalar);
    Tensor result;
    FUNCTION("TestMinS") {
        result = Maximum(operand1, operand2);
    }
}
