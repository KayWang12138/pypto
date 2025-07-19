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
 * \file test_tensor_overlap.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

using namespace npu::tile_fwk;

class TensorOverlapTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    }

    void TearDown() override {}
};

Tensor add_sub_exp(int firstTileSize, int secondTileSize) {
    std::vector<int> tshape = {24, 24};
    Tensor A(DT_INT32, tshape, "A");
    Tensor B(DT_INT32, tshape, "B");
    FUNCTION("FUNC_A") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(firstTileSize, firstTileSize);
        A = Add(A, B);
        Program::GetInstance().GetTileShape().SetVecTileShapes(secondTileSize, secondTileSize);
        A = Sub(A, B);
    }
    FUNCTION("FUNC_B") {
        A = Exp(A);
    }

    return A;
}

TEST_F(TensorOverlapTest, TensorMap) {
    config::SetPlatformConfig(KEY_ONLY_TENSOR_GRAPH, true);

    std::vector<int> tshape = {24, 24};
    Tensor A(DT_INT32, tshape, "A");
    Tensor B(DT_INT32, tshape, "B");
    FUNCTION("FUNC_A") {
        auto aa = View(A, {12, 12}, {0, 0});
        auto ab = View(A, {12, 12}, {0, 12});
        auto ac = View(A, {12, 12}, {12, 12});
        Program::GetInstance().GetCurrentFunction()->GetTensorMap().Insert(aa.GetStorage());
        Program::GetInstance().GetCurrentFunction()->GetTensorMap().Insert(ab.GetStorage());
        Program::GetInstance().GetCurrentFunction()->GetTensorMap().Insert(ac.GetStorage());

        auto a1 = View(A, {12, 24}, {0, 0});
        auto match = Program::GetInstance().GetCurrentFunction()->GetTensorMap().Find(a1.GetStorage());
        auto status = CalcOverlap(a1.GetStorage(), match);
        ASSERT(status == OverlapStatus::PERFECTLY_MATCH_WITH_ALL);

        auto a2 = View(A, {6, 24}, {3, 0});
        match = Program::GetInstance().GetCurrentFunction()->GetTensorMap().Find(a2.GetStorage());
        status = CalcOverlap(a2.GetStorage(), match);
        ASSERT(status == OverlapStatus::BE_COVERED_BY_ALL);

        auto a4 = View(A, {12, 12}, {6, 6});
        match = Program::GetInstance().GetCurrentFunction()->GetTensorMap().Find(a4.GetStorage());
        status = CalcOverlap(a4.GetStorage(), match);
        ASSERT(status == OverlapStatus::PARTIAL_OVERLAP_WITH_ALL);

        auto a5 = View(A, {12, 12}, {0, 0});
        match = Program::GetInstance().GetCurrentFunction()->GetTensorMap().Find(a5.GetStorage());
        status = CalcOverlap(a5.GetStorage(), match);
        ASSERT(status == OverlapStatus::PERFECTLY_MATCH);

        auto a6 = View(A, {6, 6}, {6, 6});
        match = Program::GetInstance().GetCurrentFunction()->GetTensorMap().Find(a6.GetStorage());
        status = CalcOverlap(a6.GetStorage(), match);
        ASSERT(status == OverlapStatus::BE_COVERED);

        auto a7 = View(A, {24, 12}, {0, 0});
        match = Program::GetInstance().GetCurrentFunction()->GetTensorMap().Find(a7.GetStorage());
        status = CalcOverlap(a7.GetStorage(), match);
        ASSERT(status == OverlapStatus::COVERED);

        auto a8 = View(A, {12, 12}, {6, 0});
        match = Program::GetInstance().GetCurrentFunction()->GetTensorMap().Find(a8.GetStorage());
        status = CalcOverlap(a8.GetStorage(), match);
        ASSERT(status == OverlapStatus::PARTIAL_OVERLAP);

        auto a9 = View(A, {12, 12}, {12, 0});
        match = Program::GetInstance().GetCurrentFunction()->GetTensorMap().Find(a9.GetStorage());
        status = CalcOverlap(a9.GetStorage(), match);
        ASSERT(status == OverlapStatus::NO_OVER_LAP);
    }
}

TEST_F(TensorOverlapTest, AddSubExp_12_12) {
    auto output = add_sub_exp(12, 12);
    std::cout << npu::tile_fwk::Program::GetInstance().Dump() << std::endl;
}

TEST_F(TensorOverlapTest, AddSubExp_12_24) {
    auto output = add_sub_exp(12, 24);
    std::cout << npu::tile_fwk::Program::GetInstance().Dump() << std::endl;
}

TEST_F(TensorOverlapTest, AddSubExp_24_12) {
    auto output = add_sub_exp(24, 12);
    std::cout << npu::tile_fwk::Program::GetInstance().Dump() << std::endl;
}

// fix issue of copy_alloc_adder
//TEST(TestIncastOverlap, AddSubExp_12_8) {
//    auto output = add_sub_exp(12, 8);
//    std::cout << npu::tile_fwk::Program::GetInstance().dump() << std::endl;
//}

TEST_F(TensorOverlapTest, AddSubExp_8_12) {
    auto output = add_sub_exp(8, 12);
    std::cout << npu::tile_fwk::Program::GetInstance().Dump() << std::endl;
}
