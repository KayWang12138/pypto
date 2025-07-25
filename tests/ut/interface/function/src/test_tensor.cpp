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
 * \file test_tensor.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "interface/tensor/tensormap.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"

namespace {
constexpr int VALUE1 = 1;
constexpr int VALUE16 = 16;
constexpr int VALUE32 = 32;
constexpr int VALUE64 = 64;
constexpr int VALUEN4 = -4;

class TestTensor : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestTensor, AssignWithData) {
    std::vector<int> tshape = {100, 100};
    npu::tile_fwk::Tensor a(npu::tile_fwk::DT_FP32, tshape, "A");
    npu::tile_fwk::Tensor b(npu::tile_fwk::DT_FP32, tshape, "B");
    auto ptr1 = std::make_unique<uint8_t>(0);
    auto ptr2 = std::make_unique<uint8_t>(0);

    auto reset = [&a, &b]() {
        a.SetData(nullptr);
        b.SetData(nullptr);
    };

    {
        reset();
        b = a;
        EXPECT_EQ(b.GetData(), nullptr);
        EXPECT_EQ(a.GetData(), nullptr);
    }

    {
        reset();
        b.SetData(ptr1.get());
        b = a;
        EXPECT_EQ(b.GetData(), ptr1.get());
        EXPECT_EQ(a.GetData(), nullptr);
    }

    {
        reset();
        a.SetData(ptr1.get());
        b = a;
        EXPECT_EQ(b.GetData(), ptr1.get());
        EXPECT_EQ(a.GetData(), ptr1.get());
    }

    {
        reset();
        a.SetData(ptr2.get());
        b.SetData(ptr2.get());
        b = a;
        EXPECT_EQ(b.GetData(), ptr2.get());
        EXPECT_EQ(a.GetData(), ptr2.get());
    }

    {
        reset();
        a.SetData(ptr1.get());
        b.SetData(ptr2.get());
        ASSERT_DEATH({
            b = a;
        }, ".*");
        EXPECT_EQ(b.GetData(), ptr2.get());
        EXPECT_EQ(a.GetData(), ptr1.get());
    }
}

TEST_F(TestTensor, AssignWithData2) {
    std::vector<int> tshape = {100, 100};
    npu::tile_fwk::Tensor b(npu::tile_fwk::DT_FP32, tshape, "B");
    auto ptr1 = std::make_unique<uint8_t>(0);
    auto ptr2 = std::make_unique<uint8_t>(0);

    auto reset = [&b]() {
        b.SetData(nullptr);
    };

    {
        npu::tile_fwk::Tensor a(npu::tile_fwk::DT_FP32, tshape, "A");
        reset();
        b = std::move(a);
        EXPECT_EQ(b.GetData(), nullptr);
    }

    {
        npu::tile_fwk::Tensor a(npu::tile_fwk::DT_FP32, tshape, "A");
        reset();
        b.SetData(ptr1.get());
        b = std::move(a);
        EXPECT_EQ(b.GetData(), ptr1.get());
    }

    {
        npu::tile_fwk::Tensor a(npu::tile_fwk::DT_FP32, tshape, "A");
        reset();
        a.SetData(ptr1.get());
        b = std::move(a);
        EXPECT_EQ(b.GetData(), ptr1.get());
    }

    {
        npu::tile_fwk::Tensor a(npu::tile_fwk::DT_FP32, tshape, "A");
        reset();
        a.SetData(ptr1.get());
        b.SetData(ptr1.get());
        b = std::move(a);
        EXPECT_EQ(b.GetData(), ptr1.get());
    }

    {
        npu::tile_fwk::Tensor a(npu::tile_fwk::DT_FP32, tshape, "A");
        reset();
        a.SetData(ptr1.get());
        b.SetData(ptr2.get());
        ASSERT_DEATH({
            b = std::move(a);
        }, ".*");
        EXPECT_EQ(b.GetData(), ptr2.get());
    }
}

TEST_F(TestTensor, GetShapeTest) {
    std::vector<int> tshape = {VALUE16, VALUE32, VALUE16, VALUE64};
    std::vector<int> kshape = {};
    npu::tile_fwk::Tensor a(npu::tile_fwk::DT_FP32, tshape, "A");
    npu::tile_fwk::Tensor b(npu::tile_fwk::DT_FP32, kshape, "B");

    auto ashape = a.GetShape(VALUE1);
    EXPECT_EQ(ashape, VALUE32);
    ashape = a.GetShape(VALUEN4);
    EXPECT_EQ(ashape, VALUE16);
}
}