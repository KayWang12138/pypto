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
 * \file test_serialization.cpp
 * \brief
 */

#include "interface/configs/config_manager.h"
#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"

using namespace npu::tile_fwk;

class SerializationTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig("USE_SSA", true);
    }

    void TearDown() override {}
};

TEST_F(SerializationTest, TestAddSub) {
    config::SetPlatformConfig(KEY_ONLY_TENSOR_GRAPH, true);
    config::SetHostConfig(KEY_ONLY_TENSOR_GRAPH, true);
        ConfigManager::Instance();

    std::vector<int64_t> shape{24, 24};

    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");

    TileShape::Current().SetVecTile(8, 8);

    Tensor e;
    FUNCTION("A") {
        auto c = Add(a, b);
        auto d = Sub(a, b);
        e = Add(c, d);
    }
    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_A");
    EXPECT_NE(func, nullptr);
    auto r1 = func->DumpJson();
    auto r1str = r1.dump(4);
    Program p;
    auto f2 = Function::LoadJson(p, r1);
    auto r2 = f2->DumpJson();
    auto r2str = r2.dump(4);
    EXPECT_EQ(r1str, r2str);
}
