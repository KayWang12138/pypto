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
 * \file test_codegen_dispatch.cpp
 * \brief Unit test for codegen.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include <vector>
#include <string>

namespace npu::tile_fwk {
namespace Distributed {
class TestCodegenDispatch : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    }

    void TearDown() override {
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
    }

protected:
    bool oriEnableAihacBackend = false;
};

void TestMoeDispatch(bool isSharedExpert) {
    const char *group = "hcom123";
    DataType dType = DT_FP16;
    int sharedExpertNum = 1;
    int routingExpertNum = 3;
    int topK = 2;
    int bs = 4;
    int tokenLen = 256;
    int rankId = isSharedExpert ? 0 : sharedExpertNum;

    Tensor tokenTensor(DataType::DT_INT32, {bs, tokenLen}, "tokenTensor");
    Tensor tokenExpertTable(DataType::DT_INT32, {bs, topK}, "tokenExpertTable");
    Tensor expandX(dType, {bs * (routingExpertNum + sharedExpertNum), tokenLen}, "expandX");
    Tensor validSize(DataType::DT_INT32, {1, 1}, "validSize");
    FunctionConfig funConfig = {.funcType = FunctionType::STATIC};
    FUNCTION("DISPATCH_F", funConfig, {tokenTensor, tokenExpertTable, validSize, expandX}) {
        Program::GetInstance().GetTileShape().SpecifyStaticRankId(rankId);
        expandX = Distributed::MoeDispatch(tokenTensor, tokenExpertTable, validSize, group);
    }
}

TEST_F(TestCodegenDispatch, TestMoeDispatchRoutingExpert) {
    TestMoeDispatch(false);
}

TEST_F(TestCodegenDispatch, TestMoeDispatchSharedExpert) {
    TestMoeDispatch(true);
}
} // namespace Distributed
} // namespace npu::tile_fwk
