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
 * \file test_codegen_moe_combine.cpp
 * \brief Unit test for codegen.
 */

#include <gtest/gtest.h>
#include <vector>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"

using namespace npu::tile_fwk;
using namespace Distributed;

class TestCodegenMoeCombine : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    }

    void TearDown() override
    {
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
    }

protected:
    bool oriEnableAihacBackend = false;
};

void TestMoeCombine()
{
    const char *group = "hcom123";
    int expandBS = 16;
    int bs = 8;
    int h = 7168;
    int topk = 8;
    DataType dType = DT_BF16;

    Tensor in(dType, {expandBS, h}, "in");
    Tensor combineInfo(DT_INT32, {expandBS, 3}, "combineInfo");
    Tensor scale(DT_FP32, {bs, topk}, "scale");
    Tensor out(dType, {bs, h}, "out");

    ConfigManager::Instance();

    FUNCTION("ATTNCombine", FunctionType::STATIC, {in, combineInfo, scale, out}) {
        Program::GetInstance().GetTileShape().SpecifyStaticRankId(
            npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
        out = Distributed::MoeCombine(in, scale, combineInfo, group);
    }
}

TEST_F(TestCodegenMoeCombine, TestMoeCombine)
{
    TestMoeCombine();
}
