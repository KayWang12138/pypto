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
 * \file test_codegen_allgather.cpp
 * \brief Unit test for codegen.
 */

#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk::Distributed {

class TestDistributedAllGather : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}
};

TEST_F(TestDistributedAllGather, TestAllGatherDyn)
{
    const char *group = "hcom123";

    Program::GetInstance().GetTileShape().SetDistTileShapes({16, 1, 0}, {32, 1, 0}, {2, 1, 0});
    Program::GetInstance().GetTileShape().SpecifyStaticRankId(0);

    Tensor in(DT_FP16, {16, 32}, "in");
    Tensor out(DT_FP16, {32, 32}, "out");
    AllGatherDyn(in, group, out);

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "L0" + SUB_FUNC_SUFFIX);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
}
