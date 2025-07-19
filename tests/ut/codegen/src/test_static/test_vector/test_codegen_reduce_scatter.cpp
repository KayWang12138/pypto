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
 * \file test_codegen_reduce_scatter.cpp
 * \brief Unit test for codegen.
 */

#include <gtest/gtest.h>
#include <vector>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"

using namespace npu::tile_fwk;
using namespace Distributed;

class TestCodegenReduceScatter : public ::testing::Test {
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

    void TearDown() override { config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend); }

protected:
    bool oriEnableAihacBackend = false;
};

void TestReduceScatter()
{
    const char *group = "hcom123";
    int32_t m = 16;
    int32_t n = 128;
    DataType dType = DT_FP16;

    std::vector<int32_t> shape = {m, n};

    std::vector<Tensor> in;
    Tensor in1(dType, shape, "in1");
    Tensor in2(dType, shape, "in2");
    in.push_back(in1);
    in.push_back(in2);
    Tensor out(dType, shape, "out");
    ConfigManager::Instance();

    FUNCTION("REDUCESCATTER_F", FunctionType::STATIC, {in[0], in[1], out}) {
        // 为了适配 kernel 代码，这边切分改成 1，线上代码可以直接运行
        Program::GetInstance().GetTileShape().SetDistTileShapes({m / 2, 2, 0}, {n, 1, 0}, {2, 1, 0});
        Program::GetInstance().GetTileShape().SpecifyStaticRankId(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
        out = Distributed::ReduceScatter(in, group, npu::tile_fwk::Distributed::DistReduceType::DIST_REDUCE_ADD);
    }
}

TEST_F(TestCodegenReduceScatter, TestReduceScatter)
{
    TestReduceScatter();
}

void TestReduceScatterOneTensor()
{
    const char *group = "hcom123";
    int32_t m = 16;
    int32_t n = 128;
    DataType dType = DT_FP16;

    std::vector<int32_t> inShape = {m, n};
    int32_t rankSize = 2;
    std::vector<int32_t> outShape = {m / rankSize, n};

    Tensor in(dType, inShape, "in");
    Tensor out(dType, outShape, "out");
    ConfigManager::Instance();

    FUNCTION("REDUCESCATTER_F", FunctionType::STATIC, {in, out}) {
        // 为了适配 kernel 代码，这边切分改成 1，线上代码可以直接运行
        Program::GetInstance().GetTileShape().SetDistTileShapes({m / 2, 2, 0}, {m, 1, 0}, {rankSize, 1, 0});
        Program::GetInstance().GetTileShape().SpecifyStaticRankId(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
        out = Distributed::ReduceScatter(in, group, npu::tile_fwk::Distributed::DistReduceType::DIST_REDUCE_ADD);
    }
}

TEST_F(TestCodegenReduceScatter, TestReduceScatterOneTensor)
{
    TestReduceScatterOneTensor();
}
