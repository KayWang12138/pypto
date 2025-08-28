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
 * \file test_codegen_shmem_tileop.cpp
 * \brief Unit test for codegen.
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/inner/tilefwk.h"
#include "codegen/codegen.h"
#include "tilefwk/tilefwk.h"

namespace npu::tile_fwk::Distributed {

class TestCodegenShmemTileop : public ::testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
};

Tensor PutMem(const Tensor& in, const Tensor& out)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto dummy = std::make_shared<LogicalTensor>(function, in.GetDataType(), std::vector<int64_t>{1});
    auto& op= function.AddOperation("SHMEM_PUT", {in.GetStorage(), out.GetStorage()}, {dummy});
    op.SetAttr("AtomicType", std::string("TileOp::Distributed::AtomicType::SET"));
    return dummy;
}

void SetSignal(const Tensor& dummy, const Tensor& signal)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto& op= function.AddOperation("SHMEM_SIGNAL", {dummy.GetStorage()}, {signal.GetStorage()});
    std::string value = "1";
    op.SetAttr("Value", value);
    op.SetAttr("AtomicType", std::string("TileOp::Distributed::AtomicType::SET"));
}

Tensor WaitUntil(const Tensor& dummyIn, const Tensor& signal)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto dummyOut = std::make_shared<LogicalTensor>(function, dummyIn.GetDataType(), std::vector<int64_t>{1});
    auto& op= function.AddOperation("SHMEM_WAIT_UNTIL", {dummyIn.GetStorage(), signal.GetStorage()}, {dummyOut});
    op.SetAttr("AtomicType", std::string("TileOp::Distributed::AtomicType::SET"));
    return dummyOut;
}

Tensor GetMem(const Tensor& dummy, const Tensor& in)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, in.GetDataType(), in.GetShape());
    auto& op= function.AddOperation("SHMEM_GET", {dummy.GetStorage(), in.GetStorage()}, {out});
    op.SetAttr("AtomicType", std::string("TileOp::Distributed::AtomicType::SET"));
    return out;
}

TEST_F(TestCodegenShmemTileop, Success)
{
    constexpr int32_t row = 256;
    constexpr int32_t col = 256;
    DataType dtype = DT_FP16;

    const std::vector<int64_t> shape = {row, col};
    Tensor in(dtype, shape, "in");
    Tensor temp(dtype, shape, "temp");
    Tensor out(dtype, shape, "out");

    ConfigManager::Instance();
    std::string funcName = "ShmemTileop";
    FUNCTION(funcName, FunctionType::STATIC, {in, out}) {
        Tensor dummy = PutMem(in, temp);
        out = GetMem(dummy, temp);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

} // namespace Distributed::npu::tile_fwk
