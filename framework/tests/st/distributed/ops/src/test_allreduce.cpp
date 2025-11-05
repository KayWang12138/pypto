
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_allreduce.cpp
 * \brief
 */

#include "distributed_op_test_suite.h"
#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk {
namespace Distributed {

void TestShmemAddAndAllReduce(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [row, col, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = row * col;

    Shape shape{row, col};
    Tensor in(dType, shape, "in");
    Tensor out(dType, shape, "out");

    std::vector<int32_t> ipPtr = ReadToVector<int32_t>(
        GetGoldenDir() + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", {row, col});

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(in, ipPtr),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(out),
    });
    int32_t tileNum = 4;
    FUNCTION("ADD and ALLREDUCE", {in}, {out}) {
        TileShape::Current().SetDistTile(
            {row / tileNum / testParam.rankSize, tileNum, (row / tileNum) % tileNum},
            {col / tileNum, tileNum, col % tileNum},
            {1, testParam.rankSize, 0});
        ShmemAddAllReduce(in, testParam.group, out);
    }
    auto funcOp = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto hcclContext = GetHcclContext({std::string(testParam.group)});
    DeviceLauncherConfig config;
    config.runModel = false;
    config.hcclContext = hcclContext;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

} // namespace Distributed
} // namespace npu::tile_fwk