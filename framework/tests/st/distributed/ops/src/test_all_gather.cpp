/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_all_gather.cpp
 * \brief
 */

#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"
#include "machine/runtime/distributed/distributed_context.h"
using namespace TileOp;
namespace npu::tile_fwk {
namespace Distributed {

template<typename T>
void TestAllGather(OpTestParam& testParam, std::string& goldenDir)
{
    constexpr size_t paramsSize = 5;
    auto [row, col, typeNum, tileRow, tileCol] = GetParams<paramsSize>(goldenDir + "/params.bin");

    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = row * col * testParam.rankSize;

    Shape shape{row, col};
    Shape outShape{testParam.rankSize * row, col};
    Tensor in(dType, shape, "in");
    Tensor out(dType, outShape, "out");

    std::vector<uint64_t> hcclContexts = DistributedContext::GetCommContextToHost(std::vector<std::string>{testParam.group});
    auto hcclOpParam = (TileOp::CommContext*)hcclContexts[0];
    auto rankNum = hcclOpParam->rankNum;
    int64_t ctxSize = static_cast<int64_t>(sizeof(TileOp::CommContext)) + 
        static_cast<int64_t>(sizeof(uint64_t)) * rankNum * WIN_TYPE_NUM;
    Tensor commTensor(DT_INT8, Shape{1, ctxSize}, "commTensor");
    std::vector<T> inPtr = ReadToVector<T>(goldenDir + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", shape);
    
    Shape shmemDataShape{testParam.rankSize, row, col};
    FUNCTION("ALLGATHER", {in, commTensor}, {out}) {
        TileShape::Current().SetVecTile({tileRow, tileCol});
        Tensor shmemData;
        Tensor shmemSignal;
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemData(commTensor, testParam.group, testParam.rankSize, dType, shmemDataShape, shmemData);
            CreateShmemSignal(commTensor, testParam.group, shmemData, shmemSignal);
        }
        AllGather(in, in, commTensor, testParam.group, shmemData, shmemSignal, out);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inPtr),
        RawTensorData::CreateTensor(DT_INT8, Shape{1, ctxSize}, (uint8_t *)(hcclOpParam))
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(out)
    });

    RunTest();
    auto outPtr = ProgramData::GetInstance().GetOutputData(0)->GetDevPtr();
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, outPtr, testParam));

}

template void TestAllGather<int32_t>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllGather<float>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllGather<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllGather<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

} // namespace Distributed
} // namespace npu::tile_fwk