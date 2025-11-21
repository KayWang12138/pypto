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
 * \file test_all_gather.cpp
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

void TestAllGather(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [M, N, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    DataType dType = GetDataTypeNum(typeNum);
    size_t dTypeSize = BytesOf(dType);

    int32_t inSize = M * N;
    int32_t outM = M * testParam.rankSize;
    int32_t outSize = outM * N;
    int32_t outByteSize = dTypeSize * outSize;
    uint8_t *outPtr = allocDevAddr(outByteSize);
    ALOG_INFO_F("before ALL_GATHER [%d, %d], rankSize=%d, outPtr=%p", M, N, testParam.rankSize, outPtr);
    PROGRAM("ALL_GATHER") {
        std::vector<int64_t> inShape = {M, N};
        std::vector<int64_t> outShape = {outM, N};

        void *xPtr = readToDev(GetGoldenDir() + "/input_rank_"+ std::to_string(testParam.rankId) + ".bin",
            inSize * dTypeSize /sizeof(float));

        Tensor in(dType, inShape, (uint8_t *)xPtr, "in");
        Tensor out(dType, outShape, outPtr, "out");

        config::SetBuildStatic(true);
        FUNCTION("AllGather", {in, out}) {
            TileShape::Current().SetDistTile({outM / testParam.rankSize, testParam.rankSize, 0},
                {N / testParam.rankSize, testParam.rankSize, 0}, {1, testParam.rankSize, 0});
            TileShape::Current().SetDistRankId(testParam.rankId);
            out = AllGather(in, testParam.group);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(dType, "/output_rank_", outSize, outPtr, testParam));
}

void TestAllGatherEx(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [M, N, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    DataType dType = GetDataTypeNum(typeNum);
    size_t dTypeSize = BytesOf(dType);

    int32_t size = M * N;
    int32_t outByteSize = dTypeSize * size;
    std::vector<uint8_t *> outPtrs;

    PROGRAM("ALL_GATHER_EX") {
        std::vector<int64_t> shape = {M, N};

        void *xPtr = readToDev(GetGoldenDir() + "/input_rank_"+ std::to_string(testParam.rankId) + ".bin",
            outByteSize /sizeof(float));

        Tensor in(dType, shape, (uint8_t *)xPtr, "in");
        std::vector<Tensor> outs;
        for (int32_t i = 0; i < testParam.rankSize; ++i) {
            std::string tensorName = "out" + std::to_string(i);
            uint8_t *outPtr = allocDevAddr(outByteSize);
            outs.push_back(Tensor(dType, shape, (uint8_t *)outPtr, tensorName));
            outPtrs.emplace_back(outPtr);
        }
        std::vector<std::reference_wrapper<Tensor>> tensorParams(outs.begin(), outs.end());
        tensorParams.emplace_back(in);

        config::SetBuildStatic(true);
        FUNCTION("AllGather_Ex", tensorParams) {
            TileShape::Current().SetDistTile({M / 2, 2, 0}, {N / 2, 2, 0}, {1, testParam.rankSize, 0});
            TileShape::Current().SetDistRankId(testParam.rankId);
            Distributed::AllGather(in, outs, testParam.group);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    EXPECT_TRUE(outPtrs.size() > 0);
    EXPECT_TRUE(CompareWithGolden<std::vector<uint8_t *>>(dType, "/output_rank_", size * testParam.rankSize, outPtrs,
        testParam));
}

template<typename T>
void TestDynAllGather(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [M, N, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = M * N * testParam.rankSize;

    Shape shape{M, N};
    Shape outShape{testParam.rankSize * M, N};
    Tensor in(dType, shape, "in");
    Tensor barrierDummy(DT_INT32, {1, 1}, "barrierDummy");
    Tensor out(dType, outShape, "out");

    std::vector<T> inPtr = ReadToVector<T>(GetGoldenDir() + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", shape);

    int32_t tileNum1 = 8;
    int32_t tileNum2 = 8;
    FUNCTION("ALLGATHER", {in, barrierDummy}, {out}) {
        TileShape::Current().SetDistTile(
            {M / tileNum1, tileNum1, M % tileNum1},
            {N / tileNum2, tileNum2, N % tileNum2},
            {1, testParam.rankSize, 0});
        ShmemAllGather(in, barrierDummy, testParam.group, out);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inPtr),
        RawTensorData::CreateTensorZero(barrierDummy)
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(out)
    });

    auto funcOp = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto hcclContext = GetHcclContext({std::string(testParam.group)});
    DeviceLauncherConfig config;
    config.runModel = false;
    config.hcclContext = hcclContext;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    auto outPut = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/output_rank_", outSize, outPut->GetDevPtr(), testParam));
}
template void TestDynAllGather<int32_t>(OpTestParam &testParam);
template void TestDynAllGather<float>(OpTestParam &testParam);
template void TestDynAllGather<float16>(OpTestParam &testParam);
template void TestDynAllGather<bfloat16>(OpTestParam &testParam);

} // namespace Distributed
} // namespace npu::tile_fwk
