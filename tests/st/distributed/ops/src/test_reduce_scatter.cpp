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
 * \file test_reduce_scatter.cpp
 * \brief
 */

#include "distributed_op_test_suite.h"
#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "common/data_type.h"

namespace npu::tile_fwk {
namespace Distributed {

void TestReduceScatter(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [M, N, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    ASSERT(M % testParam.rankSize == 0);

    DataType dType = GetDataTypeNum(typeNum);

    int32_t inSize = M * N;
    int32_t outM = M / testParam.rankSize;
    int32_t outSize = outM * N;
    size_t dTypeSize = BytesOf(dType); 
    int32_t outByteSize = dTypeSize * outSize;
    uint8_t *outPtr = allocDevAddr(outByteSize);
    ALOG_INFO_F("before REDUCESCATTER [%d, %d], rankSize=%d, outPtr=%p", M, N, testParam.rankSize, outPtr);
    PROGRAM("REDUCESCATTER") {
        std::vector<int32_t> inShape = {M, N};
        std::vector<int32_t> outShape = {outM, N};

        void *xPtr = readToDev(GetGoldenDir() + "/input_rank_"+ std::to_string(testParam.rankId) + ".bin", inSize * dTypeSize / sizeof(float));

        Tensor in(dType, inShape,  (uint8_t *)xPtr, "in");
        Tensor out(dType, outShape, outPtr, "out");
        
        ConfigManager::Instance();

        FUNCTION("REDUCESCATTER_F", FunctionType::STATIC, {in, out}) {
            Program::GetInstance().GetTileShape().SetDistTileShapes(
                {outM / 2, 2, 0}, 
                {N / 2, 2, 0}, 
                {1, testParam.rankSize, 0});
            Program::GetInstance().GetTileShape().SpecifyStaticRankId(testParam.rankId);
            out = Distributed::ReduceScatter(in, testParam.group, npu::tile_fwk::Distributed::DistReduceType::DIST_REDUCE_ADD);
        }
    }

    EXPECT_TRUE(CompareWithGolden<uint8_t *>(dType, "/output_rank_", outSize, outPtr, testParam));
}

void TestReduceScatterEx(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [M, N, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    ASSERT(M % testParam.rankSize == 0);

    DataType dType = GetDataTypeNum(typeNum);
    size_t dTypeSize = BytesOf(dType); 

    int32_t outM = M / testParam.rankSize;
    int32_t outSize = outM * N;
    int32_t inSize = outSize;
    int32_t outByteSize = dTypeSize * outSize;
    uint8_t *outPtr = allocDevAddr(outByteSize);
    ALOG_INFO_F("before REDUCESCATTER [%d, %d], rankSize=%d, outPtr=%p", M, N, testParam.rankSize, outPtr);
    PROGRAM("REDUCESCATTEREX") {
        std::vector<int32_t> inShape = {outM, N};
        std::vector<int32_t> outShape = {outM, N};

        void *xPtr = readToDev(GetGoldenDir() + "/input_rank_"+ std::to_string(testParam.rankId) + ".bin", M * N * dTypeSize / sizeof(float));
        Tensor out(dType, outShape, outPtr, "out");

        std::vector<Tensor> inVec;
        for (int32_t i = 0; i < testParam.rankSize; i++) {
            std::string inName = "in" + std::to_string(i);
            Tensor inTensor(dType, inShape, (uint8_t *)xPtr + inSize * dTypeSize * i, inName);
            inVec.push_back(inTensor);
        }
        std::vector<std::reference_wrapper<Tensor>> paras(inVec.begin(), inVec.end());
        paras.emplace_back(out);
        ConfigManager::Instance();
 
        FUNCTION("REDUCESCATTER_EX", FunctionType::STATIC, paras) {
            Program::GetInstance().GetTileShape().SetDistTileShapes(
                {outM / 2, 2, 0}, 
                {N / 2, 2, 0}, 
                {1, testParam.rankSize, 0});
            Program::GetInstance().GetTileShape().SpecifyStaticRankId(testParam.rankId);
            out = Distributed::ReduceScatter(inVec, testParam.group, npu::tile_fwk::Distributed::DistReduceType::DIST_REDUCE_ADD);
        }
    }

    EXPECT_TRUE(CompareWithGolden<uint8_t *>(dType, "/output_rank_", outSize, outPtr, testParam));
}

} // namespace Distributed
} // namespace npu::tile_fwk