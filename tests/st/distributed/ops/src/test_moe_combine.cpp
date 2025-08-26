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
 * \file test_moe_combine.cpp
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

void TestMoeCombine(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 4;
    auto [bs, h, topK, dtype_num]
        = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    int32_t procSize = testParam.rankSize;
    DataType dType = GetDataTypeNum(dtype_num);
    int dtypeSize = BytesOf(dType);

    std::vector<int> inShape = {bs * procSize, h};
    std::vector<int> combineInfoShape = {bs * procSize * (procSize - 1)};
    std::vector<int> scaleShape = {bs, topK}; // 如果scale的shape是[8,4]，UBCopyIn会有问题
    std::vector<int> outShape = {bs, h};

    int inEleNum = inShape[0] * inShape[1];
    int combinInfoEleNum = inEleNum;
    int scaleEleNum = inEleNum;
    int outEleNum = outShape[0] * outShape[1];

    uint64_t inByteSize = inEleNum * dtypeSize;
    uint64_t combinInfoByteSize = combinInfoEleNum * sizeof(int32_t);
    uint64_t scaleByteSize = scaleEleNum * sizeof(float);
    uint64_t outByteSize = outEleNum * dtypeSize;

    uint8_t* outPtr = allocDevAddr(outByteSize);
    std::string dispatchPath =
        GetGoldenDir() + "/DistributedTest.test_dispatch_rank_size_" + std::to_string(testParam.rankSize);
    PROGRAM("PROGRAM of Combine") {
        uint8_t* inPtr = static_cast<uint8_t*>(readToDev(
            dispatchPath + "/y_rank_" + std::to_string(testParam.rankId) + ".bin",
            inByteSize / sizeof(float)
        ));
        uint8_t* combinInfoPtr = static_cast<uint8_t*>(readToDev(
            dispatchPath + "/combine_info_rank_" + std::to_string(testParam.rankId) + ".bin",
            combinInfoByteSize / sizeof(float)
        ));
        uint8_t* scalePtr = static_cast<uint8_t*>(readToDev(
            dispatchPath + "/scale_rank_" + std::to_string(testParam.rankId) + ".bin",
            scaleByteSize / sizeof(float)
        ));
        Tensor in(dType, inShape, inPtr, "in");
        Tensor combineInfo(DataType::DT_INT32, combineInfoShape, combinInfoPtr, "combineInfo");
        Tensor scale(DataType::DT_FP32, scaleShape, scalePtr, "scale");
        Tensor out(dType, outShape, outPtr, "out");

        ConfigManager::Instance();

        FUNCTION("Moe_Combine", FunctionType::STATIC, {in, combineInfo, scale, out}) {
            Program::GetInstance().GetTileShape().SpecifyStaticRankId(testParam.rankId);
            out = Distributed::MoeCombine(in, scale, combineInfo, testParam.group);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(dType, "/y_rank_", outEleNum, outPtr, testParam));
}

} // namespace Distributed
} // namespace tile_fwk