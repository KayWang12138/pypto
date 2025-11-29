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
 * \file test_moe_dispatch.cpp
 * \brief
 */

#include "distributed_op_test_suite.h"
#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk::Distributed {

void TestMoeDispatch(OpTestParam& testParam)
{
    constexpr size_t paramsSize = 5;
    auto [batchSize, hiddenSize, shareNum, topK, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    DataType dType = GetDataTypeNum(typeNum);
    int64_t dtypeSize = BytesOf(dType);

    Shape tokenTensorShape{batchSize, hiddenSize};
    Shape tokenExpertTableShape{batchSize, topK};
    Shape expandXShape{batchSize * testParam.rankSize, hiddenSize};
    Shape validCntShape{128};

    int64_t tokenTensorEleNum = tokenTensorShape[0] * tokenTensorShape[1];
    int64_t tokenExpertTableEleNum = tokenExpertTableShape[0] * tokenExpertTableShape[1];
    int64_t expandXEleNum = expandXShape[0] * expandXShape[1];
    int64_t validCntEleNum = validCntShape[0];

    int64_t tokenTensorByteSize = tokenTensorEleNum * dtypeSize;
    int64_t tokenExpertTableByteSize = tokenExpertTableEleNum * sizeof(int32_t);
    int64_t expandXByteSize = expandXEleNum * dtypeSize;
    int64_t validCntByteSize = validCntEleNum * sizeof(int32_t);

    uint8_t* expandXPtr = allocDevAddr(expandXByteSize);
    uint8_t* validCntPtr = allocDevAddr(validCntByteSize);

    PROGRAM("Moe Dispatch") {
        uint8_t* tokenTensorPtr = static_cast<uint8_t*>(readToDev(
            GetGoldenDir() + "/x_rank_" + std::to_string(testParam.rankId) + ".bin",
            tokenTensorByteSize / sizeof(float)
        ));
        uint8_t* tokenExpertTablePtr = static_cast<uint8_t*>(readToDev(
            GetGoldenDir() + "/expert_ids_rank_" + std::to_string(testParam.rankId) + ".bin",
            tokenExpertTableByteSize
        ));

        Tensor tokenTensor(dType, tokenTensorShape, tokenTensorPtr, "tokenTensor");
        Tensor tokenExpertTable(DataType::DT_INT32, tokenExpertTableShape, tokenExpertTablePtr, "tokenExpertTable");
        Tensor expandX(dType, expandXShape, expandXPtr, "expandX");
        Tensor validCnt(DataType::DT_INT32, validCntShape, validCntPtr, "validCnt");

        config::SetBuildStatic(true);
        FUNCTION("MoeDispatch", {tokenTensor, tokenExpertTable, validCnt, expandX}) {
            TileShape::Current().SetDistRankId(testParam.rankId);
            expandX = MoeDispatch(tokenTensor, tokenExpertTable, validCnt, testParam.group);
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(dType, "/y_rank_", expandXEleNum, expandXPtr, testParam));
    if (testParam.rankId >= shareNum) {
        EXPECT_TRUE(CompareWithGolden<uint8_t *>(DataType::DT_INT32, "/valid_count_rank_", validCntEleNum, validCntPtr,
            testParam));
    }
}

} // namespace npu::tile_fwk::Distributed
