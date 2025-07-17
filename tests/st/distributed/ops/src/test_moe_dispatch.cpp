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
#include "interface/program/program.h"
#include "common/data_type.h"

namespace npu::tile_fwk {
namespace Distributed {

void TestMoeDispatch(OpTestParam &testParam) {
    constexpr size_t paramsSize = 6;
    auto [batchSize, hiddenSize, shareNum, expertNum, topK, typeNum]
        = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    DataType dType = GetDataTypeNum(typeNum);

    int32_t expandXRow = batchSize * (shareNum + expertNum);
    int32_t expandXCol = hiddenSize;
    int32_t expandXSize = expandXRow * expandXCol;
    int32_t expandXByteSize = BytesOf(dType) * expandXSize;
    uint8_t* expandXPtr = allocDevAddr(expandXByteSize);

    int32_t validCntRow = 128;
    int32_t validCntCol = 1;
    int32_t validCntSize = validCntRow * validCntCol;
    int32_t validCntByteSize = sizeof(int32_t) * validCntSize;
    uint8_t* validCntPtr = allocDevAddr(validCntByteSize);

    ALOG_INFO_F(
        "before moe dispatch [%d, %d, %d, %d, %d], rankSize=%d, validCntPtr=%p, expandXPtr=%p",
        batchSize, hiddenSize, shareNum, expertNum, topK, testParam.rankSize, validCntPtr, expandXPtr
    );

    using T = npu::tile_fwk::bfloat16;

    PROGRAM("Moe Dispatch") {
        std::string xPath = GetGoldenDir() + "/x_rank_" + std::to_string(testParam.rankId) + ".bin";
        void* tokenTensorPtr = readToDev<T>(xPath, batchSize * hiddenSize);
        std::string expertIdsPath = GetGoldenDir() + "/expert_ids_rank_" + std::to_string(testParam.rankId) + ".bin";
        void* tokenExpertTablePtr = readToDev<int32_t>(expertIdsPath, batchSize * topK);

        Tensor tokenTensor(dType, {batchSize, hiddenSize}, static_cast<uint8_t*>(tokenTensorPtr), "tokenTensor");
        Tensor tokenExpertTable(
            DataType::DT_INT32,
            {batchSize, topK},
            static_cast<uint8_t*>(tokenExpertTablePtr),
            "tokenExpertTable"
        );
        Tensor validCnt(DataType::DT_INT32, {validCntRow * validCntCol}, validCntPtr, "validCnt");
        Tensor expandX(dType, {expandXRow, expandXCol}, expandXPtr, "expandX");

        ConfigManager::Instance();
        FUNCTION("MoeDispatch", FunctionType::STATIC, {tokenTensor, tokenExpertTable, validCnt, expandX}) {
            Program::GetInstance().GetTileShape().SpecifyStaticRankId(testParam.rankId);
            expandX = MoeDispatch(tokenTensor, tokenExpertTable, validCnt, testParam.group);
        }
    }

    EXPECT_TRUE(CompareWithGolden<uint8_t *>(dType, "/y_rank_", expandXSize, expandXPtr, testParam));

    if (testParam.rankId >= shareNum) {
        EXPECT_TRUE(CompareWithGolden<uint8_t *>(
            DataType::DT_INT32,
            "/valid_count_rank_",
            validCntSize,
            validCntPtr,
            testParam
        ));
    }
}
} // namespace Distributed
} // namespace npu::tile_fwk
