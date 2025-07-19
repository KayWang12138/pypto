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
 * \file test_memory_attribute.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"
#include "common/data_type.h"

TEST(TestMemoryAttribute, MemorySizeLimitTest) {
    npu::tile_fwk::Program::GetInstance().GetPlatformConfig().SetMemoryLimitList(static_cast<npu::tile_fwk::ModelID>(0));

    EXPECT_EQ(npu::tile_fwk::Program::GetInstance().GetPlatformConfig().GetMemoryLimit(npu::tile_fwk::MEM_UB), npu::tile_fwk::KB(192));
    EXPECT_EQ(npu::tile_fwk::Program::GetInstance().GetPlatformConfig().GetMemoryLimit(npu::tile_fwk::MEM_L0C), npu::tile_fwk::KB(128));

    npu::tile_fwk::Program::GetInstance().GetPlatformConfig().SetMemoryLimitList(static_cast<npu::tile_fwk::ModelID>(1));

    EXPECT_EQ(npu::tile_fwk::Program::GetInstance().GetPlatformConfig().GetMemoryLimit(npu::tile_fwk::MEM_UB), npu::tile_fwk::KB(192));
    EXPECT_EQ(npu::tile_fwk::Program::GetInstance().GetPlatformConfig().GetMemoryLimit(npu::tile_fwk::MEM_L2), npu::tile_fwk::MB(192));

    npu::tile_fwk::Program::GetInstance().GetConfig().Reset();
}

TEST(TestMemoryAttribute, MemorySizeTest) {
    std::vector<std::vector<int>> tshapes = {
        {   1,    1},
        {   1,   10},
        {   5,   10},
        {  32,   64},
        { 100,  100},
        { 255,  255},
        { 256, 1024},
        {1024, 2048}
    };
    std::vector<npu::tile_fwk::DataType> dtypes = {npu::tile_fwk::DT_INT4, npu::tile_fwk::DT_INT8, npu::tile_fwk::DT_INT16,
        npu::tile_fwk::DT_INT32, npu::tile_fwk::DT_FP8, npu::tile_fwk::DT_FP16, npu::tile_fwk::DT_FP32, npu::tile_fwk::DT_BF16, npu::tile_fwk::DT_HF8,
        npu::tile_fwk::DT_HF4};
    for (auto &tshape : tshapes) {
        for (auto dt : dtypes) {
            npu::tile_fwk::Tensor A(dt, tshape, "A_" + DataType2String(dt));
            A.GetStorage()->SetMemoryTypeToBe(npu::tile_fwk::MEM_UB);
            EXPECT_EQ(A->MemorySize(), (tshape[0] * tshape[1] * BytesOf(dt) + 31) / 32 * 32);

            A.GetStorage()->SetMemoryTypeToBe(npu::tile_fwk::MEM_L2);
            EXPECT_EQ(A->MemorySize(), tshape[0] * tshape[1] * BytesOf(dt));
        }
    }
}
