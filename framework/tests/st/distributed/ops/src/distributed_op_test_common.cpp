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
 * \file distributed_op_test_common.cpp
 * \brief
 */

#include "distributed_op_test_common.h"

namespace npu::tile_fwk {
namespace Distributed {
int64_t GetEleNumFromShape(std::vector<int64_t>& shape)
{
    int64_t eleNum = 1;
    for (int num : shape) {
        eleNum *= num;
    }
    return eleNum;
}

Tensor CreateTensorFromFile(std::vector<int64_t>& shape, DataType dtype, std::string& file, std::string tname)
{
    int eleNum = GetEleNumFromShape(shape);
    uint64_t byteSize = eleNum * BytesOf(dtype);
    uint8_t* ptr = (uint8_t*)readToDev(GetGoldenDir() + file, byteSize / sizeof(float));
    Tensor tensor(dtype, shape, ptr, tname);
    return tensor;
}

std::string GetGoldenDirPath(const nlohmann::json& testData)
{
    std::filesystem::path goldenDir = GetGoldenDir();
    goldenDir = goldenDir.parent_path().parent_path();
    std::string operation = testData["operation"].get<std::string>();
    int32_t caseIndex = testData["case_index"].get<int32_t>();
    std::string caseName = testData["case_name"].get<std::string>();
    goldenDir /= operation;
    goldenDir /= std::to_string(caseIndex) + "_" + caseName;
    return goldenDir.string();
}

} // namespace Distributed
} // namespace npu::tile_fwk