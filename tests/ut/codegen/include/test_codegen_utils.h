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
 * \file test_codegen_utils.h
 * \brief Unit test for codegen.
 */

#ifndef TEST_CODEGEN_UTILS_H
#define TEST_CODEGEN_UTILS_H

#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {
const constexpr int DummyFuncMagic = 1;
struct LogicalTensorInfo {
    LogicalTensorInfo(Function &func, DataType dataType, MemoryType memoryType, const std::vector<int> &tShape)
        : function(func), dType(dataType), memType(memoryType), shape(tShape){};
    LogicalTensorInfo(
        Function &func, DataType dataType, MemoryType memoryType, const std::vector<int> &tShape, std::string tName)
        : function(func), dType(dataType), memType(memoryType), shape(tShape), tensorName(std::move(tName)){};

    Function &function;
    DataType dType;
    MemoryType memType;
    const std::vector<int> &shape;
    const std::string tensorName;
};

std::shared_ptr<LogicalTensor> CreateLogicalTensor(const LogicalTensorInfo &info);

std::string GetResultFromCpp(const Function &function);

} // namespace npu::tile_fwk

#endif // TEST_CODEGEN_UTILS_H