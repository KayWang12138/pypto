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
 * \file execute_function.h
 * \brief
 */

#pragma once

#include "interface/function/function.h"

namespace npu::tile_fwk {


// ExecuteFunction is the dedicated subtype for EXECUTE_GRAPH graphs.
// It contains members and methods specific to excecute (execute graph) functions.
class ExecuteFunction : public Function {
public:
    ExecuteFunction(const Program &belongTo, const std::string &funcMagicName,
        const std::string &funcRawName, Function *parentFunc);

    ~ExecuteFunction() override = default;
    ExecuteFunction(const ExecuteFunction &other) = delete;
    ExecuteFunction(ExecuteFunction &&other) = delete;
    ExecuteFunction &operator=(const ExecuteFunction &other) = delete;
    ExecuteFunction &operator=(ExecuteFunction &&other) = delete;

    void DumpTopoFile(const std::string &fileName) const override;
};
} // namespace npu::tile_fwk