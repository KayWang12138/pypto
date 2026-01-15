/**
 * Copyright (c) 2025 - 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_call.h
 * \brief
 */

#pragma once
#include "ir/program.h"
#include "ir/function.h"
#include "ir/value.h"

#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include <unordered_map>

namespace pto {
using BlockFunctionType = std::shared_ptr<Function> (*) (
const std::vector<TensorValuePtr> &inputArgs,
const std::vector<TensorValuePtr> &outputArgs,
const std::vector<ScalarValuePtr> &indices);

class BlockRegistry {
public:
    static BlockRegistry& GetInstance() {
        static BlockRegistry instance;
        return instance;
    }

    // 注册 block，使用大驼峰函数名作为标识
    void RegisterBlock(const std::string& blockName, std::function<std::shared_ptr<pto::Function>(
        const std::vector<std::shared_ptr<pto::TensorValue>>&,
        const std::vector<std::shared_ptr<pto::TensorValue>>&,
        const std::vector<std::shared_ptr<pto::ScalarValue>>&)> block) {
        registry_[blockName] = std::move(block);
    }

    // 获取 block 函数
    std::function<std::shared_ptr<pto::Function>(
        const std::vector<std::shared_ptr<pto::TensorValue>>&,
        const std::vector<std::shared_ptr<pto::TensorValue>>&,
        const std::vector<std::shared_ptr<pto::ScalarValue>>&)> GetBlock(const std::string& blockName) const {
        auto it = registry_.find(blockName);
        ASSERT(it != registry_.end()) << "Cannot find block function by name" << blockName;
        return it->second;
    }

private:
    BlockRegistry() = default;
    BlockRegistry(const BlockRegistry&) = delete;
    void operator=(const BlockRegistry&) = delete;

    std::unordered_map<std::string, std::function<std::shared_ptr<Function>(
        const std::vector<std::shared_ptr<TensorValue>>&,
        const std::vector<std::shared_ptr<TensorValue>>&,
        const std::vector<std::shared_ptr<ScalarValue>>&)>> registry_;
};

void CallBlock(const std::string &blockName,
    const std::vector<std::reference_wrapper<const npu::tile_fwk::Tensor>> &inputTensorArgs,
    const std::vector<std::reference_wrapper<const npu::tile_fwk::Tensor>> &outputTensorArgs,
    const std::vector<npu::tile_fwk::SymbolicScalar> &indices);
}