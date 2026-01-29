/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "ir/program.h"
#include <memory>
#include <string>

// Forward declaration from interface layer
namespace npu::tile_fwk {
    struct LeafFuncAttribute;
}

namespace pto {

/**
 * @brief Temporary stub for Function
 * This is a minimal placeholder to maintain compilation compatibility
 * Will be replaced with actual implementation from new IR
 */
class Function {
public:
    Function() = default;
    explicit Function(std::string name) : name_(std::move(name)) {}
    virtual ~Function() = default;

    const std::string& GetName() const { return name_; }
    void SetName(const std::string& name) { name_ = name; }

    // Stub method for compatibility
    virtual std::shared_ptr<npu::tile_fwk::LeafFuncAttribute> GetLeafFuncAttribute() const {
        return nullptr;
    }

private:
    std::string name_;
};

/**
 * @brief Temporary stub for BlockFunction
 * This is a minimal placeholder to maintain compilation compatibility
 * Will be replaced with actual implementation from new IR
 */
class BlockFunction : public Function {
public:
    BlockFunction() = default;
    explicit BlockFunction(std::string name) : Function(std::move(name)) {}

    // Stub method for compatibility
    std::shared_ptr<npu::tile_fwk::LeafFuncAttribute> GetLeafFuncAttribute() const override {
        return leafAttr_;
    }

    void SetLeafFuncAttribute(std::shared_ptr<npu::tile_fwk::LeafFuncAttribute> attr) {
        leafAttr_ = attr;
    }

    // Stub method for program ID - returns 0 as default
    int GetID() const { return id_; }
    void SetID(int id) { id_ = id; }

private:
    std::shared_ptr<npu::tile_fwk::LeafFuncAttribute> leafAttr_;
    int id_ = 0;
};

} // namespace pto
