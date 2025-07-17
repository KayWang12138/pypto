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
 * \file ast_op_registry.h
 * \brief
 */

#pragma once

#include <map>
#include <vector>
#include "tilefwk/tensor.h"

namespace npu::tile_fwk {
using AstOpImplFunc = void (*)(uint64_t);
class AstOpImplRegistry {
public:
    static AstOpImplRegistry &GetInstance();
    std::map<uint64_t, AstOpImplFunc> &CreateOrGetImpl(const char *opType);
    AstOpImplFunc GetOpImplFunc(const std::string &opType, const uint64_t configKey);
    std::vector<uint64_t> GetAllConfigKeys(const std::string &opType) const;
private:
    AstOpImplRegistry() {}
    ~AstOpImplRegistry() {
        regKeyFuncs_.clear();
    }
    std::map<std::string, std::map<uint64_t, AstOpImplFunc>> regKeyFuncs_;
};

class AstOpRegistryImpl {
public:
    std::string opType_;
    std::map<uint64_t, AstOpImplFunc> keyToFuncs_;
};

class AstOpRegistry {
public:
    explicit AstOpRegistry(const char *opType);
    AstOpRegistry(AstOpRegistry &&registerData) noexcept;
    AstOpRegistry(const AstOpRegistry &registerData);
    AstOpRegistry &operator=(const AstOpRegistry &) = delete;
    AstOpRegistry &operator=(AstOpRegistry &&) = delete;
    ~AstOpRegistry() = default;
    AstOpRegistry &ImplFunc(const std::map<uint64_t, AstOpImplFunc> &keyToFunc);
private:
    std::unique_ptr<AstOpRegistryImpl> impl_;
};
}

#define VAR_UNUSED __attribute__((unused))

#define REGISTER_OP_COUNTER(opType, name, counter) \
  static npu::tile_fwk::AstOpRegistry VAR_UNUSED name##counter = npu::tile_fwk::AstOpRegistry(#opType)
#define REGISTER_OP_COUNTER_NUMBER(opType, name, counter) REGISTER_OP_COUNTER(opType, name, counter)
#define REGISTER_OP(opType) REGISTER_OP_COUNTER_NUMBER(opType, op_impl_reg_##opType, __COUNTER__)
