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
 * \file tile_fwk_op_registry.h
 * \brief
 */

#pragma once

#include <map>
#include <vector>
#include "tilefwk/tensor.h"

namespace npu::tile_fwk {
using TileFwkOpImplFunc = void (*)(uint64_t);
class TileFwkOpRegister {
public:
    explicit TileFwkOpRegister(const std::string &opType);
    TileFwkOpRegister(const TileFwkOpRegister &registerData);
    TileFwkOpRegister &operator=(const TileFwkOpRegister &) = delete;
    TileFwkOpRegister &operator=(TileFwkOpRegister &&) = delete;
    ~TileFwkOpRegister();
    void AddImplFunc(const std::map<uint64_t, TileFwkOpImplFunc> &implFuncMap);
    void AddImplFunc(const uint64_t configKey, const TileFwkOpImplFunc implFunc);
    std::vector<uint64_t> GetAllConfigKeys() const;
    TileFwkOpImplFunc GetOpImplFunc(const uint64_t configKey) const;
private:
    std::string opType_;
    std::map<uint64_t, TileFwkOpImplFunc> implFuncMap_;
};
using TileFwkOpRegisterPtr = std::shared_ptr<TileFwkOpRegister>;

class TileFwkOpRegistry {
public:
    static TileFwkOpRegistry &GetInstance();
    TileFwkOpRegisterPtr CreateOrGetOpRegister(const std::string &opType);
    TileFwkOpImplFunc GetOpImplFunc(const std::string &opType, const uint64_t configKey) const;
    std::vector<uint64_t> GetAllConfigKeys(const std::string &opType) const;
private:
    TileFwkOpRegistry() {}
    ~TileFwkOpRegistry() {
        opRegisterMap_.clear();
    }
    std::map<std::string, TileFwkOpRegisterPtr> opRegisterMap_;
};

class TileFwkOpRegistHelper {
public:
    explicit TileFwkOpRegistHelper(const std::string &opType);
    ~TileFwkOpRegistHelper();
    TileFwkOpRegistHelper &ImplFunc(const std::map<uint64_t, TileFwkOpImplFunc> &implFuncMap);
    TileFwkOpRegistHelper &ImplFunc(const uint64_t configKey, const TileFwkOpImplFunc keyToFunc);
private:
    TileFwkOpRegisterPtr opRegister_;
};
}

#define VAR_UNUSED __attribute__((unused))
#define REGISTER_OP_COUNTER(opType, name, counter) \
  static npu::tile_fwk::TileFwkOpRegistHelper VAR_UNUSED name##counter = npu::tile_fwk::TileFwkOpRegistHelper(#opType)
#define REGISTER_OP_COUNTER_NUMBER(opType, name, counter) REGISTER_OP_COUNTER(opType, name, counter)
#define REGISTER_OP(opType) REGISTER_OP_COUNTER_NUMBER(opType, op_impl_reg_##opType, __COUNTER__)
