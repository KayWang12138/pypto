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
 * \file ast_op_registry.cpp
 * \brief
 */

#include "interface/registry/ast_op_registry.h"

namespace npu::tile_fwk {
void RegisterOpImplToRegistry(const AstOpRegistryImpl *rd) {
    if (rd == nullptr) {
        return;
    }
    auto &keyToFuncs = AstOpImplRegistry::GetInstance().CreateOrGetImpl(rd->opType_.c_str());
    if (!rd->keyToFuncs_.empty()) {
        keyToFuncs = rd->keyToFuncs_;
    }
}

AstOpImplRegistry &AstOpImplRegistry::GetInstance() {
    static AstOpImplRegistry instance;
    return instance;
}

std::map<uint64_t, AstOpImplFunc> &AstOpImplRegistry::CreateOrGetImpl(const char *opType) {
    if (regKeyFuncs_.find(opType) == regKeyFuncs_.end()) {
        std::map<uint64_t, AstOpImplFunc> dftVal;
        regKeyFuncs_[opType] = dftVal;
    }
    return regKeyFuncs_[opType];
}

AstOpImplFunc AstOpImplRegistry::GetOpImplFunc(const std::string &opType, const uint64_t configKey) {
    const auto iter = regKeyFuncs_.find(opType);
    if (iter == regKeyFuncs_.end() || iter->second.find(configKey) == iter->second.end()) {
        return nullptr;
    }
    return iter->second[configKey];
}

std::vector<uint64_t> AstOpImplRegistry::GetAllConfigKeys(const std::string &opType) const {
    auto funcIter = regKeyFuncs_.find(opType);
    std::vector<uint64_t> configKeys;
    if (funcIter != regKeyFuncs_.end()) {
        for (const auto &item : funcIter->second) {
            configKeys.emplace_back(item.first);
        }
    }
    return configKeys;
}

AstOpRegistry::AstOpRegistry(const char *opType) : impl_(new(std::nothrow) AstOpRegistryImpl) {
    if (impl_ == nullptr) {
        return;
    }
    impl_->opType_ = opType;
    (void)AstOpImplRegistry::GetInstance().CreateOrGetImpl(opType);
}

AstOpRegistry &AstOpRegistry::ImplFunc(const std::map<uint64_t, AstOpImplFunc> &keyToFunc) {
    if (impl_ != nullptr) {
        impl_->keyToFuncs_ = keyToFunc;
    }
    return *this;
}

AstOpRegistry::AstOpRegistry(const AstOpRegistry &registerData) {
    RegisterOpImplToRegistry(registerData.impl_.get());
}

AstOpRegistry::AstOpRegistry(AstOpRegistry &&registerData) noexcept {
    RegisterOpImplToRegistry(registerData.impl_.get());
}
}
