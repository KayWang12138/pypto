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

#include "interface/registry/tile_fwk_op_registry.h"

namespace npu::tile_fwk {
TileFwkOpRegister::TileFwkOpRegister(const std::string &opType) : opType_(opType) {}

TileFwkOpRegister::TileFwkOpRegister(const TileFwkOpRegister &registerData) {
    this->opType_ = registerData.opType_;
    this->implFuncMap_ = registerData.implFuncMap_;
}

TileFwkOpRegister::~TileFwkOpRegister() {
    opType_.clear();
    implFuncMap_.clear();
}

void TileFwkOpRegister::AddImplFunc(const std::map<uint64_t, TileFwkOpImplFunc> &implFuncMap) {
    if (!implFuncMap.empty()) {
        implFuncMap_.insert(implFuncMap.begin(), implFuncMap.end());
    }
}

void TileFwkOpRegister::AddImplFunc(const uint64_t configKey, const TileFwkOpImplFunc implFunc) {
    if (implFunc == nullptr) {
        return;
    }
    implFuncMap_.emplace(configKey, implFunc);
}

std::vector<uint64_t> TileFwkOpRegister::GetAllConfigKeys() const {
    std::vector<uint64_t> configKeys;
    for (const auto &item : implFuncMap_) {
        configKeys.emplace_back(item.first);
    }
    return configKeys;
}

TileFwkOpImplFunc TileFwkOpRegister::GetOpImplFunc(const uint64_t configKey) const {
    auto iter = implFuncMap_.find(configKey);
    return iter == implFuncMap_.end() ? nullptr : iter->second;
}

TileFwkOpRegistry &TileFwkOpRegistry::GetInstance() {
    static TileFwkOpRegistry instance;
    return instance;
}

TileFwkOpRegisterPtr TileFwkOpRegistry::CreateOrGetOpRegister(const std::string &opType) {
    auto iter = opRegisterMap_.find(opType);
    if (iter == opRegisterMap_.end()) {
        TileFwkOpRegisterPtr opRegister = std::make_shared<TileFwkOpRegister>(opType);
        opRegisterMap_.emplace(opType, opRegister);
        return opRegister;
    } else {
        return iter->second;
    }
}

TileFwkOpImplFunc TileFwkOpRegistry::GetOpImplFunc(const std::string &opType, const uint64_t configKey) const {
    auto iter = opRegisterMap_.find(opType);
    if (iter == opRegisterMap_.end()) {
        return nullptr;
    }
    return iter->second->GetOpImplFunc(configKey);
}

std::vector<uint64_t> TileFwkOpRegistry::GetAllConfigKeys(const std::string &opType) const {
    std::vector<uint64_t> configKeys;
    auto iter = opRegisterMap_.find(opType);
    if (iter != opRegisterMap_.end()) {
        configKeys = iter->second->GetAllConfigKeys();
    }
    return configKeys;
}

TileFwkOpRegistHelper::TileFwkOpRegistHelper(const std::string &opType) {
    opRegister_ = TileFwkOpRegistry::GetInstance().CreateOrGetOpRegister(opType);
}

TileFwkOpRegistHelper::~TileFwkOpRegistHelper() {}

TileFwkOpRegistHelper &TileFwkOpRegistHelper::ImplFunc(const std::map<uint64_t, TileFwkOpImplFunc> &implFuncMap) {
    if (opRegister_ != nullptr) {
        opRegister_->AddImplFunc(implFuncMap);
    }
    return *this;
}
TileFwkOpRegistHelper &TileFwkOpRegistHelper::ImplFunc(const uint64_t configKey, const TileFwkOpImplFunc implFunc) {
    if (opRegister_ != nullptr) {
        opRegister_->AddImplFunc(configKey, implFunc);
    }
    return *this;
}
}
