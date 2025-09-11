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
 * \file config.h
 * \brief
 */

#pragma once
#include "tilefwk/config.h"
#include "tilefwk/function.h"

#include <variant>
#include <nlohmann/json.hpp>

#include "interface/configs/config_storage.h"


namespace npu::tile_fwk {

void SetOptionOverlay(const std::string &key, bool value);
void SetOptionOverlay(const std::string &key, int64_t value);
void SetOptionOverlay(const std::string &key, const std::string &value);
void SetOptionOverlay(const std::string &key, std::vector<int64_t> &value);

FunctionType GetFunctionType();

bool GetOptionInner(const std::string &key, bool &value);
bool GetOptionInner(const std::string &key, int64_t &value);
bool GetOptionInner(const std::string &key, std::string &value);
bool GetOptionInner(const std::string &key, std::vector<int64_t> &value);

template<typename T>
static T GetOption(const std::string &key, T &&defaultValue) {
    T ret;
    return GetOptionInner(key, ret) ? ret : defaultValue;
}

} // end npu::tile_fwk
