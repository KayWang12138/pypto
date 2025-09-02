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
 * \file json_node_parser.cpp
 * \brief
 */

#include "passes/pass_config/json_node_paser.h"
#include "interface/utils/file_utils.h"
#include "interface/utils/log.h"
namespace npu {
namespace tile_fwk {
Status JsonNodeParser::Initialize(const std::string &jsonPath) {
    ALOG_INFO_F("Start to parse op_json_file %s.", jsonPath.c_str());
    if (!ReadJsonFile(jsonPath, json_)) {
        ALOG_ERROR_F("ReadJsonFile failed.");
        return FAILED;
    }
    originJson_ = json_;
    return SUCCESS;
}

const nlohmann::json *JsonNodeParser::GetJsonInnerNode(const nlohmann::json &root, const std::vector<std::string> &keys) {
    auto *curr = &root;
    for (auto &&key : keys) {
        auto it = curr->find(key);
        if (it == curr->end()) {
            return nullptr;
        }
        curr = &*it;
    }
    return curr;
}

const nlohmann::json *JsonNodeParser::GetRootNode() const {
    return &json_;
}
}  // namespace tile_fwk
}  // namespace npu