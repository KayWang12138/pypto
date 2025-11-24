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
 * \file json_node_parser.h
 * \brief
 */
#ifndef JSON_NODE_PARSER_H_
#define JSON_NODE_PARSER_H_
#include <nlohmann/json.hpp>
#include "interface/utils/common.h"
namespace npu {
namespace tile_fwk {
class JsonNodeParser {
  public:
    JsonNodeParser() = default;
    ~JsonNodeParser() = default;
    Status Initialize(const std::string &jsonPath); 
    const nlohmann::json *GetJsonInnerNode(const nlohmann::json &root, const std::vector<std::string> &keys);
    const nlohmann::json *GetRootNode() const;
  private:
    nlohmann::json json_;
    nlohmann::json originJson_;
};
} // namespace tile_fwk
} // namepsace npu 
#endif