/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file schema_parser.h
 * \brief
 */

#pragma once
#ifndef SCHEMA_PARSER_H
#define SCHEMA_PARSER_H

#include <string>
#include <memory>
#include <sstream>
#include <vector>
#include <map>

namespace npu::tile_fwk::schema {

struct SchemaNode : std::vector<std::shared_ptr<SchemaNode>> {
    std::string name;
    SchemaNode(const std::string &name_) : name(name_) {}

    const std::string &GetName() const { return name; }
    std::string &GetName() { return name; }

    std::string Dump() const;

    static std::vector<std::shared_ptr<SchemaNode>> ParseSchema(const std::string &schema);
    static std::vector<std::shared_ptr<SchemaNode>> ParseSchema(const std::vector<std::string> &schemaList);
    static std::map<std::string, std::vector<std::shared_ptr<SchemaNode>>> BuildDict(const std::vector<std::shared_ptr<SchemaNode>> &nodeList);
};

}

#endif//SCHEMA_TRACE_H