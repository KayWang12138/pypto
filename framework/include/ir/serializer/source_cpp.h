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
 * \file serializer_base.h
 * \brief
 */

#pragma once

#include "ir/function.h"
#include "ir/program.h"

#include "serializer_base.h"

namespace pto {
namespace serializer {

#define IR_SOURCE_CPP_PREFIX                    "RT_"
#define IR_SOURCE_CPP_FUNCTION                  "FUNCTION"
#define IR_SOURCE_CPP_OPERATION                 "OPERATION"
#define IR_SOURCE_CPP_DECL_TILE_TYPE            "DECL_TILE_TYPE"
#define IR_SOURCE_CPP_DECL_TENSOR_TYPE          "DECL_TENSOR_TYPE"
#define IR_SOURCE_CPP_DECL_ADDR                 "DECL_ADDR"
#define IR_SOURCE_CPP_DECL_TILE                 "DECL_TILE"
#define IR_SOURCE_CPP_DECL_TENSOR               "DECL_TENSOR"
#define IR_SOURCE_CPP_DECL_SCALAR               "DECL_SCALAR"
#define IR_SOURCE_CPP_STMT_OP                   "STMT_OP"
#define IR_SOURCE_CPP_STMT_IF                   "STMT_IF"
#define IR_SOURCE_CPP_STMT_ELSE                 "STMT_ELSE"
#define IR_SOURCE_CPP_STMT_FOR                  "STMT_FOR"
#define IR_SOURCE_CPP_STMT_YIELD                "STMT_YIELD"
#define IR_SOURCE_CPP_STMT_RETURN               "STMT_RETURN"
#define IR_SOURCE_CPP_STMT_YIELD_LOOP_BEGIN     "LOOP_BEGIN"
#define IR_SOURCE_CPP_STMT_YIELD_LOOP_ITER      "LOOP_ITER"
#define IR_SOURCE_CPP_STMT_YIELD_LOOP_ASSIGN    "LOOP_ASSIGN"
#define IR_SOURCE_CPP_STMT_YIELD_IF_ASSIGN      "IF_ASSIGN"

class SourceCppASTNode : public std::vector<std::shared_ptr<SourceCppASTNode>> {
public:
    SourceCppASTNode(const std::string &name, const std::vector<std::string> &argList)
      : std::vector<std::shared_ptr<SourceCppASTNode>>(), name_(name), argList_(argList) {}

    const std::string &GetName() const { return name_; }
    const std::vector<std::string> &GetArgList() const { return argList_; }

private:
    std::string name_;
    std::vector<std::string> argList_;
};
using SourceCppASTNodePtr = std::shared_ptr<SourceCppASTNode>;

IRBuffer &operator<<(IRBuffer &buf, const std::shared_ptr<SourceCppASTNode> &node);

struct SourceCppToken {
    enum TokenKind {
        Space,
        CommentLine,
        CommentBlock,
        Identifier,
        Number,
        ParenthesisLeft,
        ParenthesisRight,
        Comma,
        BraceLeft,
        BraceRight,
        Preprocess,
        Unknown,
    };
    SourceCppToken(TokenKind kind, const std::string &token) : kind_(kind), token_(token) {}
    TokenKind GetKind() { return kind_; }
    std::string GetToken() { return token_; }
private:
    TokenKind kind_;
    std::string token_;
};

class SourceCppIRSerializer : IRSerializer {
public:
    SourceCppIRSerializer() : IRSerializer(IRSerializer::SOURCE_CPP) {}

    virtual void Serialize(IRBuffer &buffer, const ProgramModulePtr &module);
    virtual ProgramModulePtr Deserialize(IRBuffer &buffer);

    SourceCppASTNodePtr SerializeASTNode(const ProgramModulePtr &module);

    void Tokenization(IRBuffer &buffer, std::vector<SourceCppToken> &tokenList);
};

}
}
