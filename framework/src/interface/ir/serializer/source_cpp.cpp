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
 * \file source_cpp.cpp
 * \brief Serialize to source format whose style is cplusplus
 */

#include "ir/serializer.h"
#include "tilefwk/error.h"
#include "ir/utils.h"
#include "interface/utils/common.h"

using namespace npu::tile_fwk;

namespace pto {
namespace serializer {

static inline std::string Indent(int indent) {
    return std::string(indent * 4, ' ');
}

static void SerializeNodeHead(IRBuffer &buffer, const std::shared_ptr<SourceCppASTNode> &node, int indent) {
    if (node->GetName() != "") {
        buffer << Indent(indent) << node->GetName() << "(" << node->GetArgList() << ")";
    }
}

static void SerializeNode(IRBuffer &buffer, const std::shared_ptr<SourceCppASTNode> &node, int indent) {
    SerializeNodeHead(buffer, node, indent);
    if (node->size() != 0) {
        if (node->GetName() != "") {
            buffer << " {\n";
        }
        for (size_t k = 0; k < node->size(); k++) {
            SerializeNode(buffer, node->at(k), indent + 1);
        }
        if (node->GetName() != "") {
            buffer << Indent(indent) << "}\n";
        }
    } else {
        buffer << "\n";
    }
}

const static std::unordered_map<char, SourceCppToken::TokenKind> punctuationDict = {
    {'{', SourceCppToken::BraceLeft},
    {'}', SourceCppToken::BraceRight},
    {'(', SourceCppToken::ParenthesisLeft},
    {')', SourceCppToken::ParenthesisRight},
    {',', SourceCppToken::Comma},
};
static int DeserializeTokenList(IRBuffer &buffer, std::vector<SourceCppToken> &tokenList) {
    char ch = 0;
    while (true) {
        int count = buffer.Read(&ch, 1, true);
        if (count == 0) {
            break;
        } else if (std::isspace(ch)) {
            std::string space = buffer.ReadSpace();
            tokenList.emplace_back(SourceCppToken::Space, space);
        } else if (ch == '/') {
            char comment[2];
            count = buffer.Read(comment, 2, true);
            if (comment[1] == '/') {
                std::string commentLine = buffer.ReadLine();
                tokenList.emplace_back(SourceCppToken::CommentLine, commentLine);
            } else if (comment[1] == '*') {
                std::string commentBlock = buffer.ReadUntil("*/");
                tokenList.emplace_back(SourceCppToken::CommentBlock, commentBlock);
            } else {
                buffer.Read(&ch, 1);
                tokenList.emplace_back(SourceCppToken::Unknown, std::string(1, ch));
            }
        } else if (punctuationDict.count(ch)) {
            buffer.Read(&ch, 1);
            auto tokenKind = punctuationDict.find(ch)->second;
            tokenList.emplace_back(tokenKind, std::string(1, ch));
        } else if (std::isalpha(ch) || ch == '_') {
            std::string identifier = SourceReadIdentifier(buffer);
            tokenList.emplace_back(SourceCppToken::Identifier, identifier);
        } else if (std::isdigit(ch)) {
            std::string number = SourceReadNumber(buffer);
            tokenList.emplace_back(SourceCppToken::Number, number);
        } else if (ch == '#') {
            std::string preprocess = buffer.ReadLine();
            tokenList.emplace_back(SourceCppToken::Preprocess, preprocess);
        } else {
            break;
        }
    }
    return buffer.ReadSeek(0, IRBuffer::ReadSeekMode::Position);
}

static void DeserializeNode(IRBuffer &buffer, std::shared_ptr<SourceCppASTNode> &node) {
    (void)buffer;
    (void)node;
}

IRBuffer &operator<<(IRBuffer &buf, const std::shared_ptr<SourceCppASTNode> &node) {
    SerializeNode(buf, node, 0);
    return buf;
}

IRBuffer &operator>>(IRBuffer &buf, std::shared_ptr<SourceCppASTNode> &node) {
    DeserializeNode(buf, node);
    return buf;
}

class SerializeCommand {
public:
    SerializeCommand(const std::string &name) : polishName_(IR_SOURCE_CPP_PREFIX + name) {}

    template<typename ...TyArgs>
    SourceCppASTNodePtr operator()(TyArgs ...args) {
        std::vector<std::string> argumentList = SerializeUtils::List(args...);
        return std::make_shared<SourceCppASTNode>(polishName_, argumentList);
    }
private:
    std::string polishName_;
};

static SerializeCommand rtFunction(IR_SOURCE_CPP_FUNCTION);
static SerializeCommand rtOperation(IR_SOURCE_CPP_OPERATION);
static SerializeCommand rtDeclTileType(IR_SOURCE_CPP_DECL_TILE_TYPE);
static SerializeCommand rtDeclTensorType(IR_SOURCE_CPP_DECL_TENSOR_TYPE);
static SerializeCommand rtDeclAddr(IR_SOURCE_CPP_DECL_ADDR);
static SerializeCommand rtDeclTileValue(IR_SOURCE_CPP_DECL_TILE);
static SerializeCommand rtDeclTensorValue(IR_SOURCE_CPP_DECL_TENSOR);
static SerializeCommand rtDeclScalarValue(IR_SOURCE_CPP_DECL_SCALAR);
static SerializeCommand rtStmtOp(IR_SOURCE_CPP_STMT_OP);
static SerializeCommand rtStmtIf(IR_SOURCE_CPP_STMT_IF);
static SerializeCommand rtStmtElse(IR_SOURCE_CPP_STMT_ELSE);
static SerializeCommand rtStmtFor(IR_SOURCE_CPP_STMT_FOR);
static SerializeCommand rtStmtYield(IR_SOURCE_CPP_STMT_YIELD);
static SerializeCommand rtStmtReturn(IR_SOURCE_CPP_STMT_RETURN);

static std::string SerializeValue(const ValuePtr &value) {
    std::string result;
    switch (value->GetValueKind()) {
        case ValueKind::Scalar: {
            ScalarValuePtr scalar = ObjectCast<ScalarValue>(value);
            if (scalar->GetScalarValueKind() == ScalarValueKind::Immediate) {
                auto immediate = scalar->GetImmediateValue();
                if (std::holds_alternative<bool>(immediate)) {
                    result = std::get<bool>(immediate) ? "true" : "false";
                } else if (std::holds_alternative<int>(immediate)) {
                    result = std::to_string(std::get<int>(immediate));
                } else if (std::holds_alternative<int64_t>(immediate)) {
                    result = std::to_string(std::get<int64_t>(immediate));
                } else if (std::holds_alternative<uint64_t>(immediate)) {
                    result = std::to_string(std::get<uint64_t>(immediate));
                } else if (std::holds_alternative<double>(immediate)) {
                    result = std::to_string(std::get<double>(immediate));
                } else {
                    ASSERT(false) << "Unknown value";
                }
            } else {
                result = scalar->GetName();
            }
            break;
        }
        case ValueKind::Tile: {
            result = value->GetName();
            break;
        }
        case ValueKind::Tensor: {
            result = value->GetName();
            break;
        }
        default: {
            ASSERT(false) << "Invalid value: " << static_cast<int>(value->GetValueKind());
        }
    }
    return result;
}

static std::string SerializeDataType(const TypePtr &type) {
    return DTypeInfoOf(type->GetDataType()).name;
}

static void SerializeOperation(SourceCppASTNodePtr &stmtOpNode, const OperationPtr &op) {
    std::vector<std::string> argList({GetOpcodeName(op->GetOpcode())});
    for (size_t k = 0; k < op->GetNumOutputOperand(); k++) {
        auto oop = op->GetOutputOperand(k);
        switch (oop->GetValueKind()) {
            case ValueKind::Scalar: {
                /* scalar is declared */
                ScalarValuePtr oopScalar = ObjectCast<ScalarValue>(oop);
                ASSERT(oopScalar->GetScalarValueKind() == ScalarValueKind::Symbolic) << "Output must be symbol";
                argList.push_back(oopScalar->GetName());
                break;
            }
            case ValueKind::Tile: {
                TileValuePtr oopTile = ObjectCast<TileValue>(oop);
                auto declNode = rtDeclTileValue(oopTile->GetName());
                stmtOpNode->push_back(declNode);
                argList.push_back(oopTile->GetName());
                break;
            }
            case ValueKind::Tensor: {
                TensorValuePtr oopTensor = ObjectCast<TensorValue>(oop);
                auto declNode = rtDeclTensorValue(oopTensor->GetName());
                stmtOpNode->push_back(declNode);
                argList.push_back(oopTensor->GetName());
                break;
            }
            default: {
                ASSERT(false) << "Invalid value: " << static_cast<int>(oop->GetValueKind());
                break;
            }
        }
    }
    for (size_t k = 0; k < op->GetNumInputOperand(); k++) {
        auto iop = op->GetInputOperand(k);
        argList.push_back(SerializeValue(iop));
    }

    std::vector<AttributeKeyValue> kvList = op->GetAttributeList();
    for (auto &kv : kvList) {
        const auto &value = kv.GetValue();
        if (std::holds_alternative<std::string>(value)) {
            argList.push_back(std::get<std::string>(value));
        } else if (std::holds_alternative<int64_t>(value)) {
            argList.push_back(std::to_string(std::get<int64_t>(value)));
        } else if (std::holds_alternative<bool>(value)) {
            argList.push_back(std::get<bool>(value) ? "true" : "false");
        } else {
            ASSERT(false) << "Unknown attribute: " << kv.GetName();
        }
    }
    auto opNode = rtOperation(argList);
    stmtOpNode->push_back(opNode);
}

static void SerializeStatement(SourceCppASTNodePtr &parentNode, const StatementPtr &stmt, const StatementPtr &yieldTarget) {
    switch (stmt->GetKind()) {
        case StatementKind::Compound: {
            CompoundStatementPtr stmtCompound = ObjectCast<CompoundStatement>(stmt);
            for (size_t k = 0; k < stmtCompound->GetStatementsNum(); k++) {
                SerializeStatement(parentNode, stmtCompound->GetStatement(k), yieldTarget);
            }
            break;
        }
        case StatementKind::Op: {
            OpStatementPtr stmtOp = ObjectCast<OpStatement>(stmt);
            auto stmtOpNode = rtStmtOp();
            parentNode->push_back(stmtOpNode);

            for (auto &op : stmtOp->Operations()) {
                SerializeOperation(stmtOpNode, op);
            }
            break;
        }
        case StatementKind::If: {
            IfStatementPtr stmtIf = ObjectCast<IfStatement>(stmt);
            auto cond = stmtIf->GetCondition();
            auto stmtIfNode = rtStmtIf(SerializeValue(cond));
            auto stmtElseNode = rtStmtElse();
            parentNode->push_back(stmtIfNode);
            parentNode->push_back(stmtElseNode);

            SerializeStatement(stmtIfNode, stmtIf->GetThenCompound(), stmt);
            SerializeStatement(stmtElseNode, stmtIf->GetElseCompound(), stmt);
            break;
        }
        case StatementKind::For: {
            ForStatementPtr stmtFor = ObjectCast<ForStatement>(stmt);
            auto var = stmtFor->GetIterationVar();
            auto begin = SerializeValue(stmtFor->GetStart());
            auto end = SerializeValue(stmtFor->GetEnd());
            auto step = SerializeValue(stmtFor->GetStep());
            for (size_t k = 0; k < stmtFor->Results().size(); k++) {
                auto yieldDst = stmtFor->GetIterValue(k)->GetName();
                auto yieldSrc = SerializeValue(stmtFor->GetIterInitValue(k));
                auto stmtInitNode = rtStmtYield(IR_SOURCE_CPP_STMT_YIELD_LOOP_BEGIN, yieldDst, yieldSrc);
                parentNode->push_back(stmtInitNode);
            }
            auto stmtForNode = rtStmtFor(var->GetName(), begin, end, step);
            parentNode->push_back(stmtForNode);

            SerializeStatement(stmtForNode, stmtFor->GetCompound(), stmt);
            for (size_t k = 0; k < stmtFor->Results().size(); k++) {
                auto yieldDst = stmtFor->Results()[k]->GetName();
                auto yieldSrc = stmtFor->GetIterValue(k)->GetName();
                auto stmtYieldNode = rtStmtYield(IR_SOURCE_CPP_STMT_YIELD_LOOP_ASSIGN, yieldDst, yieldSrc);
                parentNode->push_back(stmtYieldNode);
            }
            break;
        }
        case StatementKind::Yield: {
            YieldStatementPtr stmtYield = ObjectCast<YieldStatement>(stmt);
            switch (yieldTarget->GetKind()) {
                case StatementKind::If: {
                    IfStatementPtr yieldTargetIf = ObjectCast<IfStatement>(yieldTarget);
                    for (size_t k = 0; k < yieldTargetIf->Results().size(); k++) {
                        auto yieldDst = yieldTargetIf->Results()[k]->GetName();
                        auto yieldSrc = SerializeValue(stmtYield->Values()[k]);
                        auto yieldNode = rtStmtYield(IR_SOURCE_CPP_STMT_YIELD_IF_ASSIGN, yieldDst, yieldSrc);
                        parentNode->push_back(yieldNode);
                    }
                    break;
                }
                case StatementKind::For: {
                    ForStatementPtr yieldTargetFor = ObjectCast<ForStatement>(yieldTarget);
                    for (size_t k = 0; k < yieldTargetFor->Results().size(); k++) {
                        auto yieldDst = yieldTargetFor->GetIterValue(k)->GetName();
                        auto yieldSrc = SerializeValue(stmtYield->Values()[k]);
                        auto yieldNode = rtStmtYield(IR_SOURCE_CPP_STMT_YIELD_LOOP_ITER, yieldDst, yieldSrc);
                        parentNode->push_back(yieldNode);
                    }
                    break;
                }
                default: {
                    ASSERT(false) << "Unknown statement: " << static_cast<int>(stmt->GetKind());
                    break;
                }
            }
            break;
        }
        case StatementKind::Return: {
            ReturnStatementPtr stmtReturn = ObjectCast<ReturnStatement>(stmt);
            auto stmtReturnNode = rtStmtReturn();
            parentNode->push_back(stmtReturnNode);
            break;
        }
        default: {
            ASSERT(false) << "Unknown statement: " << static_cast<int>(stmt->GetKind());
            break;
        }
    }
}

struct ValueSet {
    OrderedSet<ScalarValuePtr> scalarList;
    OrderedSet<TileValuePtr> tileList;
    OrderedSet<TensorValuePtr> tensorList;
};

static void SerializeFunctionFindValue(
        const FunctionPtr &func,
        ValueSet &valueSet) {
    struct Visit {
        static void VisitValue(const ValuePtr value, ValueSet &valueSet) {
            switch(value->GetValueKind()) {
                case ValueKind::Scalar:
                    valueSet.scalarList.Insert(ObjectCast<ScalarValue>(value));
                    break;
                case ValueKind::Tile:
                    valueSet.tileList.Insert(ObjectCast<TileValue>(value));
                    break;
                case ValueKind::Tensor:
                    valueSet.tensorList.Insert(ObjectCast<TensorValue>(value));
                    break;
                default:
                    break;
            }
        }
        static void VisitFunc(const FunctionPtr &func, ValueSet &valueSet) {
            VisitStmt(func->GetCompound(), valueSet);
        }
        static void VisitStmt(const StatementPtr &stmt, ValueSet &valueSet) {
            switch (stmt->GetKind()) {
                case StatementKind::Compound: {
                    CompoundStatementPtr stmtCompound = ObjectCast<CompoundStatement>(stmt);
                    for (size_t k = 0; k < stmtCompound->GetStatementsNum(); k++) {
                        VisitStmt(stmtCompound->GetStatement(k), valueSet);
                    }
                    break;
                }
                case StatementKind::Op: {
                    OpStatementPtr stmtOp = ObjectCast<OpStatement>(stmt);
                    for (auto &op : stmtOp->Operations()) {
                        VisitOp(op, valueSet);
                    }
                    break;
                }
                case StatementKind::If: {
                    IfStatementPtr stmtIf = ObjectCast<IfStatement>(stmt);
                    VisitStmt(stmtIf->GetThenCompound(), valueSet);
                    VisitStmt(stmtIf->GetElseCompound(), valueSet);
                    break;
                }
                case StatementKind::For: {
                    ForStatementPtr stmtFor = ObjectCast<ForStatement>(stmt);
                    VisitValue(stmtFor->GetIterationVar(), valueSet);
                    for (size_t k = 0; k < stmtFor->Results().size(); k++) {
                        VisitValue(stmtFor->GetIterValue(k), valueSet);
                    }
                    VisitStmt(stmtFor->GetCompound(), valueSet);
                    for (size_t k = 0; k < stmtFor->Results().size(); k++) {
                        VisitValue(stmtFor->Results()[k], valueSet);
                    }
                    break;
                }
                case StatementKind::Yield: {
                    break;
                }
                case StatementKind::Return: {
                    break;
                }
                default: {
                    break;
                }
            }

        }
        static void VisitOp(const OperationPtr &op, ValueSet &valueSet) {
            for (size_t k = 0; k < op->GetNumOutputOperand(); k++) {
                VisitValue(op->GetOutputOperand(k), valueSet);
            }
            for (size_t k = 0; k < op->GetNumInputOperand(); k++) {
                VisitValue(op->GetInputOperand(k), valueSet);
            }
        }
    };
    Visit::VisitFunc(func, valueSet);
}

static void SerializeFunctionTypeList(
        SourceCppASTNodePtr &funcNode,
        const OrderedSet<ValuePtr> &outputList,
        std::unordered_map<TensorTypePtr, std::string> &tensorTypeNameDict,
        std::unordered_map<TileTypePtr, std::string> &tileTypeNameDict) {
    OrderedSet<TypePtr> tensorTypeList;
    OrderedSet<TypePtr> tileTypeList;
    for (auto output: outputList) {
        switch (output->GetValueKind()) {
            case ValueKind::Tensor:
                tensorTypeList.Insert(output->GetType());
                break;
            case ValueKind::Tile:
                tileTypeList.Insert(output->GetType());
                break;
            default:
                break;
        }
    }
    for (auto tensorType : tensorTypeList) {
        tensorTypeNameDict[ObjectCast<TensorType>(tensorType)] = "TG" + std::to_string(tensorTypeList.GetIndex(tensorType));
    }
    for (auto tileType : tileTypeList) {
        tileTypeNameDict[ObjectCast<TileType>(tileType)] = "TT" + std::to_string(tileTypeList.GetIndex(tileType));
    }
    for (auto &[tensorType, name] : tensorTypeNameDict) {
        funcNode->push_back(rtDeclTensorType(name, SerializeDataType(tensorType), tensorType->GetDimNum()));
    }
    for (auto &[tileType, name] : tileTypeNameDict) {
        funcNode->push_back(rtDeclTileType(name, SerializeDataType(tileType), tileType->GetShape()));
    }
}

static void SerializeFunctionOutputList(
        SourceCppASTNodePtr &funcNode,
        const ValueSet &valueSet,
        const std::unordered_map<TensorTypePtr, std::string> &tensorTypeNameDict,
        const std::unordered_map<TileTypePtr, std::string> &tileTypeNameDict) {
    for (auto scalar : valueSet.scalarList) {
        funcNode->push_back(rtDeclScalarValue(scalar->GetName(), SerializeDataType(scalar->GetType())));
    }
    for (auto tile : valueSet.tileList) {
        funcNode->push_back(rtDeclTileValue(tile->GetName(), tileTypeNameDict.find(ObjectCast<TileType>(tile->GetType()))->second));
    }
    for (auto tensor : valueSet.tensorList) {
        funcNode->push_back(rtDeclTensorValue(tensor->GetName(), tensorTypeNameDict.find(ObjectCast<TensorType>(tensor->GetType()))->second));
    }
}

static SourceCppASTNodePtr SerializeFunction(const FunctionPtr &func) {
    SourceCppASTNodePtr funcNode = rtFunction(func->GetName());
    ValueSet valueSet;
    SerializeFunctionFindValue(func, valueSet);
    std::unordered_map<TensorTypePtr, std::string> tensorTypeNameDict;
    std::unordered_map<TileTypePtr, std::string> tileTypeNameDict;

    SerializeFunctionTypeList(funcNode, valueSet, tensorTypeNameDict, tileTypeNameDict);
    SerializeFunctionOutputList(funcNode, valueSet, tensorTypeNameDict, tileTypeNameDict);
    SerializeStatement(funcNode, func->GetCompound(), nullptr);
    return funcNode;
}

static SourceCppASTNodePtr SerializeProgram(const ProgramModulePtr &prog) {
    SourceCppASTNodePtr progNode = std::make_shared<SourceCppASTNode>(std::string(), std::vector<std::string>());
    for (auto func : prog->GetFunctions()) {
        progNode->push_back(SerializeFunction(func));
    }
    return progNode;
}

SourceCppASTNodePtr SourceCppIRSerializer::SerializeASTNode(const ProgramModulePtr &prog) {
    return SerializeProgram(prog);
}

void SourceCppIRSerializer::Tokenization(IRBuffer &buffer, std::vector<SourceCppToken> &tokenList) {
    DeserializeTokenList(buffer, tokenList);
}

void SourceCppIRSerializer::Serialize(IRBuffer &buffer, const ProgramModulePtr &prog) {
    auto progNode = SerializeASTNode(prog);
    buffer << progNode;
}

ProgramModulePtr SourceCppIRSerializer::Deserialize(IRBuffer &buffer) {
    (void)buffer;
    ASSERT(false);
    return nullptr;
}

} // namespace serializer
} // namespace pto
