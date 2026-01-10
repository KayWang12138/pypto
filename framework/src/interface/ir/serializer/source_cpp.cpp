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
static SerializeCommand rtDeclTypeTile(IR_SOURCE_CPP_DECL_TYPE_TILE);
static SerializeCommand rtDeclTypeTensor(IR_SOURCE_CPP_DECL_TYPE_TENSOR);
static SerializeCommand rtDeclValueScalar(IR_SOURCE_CPP_DECL_VALUE_SCALAR);
static SerializeCommand rtDeclValueTile(IR_SOURCE_CPP_DECL_VALUE_TILE);
static SerializeCommand rtDeclValueTensor(IR_SOURCE_CPP_DECL_VALUE_TENSOR);

static SerializeCommand rtInitValueTile(IR_SOURCE_CPP_INIT_VALUE_TILE);
static SerializeCommand rtInitValueTensor(IR_SOURCE_CPP_INIT_VALUE_TENSOR);
static SerializeCommand rtInitAddr(IR_SOURCE_CPP_INIT_ADDR);

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

static bool SerializeIsSymbol(const ScalarValuePtr &ptr) {
    return !ptr->HasImmediateValue();
}

static std::vector<ScalarValuePtr> SerializeGetScalarDepend(const TileValuePtr &tile) {
    std::vector<ScalarValuePtr> argList;
    for (auto shape : tile->GetValidShape()) {
        if (SerializeIsSymbol(shape)) {
            argList.push_back(shape);
        }
    }
    return argList;
}

static std::vector<ScalarValuePtr> SerializeGetScalarDepend(const TensorValuePtr &tensor) {
    std::vector<ScalarValuePtr> argList;
    for (auto shape : tensor->GetShape()) {
        if (SerializeIsSymbol(shape)) {
            argList.push_back(shape);
        }
    }
    for (auto stride : tensor->GetStride()) {
        if (SerializeIsSymbol(stride)) {
            argList.push_back(stride);
        }
    }
    return argList;
}

static std::vector<std::string> SerializeGetScalarArgument(const TileValuePtr &tile) {
    std::vector<std::string> argList;
    for (auto shape : tile->GetValidShape()) {
        argList.push_back(SerializeValue(shape));
    }
    return argList;
}

static std::vector<std::string> SerializeGetScalarArgument(const TensorValuePtr &tensor) {
    std::vector<std::string> argList;
    for (auto shape : tensor->GetShape()) {
        argList.push_back(SerializeValue(shape));
    }
    for (auto stride : tensor->GetStride()) {
        argList.push_back(SerializeValue(stride));
    }
    return argList;
}

struct SerializeContext {
    OrderedSet<ScalarValuePtr> scalarValueList;
    OrderedSet<TileValuePtr> tileValueList;
    OrderedSet<TensorValuePtr> tensorValueList;
    std::unordered_map<TensorTypePtr, std::string> tensorTypeNameDict;
    std::unordered_map<TileTypePtr, std::string> tileTypeNameDict;

    struct TileTensorDepend {
        std::vector<TileValuePtr> tileValueList;
        std::vector<TensorValuePtr> tensorValueList;
    };
    std::unordered_map<ScalarValuePtr, TileTensorDepend> dependDict;

    OrderedSet<MemoryPtr> memList;
    std::unordered_map<MemoryPtr, std::string> memNameDict;

    struct State {
        std::unordered_map<TileValuePtr, int> tilePredDict;
        std::unordered_map<TensorValuePtr, int> tensorPredDict;
    } state;
};

static void SerializeOperation(SerializeContext &ctx, SourceCppASTNodePtr &stmtOpNode, const OperationPtr &op) {
    std::vector<std::string> argList({GetOpcodeName(op->GetOpcode())});
    std::vector<TileValuePtr> readyTileList;
    std::vector<TensorValuePtr> readyTensorList;
    for (size_t k = 0; k < op->GetNumOutputOperand(); k++) {
        auto oop = op->GetOutputOperand(k);
        switch (oop->GetValueKind()) {
            case ValueKind::Scalar: {
                /* scalar is declared */
                ScalarValuePtr oopScalar = ObjectCast<ScalarValue>(oop);
                ASSERT(oopScalar->GetScalarValueKind() == ScalarValueKind::Symbolic) << "Output must be symbol";
                argList.push_back(oopScalar->GetName());

                if (ctx.dependDict.count(oopScalar)) {
                    for (auto tile : ctx.dependDict[oopScalar].tileValueList) {
                        ctx.state.tilePredDict[tile]--;
                        if (ctx.state.tilePredDict[tile] == 0) {
                            readyTileList.push_back(tile);
                        }
                    }
                    for (auto tensor : ctx.dependDict[oopScalar].tensorValueList) {
                        ctx.state.tensorPredDict[tensor]--;
                        if (ctx.state.tensorPredDict[tensor] == 0) {
                            readyTensorList.push_back(tensor);
                        }
                    }
                }

                break;
            }
            case ValueKind::Tile: {
                TileValuePtr oopTile = ObjectCast<TileValue>(oop);
                argList.push_back(oopTile->GetName());
                break;
            }
            case ValueKind::Tensor: {
                TensorValuePtr oopTensor = ObjectCast<TensorValue>(oop);
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

    for (auto readyTile : readyTileList) {
        auto initNode = rtInitValueTile(readyTile->GetName(), readyTile->GetValidShape().size(), SerializeGetScalarArgument(readyTile));
        stmtOpNode->push_back(initNode);
    }
    for (auto readyTensor : readyTensorList) {
        auto initNode = rtInitValueTensor(readyTensor->GetName(), readyTensor->GetShape().size(), SerializeGetScalarArgument(readyTensor));
        stmtOpNode->push_back(initNode);
    }
}

static void SerializeStatement(SerializeContext &ctx, SourceCppASTNodePtr &parentNode, const StatementPtr &stmt, const StatementPtr &yieldTarget) {
    switch (stmt->GetKind()) {
        case StatementKind::Compound: {
            CompoundStatementPtr stmtCompound = ObjectCast<CompoundStatement>(stmt);
            for (size_t k = 0; k < stmtCompound->GetStatementsNum(); k++) {
                SerializeStatement(ctx, parentNode, stmtCompound->GetStatement(k), yieldTarget);
            }
            break;
        }
        case StatementKind::Op: {
            OpStatementPtr stmtOp = ObjectCast<OpStatement>(stmt);
            auto stmtOpNode = rtStmtOp();
            parentNode->push_back(stmtOpNode);

            for (auto &op : stmtOp->Operations()) {
                SerializeOperation(ctx, stmtOpNode, op);
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

            SerializeStatement(ctx, stmtIfNode, stmtIf->GetThenCompound(), stmt);
            SerializeStatement(ctx, stmtElseNode, stmtIf->GetElseCompound(), stmt);
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

            SerializeStatement(ctx, stmtForNode, stmtFor->GetCompound(), stmt);
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

static void SerializeFunctionFindValue(SerializeContext &ctx, const FunctionPtr &func) {
    struct Visit {
        static void VisitValue(const ValuePtr value, SerializeContext &ctx) {
            switch(value->GetValueKind()) {
                case ValueKind::Scalar: {
                    auto scalar = ObjectCast<ScalarValue>(value);
                    if (SerializeIsSymbol(scalar)) {
                        ctx.scalarValueList.Insert(scalar);
                    }
                    break;
                }
                case ValueKind::Tile:
                    ctx.tileValueList.Insert(ObjectCast<TileValue>(value));
                    break;
                case ValueKind::Tensor:
                    ctx.tensorValueList.Insert(ObjectCast<TensorValue>(value));
                    break;
                default:
                    break;
            }
        }
        static void VisitFunc(const FunctionPtr &func, SerializeContext &ctx) {
            VisitStmt(func->GetCompound(), ctx);
        }
        static void VisitStmt(const StatementPtr &stmt, SerializeContext &ctx) {
            switch (stmt->GetKind()) {
                case StatementKind::Compound: {
                    CompoundStatementPtr stmtCompound = ObjectCast<CompoundStatement>(stmt);
                    for (size_t k = 0; k < stmtCompound->GetStatementsNum(); k++) {
                        VisitStmt(stmtCompound->GetStatement(k), ctx);
                    }
                    break;
                }
                case StatementKind::Op: {
                    OpStatementPtr stmtOp = ObjectCast<OpStatement>(stmt);
                    for (auto &op : stmtOp->Operations()) {
                        VisitOp(op, ctx);
                    }
                    break;
                }
                case StatementKind::If: {
                    IfStatementPtr stmtIf = ObjectCast<IfStatement>(stmt);
                    VisitStmt(stmtIf->GetThenCompound(), ctx);
                    VisitStmt(stmtIf->GetElseCompound(), ctx);
                    break;
                }
                case StatementKind::For: {
                    ForStatementPtr stmtFor = ObjectCast<ForStatement>(stmt);
                    VisitValue(stmtFor->GetIterationVar(), ctx);
                    for (size_t k = 0; k < stmtFor->Results().size(); k++) {
                        VisitValue(stmtFor->GetIterValue(k), ctx);
                    }
                    VisitStmt(stmtFor->GetCompound(), ctx);
                    for (size_t k = 0; k < stmtFor->Results().size(); k++) {
                        VisitValue(stmtFor->Results()[k], ctx);
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
        static void VisitOp(const OperationPtr &op, SerializeContext &ctx) {
            for (size_t k = 0; k < op->GetNumOutputOperand(); k++) {
                VisitValue(op->GetOutputOperand(k), ctx);
            }
            for (size_t k = 0; k < op->GetNumInputOperand(); k++) {
                VisitValue(op->GetInputOperand(k), ctx);
            }
        }
    };
    Visit::VisitFunc(func, ctx);

    for (auto tile : ctx.tileValueList) {
        for (auto value : SerializeGetScalarDepend(tile)) {
            ctx.dependDict[value].tileValueList.push_back(tile);
            ctx.state.tilePredDict[tile]++;
        }
    }
    for (auto tensor : ctx.tensorValueList) {
        for (auto value : SerializeGetScalarDepend(tensor)) {
            ctx.dependDict[value].tensorValueList.push_back(tensor);
            ctx.state.tensorPredDict[tensor]++;
        }
    }
}

static void SerializeFunctionDeclTypeList(SerializeContext &ctx, SourceCppASTNodePtr &funcNode) {
    OrderedSet<TypePtr> tileTypeList;
    OrderedSet<TypePtr> tensorTypeList;
    for (auto tile : ctx.tileValueList) {
        auto tileType = tile->GetType();
        tileTypeList.Insert(tileType);
    }
    for (auto tensor : ctx.tensorValueList) {
        auto tensorType = tensor->GetType();
        tensorTypeList.Insert(tensorType);
    }
    for (auto tile : ctx.tileValueList) {
        auto tileType = tile->GetType();
        ctx.tileTypeNameDict[ObjectCast<TileType>(tileType)] = "RT_L" + std::to_string(tileTypeList.GetIndex(tileType));
    }
    for (auto tensor : ctx.tensorValueList) {
        auto tensorType = tensor->GetType();
        ctx.tensorTypeNameDict[ObjectCast<TensorType>(tensorType)] = "RT_G" + std::to_string(tensorTypeList.GetIndex(tensorType));
    }
    for (auto &[tileType, name] : ctx.tileTypeNameDict) {
        funcNode->push_back(rtDeclTypeTile(name, SerializeDataType(tileType), tileType->GetShape()));
    }
    for (auto &[tensorType, name] : ctx.tensorTypeNameDict) {
        funcNode->push_back(rtDeclTypeTensor(name, SerializeDataType(tensorType), tensorType->GetDimNum()));
    }
}

static void SerializeFunctionDeclValueList(const SerializeContext &ctx, SourceCppASTNodePtr &funcNode) {
    for (auto scalar : ctx.scalarValueList) {
        funcNode->push_back(rtDeclValueScalar(scalar->GetName(), SerializeDataType(scalar->GetType())));
    }
    for (auto tile : ctx.tileValueList) {
        funcNode->push_back(rtDeclValueTile(tile->GetName(), ctx.tileTypeNameDict.find(ObjectCast<TileType>(tile->GetType()))->second));
    }
    for (auto tensor : ctx.tensorValueList) {
        funcNode->push_back(rtDeclValueTensor(tensor->GetName(), ctx.tensorTypeNameDict.find(ObjectCast<TensorType>(tensor->GetType()))->second));
    }
}

static void SerializeFunctionInitAddr(SerializeContext &ctx, SourceCppASTNodePtr &funcNode) {
    for (auto tile : ctx.tileValueList) {
        ctx.memList.Insert(tile->GetMemory());
    }
    for (auto mem : ctx.memList) {
        ctx.memNameDict[mem] = "RT_S" + std::to_string(mem->GetAddr()) + "_E" + std::to_string(mem->GetAddr() + mem->GetSize()) +  "_" + std::to_string(ctx.memList.GetIndex(mem));
    }
    for (auto mem : ctx.memList) {
        funcNode->push_back(rtInitAddr(ctx.memNameDict[mem], mem->GetAddr(), mem->GetSize(), GetMemSpaceKindName(mem->GetSpace())));
    }
}

static SourceCppASTNodePtr SerializeFunction(const FunctionPtr &func) {
    SourceCppASTNodePtr funcNode = rtFunction(func->GetName());
    SerializeContext ctx;
    SerializeFunctionFindValue(ctx, func);
    SerializeFunctionDeclTypeList(ctx, funcNode);
    SerializeFunctionDeclValueList(ctx, funcNode);
    SerializeFunctionInitAddr(ctx, funcNode);
    SerializeStatement(ctx, funcNode, func->GetCompound(), nullptr);
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
