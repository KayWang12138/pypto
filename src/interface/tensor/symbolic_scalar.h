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
 * \file symbolic_scalar.h
 * \brief
 */

#pragma once

#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <set>
#include <map>

#include <tilefwk/symbolic_scalar.h>

#include "tilefwk/error.h"
#include "interface/cache/hash_buffer.h"
#include "symbol_handler.h"
#include "interface/utils/string_utils.h"
#include "interface/utils/common.h"
#include "value_guesser.h"

#include <nlohmann/json.hpp>
using Json = nlohmann::json;

namespace npu::tile_fwk {

constexpr int INVALID_SCALAR_IMMEDIATE = -1;
constexpr const char* const SPECIAL_SYMBOL_NAME_RUNTIME_PREFIX = "RUNTIME_";
constexpr const char* const SPECIAL_SYMBOL_NAME_RUNTIME_COA_PREFIX = "RUNTIME_COA_";
constexpr const char* const SPECIAL_SYMBOL_NAME_ARG_PREFIX = "ARG_";
using ScalarImmediateType = long long;

enum class SymbolicScalarKind {
    T_SCALAR_SYMBOLIC_IMMEDIATE,
    T_SCALAR_SYMBOLIC_SYMBOL,
    T_SCALAR_SYMBOLIC_EXPRESSION,
};

inline std::string SymbolicScalarKind2Name(SymbolicScalarKind kind) {
    std::string name;
    switch (kind) {
    case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE:
        name = "immediate";
        break;
    case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL:
        name = "symbol";
        break;
    case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION:
        name = "expression";
        break;
    default:
        ASSERT(false);
        break;
    }
    return name;
}

enum class SymbolicOpcode {
    T_UOP_POS,
    T_UOP_NEG,
    T_UOP_NOT,

    T_BOP_ADD,
    T_BOP_SUB,
    T_BOP_MUL,
    T_BOP_DIV,
    T_BOP_MOD,

    T_BOP_EQ,
    T_BOP_NE,
    T_BOP_LT,
    T_BOP_LE,
    T_BOP_GT,
    T_BOP_GE,

    T_BOP_MIN,
    T_BOP_MAX,

    T_MOP_CALL,

    T_UOP_BEGIN = T_UOP_POS,
    T_UOP_END = T_UOP_NOT + 1,
    T_BOP_BEGIN = T_BOP_ADD,
    T_BOP_END = T_BOP_MAX + 1
};

class RawSymbolicScalar {
public:
    SymbolicScalarKind kind;

    RawSymbolicScalar(SymbolicScalarKind tkind) : kind(tkind) {}

    RawSymbolicScalar(SymbolicScalarKind tkind, ValueGuesser valueGuesser) : kind(tkind), valueGuesser_(valueGuesser) {}

    SymbolicScalarKind Kind() const { return kind; }

    bool IsImmediate() const { return Kind() == SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE; }
    bool IsSymbol() const { return Kind() == SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL; }
    bool IsExpression() const { return Kind() == SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION; }

    ScalarImmediateType GetImmediateValue() const;
    const std::string &GetSymbolName() const;
    SymbolicOpcode GetExpressionOpcode() const;
    const std::vector<RawSymbolicScalarPtr> &GetExpressionOperandList() const;
    bool IsExpressionCall(const std::string &calleeName) const;

    [[nodiscard]] bool IsIntermediateVariable() const { return intermediateVariable_; }
    void AsIntermediateVariable() { intermediateVariable_ = true; }

    virtual Json DumpJson() const = 0;
    virtual ~RawSymbolicScalar() = default;

    void ResetValueGuesser(ValueGuesser valueGuesser);

    std::string Dump() const {
        std::string buf;
        DumpBuffer(buf);
        return buf;
    }

private:
    friend class SymbolicScalar;
    friend class RawSymbolicExpression;
    virtual void DumpBuffer(std::string &buffer) const = 0;

    ValueGuesser valueGuesser_;
    bool intermediateVariable_{false};
};

class RawSymbolicImmediate : public RawSymbolicScalar {
public:
    explicit RawSymbolicImmediate(ScalarImmediateType immediate)
        : RawSymbolicScalar(SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE), immediate_(immediate) {}

    ScalarImmediateType Immediate() const { return immediate_; }

    Json DumpJson() const override {
        Json immediateDump = Json::array();
        immediateDump.push_back(kind);
        immediateDump.push_back(immediate_);
        return immediateDump;
    }
    static RawSymbolicScalarPtr Create(ScalarImmediateType immediate) {
        return std::make_shared<RawSymbolicImmediate>(immediate);
    }

private:
    void DumpBuffer(std::string &buffer) const override { buffer += std::to_string(immediate_); }

    ScalarImmediateType immediate_;
};

class RawSymbolicSymbol : public RawSymbolicScalar {
public:
    explicit RawSymbolicSymbol(const std::string &name)
        : RawSymbolicScalar(SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL), name_(name) {}
    RawSymbolicSymbol(const std::string &name, ValueGuesser valueGuesser)
        : RawSymbolicScalar(SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL, valueGuesser), name_(name) {}

    const std::string &Name() const { return name_; }

    Json DumpJson() const override {
        Json symbolDump = Json::array();
        symbolDump.push_back(kind);
        symbolDump.push_back(name_);
        return symbolDump;
    }

    static RawSymbolicScalarPtr Create(const std::string &name) { return std::make_shared<RawSymbolicSymbol>(name); }
    static RawSymbolicScalarPtr Create(const std::string &name, ValueGuesser valueGuesser) {
        return std::make_shared<RawSymbolicSymbol>(name, valueGuesser);
    }

private:
    void DumpBuffer(std::string &buffer) const override { buffer += name_; }

    std::string name_;
};

class RawSymbolicExpression : public RawSymbolicScalar {
public:
    RawSymbolicExpression(SymbolicOpcode opcode, const std::vector<RawSymbolicScalarPtr> &operandList)
        : RawSymbolicScalar(SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION),
          opcode_(opcode),
          operandList_(operandList) {
        // LoopBegin和LoopEnd目前只能作为最外层的表达式
        for (const auto &operand : operandList) {
            if (operand->IsExpression()) {
                auto expression = std::static_pointer_cast<RawSymbolicExpression>(operand);
                ASSERT(!expression->IsLoopBeginCall());
                ASSERT(!expression->IsLoopEndCall());
            }
        }
    }

    SymbolicOpcode Opcode() const { return opcode_; }
    const std::vector<RawSymbolicScalarPtr> &OperandList() const { return operandList_; }

    bool IsLoopBeginCall() const {
        if (opcode_ == SymbolicOpcode::T_MOP_CALL) {
            auto raw = operandList_[0];
            RawSymbolicSymbol *rawSymbol = dynamic_cast<RawSymbolicSymbol *>(raw.get());
            auto callee = rawSymbol->Name();
            return callee.find(SymbolHandler::GetNameByHandlerId(SymbolHandlerId::IsLoopBegin)) != std::string::npos;
        }
        return false;
    }

    bool IsLoopEndCall() const {
        if (opcode_ == SymbolicOpcode::T_MOP_CALL) {
            auto raw = operandList_[0];
            RawSymbolicSymbol *rawSymbol = dynamic_cast<RawSymbolicSymbol *>(raw.get());
            auto callee = rawSymbol->Name();
            return callee.find(SymbolHandler::GetNameByHandlerId(SymbolHandlerId::IsLoopEnd)) != std::string::npos;
        }
        return false;
    }

    Json DumpJson() const override {
        Json exprDump = Json::array();
        exprDump.push_back(kind);
        exprDump.push_back(opcode_);
        for (auto &op : operandList_) {
            exprDump.push_back(op->DumpJson());
        }
        return exprDump;
    }

#define RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_UOP(name, uop) \
        static ScalarImmediateType name(const ScalarImmediateType &val) \
        { \
            return uop val; \
        }
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_UOP(CalcUopPos, +)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_UOP(CalcUopNeg, -)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_UOP(CalcUopNot, !)
#undef  RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_UOP

#define RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(name, bop)                                                \
    static ScalarImmediateType name(const ScalarImmediateType &lhs, const ScalarImmediateType &rhs) { \
        return lhs bop rhs;                                                                           \
    }
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopAdd, +)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopSub, -)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopMul, *)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopDiv, /)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopMod, %)

    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopEq, ==)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopNe, !=)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopLt, <)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopLe, <=)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopGt, >)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopGe, >=)
#undef  RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP

#define RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOPF(name, bfn) \
        static ScalarImmediateType name(const ScalarImmediateType &lhs, const ScalarImmediateType &rhs) \
        { \
            return bfn(lhs, rhs); \
        }
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOPF(CalcBopMin, std::min)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOPF(CalcBopMax, std::max)
#undef  RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOPF

    using SymbolicCalcUnary = ScalarImmediateType (*) (const ScalarImmediateType &);
    static SymbolicCalcUnary GetSymbolicCalcUnary(SymbolicOpcode opcode) {
        static const RawSymbolicExpression::SymbolicCalcUnary CALC_LIST[] = {
            RawSymbolicExpression::CalcUopPos,
            RawSymbolicExpression::CalcUopNeg,
            RawSymbolicExpression::CalcUopNot,
        };
        return CALC_LIST[static_cast<int>(opcode) - static_cast<int>(SymbolicOpcode::T_UOP_BEGIN)];
    }

    using SymbolicCalcBinary = ScalarImmediateType (*)(const ScalarImmediateType &, const ScalarImmediateType &);

    static SymbolicCalcBinary GetSymbolicCalcBinary(SymbolicOpcode opcode) {
        static const RawSymbolicExpression::SymbolicCalcBinary CALC_LIST[] = {
            RawSymbolicExpression::CalcBopAdd,
            RawSymbolicExpression::CalcBopSub,
            RawSymbolicExpression::CalcBopMul,
            RawSymbolicExpression::CalcBopDiv,
            RawSymbolicExpression::CalcBopMod,

            RawSymbolicExpression::CalcBopEq,
            RawSymbolicExpression::CalcBopNe,
            RawSymbolicExpression::CalcBopLt,
            RawSymbolicExpression::CalcBopLe,
            RawSymbolicExpression::CalcBopGt,
            RawSymbolicExpression::CalcBopGe,

            RawSymbolicExpression::CalcBopMin,
            RawSymbolicExpression::CalcBopMax,
        };
        return CALC_LIST[static_cast<size_t>(opcode) - static_cast<size_t>(SymbolicOpcode::T_BOP_BEGIN)];
    }

    static std::string GetSymbolicCalcOpcode(SymbolicOpcode opcode) {
        static const std::string OPCODE_NAME_LIST[] = {
            "+",
            "-",
            "!",

            "+",
            "-",
            "*",
            "/",
            "%",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
            "<:min:>",
            "<:max:>",
        };
        return OPCODE_NAME_LIST[static_cast<size_t>(opcode)];
    }

    static inline ScalarImmediateType CalcMopCall(const std::vector<ScalarImmediateType> &immediateList) {
        ScalarImmediateType result = 0;
        switch (immediateList.size()) {
            // 1 func with not arguments
            case 1: result = reinterpret_cast<ScalarImmediateType(*)()>(immediateList[0])(); break;
            // 2 func with unary operand
            case 2: result = reinterpret_cast<ScalarImmediateType(*)(ScalarImmediateType)>(immediateList[0])(immediateList[1]); break;
            // 3 func with binary operands
            case 3:
                result = reinterpret_cast<ScalarImmediateType(*)(ScalarImmediateType, ScalarImmediateType)>(immediateList[0])(
                    immediateList[1], immediateList[2]); // 2 is arg index
                break;
            // 4 func with ternary operands
            case 4:
                result = reinterpret_cast<ScalarImmediateType(*)(ScalarImmediateType, ScalarImmediateType, ScalarImmediateType)>(
                    immediateList[0])(immediateList[1], immediateList[2], immediateList[3]); // 2 and 3 is arg index
                break;
            default: ASSERT(false); break;
        }
        return result;
    }

    static void Handle2NonzeroOperand(RawSymbolicScalarPtr& raw, SymbolicOpcode opcode, std::vector<RawSymbolicScalarPtr>& nonzeroOperandList) {
        constexpr int size2 = 2;
        ASSERT(nonzeroOperandList.size() == size2);
        if (nonzeroOperandList[0]->IsImmediate()) {
            if (nonzeroOperandList[1]->IsImmediate()) {
                raw = std::make_shared<RawSymbolicImmediate>(
                    std::static_pointer_cast<RawSymbolicImmediate>(nonzeroOperandList[0])->Immediate() +
                    std::static_pointer_cast<RawSymbolicImmediate>(nonzeroOperandList[1])->Immediate());
            } else {
                std::swap(nonzeroOperandList[0], nonzeroOperandList[1]);
                raw = std::make_shared<RawSymbolicExpression>(opcode, nonzeroOperandList);
            }
        } else {
            if (!nonzeroOperandList[1]->IsImmediate() || !nonzeroOperandList[0]->IsExpression()) {
                raw = std::make_shared<RawSymbolicExpression>(opcode, nonzeroOperandList);
            } else {
                auto expression = std::static_pointer_cast<RawSymbolicExpression>(nonzeroOperandList[0]);
                if (expression->Opcode() == SymbolicOpcode::T_BOP_ADD) {
                    ASSERT(expression->OperandList().size() == size2);
                    if (!expression->OperandList()[1]->IsImmediate()) {
                        raw = std::make_shared<RawSymbolicExpression>(opcode, nonzeroOperandList);
                    } else {
                        auto operationList = expression->OperandList();
                        operationList[1] = std::make_shared<RawSymbolicImmediate>(
                            std::static_pointer_cast<RawSymbolicImmediate>(operationList[1])->Immediate() +
                            std::static_pointer_cast<RawSymbolicImmediate>(nonzeroOperandList[1])->Immediate());
                        raw = std::make_shared<RawSymbolicExpression>(opcode, operationList);
                    }
                } else {
                    raw = std::make_shared<RawSymbolicExpression>(opcode, nonzeroOperandList);
                }
            }
        }
    }

    static RawSymbolicScalarPtr Create(SymbolicOpcode opcode, const std::vector<RawSymbolicScalarPtr> &operandList) {
        RawSymbolicScalarPtr raw;
        if (std::all_of(operandList.begin(), operandList.end(),
                [](const RawSymbolicScalarPtr &operand) { return operand->IsImmediate(); })) {
            std::vector<ScalarImmediateType> immediateList(operandList.size());
            std::transform(
                operandList.begin(), operandList.end(), immediateList.begin(), [](const RawSymbolicScalarPtr &val) {
                    return std::static_pointer_cast<RawSymbolicImmediate>(val)->Immediate();
                });

            ScalarImmediateType result = 0;
            if (SymbolicOpcode::T_UOP_BEGIN <= opcode && opcode < SymbolicOpcode::T_UOP_END) {
                ASSERT(immediateList.size() == 1);
                result = RawSymbolicExpression::GetSymbolicCalcUnary(opcode)(immediateList[0]);
            } else if (SymbolicOpcode::T_BOP_BEGIN <= opcode && opcode < SymbolicOpcode::T_BOP_END) {
                result = std::accumulate(immediateList.begin() + 1, immediateList.end(), immediateList[0],
                    [opcode](const ScalarImmediateType &lhs, const ScalarImmediateType &rhs) {
                        return RawSymbolicExpression::GetSymbolicCalcBinary(opcode)(lhs, rhs);
                    });
            } else if (opcode == SymbolicOpcode::T_MOP_CALL) {
                result = CalcMopCall(immediateList);
            } else {
                ASSERT(false);
            }
            raw = std::make_shared<RawSymbolicImmediate>(result);
        } else if (opcode == SymbolicOpcode::T_BOP_ADD) {
            // Tiny optimization: everything + 0 == everything
            std::vector<RawSymbolicScalarPtr> nonzeroOperandList;
            for (auto &op : operandList) {
                if (op->IsImmediate() && std::static_pointer_cast<RawSymbolicImmediate>(op)->Immediate() == 0) {
                } else {
                    nonzeroOperandList.push_back(op);
                }
            }
            if (nonzeroOperandList.size() == 1) {
                raw = nonzeroOperandList[0];
            } else {
                Handle2NonzeroOperand(raw, opcode, nonzeroOperandList);
            }
        } else {
            raw = std::make_shared<RawSymbolicExpression>(opcode, operandList);
        }
        return raw;
    }

#define RAW_SYMBOLIC_EXPRESSION_DEFINE_UOP(name, uop) \
        static RawSymbolicScalarPtr name(const RawSymbolicScalarPtr &val) \
        { \
            RawSymbolicScalarPtr result = Create(uop, {val}); \
            return result; \
        }
    RAW_SYMBOLIC_EXPRESSION_DEFINE_UOP(CreateUopPos, SymbolicOpcode::T_UOP_POS)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_UOP(CreateUopNeg, SymbolicOpcode::T_UOP_NEG)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_UOP(CreateUopNot, SymbolicOpcode::T_UOP_NOT)
#undef  RAW_SYMBOLIC_EXPRESSION_DEFINE_UOP
#define RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(name, bop) \
    static RawSymbolicScalarPtr name(const RawSymbolicScalarPtr &lhs, const RawSymbolicScalarPtr &rhs) { \
        RawSymbolicScalarPtr result = Create(bop, {lhs, rhs});                                           \
        return result;                                                                                   \
    }
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopAdd, SymbolicOpcode::T_BOP_ADD)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopSub, SymbolicOpcode::T_BOP_SUB)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopMul, SymbolicOpcode::T_BOP_MUL)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopDiv, SymbolicOpcode::T_BOP_DIV)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopMod, SymbolicOpcode::T_BOP_MOD)

    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopEq, SymbolicOpcode::T_BOP_EQ)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopNe, SymbolicOpcode::T_BOP_NE)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopLt, SymbolicOpcode::T_BOP_LT)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopLe, SymbolicOpcode::T_BOP_LE)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopGt, SymbolicOpcode::T_BOP_GT)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopGe, SymbolicOpcode::T_BOP_GE)

    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopMin, SymbolicOpcode::T_BOP_MIN)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopMax, SymbolicOpcode::T_BOP_MAX)
#undef  RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP

    static RawSymbolicScalarPtr CreateMopCall(const RawSymbolicScalarPtr &callee) {
        RawSymbolicScalarPtr result = Create(SymbolicOpcode::T_MOP_CALL, {callee});
        return result;
    }
    static RawSymbolicScalarPtr CreateMopCall(const RawSymbolicScalarPtr &callee, const RawSymbolicScalarPtr &arg0) {
        RawSymbolicScalarPtr result = Create(SymbolicOpcode::T_MOP_CALL, {callee, arg0});
        return result;
    }
    static RawSymbolicScalarPtr CreateMopCall(
        const RawSymbolicScalarPtr &callee, const RawSymbolicScalarPtr &arg0, const RawSymbolicScalarPtr &arg1) {
        RawSymbolicScalarPtr result = Create(SymbolicOpcode::T_MOP_CALL, {callee, arg0, arg1});
        return result;
    }
    static RawSymbolicScalarPtr CreateMopCall(const std::vector<RawSymbolicScalarPtr> &calleeArgs) {
        RawSymbolicScalarPtr result = Create(SymbolicOpcode::T_MOP_CALL, calleeArgs);
        return result;
    }

private:
    void DumpBuffer(std::string &buffer) const override {
        if (SymbolicOpcode::T_UOP_BEGIN <= opcode_ && opcode_ < SymbolicOpcode::T_UOP_END) {
            buffer += "(";
            buffer += GetSymbolicCalcOpcode(opcode_);
            operandList_[0]->DumpBuffer(buffer);
            buffer += ")";
        } else if (SymbolicOpcode::T_BOP_BEGIN <= opcode_ && opcode_ < SymbolicOpcode::T_BOP_END) {
            if (opcode_ == SymbolicOpcode::T_BOP_MAX) {
                buffer += "RUNTIME_Max(";
                operandList_[0]->DumpBuffer(buffer);
                buffer += ", ";
                operandList_[1]->DumpBuffer(buffer);
                buffer += ")";
            } else if (opcode_ == SymbolicOpcode::T_BOP_MIN) {
                buffer += "RUNTIME_Min(";
                operandList_[0]->DumpBuffer(buffer);
                buffer += ", ";
                operandList_[1]->DumpBuffer(buffer);
                buffer += ")";
            } else {
                buffer += "(";
                for (size_t i = 0; i < operandList_.size(); i++) {
                    if (i != 0) {
                        buffer += GetSymbolicCalcOpcode(opcode_);
                    }
                    operandList_[i]->DumpBuffer(buffer);
                }
                buffer += ")";
            }
        } else if (opcode_ == SymbolicOpcode::T_MOP_CALL) {
            operandList_[0]->DumpBuffer(buffer);
            buffer += "(";
            for (size_t i = 1; i < operandList_.size(); i++) {
                if (i != 1) {
                    buffer += ",";
                }
                operandList_[i]->DumpBuffer(buffer);
            }
            buffer += ")";
        }
    }

    SymbolicOpcode opcode_;
    std::vector<RawSymbolicScalarPtr> operandList_;
};

inline std::string AddRuntimePrefix(const std::string &name) {
    return SPECIAL_SYMBOL_NAME_RUNTIME_PREFIX + name;
}
inline bool CheckRuntimePrefix(const std::string &name) {
    return StringUtils::StartsWith(name, SPECIAL_SYMBOL_NAME_RUNTIME_PREFIX);
}
inline std::string AddRuntimeCoaPrefix(const std::string &name) {
    return SPECIAL_SYMBOL_NAME_RUNTIME_COA_PREFIX + name;
}
inline std::string AddArgPrefix(const std::string &name) {
    return SPECIAL_SYMBOL_NAME_ARG_PREFIX + name;
}
inline bool CheckArgPrefix(const std::string &name) {
    return StringUtils::StartsWith(name, SPECIAL_SYMBOL_NAME_ARG_PREFIX);
}

Json ToJson(const SymbolicScalar &sval);

SymbolicScalar LoadSymbolicScalar(const Json &jval);

} // namespace npu::tile_fwk

namespace npu::tile_fwk {
class SymbolicClosure {
public:
    std::shared_ptr<std::unordered_map<std::string, ScalarImmediateType>> symbolValueDict;

    SymbolicClosure() { symbolValueDict = std::make_shared<std::unordered_map<std::string, ScalarImmediateType>>(); }
    void Insert(const std::string &name, ScalarImmediateType val) { symbolValueDict->insert({name, val}); }
    void Remove(const std::string &name) { symbolValueDict->erase(name); }
    bool Has(const std::string &name) const { return symbolValueDict->count(name) != 0; }
    ScalarImmediateType Get(const std::string &name) const { return symbolValueDict->find(name)->second; }

    ScalarImmediateType Evaluate(const SymbolicScalar &scalar) const { return Evaluate(scalar.Raw()); }

private:
    ScalarImmediateType EvaluateExpression(const RawSymbolicExpression *expr) const {
        std::vector<ScalarImmediateType> dataList;
        for (auto &operand : expr->OperandList()) {
            dataList.emplace_back(Evaluate(operand));
        }
        ScalarImmediateType result = 0;
        if (SymbolicOpcode::T_UOP_BEGIN <= expr->Opcode() && expr->Opcode() < SymbolicOpcode::T_UOP_END) {
            result = RawSymbolicExpression::GetSymbolicCalcUnary(expr->Opcode())(dataList[0]);
        } else if (SymbolicOpcode::T_BOP_BEGIN <= expr->Opcode() && expr->Opcode() < SymbolicOpcode::T_BOP_END) {
            result = dataList[0];
            for (size_t i = 1; i < dataList.size(); i++) {
                result = RawSymbolicExpression::GetSymbolicCalcBinary(expr->Opcode())(result, dataList[i]);
            }
        }
        return result;
    }

    ScalarImmediateType Evaluate(const RawSymbolicScalarPtr &raw) const {
        ScalarImmediateType result{INVALID_SCALAR_IMMEDIATE};
        switch (raw->Kind()) {
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
                RawSymbolicImmediate *immediate = dynamic_cast<RawSymbolicImmediate *>(raw.get());
                result = immediate->Immediate();
            } break;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
                RawSymbolicSymbol *symbol = dynamic_cast<RawSymbolicSymbol *>(raw.get());
                ASSERT(symbolValueDict->count(symbol->Name()));
                result = symbolValueDict->find(symbol->Name())->second;
            } break;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
                RawSymbolicExpression *expr = dynamic_cast<RawSymbolicExpression *>(raw.get());
                result = EvaluateExpression(expr);
            } break;
            default: ASSERT(false); break;
        }
        return result;
    }
};

struct SymbolicSymbolTable {
    std::map<std::string, int> symbolIndexTable;

    std::string Dump() const {
        std::ostringstream oss;
        for (auto &[symbol, index] : symbolIndexTable) {
            oss << "Symbol:" << index << " name:" << symbol << "\n";
        }
        return oss.str();
    }

    int GetSymbolTableSize() const { return symbolIndexTable.size(); }

    std::vector<SymbolHandler> GetSymbolHandlerList() const {
        std::vector<SymbolHandler> handlerList;
        for (auto &[name, index] : symbolIndexTable) {
            if (symbolHandlerIndexDict.count(name)) {
                handlerList.emplace_back(symbolHandlerIndexDict.find(name)->second, index);
            }
        }
        return handlerList;
    }

    int LookupSymbolIndex(const std::string &name) const {
        ASSERT(symbolIndexTable.count(name));
        return symbolIndexTable.find(name)->second;
    }

    std::string BuildHeader() const {
        std::ostringstream oss;
        oss << "#include \"machine/utils/codegen/codegen.h\"\n";
        return oss.str();
    }

    std::string BuildSymbolList() const {
        std::ostringstream oss;
        for (auto &[name, index] : symbolIndexTable) {
            oss << "#define INDEX_" << name << " " << index << "\n";
        }
        for (auto &[name, index] : symbolIndexTable) {
            (void) index;
            oss << "#define VALUE_" << name << " ((symbolTable)[INDEX_" << name << "])\n";
        }
        return oss.str();
    }
};

struct SymbolicExpressionTable {
    std::map<std::string, int> expressionIndexTable;
    std::vector<std::string> sourceList;

    std::string Dump() const {
        std::ostringstream oss;
        for (auto &[expr, index] : expressionIndexTable) {
            oss << "Expression:" << index << " code:" << expr << "\n";
        }
        return oss.str();
    }

    static std::string BuildExpression(const SymbolicScalar &ss);

    int LookupExpressionIndex(const SymbolicScalar &ss) const {
        std::string str = BuildExpression(ss);
        ASSERT(expressionIndexTable.count(str));
        return expressionIndexTable.find(str)->second;
    }

    void InsertExpressionIndex(const std::string &expr, int index) {
        expressionIndexTable[expr] = index;
    }
};

struct SymbolicSymbolTableBuilder {
    std::set<std::string> symbolTable;

    void AddSymbolFromExpression(const SymbolicScalar &ss) {
        if (ss.Raw()->IsImmediate()) {
            return;
        }
        AddAllSymbol(ss.Raw());
    }

    void AddSymbol(const SymbolicScalar &ss) {
        ASSERT(ss.Raw()->Kind() == SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL);
        auto raw = ss.Raw();
        symbolTable.insert(dynamic_cast<RawSymbolicSymbol *>(raw.get())->Name());
    }

    SymbolicSymbolTable BuildAndLoad();

    std::string Dump() const {
        std::ostringstream oss;
        for (auto &name : symbolTable) {
            oss << "name:" << name << "\n";
        }
        return oss.str();
    }

private:
    void AddAllSymbol(const RawSymbolicScalarPtr &raw) {
        switch (raw->Kind()) {
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: break;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
                RawSymbolicSymbol *symbol = dynamic_cast<RawSymbolicSymbol *>(raw.get());
                symbolTable.insert(symbol->Name());
            } break;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
                RawSymbolicExpression *expr = dynamic_cast<RawSymbolicExpression *>(raw.get());
                for (auto &operand : expr->OperandList()) {
                    AddAllSymbol(operand);
                }
            } break;
            default: ASSERT(false); break;
        }
    }
};

struct SymbolicExpressionTableBuilder {
    std::map<std::string, RawSymbolicScalarPtr> expressionTable;
    OrderedSet<RawSymbolicScalarPtr> expressionSet;
    OrderedSet<RawSymbolicScalarPtr> primaryExpressionSet;

    void AddPrimaryExpression(const SymbolicScalar &ss) {
        if (ss.Raw()->IsImmediate()) {
            return;
        }
        std::string str = BuildExpressionByRaw(ss.Raw(), {});
        expressionTable.emplace(str, ss.Raw());
        primaryExpressionSet.Insert(ss.Raw());
        AddExpression(ss.Raw());
    }

    SymbolicExpressionTable BuildAndLoad();
    const OrderedSet<RawSymbolicScalarPtr> &GetExpressionSet() const {
        return expressionSet;
    }

    std::string Dump() const {
        std::ostringstream oss;
        for (auto &expr: expressionTable) {
            oss << "expr:" << BuildExpressionByRaw(expr.second, {}) << "\n";
        }
        return oss.str();
    }

    std::string BuildExpressionList(const std::string &prefix, const std::string &title) const;

    static std::string GetExprNameLoopPrefix(int funcKey) { return "EXPR_LOOP_" + std::to_string(funcKey); }
    static std::string GetExprNameRootPrefix(int funcKey) { return "EXPR_ROOT_" + std::to_string(funcKey); }
    static std::string GetExprNameLeafPrefix(int funcKey) { return "EXPR_LEAF_" + std::to_string(funcKey); }

    static std::string GetExprNameTempVarFlag(const std::string &prefix, int index) { return prefix + "_" + std::to_string(index) + "_TEMP_VAR_FLAG"; }
    static std::string GetExprNameTempVar(const std::string &prefix, int index) { return prefix + "_" + std::to_string(index) + "_TEMP_VAR"; }
    static std::string GetExprNameTempVarInit(const std::string &prefix, int index) { return prefix + "_" + std::to_string(index) + "_TEMP_INIT"; }
    static std::string GetExprNameCalc(const std::string &prefix, int index) { return prefix + "_" + std::to_string(index) + "_CALC"; }
    static std::string GetExprNameGet(const std::string &prefix, int index) { return prefix + "_" + std::to_string(index) + "_GET"; }

    static std::string BuildExpressionByRaw(const RawSymbolicScalarPtr &raw, const std::unordered_map<RawSymbolicScalarPtr, std::string> &exprDict);

private:
    static std::string BuildExpressionCode(const RawSymbolicExpression *expr, const std::unordered_map<RawSymbolicScalarPtr, std::string> &exprDict);

    void AddExpression(const RawSymbolicScalarPtr &raw) {
        switch (raw->Kind()) {
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            } break;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            } break;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
                for (auto &operand : raw->GetExpressionOperandList()) {
                    AddExpression(operand);
                }
                expressionSet.Insert(raw);
            } break;
            default: ASSERT(false); break;
        }
    }
};

bool ToolchainExist(const std::string &gcc);

std::vector<uint8_t> CompileAndLoadSection(const std::string &code, const std::string &sourceFilePath,
    const std::string &gcc, const std::string &objcopy, const std::string &sectionName, const std::string &extraCflag="");

void CompileAndLink(const std::string &code, const std::string &sourceFilePath,
        const std::string &gcc, bool isStaticLink, bool isBenchmark, bool useMakefile);

std::string CompileCopyLink(const std::string &code, const std::string &sourceFilePath,
        const std::string &gcc, const std::string &objcopy, bool isStaticLink, bool isBenchmark,
        const std::map<std::string, std::string> &sectionDataDict);

void RunMake(const std::string &makefilePath);
} // namespace npu::tile_fwk
