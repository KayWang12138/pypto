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
 * \file symbolic_scalar.cpp
 * \brief
 */

#include "interface/tensor/symbolic_scalar.h"

#include <sys/mman.h>

#include <sstream>
#include "interface/utils/log.h"

#ifdef SRCPATH
constexpr const char *SrcPath = SRCPATH;
#else
constexpr const char *SrcPath = ".";
#endif
constexpr uint64_t IMMEDIATE = 0;
constexpr uint64_t SYMBOL = 1;
constexpr uint64_t EXPRESSION = 2;
constexpr int OPERAND_NUM = 2;
namespace npu::tile_fwk {

bool ToolchainExist(const std::string &gcc) {
    std::string cmdGcc = gcc + " --version";
    if (system(cmdGcc.c_str()) != 0) {
        return false;
    }
    return true;
}

std::vector<uint8_t> CompileAndLoadSection(const std::string &code, const std::string &sourceFilePath,
    const std::string &gcc, const std::string &objcopy, const std::string &sectionName, const std::string &extraCflag) {
    FILE *fsrc = fopen(sourceFilePath.c_str(), "w");
    fprintf(fsrc, "%s", code.c_str());
    fclose(fsrc);

    std::string assembleFilePath = sourceFilePath + ".s";
    std::string objectFilePath = sourceFilePath + ".o";
    std::string binaryFilePath = sourceFilePath + ".bin";
    std::string LD_PRELOAD = "LD_PRELOAD= ";

    std::string cmdGcc = LD_PRELOAD + gcc + " -fPIC -O2 " + extraCflag +
        " -I" + std::string(SrcPath) + "/src " +
        " -I" + std::string(SrcPath) + "/src/interface " +
        " -S " + sourceFilePath + " -o " + assembleFilePath;
    ALOG_INFO("[RunCmd] ", cmdGcc);
    ASSERT(system(cmdGcc.c_str()) == 0);

    std::string cmdAs = LD_PRELOAD + gcc + " -O2 -c " + assembleFilePath + " -o " + objectFilePath;
    ALOG_INFO("[RunCmd] ", cmdAs);
    ASSERT(system(cmdAs.c_str()) == 0);

    std::string cmdObjcopy = LD_PRELOAD + objcopy + " --dump-section " + sectionName + "=" + binaryFilePath + " " + objectFilePath;
    ALOG_INFO("[RunCmd] ", cmdObjcopy);
    ASSERT(system(cmdObjcopy.c_str()) == 0);

    FILE *fbin = fopen(binaryFilePath.c_str(), "rb");
    if (fbin == nullptr) {
        ALOG_FATAL("open binary file name failed");
        return {};
    }

    fseek(fbin, 0, SEEK_END);
    int size = ftell(fbin);
    fseek(fbin, 0, SEEK_SET);
    std::vector<uint8_t> binary(size);
    fread(binary.data(), 1, size, fbin);
    fclose(fbin);
    return binary;
}

SymbolicSymbolTable SymbolicSymbolTableBuilder::BuildAndLoad() {
    SymbolicSymbolTable table;
    int symbolIndex = 0;
    for (auto &name : symbolTable) {
        table.symbolIndexTable[name] = symbolIndex++;
    }
    return table;
}

SymbolicExpressionTable SymbolicExpressionTableBuilder::BuildAndLoad() {
    SymbolicExpressionTable table;
    int expressionIndex = 0;
    for (auto &ele : expressionTable) {
        table.expressionIndexTable[ele.first] = {ele.second, expressionIndex++};
    }
    table.BuildSource("tile_fwk");
    return table;
}

ScalarImmediateType RawSymbolicScalar::GetImmediateValue() const {
    ASSERT(IsImmediate()) << "Mismatch immediate type: " << SymbolicScalarKind2Name(Kind());
    auto immediate = static_cast<const RawSymbolicImmediate *>(this);
    return immediate->Immediate();
}
const std::string &RawSymbolicScalar::GetSymbolName() const {
    ASSERT(IsSymbol()) << "Mismatch symbol type: " << SymbolicScalarKind2Name(Kind());
    auto symbol = static_cast<const RawSymbolicSymbol *>(this);
    return symbol->Name();
}
SymbolicOpcode RawSymbolicScalar::GetExpressionOpcode() const {
    ASSERT(IsExpression()) << "Mismatch expression type: " << SymbolicScalarKind2Name(Kind());
    auto expression = static_cast<const RawSymbolicExpression *>(this);
    return expression->Opcode();
}
const std::vector<RawSymbolicScalarPtr> &RawSymbolicScalar::GetExpressionOperandList() const {
    ASSERT(IsExpression()) << "Mismatch expression type: " << SymbolicScalarKind2Name(Kind());
    auto expression = static_cast<const RawSymbolicExpression *>(this);
    return expression->OperandList();
}

bool RawSymbolicScalar::IsExpressionCall(const std::string &calleeName) const {
    if (!IsExpression()) {
        return false;
    }
    if (GetExpressionOpcode() != SymbolicOpcode::T_MOP_CALL) {
        return false;
    }
    auto caller = GetExpressionOperandList()[0];
    if (!caller->IsSymbol()) {
        return false;
    }
    if (caller->GetSymbolName() != calleeName) {
        return false;
    }
    return true;
}

static void DumpSymbolicScalar(const RawSymbolicScalarPtr &raw, Json &jarray) {
    switch (raw->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            jarray.emplace_back(IMMEDIATE);
            RawSymbolicImmediate *immediate = dynamic_cast<RawSymbolicImmediate *>(raw.get());
            jarray.emplace_back(static_cast<uint64_t>(immediate->Immediate()));
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            jarray.emplace_back(SYMBOL);
            RawSymbolicSymbol *symbol = dynamic_cast<RawSymbolicSymbol *>(raw.get());
            jarray.emplace_back(symbol->Name());
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            jarray.emplace_back(EXPRESSION);
            RawSymbolicExpression *expr = dynamic_cast<RawSymbolicExpression *>(raw.get());
            jarray.emplace_back(static_cast<int32_t>(expr->Opcode()));
            if (expr->Opcode() == SymbolicOpcode::T_MOP_CALL) {
                jarray.emplace_back(static_cast<int32_t>(expr->OperandList().size()));
            }
            for (auto &op : expr->OperandList()) {
                DumpSymbolicScalar(op, jarray);
            }
        } break;
        default: ASSERT(false); break;
    }
}

Json ToJson(const SymbolicScalar &sval) {
    Json jdata;
    DumpSymbolicScalar(sval.Raw(), jdata);
    return jdata;
}

static RawSymbolicScalarPtr LoadRawSymbolicScalar(const Json &symbolicJson, int &despos) {
    RawSymbolicScalarPtr raw;
    SymbolicScalarKind kind = static_cast<SymbolicScalarKind>(symbolicJson[despos++]);
    switch (kind) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            uint64_t immediateData = static_cast<uint64_t>(symbolicJson[despos++]);
            raw = std::static_pointer_cast<RawSymbolicScalar>(std::make_shared<RawSymbolicImmediate>(immediateData));
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            std::string nameData = static_cast<std::string>(symbolicJson[despos++]);
            raw = std::static_pointer_cast<RawSymbolicScalar>(std::make_shared<RawSymbolicSymbol>(nameData));
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            SymbolicOpcode opcode = static_cast<SymbolicOpcode>(symbolicJson[despos++]);
            std::vector<RawSymbolicScalarPtr> operandList;
            if (opcode == SymbolicOpcode::T_MOP_CALL) {
                int size = symbolicJson[despos++];
                for (int i = 0; i < size; i++) {
                    operandList.push_back(LoadRawSymbolicScalar(symbolicJson, despos));
                }
            } else {
                for (int i = 0; i < OPERAND_NUM; i++) {
                    operandList.push_back(LoadRawSymbolicScalar(symbolicJson, despos));
                }
            }
            raw = std::static_pointer_cast<RawSymbolicScalar>(std::make_shared<RawSymbolicExpression>(opcode, operandList));
        } break;
        default: break;
    }
    return raw;
}

SymbolicScalar LoadSymbolicScalar(const Json &jval) {
    int pos = 0;
    return SymbolicScalar(LoadRawSymbolicScalar(jval, pos));
}

void SymbolicScalar::AsIntermediateVariable() {
    raw_->AsIntermediateVariable();
}

bool SymbolicScalar::IsIntermediateVariable() const {
    return raw_->IsIntermediateVariable();
}

#define SYMBOLIC_SCALAR_DEFINE_UOP(name, uop, rawname)  \
    SymbolicScalar SymbolicScalar::name() const {       \
        auto raw = rawname(raw_);                       \
        if (ConcreteValid()) {                          \
            return SymbolicScalar(raw, uop Concrete()); \
        } else {                                        \
            return SymbolicScalar(raw);                 \
        }                                               \
    }
SYMBOLIC_SCALAR_DEFINE_UOP(Pos, +, RawSymbolicExpression::CreateUopPos)
SYMBOLIC_SCALAR_DEFINE_UOP(Neg, -, RawSymbolicExpression::CreateUopNeg)
SYMBOLIC_SCALAR_DEFINE_UOP(Not, !, RawSymbolicExpression::CreateUopNot)
#undef SYMBOLIC_SCALAR_DEFINE_UOP

#define SYMBOLIC_SCALAR_DEFINE_BOP(name, bop, rawname)                      \
    SymbolicScalar SymbolicScalar::name(const SymbolicScalar &sval) const { \
        auto raw = rawname(raw_, sval.raw_);                                \
        if (ConcreteValid() && sval.ConcreteValid()) {                      \
            return SymbolicScalar(raw, Concrete() bop sval.Concrete());     \
        } else {                                                            \
            return SymbolicScalar(raw);                                     \
        }                                                                   \
    }

SYMBOLIC_SCALAR_DEFINE_BOP(Add, +, RawSymbolicExpression::CreateBopAdd)
SYMBOLIC_SCALAR_DEFINE_BOP(Sub, -, RawSymbolicExpression::CreateBopSub)
SYMBOLIC_SCALAR_DEFINE_BOP(Mul, *, RawSymbolicExpression::CreateBopMul)
SYMBOLIC_SCALAR_DEFINE_BOP(Div, /, RawSymbolicExpression::CreateBopDiv)
SYMBOLIC_SCALAR_DEFINE_BOP(Mod, %, RawSymbolicExpression::CreateBopMod)
SYMBOLIC_SCALAR_DEFINE_BOP(Eq, ==, RawSymbolicExpression::CreateBopEq)
SYMBOLIC_SCALAR_DEFINE_BOP(Ne, !=, RawSymbolicExpression::CreateBopNe)
SYMBOLIC_SCALAR_DEFINE_BOP(Lt, <, RawSymbolicExpression::CreateBopLt)
SYMBOLIC_SCALAR_DEFINE_BOP(Le, <=, RawSymbolicExpression::CreateBopLe)
SYMBOLIC_SCALAR_DEFINE_BOP(Gt, >, RawSymbolicExpression::CreateBopGt)
SYMBOLIC_SCALAR_DEFINE_BOP(Ge, >=, RawSymbolicExpression::CreateBopGe)
#undef SYMBOLIC_SCALAR_DEFINE_BOP

static bool AllConcreteValid(const std::vector<SymbolicScalar> &slist) {
    for (auto &s : slist) {
        if (!s.ConcreteValid()) {
            return false;
        }
    }
    return true;
}

SymbolicScalar SymbolicScalar::operator()() const {
    auto raw = RawSymbolicExpression::CreateMopCall(raw_);
    if (ConcreteValid()) {
        return SymbolicScalar(raw, RawSymbolicExpression::CalcMopCall({Concrete()}));
    } else {
        return SymbolicScalar(raw);
    }
}
SymbolicScalar SymbolicScalar::operator()(const SymbolicScalar &arg0) const {
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    if (AllConcreteValid({*this, arg0})) {
        return SymbolicScalar(raw, RawSymbolicExpression::CalcMopCall({Concrete(), arg0.Concrete()}));
    } else {
        return SymbolicScalar(raw);
    }
}
SymbolicScalar SymbolicScalar::operator()(const SymbolicScalar &arg0, const SymbolicScalar &arg1) const {
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_, arg1.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    if (AllConcreteValid({*this, arg0, arg1})) {
        return SymbolicScalar(raw, RawSymbolicExpression::CalcMopCall({Concrete(), arg0.Concrete(), arg1.Concrete()}));
    } else {
        return SymbolicScalar(raw);
    }
}
SymbolicScalar SymbolicScalar::operator()(
    const SymbolicScalar &arg0, const SymbolicScalar &arg1, const SymbolicScalar &arg2) const {
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_, arg1.raw_, arg2.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    if (AllConcreteValid({*this, arg0, arg1, arg2})) {
        return SymbolicScalar(
            raw, RawSymbolicExpression::CalcMopCall({Concrete(), arg0.Concrete(), arg1.Concrete(), arg2.Concrete()}));
    } else {
        return SymbolicScalar(raw);
    }
}
SymbolicScalar SymbolicScalar::operator()(const SymbolicScalar &arg0, const SymbolicScalar &arg1,
    const SymbolicScalar &arg2, const SymbolicScalar &arg3) const {
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_, arg1.raw_, arg2.raw_, arg3.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    if (AllConcreteValid({*this, arg0, arg1, arg2, arg3})) {
        return SymbolicScalar(raw, RawSymbolicExpression::CalcMopCall({Concrete(), arg0.Concrete(), arg1.Concrete(),
                                       arg2.Concrete(), arg3.Concrete()}));
    } else {
        return SymbolicScalar(raw);
    }
}
SymbolicScalar SymbolicScalar::operator()(const SymbolicScalar &arg0, const SymbolicScalar &arg1,
    const SymbolicScalar &arg2, const SymbolicScalar &arg3, const SymbolicScalar &arg4) const {
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_, arg1.raw_, arg2.raw_, arg3.raw_, arg4.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    if (AllConcreteValid({*this, arg0, arg1, arg2, arg3, arg4})) {
        return SymbolicScalar(raw, RawSymbolicExpression::CalcMopCall({Concrete(), arg0.Concrete(), arg1.Concrete(),
                                       arg2.Concrete(), arg3.Concrete()}));
    } else {
        return SymbolicScalar(raw);
    }
}

SymbolicScalar SymbolicScalar::operator()(const std::vector<SymbolicScalar> &argList) const {
    std::vector<RawSymbolicScalarPtr> args = {raw_};
    for (auto &a : argList) {
        args.push_back(a.raw_);
    }
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    if (AllConcreteValid({*this}) && AllConcreteValid(argList)) {
        std::vector<ScalarImmediateType> calcArgList = {Concrete()};
        for (auto &a : argList) {
            calcArgList.push_back(a.Concrete());
        }
        return SymbolicScalar(raw, RawSymbolicExpression::CalcMopCall(calcArgList));
    } else {
        return SymbolicScalar(raw);
    }
}

std::string SymbolicScalar::Dump() const {
    std::string buf;
    raw_->DumpBuffer(buf);
    return buf;
}

bool SymbolicScalar::IsImmediate() const {
    return raw_ && raw_->IsImmediate();
}
bool SymbolicScalar::IsSymbol() const {
    return raw_ && raw_->IsSymbol();
}
bool SymbolicScalar::IsExpression() const {
    return raw_ && raw_->IsExpression();
}

SymbolicScalar SymbolicScalar::Min(const SymbolicScalar &sval) const {
    auto raw = RawSymbolicExpression::CreateBopMin(raw_, sval.raw_);
    if (ConcreteValid() && sval.ConcreteValid()) {
        return SymbolicScalar(raw, std::min(Concrete(), sval.Concrete()));
    } else if (sval.Dump() == Dump()) {
        return sval;
    } else {
        return SymbolicScalar(raw);
    }
}

SymbolicScalar SymbolicScalar::Max(const SymbolicScalar &sval) const {
    auto raw = RawSymbolicExpression::CreateBopMax(raw_, sval.raw_);
    if (ConcreteValid() && sval.ConcreteValid()) {
        return SymbolicScalar(raw, std::max(Concrete(), sval.Concrete()));
    } else if (sval.Dump() == Dump()) {
        return sval;
    } else {
        return SymbolicScalar(raw);
    }
}

SymbolicScalar::SymbolicScalar(int64_t value)
    : raw_(RawSymbolicImmediate::Create(value)), concreteValid_(true), concrete_(value) {}
SymbolicScalar::SymbolicScalar(const std::string &name) : raw_(RawSymbolicSymbol::Create(name)) {}
SymbolicScalar::SymbolicScalar(const std::string &name, int64_t value)
    : raw_(RawSymbolicSymbol::Create(name)), concreteValid_(true), concrete_(value) {}
SymbolicScalar::SymbolicScalar(const std::string &name, NotLessThan minVal)
    : raw_(RawSymbolicSymbol::Create(name, ValueGuesser(minVal))) {}
SymbolicScalar::SymbolicScalar(const std::string &name, NotGreaterThan maxVal)
    : raw_(RawSymbolicSymbol::Create(name, ValueGuesser(maxVal))) {}
SymbolicScalar::SymbolicScalar(const std::string &name, NotLessThan minVal, NotGreaterThan maxVal)
    : raw_(RawSymbolicSymbol::Create(name, ValueGuesser(minVal, maxVal))) {}
SymbolicScalar::SymbolicScalar(RawSymbolicScalarPtr raw, int64_t concrete)
    : raw_(raw), concreteValid_(true), concrete_(concrete) {}
SymbolicScalar::SymbolicScalar(RawSymbolicScalarPtr raw) : raw_(raw) {
    if (raw_->IsImmediate()) {
        concreteValid_ = true;
        concrete_ = dynamic_cast<RawSymbolicImmediate *>(raw.get())->Immediate();
    }
}

std::vector<int> SymbolicScalar::Concrete(const std::vector<SymbolicScalar> &scalarList, int64_t defValue) {
    std::vector<int> concreteList;
    for (auto &s : scalarList) {
        if (s.ConcreteValid()) {
            concreteList.push_back(s.Concrete());
        } else {
            concreteList.push_back(defValue);
        }
    }
    return concreteList;
}

std::vector<SymbolicScalar> SymbolicScalar::FromConcrete(const std::vector<int> &values) {
    std::vector<SymbolicScalar> result;
    for (auto x : values) {
        result.push_back(SymbolicScalar(x));
    }
    return result;
}

void RawSymbolicScalar::ResetValueGuesser(ValueGuesser valueGuesser) {
    ASSERT(valueGuesser.IsCalculated());
    valueGuesser_ = valueGuesser;
}

void IntermediateVariableTable::GetAllIntermediateVariables(RawSymbolicScalar &scalar, std::vector<IntermediateVariableInfo> &result) {
    if (!scalar.IsExpression()) {
        return;
    }
    if (mapping_.count(&scalar) > 0) {
        return;
    }
    auto &curExpression = dynamic_cast<RawSymbolicExpression &>(scalar);
    for (const auto &operand : curExpression.OperandList()) {
        if (!operand->IsExpression()) {
            continue;
        }
        if (mapping_.count(operand.get()) > 0) {
            continue;
        }
        GetAllIntermediateVariables(*operand, result);
    }

    if (!scalar.IsIntermediateVariable()) {
        return;
    }
    auto expressionStr = BuildExpressionByRaw(scalar, false);
    for (size_t i = 0; i < infos_.size(); i++) {
        if (expressionStr == infos_[i].expression) { // 规避由于相同表达式不同变量导致的重复定义问题
            mapping_.emplace(&scalar, i);
            return;
        }
    }
    mapping_.emplace(&scalar, infos_.size());
    infos_.emplace_back(namePrefix_ + std::to_string(infos_.size()), expressionStr);
    result.emplace_back(infos_.back());
}

} // namespace npu::tile_fwk
