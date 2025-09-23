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
 * \file codegen_vector.cpp
 * \brief
 */

#include "interface/utils/log.h"
#include "interface/tensor/logical_tensor.h"
#include "codegen_op_cloudnpu.h"
#include "securec.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/symbol_mgr/codegen_symbol.h"

namespace npu::tile_fwk {
std::string CodeGenOpCloudNPU::PrintCastDynamicUnaligned(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    std::vector<int64_t> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    auto dynDstShape = dynamicValidShape[0];
    std::vector<SymbolicScalar> newDynDstShape = dynDstShape;
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynDstShape, SHAPE_DIM4 - dynDstShape.size(), 1);
    paramList.insert(paramList.end(), {dstDtypeStr, srcDtypeStr});
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ss[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dst, src});
    for (auto dynShape : newDynDstShape) {
        paramList.emplace_back(dynShape.Dump());
    }
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}
std::string CodeGenOpCloudNPU::GenCastOp() const {
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector srcShape = this->rawShape[ID1];
    ALOG_INFO_F("genCastOp %s, srcShape is %s", tileOpName.c_str(), IntVecToStr(srcShape).c_str());

    std::vector dstShape = this->rawShape[ID0];
    ALOG_INFO_F("genCastOp %s, dstShape is %s", tileOpName.c_str(), IntVecToStr(dstShape).c_str());

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufferVarOffset(std::vector{&dVar, &s0Var});
    std::vector<int64_t> os = NormalizeShape(originShape[0], SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int ret = 0;
    if (isSupportDynamicUnaligned) {
        return PrintCastDynamicUnaligned({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    }

    ret = sprintf_s(buffer, sizeof(buffer),
        "%s_<%s, %s, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u %s>((__ubuf__ %s *)%s,  (__ubuf__ %s *)%s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), srcDtypeStr.c_str(), os[0], os[1], os[2], os[3], ds[1], ds[2], ds[3],
        ss[1], ss[2], ss[3], GenOpAttr().c_str(), dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(),
        s0Var.c_str());
    ASSERT(ret >= 0) << "GenCastOp sprintf_s failed " << ret;
    std::string ostring(buffer);
    return ostring;
}

std::string CodeGenOpCloudNPU::PrintDupOpDynUnaligned(const PrintDupOpParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &dupV = param.dupV;
    // dst origin shape
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    paramList.insert(paramList.end(), {dst, dupV});
    auto dynDstShape = dynamicValidShape[0];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynDstShape, SHAPE_DIM4 - dynDstShape.size(), 1);
    for (auto dstOriShape : dynDstShape) {
        paramList.emplace_back(dstOriShape.Dump());
    }
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintDupOpStatic(const PrintDupOpParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &dupV = param.dupV;
    // dst origin shape
    std::vector<int64_t> dos = NormalizeShape(originShape[0], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (auto oriShape : dos) {
        paramList.emplace_back(std::to_string(oriShape));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    paramList.insert(paramList.end(), {dst, dupV});
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintDupOp(const PrintDupOpParam &param) const {
    if (isSupportDynamicUnaligned) {
        return PrintDupOpDynUnaligned(param);
    }
    return PrintDupOpStatic(param);
}

std::string CodeGenOpCloudNPU::GenDupOp() const {
    auto kDst = sm->CreateAllocKey(operandWithMagic[ID0]);
    std::string dVar = sm->QueryVariableName(kDst);
    AppendLocalBufferVarOffset(std::vector{&dVar});
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    std::string dupV;
    if (opAttrs.count(OpAttributeKey::dynScalar)) {
        auto scalar = opAttrs.at(OpAttributeKey::dynScalar);
        ASSERT((scalar.HasValue()) && (scalar.Type() == typeid(SymbolicScalar)))
            << npu::tile_fwk::AnyCast<SymbolicScalar>(scalar).IsValid()
            << "SCALAR attribute has to have symbolic value.";
        auto scalarExpr = npu::tile_fwk::AnyCast<SymbolicScalar>(scalar);
        dupV = SymbolicExpressionTable::BuildExpression(scalarExpr);
    } else if (dstDtypeStr == "float") {
        auto scalar = opAttrs.at(OpAttributeKey::scalar);
        ASSERT((scalar.HasValue()) && (scalar.Type() == typeid(Element)))
            << npu::tile_fwk::AnyCast<Element>(scalar).IsFloat() << "SCALAR attribute has to have float value.";
        dupV = std::to_string(npu::tile_fwk::AnyCast<Element>(scalar).Cast<float>());
    } else if (dstDtypeStr == "int32_t") {
        auto scalar = opAttrs.at(OpAttributeKey::scalar);
        ASSERT((scalar.HasValue()) && (scalar.Type() == typeid(Element)))
            << npu::tile_fwk::AnyCast<Element>(scalar).IsSigned() << "SCALAR attribute has to have int value.";
        dupV = std::to_string(npu::tile_fwk::AnyCast<Element>(scalar).Cast<int>());
    } else {
        ASSERT(false) << "unsupported type";
    }
    return PrintDupOp({dVar, dstDtypeStr, dupV});
}

std::string CodeGenOpCloudNPU::PrintRowSumlineStatic(const PrintUnaryParam &param) const {
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.HasValue()) {
        reduceAxis = npu::tile_fwk::AnyCast<int64_t>(axis);
    }
    ASSERT(((reduceAxis >= 0) && (reduceAxis < (int(shape[1].size()) - 1)))) << "unsupported reduce axis";
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    reduceAxis += SHAPE_DIM4 - rawShape[0].size();
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> dstShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::vector<int64_t> os = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    paramList.emplace_back(std::to_string(reduceAxis));
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dst, src});

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName << "_<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintRowSumlineDynamicUnaligned(const PrintUnaryParam &param) const {
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.HasValue()) {
        reduceAxis = npu::tile_fwk::AnyCast<int64_t>(axis);
    }
    ASSERT(((reduceAxis >= 0) && (reduceAxis < (int(shape[1].size()) - 1)))) << "unsupported reduce axis";
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    auto dynSrcShape = dynamicValidShape[1];
    // adjust reduceAxis for dim4
    reduceAxis += SHAPE_DIM4 - rawShape[0].size();
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> dstShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - rawShape[0].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    paramList.emplace_back(std::to_string(reduceAxis));
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src);
    for (auto dynShape : dynSrcShape) {
        paramList.emplace_back(dynShape.Dump());
    }

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintRowSumline(const PrintUnaryParam &param) const {
    if (isSupportDynamicUnaligned) {
        return PrintRowSumlineDynamicUnaligned(param);
    }
    return PrintRowSumlineStatic(param);
}

std::string CodeGenOpCloudNPU::PrintReduceExStatic(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    std::vector<int64_t> oriShape = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(oriShape[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dst, src0});
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintReduceEx(const PrintUnaryParam &param) const {
    return PrintReduceExStatic(param);
}

std::string CodeGenOpCloudNPU::PrintReduceSumStatic(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();
    std::string dst0 = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dst0, src0});
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintReduceSum(const PrintUnaryParam &param) const {
    return PrintReduceSumStatic(param);
}

std::string CodeGenOpCloudNPU::PrintVcopyStatic(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(srcRawShape[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dst, src0});
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintVcopy(const PrintUnaryParam &param) const {
    return PrintVcopyStatic(param);
}

std::string CodeGenOpCloudNPU::PrintUnaryDynamicUnaligned(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    std::vector<int64_t> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*SS*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ss[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);

    auto dynSrcShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);
    for (auto dynShape : dynSrcShape) {
        paramList.emplace_back(dynShape.Dump());
    }
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintUnaryStatic(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    std::vector<int64_t> os0 = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*OS*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os0[i]));
    }
    paramList.emplace_back("/*DS*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*SS*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ss[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintUnary(const PrintUnaryParam &param) const {
    if (isSupportDynamicUnaligned) {
        return PrintUnaryDynamicUnaligned(param);
    }
    return PrintUnaryStatic(param);
}

std::string CodeGenOpCloudNPU::GenTransposeDataMove() const {
    bool isCopyLocalToGM = opCode == Opcode::OP_TRANSPOSE_MOVEOUT;
    unsigned gmIdx = isCopyLocalToGM ? 0 : 1;
    unsigned localIdx = isCopyLocalToGM ? 1 : 0;

    auto kS0 = sm->CreateAllocKey(operandWithMagic[localIdx]);
    std::string localVar = sm->QueryVariableName(kS0);
    std::string gmVar = GenGmParamVar(gmIdx);

    std::vector<int64_t> srcShape = this->rawShape[localIdx];
    ALOG_INFO_F("GenUnaryOp: srcShape is %s", IntVecToStr(srcShape).c_str());
    std::vector<int64_t> gmShape = this->rawShape[gmIdx];
    ALOG_INFO_F("GenUnaryOp: gmShape is %s", IntVecToStr(gmShape).c_str());

    AppendLocalBufferVarOffset({
        {   gmIdx,    &gmVar},
        {localIdx, &localVar}
    });

    std::string localDtypeStr = DataType2CCEStr(operandDtype[localIdx]);
    std::string gmDtypeStr = DataType2CCEStr(operandDtype[gmIdx]);
    return PrintTransposeDataMove({gmIdx, localIdx, localVar, gmShape, localDtypeStr, gmDtypeStr});
}

std::string CodeGenOpCloudNPU::GenUnaryOp() const {
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    AppendLocalBufferVarOffset(std::vector{&dVar, &s0Var});

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    if (opCode == Opcode::OP_COPY_UB_TO_UB) {
        srcDtypeStr = GetTypeForB16B32(operandDtype[ID1]);
        dstDtypeStr = GetTypeForB16B32(operandDtype[ID0]);
    }

    if (opCode == Opcode::OP_EXPAND) {
        return PrintExpand(s0Var, dVar, srcDtypeStr, dstDtypeStr);
    } else if (opCode == Opcode::OP_ROWMAX || opCode == Opcode::OP_ROWEXPMAX || opCode == Opcode::OP_ROWEXPSUM) {
        return PrintReduceEx({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    } else if (opCode == Opcode::OP_ROWSUMLINE || opCode == Opcode::OP_ROWMAXLINE || opCode == Opcode::OP_ROWMINLINE) {
        return PrintRowSumline({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    } else if (opCode == Opcode::OP_EXP || opCode == Opcode::OP_SQRT || opCode == Opcode::OP_ABS ||
               opCode == Opcode::OP_RECIPROCAL || opCode == Opcode::OP_NEG || opCode == Opcode::OP_RSQRT || opCode == Opcode::OP_LN) {
        return PrintUnary({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    } else if (opCode == Opcode::OP_COPY_UB_TO_UB) {
        return PrintVcopy({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    } else if (opCode == Opcode::OP_ROWSUM) {
        return PrintReduceSum({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    }
    ALOG_INFO_F("unsupported tileop: %s", OpcodeManager::Inst().GetOpcodeStr(opCode));
    return "CG_ERROR";
}

std::string CodeGenOpCloudNPU::PrintExpandDynamicUnaligned(const PrintUnaryParam &param, int expandAxis) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    auto dynDstShape = dynamicValidShape[0];
    std::vector<SymbolicScalar> newDynDstShape = dynDstShape;
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynDstShape, SHAPE_DIM4 - dynDstShape.size(), 1);
    auto dynSrcShape = dynamicValidShape[1];
    std::vector<SymbolicScalar> newDynSrcShape = dynSrcShape;
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynSrcShape, SHAPE_DIM4 - dynSrcShape.size(), 1);
    std::ostringstream os;
    std::vector<std::string> paramList;
    std::vector<int64_t> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = ID1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*SS*/");
    for (int i = ID1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(ss[i]));
    }
    paramList.emplace_back(std::to_string(expandAxis));
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dst, src});
    for (int i = ID0; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(newDynDstShape[i].Dump());
    }
    for (int i = ID0; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(newDynSrcShape[i].Dump());
    }

    std::string tiloOpCallParam = JoinString(paramList, ", ");

    os << tileOpName << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintExpand(const std::string &s0Var, const std::string &dVar,
    const std::string &srcDtypeStr, const std::string &dstDtypeStr) const {
    char buffer[256] = "CG_ERROR";
    int ret = 0;
    int expandAxis{-1};
    std::vector<int64_t> dos = NormalizeShape(originShape[0], SHAPE_DIM4);
    std::vector<int64_t> os = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "EXPANDDIM");
    if (axis.HasValue()) {
        expandAxis = AnyCast<int64_t>(axis);
    }
    ASSERT((expandAxis >= 0) && (expandAxis <= (static_cast<int>(shape[1].size() - 1)))) << "unsupported reduce axis";
    // modify expandAxis for SHAPE_DIM4
    expandAxis += SHAPE_DIM4 - shape[1].size();

    if (isSupportDynamicUnaligned) {
        return PrintExpandDynamicUnaligned({s0Var, dVar, srcDtypeStr, dstDtypeStr}, expandAxis);
    }

    ret = sprintf_s(buffer, sizeof(buffer),
        "%s_<%s, %d, %d, %d, %d, %d, %d, %d, %d, /*DS*/ %d, %d, %d, /*SS*/ %d, %d, %d, %d>"
        "((__ubuf__ %s*)%s, (__ubuf__ %s*)%s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), dos[ID0], dos[ID1], dos[ID2], dos[ID3], os[ID0], os[ID1], os[ID2],
        os[ID3], ds[ID1], ds[ID2], ds[ID3], ss[ID1], ss[ID2], ss[ID3], expandAxis, dstDtypeStr.c_str(), dVar.c_str(),
        srcDtypeStr.c_str(), s0Var.c_str());
    ASSERT(ret >= 0) << "GenUnaryOp" << OpcodeManager::Inst().GetOpcodeStr(opCode) << " sprintf_s failed " << ret;
    return buffer;
}

std::string CodeGenOpCloudNPU::PrintTransposeDataMove(const PrintTransposeDataMoveParam &param) const {
    if (isSupportDynamicUnaligned) {
        return PrintTransposeDataMoveDynamicUnaligned(param);
    } else if (functionType == FunctionType::DYNAMIC_LOOP_PATH) {
        return PrintTransposeDataMoveDynamic(param);
    }
    return PrintTransposeDataMoveStatic(param);
}

std::string CodeGenOpCloudNPU::PrintTransposeDataMoveStatic(const PrintTransposeDataMoveParam &param) const {
    const std::string &localVar = param.localVar;
    const std::string &localDtypeStr = param.localDtypeStr;
    const std::string &gmDtypeStr = param.gmDtypeStr;
    std::string dstVar = GenGmParamVar(ID0);
    std::vector<int64_t> os = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int64_t> gmShape = NormalizeShape(param.gmShape, SHAPE_DIM4);
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;

    paramList.emplace_back(gmDtypeStr);
    for (auto oriShape : os) {
        paramList.emplace_back(std::to_string(oriShape));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(gmShape[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    std::vector<int64_t> transposeAxis =
        npu::tile_fwk::AnyCast<std::vector<int64_t>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM4 - originShape[0].size();
    for (auto &axis : transposeAxis) {
        axis += correctionAxis;
        paramList.emplace_back(std::to_string(axis));
    }
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();

    std::string dst = "(__gm__ " + gmDtypeStr + "*)" + dstVar;
    std::string src = "(__ubuf__ " + localDtypeStr + "*)" + localVar;
    paramList.insert(paramList.end(), {dst, src});
    std::string tiloOpCallParam = JoinString(paramList, ", ");

    oss << tileOpName.c_str() << "_<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintTransposeDataMoveDynamic(const PrintTransposeDataMoveParam &param) const {
    const std::string &localVar = param.localVar;
    const std::string &localDtypeStr = param.localDtypeStr;
    const std::string &gmDtypeStr = param.gmDtypeStr;
    std::string dstVar = GenGmParamVar(ID0);

    int dim = static_cast<int>(paramIdxForDynShape[ID0].size());
    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(ID0, dim, PREFIX_STR_RAW_SHAPE);
    FillIntVecWithDummyInHead<std::string>(gmShapeExpr, SHAPE_DIM4 - dim, "1");
    ALOG_INFO_F("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(ID0, dim, PREFIX_STR_OFFSET);
    FillIntVecWithDummyInHead<std::string>(gmOffsetExpr, SHAPE_DIM4 - dim, "0");
    ALOG_INFO_F("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());

    std::vector<int64_t> os = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(gmDtypeStr);
    for (auto oriShape : os) {
        paramList.emplace_back(std::to_string(oriShape));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    std::vector<int64_t> transposeAxis =
        npu::tile_fwk::AnyCast<std::vector<int64_t>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM4 - originShape[1].size();
    for (auto &axis : transposeAxis) {
        axis += correctionAxis;
        paramList.emplace_back(std::to_string(axis));
    }
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();

    std::string dst = "(__gm__ " + gmDtypeStr + "*)" + dstVar;
    std::string src = "(__ubuf__ " + localDtypeStr + "*)" + localVar;
    paramList.insert(paramList.end(), {dst, src});
    for (auto gs : gmShapeExpr) {
        paramList.emplace_back(gs);
    }
    for (auto go : gmOffsetExpr) {
        paramList.emplace_back(go);
    }
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintTransposeDataMoveDynamicUnaligned(const PrintTransposeDataMoveParam &param) const {
    const int gmIdx = param.gmIdx;
    const int localIdx = param.localIdx;
    const std::string &localVar = param.localVar;
    const std::string &localDtypeStr = param.localDtypeStr;
    const std::string &gmDtypeStr = param.gmDtypeStr;
    std::string gmVar = GenGmParamVar(gmIdx);

    int dim = static_cast<int>(paramIdxForDynShape[gmIdx].size());
    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(gmIdx, dim, PREFIX_STR_RAW_SHAPE);
    FillIntVecWithDummyInHead<std::string>(gmShapeExpr, SHAPE_DIM5 - dim, "1");
    ALOG_INFO_F("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(gmIdx, dim, PREFIX_STR_OFFSET);
    FillIntVecWithDummyInHead<std::string>(gmOffsetExpr, SHAPE_DIM5 - dim, "0");
    ALOG_INFO_F("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());
    auto newDynLocalValidShape = dynamicValidShape[localIdx];
    FillIntVecWithDummyInHead<SymbolicScalar>(
        newDynLocalValidShape, SHAPE_DIM5 - dynamicValidShape[localIdx].size(), 1);

    std::vector<int64_t> localShape = NormalizeShape(rawShape[localIdx], SHAPE_DIM5);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(gmDtypeStr);
    for (int i = 1; i < SHAPE_DIM5; i++) {
        paramList.emplace_back(std::to_string(localShape[i]));
    }
    std::vector<int64_t> transposeAxis =
        npu::tile_fwk::AnyCast<std::vector<int64_t>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM5 - originShape[localIdx].size();
    for (auto &axis : transposeAxis) {
        axis += correctionAxis;
        paramList.emplace_back(std::to_string(axis));
    }
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();

    std::string gm = "(__gm__ " + gmDtypeStr + "*)" + gmVar;
    std::string ub = "(__ubuf__ " + localDtypeStr + "*)" + localVar;

    if (gmIdx == 0) {
        paramList.insert(paramList.end(), {gm, ub});
    } else {
        paramList.insert(paramList.end(), {ub, gm});
    }

    for (auto localDynShape : newDynLocalValidShape) {
        paramList.emplace_back(localDynShape.Dump());
    }
    for (auto gs : gmShapeExpr) {
        paramList.emplace_back(gs);
    }
    for (auto go : gmOffsetExpr) {
        paramList.emplace_back(go);
    }
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName << "_<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintVnchwconvStatic(const PrintUnaryTmpBuffParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::vector<int64_t> os0 = NormalizeShape(originShape[ID2], SHAPE_DIM5);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID2], SHAPE_DIM5);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM5);
    std::ostringstream os;
    std::vector<std::string> paramList;
    // template param
    paramList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(os0[i]));
    }
    for (int i = 1; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = 1; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();
    // func actual param
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dst, src, tmp});
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintVnchwconvDynUnaligned(const PrintUnaryTmpBuffParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID2], SHAPE_DIM5);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM5);
    auto newDynSrcValidShape = dynamicValidShape[ID2];
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynSrcValidShape, SHAPE_DIM5 - dynamicValidShape[ID2].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 1; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = 1; i < SHAPE_DIM5; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dst, src, tmp});
    for (auto dynShape : newDynSrcValidShape) {
        paramList.emplace_back(dynShape.Dump());
    }

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintVnchwconv(const PrintUnaryTmpBuffParam &param) const {
    if (isSupportDynamicUnaligned) {
        return PrintVnchwconvDynUnaligned(param);
    }
    return PrintVnchwconvStatic(param);
}

std::string CodeGenOpCloudNPU::PrintCompactStatic(const PrintUnaryTmpBuffParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(srcRawShape[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dst, src, tmp});

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintCompact(const PrintUnaryTmpBuffParam &param) const {
    return PrintCompactStatic(param);
}

std::string CodeGenOpCloudNPU::PrintReduceCombine(const PrintUnaryTmpBuffParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;

    std::vector<int64_t> srcOriginShape = NormalizeShape(originShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> tmpRawShape = NormalizeShape(rawShape[ID1], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> parmList;
    parmList.emplace_back(dstDtypeStr);
    // src origin shape
    for (int i = ID0; i < SHAPE_DIM4; ++i) {
        parmList.emplace_back(std::to_string(srcOriginShape[i]));
    }
    // dst raw shape
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        parmList.emplace_back(std::to_string(dstRawShape[i]));
    }
    // src raw shape
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        parmList.emplace_back(std::to_string(srcRawShape[i]));
    }
    parmList.emplace_back(std::to_string(tmpRawShape[ID3]));
    std::string templateParam = JoinString(parmList, ", ");
    parmList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    parmList.insert(parmList.end(), {dst, src, tmp});

    std::string tiloOpCallParam = JoinString(parmList, ", ");
    os << tileOpName << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::GenUnaryOpWithTmpBuff() const {
    // In this scenario, frontend set tmp buffer in output to optimize ooo schedule result.
    auto kS0 = sm->CreateAllocKey(operandWithMagic[ID2]);
    auto kTmp = sm->CreateAllocKey(operandWithMagic[ID1]);
    auto kDst = sm->CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string tmpVar = sm->QueryVariableName(kTmp);
    std::string dVar = sm->QueryVariableName(kDst);

    std::vector srcShape = this->rawShape[2];
    ALOG_INFO_F("GenUnaryOpWithTmpBuff %s src raw shape: %s", tileOpName.c_str(), IntVecToStr(srcShape).c_str());

    std::vector dstShape = this->rawShape[0];
    ALOG_INFO_F("GenUnaryOpWithTmpBuff %s dst raw shape: %s", tileOpName.c_str(), IntVecToStr(dstShape).c_str());

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string tmpDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufferVarOffset(std::vector{&dVar, &tmpVar, &s0Var});

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    if (opCode == Opcode::OP_TRANSPOSE_VNCHWCONV) {
        return PrintVnchwconv({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    if (opCode == Opcode::OP_ROWSUM_SINGLE || opCode == Opcode::OP_ROWMAX_SINGLE ||
        opCode == Opcode::OP_ROWMIN_SINGLE) {
        return PrintReduceLastAxis({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    if (opCode == Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE || opCode == Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE) {
        return PrintReduceCombine({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    if (opCode == Opcode::OP_COMPACT) {
        return PrintCompact({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    std::string ostring(buffer);
    return ostring;
}

std::string CodeGenOpCloudNPU::PrintReduceLastAxisDynamicUnalign(const PrintUnaryTmpBuffParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;

    auto newDynSrcValidShape = dynamicValidShape[ID2];
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynSrcValidShape, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> tmpRawShape = NormalizeShape(rawShape[ID1], SHAPE_DIM4);

    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(srcRawShape[i]));
    }
    // tmp only need the last axis shape
    paramList.emplace_back(std::to_string(tmpRawShape[ID3]));

    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();

    std::string dstName = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcName = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmpName = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dstName, srcName, tmpName});
    for (auto dynShape : newDynSrcValidShape) {
        paramList.emplace_back(dynShape.Dump());
    }

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintReduceLastAxis(const PrintUnaryTmpBuffParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int ret = 0;

    std::vector<int64_t> dstOriginShape = NormalizeShape(originShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> srcOriginShape = NormalizeShape(originShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> tmpRawShape = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    ALOG_INFO_F("rawShape[2] is %s", IntVecToStr(rawShape[ID2]).c_str());
    ASSERT(dstOriginShape[ID3] == 1) << "Dst last axis length must be 1";

    if (isSupportDynamicUnaligned) {
        return PrintReduceLastAxisDynamicUnalign({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    ret = sprintf_s(buffer, sizeof(buffer),
        "%s_<%s, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s, (__ubuf__ "
        "%s *)%s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), srcOriginShape[ID0], srcOriginShape[ID1], srcOriginShape[ID2],
        srcOriginShape[ID3], dstRawShape[ID1], dstRawShape[ID2], dstRawShape[ID3], srcRawShape[ID1], srcRawShape[ID2],
        srcRawShape[ID3], tmpRawShape[ID3], dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str(),
        tmpDtypeStr.c_str(), tmpVar.c_str());
    ASSERT(ret >= 0) << "PrintReduceLastAxis" << OpcodeManager::Inst().GetOpcodeStr(opCode) << " sprintf_s failed "
                     << ret;
    return buffer;
}

std::string CodeGenOpCloudNPU::PrintBinaryStatic(const PrintBinaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;

    std::vector<int64_t> os0 = NormalizeShape(originShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> os1 = NormalizeShape(originShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*OS0*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os0[i]));
    }
    paramList.emplace_back("/*OS1*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os1[i]));
    }
    paramList.emplace_back("/*DS*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryDynamicUnaligned(const PrintBinaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;

    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);

    std::vector<SymbolicScalar> dynSrcShape0 = dynamicValidShape[ID1];
    std::vector<SymbolicScalar> dynSrcShape1 = dynamicValidShape[ID2];

    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape0, SHAPE_DIM4 - dynamicValidShape[ID1].size(), 1);
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape1, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);
    for (auto dynShape : dynSrcShape0) {
        paramList.emplace_back(dynShape.Dump());
    }
    for (auto dynShape : dynSrcShape1) {
        paramList.emplace_back(dynShape.Dump());
    }
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinary(const PrintBinaryParam &param) const {
    if (isSupportLayout) {
        return PrintBinaryTileTensor();
    }
    if (isSupportDynamicUnaligned) {
        return PrintBinaryDynamicUnaligned(param);
    }
    return PrintBinaryStatic(param);
}

std::string CodeGenOpCloudNPU::PrintBinaryBrcStatic(const PrintBinaryBrcParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    const std::string &tmpVar = param.tmpVar;

    std::vector<int64_t> os0 = NormalizeShape(originShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID3], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> brcParamList;
    brcParamList.emplace_back(dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        brcParamList.emplace_back(std::to_string(os0[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        brcParamList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        brcParamList.emplace_back(std::to_string(s0[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        brcParamList.emplace_back(std::to_string(s1[i]));
    }
    brcParamList.emplace_back(std::to_string(isInputForceCombineAxis));
    std::string templateParam = JoinString(brcParamList, ", ");

    brcParamList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    brcParamList.emplace_back(dst);
    brcParamList.emplace_back(src0);
    brcParamList.emplace_back(src1);
    brcParamList.emplace_back(tmp);

    std::string tiloOpCallParam = JoinString(brcParamList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryBrcDynamicUnaligned(const PrintBinaryBrcParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    const std::string &tmpVar = param.tmpVar;

    std::vector<int64_t> os0 = NormalizeShape(originShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID3], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);

    auto dynSrcShape = dynamicValidShape[ID2];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    paramList.emplace_back("/*isCombineAxis*/");
    paramList.emplace_back(std::to_string(isInputForceCombineAxis));
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.insert(paramList.end(), {dst, src0, src1, tmp});
    for (auto dynShape : dynSrcShape) {
        paramList.emplace_back(dynShape.Dump());
    }

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryTileTensor() const {
    std::string dstTensor = sm->QueryTileTensorByMagic(operandWithMagic[ToUnderlying(DISIIdx::DST_IDX)]);
    std::string src0Tensor = sm->QueryTileTensorByMagic(operandWithMagic[ToUnderlying(DISIIdx::SRC0_IDX)]);
    std::string src1Tensor = sm->QueryTileTensorByMagic(operandWithMagic[ToUnderlying(DISIIdx::SRC1_IDX)]);
    std::ostringstream oss;
    oss << tileOpName << "(" << dstTensor << ", " << src0Tensor << ", " << src1Tensor << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryBrc(const PrintBinaryBrcParam &param) const {
    if (isSupportDynamicUnaligned) {
        return PrintBinaryBrcDynamicUnaligned(param);
    }
    return PrintBinaryBrcStatic(param);
}

std::string CodeGenOpCloudNPU::GenBinaryOp() const {
    auto kS0 = sm->CreateAllocKey(operandWithMagic[ID1]);
    auto kDst = sm->CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string dVar = sm->QueryVariableName(kDst);

    std::vector src0RawShape = this->rawShape[ID1];
    ALOG_INFO_F("genBinaryOp %s, src0RawShape is %s", tileOpName.c_str(), IntVecToStr(src0RawShape).c_str());

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);

    auto kS1 = sm->CreateAllocKey(operandWithMagic[ID2]);
    std::string s1Var = sm->QueryVariableName(kS1);

    AppendLocalBufferVarOffset(std::vector{&dVar, &s0Var, &s1Var});
    return PrintBinary({s0Var, s1Var, dVar, src0DtypeStr, src1DtypeStr, dstDtypeStr});
}

std::string CodeGenOpCloudNPU::GenBinaryWithBrc() const {
    auto kS0 = sm->CreateAllocKey(operandWithMagic[ID2]);
    auto kDst = sm->CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string dVar = sm->QueryVariableName(kDst);

    std::vector src0RawShape = this->rawShape[ID2];
    std::vector src1RawShape = this->rawShape[ID3];
    ALOG_INFO_F("GenBinaryWithBrc %s, src0RawShape is %s", tileOpName.c_str(), IntVecToStr(src0RawShape).c_str());

    char buffer[256] = "CG_ERROR";
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID3]);

    auto kS1 = sm->CreateAllocKey(operandWithMagic[ID3]);
    std::string s1Var = sm->QueryVariableName(kS1);
    AllocKey kTmp;
    std::string tmpVar;
    std::string tmpDtypeStr;
    kTmp = sm->CreateAllocKey(operandWithMagic[ID1]);
    tmpVar = sm->QueryVariableName(kTmp);
    tmpDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    AppendLocalBufferVarOffset(std::vector{&dVar, &s0Var, &s1Var, &tmpVar});
    int ret = 0;
    if (opCode == Opcode::OP_ADD_BRC || opCode == Opcode::OP_SUB_BRC || opCode == Opcode::OP_MUL_BRC ||
        opCode == Opcode::OP_DIV_BRC || opCode == Opcode::OP_MAX_BRC) {
        return PrintBinaryBrc({s0Var, s1Var, dVar, tmpVar, src0DtypeStr, src1DtypeStr, dstDtypeStr, tmpDtypeStr});
    }
    ASSERT(ret >= 0) << "GenBinaryWithBrc sprintf_s failed ";
    return buffer;
}

std::string CodeGenOpCloudNPU::GenFusedOp() const {
    ASSERT(opAttrs.count("VF_WRAPPER_NAME")) << "cannot get VF_WRAPPER_NAME.";
    std::string fusedOpName = "TileOp::" + npu::tile_fwk::AnyCast<std::string>(opAttrs.at("VF_WRAPPER_NAME"));
    int operandsNum = 0;
    for (auto ele : operand) {
        if (ele == NULL_OPERAND) {
            break;
        }
        operandsNum++;
    }
    std::string paramString;
    std::map<unsigned, std::string *> varMap;
    for (int i = 0; i < operandsNum; i++) {
        auto kS0 = sm->CreateAllocKey(operandWithMagic[i]);
        std::string s0Var = sm->QueryVariableName(kS0);
        std::string dtypeStr = DataType2CCEStr(operandDtype[i]);
        paramString += "(__ubuf__ " + dtypeStr + "*)" + s0Var;
        if (i < operandsNum - 1) {
            paramString += ", ";
        }
        varMap[i] = &s0Var;
    }
    AppendLocalBufferVarOffset(varMap);

    char buffer[256] = "CG_ERROR";
    int ret = 0;
    ret = sprintf_s(buffer, sizeof(buffer), "%s(%s);\n", fusedOpName.c_str(), paramString.c_str());
    ASSERT(ret >= 0) << "GenFusedOp sprintf_s failed ";
    std::string ostring(buffer);
    return ostring;
}

std::string CodeGenOpCloudNPU::PrintGatherStatic(const PrintGatherParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;

    std::vector dstShape = this->rawShape[ID0];
    std::vector src0Shape = this->rawShape[ID1];

    std::vector<int64_t> dos = NormalizeShape(originShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(src0Shape, SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(src0DtypeStr);
    paramList.emplace_back(src1DtypeStr);
    paramList.emplace_back("/*DOS*/");
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dos[i]));
    }
    paramList.emplace_back("/*SS*/");
    paramList.emplace_back(std::to_string(ss[ID3]));
    paramList.emplace_back("/*DS*/");
    paramList.emplace_back(std::to_string(ds[ID3]));
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintGatherDynamicUnaligned(const PrintGatherParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    std::vector dstShape = this->rawShape[ID0];
    std::vector src0Shape = this->rawShape[ID1];
    std::vector<int64_t> dos = NormalizeShape(originShape[ID0], SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(src0Shape, SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM4);

    auto dynDstShape = dynamicValidShape[ID0];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynDstShape, SHAPE_DIM4 - dynamicValidShape[ID0].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(src0DtypeStr);
    paramList.emplace_back(src1DtypeStr);
    paramList.emplace_back("/*SS*/");
    paramList.emplace_back(std::to_string(ss[ID3]));
    paramList.emplace_back("/*DS*/");
    paramList.emplace_back(std::to_string(ds[ID3]));
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(dynDstShape[i].Dump());
    }

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintGather(const PrintGatherParam &param) const {
    if (isSupportDynamicUnaligned) {
        return PrintGatherDynamicUnaligned(param);
    }
    return PrintGatherStatic(param);
}

std::string CodeGenOpCloudNPU::GenGatherOp() const {
    auto kS0 = sm->CreateAllocKey(operandWithMagic[ID1]);
    auto kS1 = sm->CreateAllocKey(operandWithMagic[ID2]);
    auto kDst = sm->CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string s1Var = sm->QueryVariableName(kS1);
    std::string dVar = sm->QueryVariableName(kDst);

    // shape: dst, src0, src1
    int dstRank = shape[ID0].size();
    int src0Rank = shape[ID1].size();
    int src1Rank = shape[ID2].size();

    // support following cases:
    // [case1] src0: [S2,D], src1: [S], axis: 0, dst: [S,D]
    // [case2] src0: [S2,D], src1: [B,S], axis: 0, dst: [B,S,D]
    ASSERT(src0Rank == RANK2) << "GenGatherOp: src0 shape rank is not supported!";
    ASSERT((src1Rank == RANK1 || src1Rank == RANK2)) << "GenGatherOp: src1 shape rank is not supported!";
    ASSERT((dstRank == RANK2 || dstRank == RANK3)) << "GenGatherOp: dst shape rank is not supported!";

    std::vector dstShape = this->rawShape[0];
    if (dstRank == RANK2) {
        ALOG_INFO_F("GenGatherOp, dst Shape is [%d,%d]", dstShape[ID0], dstShape[ID1]);
    } else if (dstRank == RANK3) {
        ALOG_INFO_F("GenGatherOp, dst Shape is [%d,%d,%d]", dstShape[ID0], dstShape[ID1], dstShape[ID2]);
    }

    std::vector src0Shape = this->rawShape[1];
    ALOG_INFO_F("GenGatherOp, src0 Shape is [%d,%d]", src0Shape[0], src0Shape[1]);

    char buffer[256] = "CG_ERROR";
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);

    AppendLocalBufferVarOffset(std::vector{&dVar, &s0Var, &s1Var});
    int ret = 0;
    if (src0Rank == RANK2 && (src1Rank == RANK1 || src1Rank == RANK2)) {
        // [case1] src0: [S2,D], src1: [S], axis: 0, dst: [S,D]
        return PrintGather({s0Var, s1Var, dVar, src0DtypeStr, src1DtypeStr, dstDtypeStr});
    } else {
        ASSERT(0) << "GenGatherOp: input shape is not supported yet!";
    }
    ASSERT(ret >= 0) << "genGatherOp sprintf_s failed ";
    std::string ostring(buffer);
    return ostring;
}

std::string CodeGenOpCloudNPU::PrintGatherElementStatic(const PrintGatherEleParam &param) const {
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    std::vector<int64_t> &dstOriginShape = param.dstOriginShape;
    std::vector<int64_t> &dstRawShape = param.dstRawShape;
    std::vector<int64_t> &src0RawShape = param.src0RawShape;
    const std::string *dataTypeExpr = param.dataTypeExpr;
    // template param
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID1], dataTypeExpr[ID2]});
    paramList.insert(paramList.end(), {std::to_string(dstOriginShape[ID0]), std::to_string(dstOriginShape[ID1])});
    paramList.emplace_back(std::to_string(src0RawShape[ID1]));
    paramList.emplace_back(std::to_string(dstRawShape[ID1]));
    paramList.emplace_back(std::to_string(param.axis));
    std::string templateParam = JoinString(paramList, ", ");
    // func actual param
    paramList.clear();
    std::string dst = "(__ubuf__ " + dataTypeExpr[ID0] + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";

    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintGatherElementDynamicUnaligned(const PrintGatherEleParam &param) const {
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    std::vector<int64_t> &dstRawShape = param.dstRawShape;
    std::vector<int64_t> &src0RawShape = param.src0RawShape;
    std::vector<int64_t> &src1RawShape = param.src1RawShape;
    const std::string *dataTypeExpr = param.dataTypeExpr;
    // template param
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID1], dataTypeExpr[ID2]});
    paramList.emplace_back(std::to_string(src0RawShape[ID1]));
    paramList.emplace_back(std::to_string(src1RawShape[ID1]));
    paramList.emplace_back(std::to_string(dstRawShape[ID1]));
    paramList.emplace_back(std::to_string(param.axis));
    std::string templateParam = JoinString(paramList, ", ");
    // func actual param
    paramList.clear();
    std::string dst = "(__ubuf__ " + dataTypeExpr[ID0] + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    auto dstValidShape = dynamicValidShape[ID0];
    paramList.emplace_back(dstValidShape[ID0].Dump());
    paramList.emplace_back(dstValidShape[ID1].Dump());
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenGatherElementOp() const {
    auto kS0 = sm->CreateAllocKey(operandWithMagic[ID1]);
    auto kS1 = sm->CreateAllocKey(operandWithMagic[ID2]);
    auto kDst = sm->CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string s1Var = sm->QueryVariableName(kS1);
    std::string dVar = sm->QueryVariableName(kDst);

    // shape: dst, src0, src1
    int dstRank = shape[0].size();
    int src0Rank = shape[1].size();
    int src1Rank = shape[2].size();

    ASSERT(src0Rank == RANK2) << "GenGatherElementOp: src0 shape rank is not supported!";
    ASSERT(src1Rank == RANK2) << "GenGatherElementOp: src1 shape rank is not supported!";
    ASSERT(dstRank == RANK2) << "GenGatherElementOp: dst shape rank is not supported!";

    std::vector dstShape = this->rawShape[0];
    ALOG_INFO_F("GenGatherElementOp, dst Shape is %s ", IntVecToStr(dstShape).c_str());

    std::vector src0Shape = this->rawShape[1];
    ALOG_INFO_F("GenGatherElementOp, src0 Shape is %s ", IntVecToStr(src0Shape).c_str());

    std::vector src1Shape = this->rawShape[2];
    ALOG_INFO_F("GenGatherElementOp, src1 Shape is %s ", IntVecToStr(src1Shape).c_str());

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);

    AppendLocalBufferVarOffset(std::vector{&dVar, &s0Var, &s1Var});

    // [case1] src0: [S2,D], src1: [B,S], axis: 0, dst: [B,S,D]
    std::vector<int64_t> dos = NormalizeShape(originShape[0], SHAPE_DIM2);
    std::vector<int64_t> s0s = NormalizeShape(src0Shape, SHAPE_DIM2);
    std::vector<int64_t> s1s = NormalizeShape(src1Shape, SHAPE_DIM2);
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM2);
    std::string dataTypeExpr[3] = {dstDtypeStr, src0DtypeStr, src1DtypeStr};
    int gatherAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "axis");
    if (axis.HasValue()) {
        gatherAxis = npu::tile_fwk::AnyCast<int64_t>(axis);
    }
    if (isSupportDynamicUnaligned) {
        return PrintGatherElementDynamicUnaligned({gatherAxis, dVar, s0Var, s1Var, dos, ds, s0s, s1s, dataTypeExpr});
    }
    return PrintGatherElementStatic({gatherAxis, dVar, s0Var, s1Var, dos, ds, s0s, s1s, dataTypeExpr});
}

std::string CodeGenOpCloudNPU::PrintScatterElementSOpStatic(const PrintScatterElemParam &param) const {
    // Static only support 2Dim
    int dstRank = shape[ToUnderlying(DISIIdx::DST_IDX)].size();
    int src1Rank = shape[ToUnderlying(DISIIdx::SRC1_IDX)].size();
    ASSERT(src1Rank == RANK2) << "GenScatterElementSOp: src1 shape rank is not supported!";
    ASSERT(dstRank == RANK2) << "GenScatterElementSOp: dst shape rank is not supported!";

    const std::string &dstVar = param.dVar;
    const std::string &src0Var = param.s0Var;
    const std::string &src1Var = param.s1Var;
    std::vector<int64_t> &dstShape = param.dstRawShape;
    std::vector<int64_t> &src1RawShape = param.src1RawShape;
    const std::string *dataTypeExpr = param.dataTypeExpr;
    const Element &scala = extOperandVal;

    std::vector src1Shape = this->originShape[ToUnderlying(DISIIdx::SRC1_IDX)];
    std::vector<int64_t> s1os = NormalizeShape(src1Shape, SHAPE_DIM2);
    std::vector<int64_t> s1rs = NormalizeShape(src1RawShape, SHAPE_DIM2);
    std::vector<int64_t> drs = NormalizeShape(dstShape, SHAPE_DIM2);

    char scalarTmpBuffer[BUFFER_SIZE_512] = "CG_ERROR";
    int ret =
        snprintf_s(scalarTmpBuffer, sizeof(scalarTmpBuffer), sizeof(scalarTmpBuffer) - 1, "%.9g", scala.Cast<float>());
    if (ret < 0) {
        ALOG_INFO_F("GenScatterElementSOp snprintf_s scalarTmpBuffer failed %d", ret);
    }
    std::vector<std::string> templateParams;
    templateParams.emplace_back(dataTypeExpr[ToUnderlying(DISIIdx::DST_IDX)]);
    templateParams.emplace_back(dataTypeExpr[ToUnderlying(DISIIdx::SRC1_IDX)]);
    templateParams.emplace_back(std::to_string(s1rs[ToUnderlying(DISIIdx::SRC0_IDX)]));
    templateParams.emplace_back(std::to_string(drs[ToUnderlying(DISIIdx::SRC0_IDX)]));
    templateParams.emplace_back(std::to_string(s1os[ToUnderlying(DISIIdx::DST_IDX)]));
    templateParams.emplace_back(std::to_string(s1os[ToUnderlying(DISIIdx::SRC0_IDX)]));
    std::string templateParamStr = JoinString(templateParams, ", ");
    templateParamStr += GenOpAttr();

    std::vector<std::string> callParams;
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(DISIIdx::DST_IDX)] + "*)" + dstVar);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(DISIIdx::SRC0_IDX)] + "*)" + src0Var);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(DISIIdx::SRC1_IDX)] + "*)" + src1Var);
    callParams.emplace_back("(" + dataTypeExpr[ToUnderlying(DISIIdx::DST_IDX)] + ")" + scalarTmpBuffer);

    std::string callParamStr = JoinString(callParams, ", ");

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParamStr << ">(" << callParamStr << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintScatterElementSOpDynamicUnaligned(const PrintScatterElemParam &param) const {
    const std::string &dstVar = param.dVar;
    const std::string &src0Var = param.s0Var;
    const std::string &src1Var = param.s1Var;
    std::vector<int64_t> &drs = param.dstRawShape;
    std::vector<int64_t> &s1rs = param.src1RawShape;
    const std::string *dataTypeExpr = param.dataTypeExpr;
    const Element &scala = extOperandVal;

    size_t dynDim = shape[ToUnderlying(DISIIdx::DST_IDX)].size();
    auto dynSrc1Shape = dynamicValidShape[ToUnderlying(DISIIdx::SRC1_IDX)];
    FillIntVecWithDummyInHead<SymbolicScalar>(
        dynSrc1Shape, dynDim - dynamicValidShape[ToUnderlying(DISIIdx::SRC1_IDX)].size(), 1);

    char scalarTmpBuffer[BUFFER_SIZE_512] = "CG_ERROR";
    int ret =
        snprintf_s(scalarTmpBuffer, sizeof(scalarTmpBuffer), sizeof(scalarTmpBuffer) - 1, "%.9g", scala.Cast<float>());
    if (ret < 0) {
        ALOG_INFO_F("GenScatterElementSOp snprintf_s scalarTmpBuffer failed %d", ret);
    }

    std::vector<std::string> templateParams;
    templateParams.emplace_back(dataTypeExpr[ToUnderlying(DISIIdx::DST_IDX)]);
    templateParams.emplace_back(dataTypeExpr[ToUnderlying(DISIIdx::SRC1_IDX)]);
    for (size_t i = 1; i < dynDim; ++i) {
        templateParams.emplace_back(std::to_string(s1rs[i]));
    }
    for (size_t i = 1; i < dynDim; ++i) {
        templateParams.emplace_back(std::to_string(drs[i]));
    }
    templateParams.emplace_back(std::to_string(param.axis));
    templateParams.emplace_back(std::to_string(param.reduceOp));
    std::string templateParamStr = JoinString(templateParams, ", ");

    std::vector<std::string> callParams;
    const std::string src2_dtypestr = "float";
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(DISIIdx::DST_IDX)] + "*)" + dstVar);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(DISIIdx::SRC0_IDX)] + "*)" + src0Var);
    callParams.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(DISIIdx::SRC1_IDX)] + "*)" + src1Var);
    callParams.emplace_back("(" + src2_dtypestr + ")" + scalarTmpBuffer);
    for (size_t i = 0; i < dynDim; ++i) {
        callParams.emplace_back(dynSrc1Shape[i].Dump());
    }
    std::string callParamStr = JoinString(callParams, ", ");

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParamStr << ">(" << callParamStr << ");\n";
    return oss.str();
}

int CodeGenOpCloudNPU::GetScatterElementSReduceOperation(const std::string &reduce) const
{
    constexpr int replaceOp = 0;
    constexpr int addOp = 1;
    constexpr int multiplyOp = 2;
    int reduceOp = replaceOp;

    if (reduce == "None") {
        reduceOp = replaceOp;
    } else if (reduce == "add") {
        reduceOp = addOp;
    } else if (reduce == "multiply") {
        reduceOp = multiplyOp;
    } else {
        ALOG_ERROR_F("GetScatterElementSReduceOperation reduce is not supported, reduce is %s", reduce.c_str());
    }

    return reduceOp;
}

std::string CodeGenOpCloudNPU::GenScatterElementSOp() const {
    ASSERT(opAttrs.count(OpAttributeKey::reduce)) << "cannot get reduce attr";
    ASSERT(opAttrs.count(OP_ATTR_PREFIX + "axis")) << "cannot get axis attr";
    int axis = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "axis"));
    std::string reduce = npu::tile_fwk::AnyCast<std::string>(opAttrs.at(OpAttributeKey::reduce));
    int reduceOp = GetScatterElementSReduceOperation(reduce);
    const DataType dstDtype = operandDtype[ToUnderlying(DISIIdx::DST_IDX)];
    const DataType src0Dtype = operandDtype[ToUnderlying(DISIIdx::SRC0_IDX)];
    const DataType src1Dtype = operandDtype[ToUnderlying(DISIIdx::SRC1_IDX)];

    std::string src0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(DISIIdx::SRC0_IDX)]);
    std::string src1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(DISIIdx::SRC1_IDX)]);
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(DISIIdx::DST_IDX)]);

    ALOG_INFO_F("GenScatterElementSOp, dst Shape is %s", IntVecToStr(shape[ToUnderlying(DISIIdx::DST_IDX)]).c_str());
    ALOG_INFO_F("GenScatterElementSOp, src0 Shape is %s", IntVecToStr(shape[ToUnderlying(DISIIdx::SRC0_IDX)]).c_str());
    ALOG_INFO_F("GenScatterElementSOp, src1 Shape is %s", IntVecToStr(shape[ToUnderlying(DISIIdx::SRC1_IDX)]).c_str());

    std::vector dstRawShape = this->rawShape[ToUnderlying(DISIIdx::DST_IDX)];
    std::vector src1RawShape = this->rawShape[ToUnderlying(DISIIdx::SRC1_IDX)];

    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string src0DtypeStr = DataType2CCEStr(src0Dtype);
    std::string src1DtypeStr = DataType2CCEStr(src1Dtype);
    ALOG_INFO_F("GenScatterElementSOp, dstDtypeStr%s", dstDtypeStr.c_str());
    ALOG_INFO_F("GenScatterElementSOp, src1DtypeStr%s", src1DtypeStr.c_str());

    AppendLocalBufferVarOffset(std::vector{&dstVar, &src0Var, &src1Var});

    constexpr int NumOperands = 3;
    std::string dataTypeExpr[NumOperands] = {dstDtypeStr, src0DtypeStr, src1DtypeStr};

    if (isSupportDynamicUnaligned) {
        return PrintScatterElementSOpDynamicUnaligned(
            {axis, reduceOp, dstVar, src0Var, src1Var, dstRawShape, src1RawShape, dataTypeExpr});
    }
    return PrintScatterElementSOpStatic({axis, reduceOp, dstVar, src0Var, src1Var, dstRawShape, src1RawShape, 
        dataTypeExpr});
}

std::string CodeGenOpCloudNPU::PrintSortDynamicUnaligned(const SortParam &param) const {
    auto dstShape = param.dstShape;
    auto src0Shape = param.srcShape;
    const std::string &s0Var = param.s0Var;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;

    auto dynSrcShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);

    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(src0Shape[i]));
    }

    std::string templateParam = JoinString(paramList, ", ");
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dstParam, srcParam});
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(dynSrcShape[i].Dump());
    }
    std::string tileCallParam = JoinString(paramList, ", ");

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">" << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintSortStatic(const SortParam &param) const {
    auto dstShape = param.dstShape;
    auto src0Shape = param.srcShape;
    unsigned orisrcShape0 = 0;
    unsigned orisrcShape1 = 0;
    const std::string &s0Var = param.s0Var;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::vector<int64_t> &oriSrc0Shape = originShape[1];
    constexpr unsigned defaultDim = 1u;
    if (oriSrc0Shape.size() == 1) {
        orisrcShape0 = defaultDim;
        orisrcShape1 = oriSrc0Shape[0];
    } else {
        orisrcShape0 = oriSrc0Shape[0];
        orisrcShape1 = oriSrc0Shape[1];
    }
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    // static only support 1~2 dim
    paramList.insert(paramList.end(), {std::to_string(dstShape[2]), std::to_string(dstShape[3])});
    paramList.insert(paramList.end(), {std::to_string(src0Shape[2]), std::to_string(src0Shape[3])});
    paramList.insert(paramList.end(), {std::to_string(orisrcShape0), std::to_string(orisrcShape1)});

    std::string templateParam = JoinString(paramList, ", ");
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dstParam, srcParam});
    std::string tileCallParam = JoinString(paramList, ", ");
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">" << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintBitSortDynamicUnaligned(const SortParam &param) const {
    return PrintSortDynamicUnaligned(param);
}

std::string CodeGenOpCloudNPU::PrintBitSortStatic(const SortParam &param) const {
    return PrintSortStatic(param);
}

std::string CodeGenOpCloudNPU::GenBitSortOp() const {
    SortParam sortParm = PrepareSortParam();
    if (isSupportDynamicUnaligned) {
        return PrintBitSortDynamicUnaligned(sortParm);
    }
    return PrintBitSortStatic(sortParm);
}

std::string CodeGenOpCloudNPU::PrintMrgSortDynamicUnaligned(const SortParam &param) const {
    return PrintSortDynamicUnaligned(param);
}

std::string CodeGenOpCloudNPU::PrintMrgSortStatic(const SortParam &param) const {
    return PrintSortStatic(param);
}

SortParam CodeGenOpCloudNPU::PrepareSortParam() const {
    const DataType dstDtype = operandDtype[ID0];
    const DataType src0Dtype = operandDtype[ID1];
    int dst = operandWithMagic[ID0];
    int src0 = operandWithMagic[ID1];

    auto kSrc0 = sm->CreateAllocKey(src0);
    auto kDst = sm->CreateAllocKey(dst);
    std::string src0Var = sm->QueryVariableName(kSrc0);
    std::string dstVar = sm->QueryVariableName(kDst);

    std::vector dstShape = this->rawShape[0];
    std::vector src0Shape = this->rawShape[1];
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(src0Shape, SHAPE_DIM4);

    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string src0DtypeStr = DataType2CCEStr(src0Dtype);
    AppendLocalBufferVarOffset(std::vector{&dstVar, &src0Var});
    return {
        {ds[ID0], ds[ID1], ds[ID2], ds[ID3]},
        {ss[ID0], ss[ID1], ss[ID2], ss[ID3]},
        src0Var, dstVar, src0DtypeStr,
        dstDtypeStr
    };
}

std::string CodeGenOpCloudNPU::GenMrgSortOp() const {
    SortParam sortParm = PrepareSortParam();
    if (isSupportDynamicUnaligned) {
        return PrintMrgSortDynamicUnaligned(sortParm);
    }
    return PrintMrgSortStatic(sortParm);
}

std::string CodeGenOpCloudNPU::PrintExtractStatic() const {
    SymbolManager::AllocRecord src0, dst;
    auto kS0 = sm->CreateAllocKey(operandWithMagic[ID1]);
    auto kDst = sm->CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string dVar = sm->QueryVariableName(kDst);
    std::vector src0RawShape = this->rawShape[1];
    unsigned tShape0 = 0;
    unsigned tShape1 = 0;

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::vector src0Shape = this->rawShape[0];
    AppendLocalBufferVarOffset(std::vector{&dVar, &s0Var});
    constexpr unsigned defaultDim = 1u;
    if (this->rawShape[1].size() == 1) {
        tShape1 = std::min(src0RawShape[0], shape[0][0]);
        tShape0 = defaultDim;
    } else {
        tShape0 = std::min(src0RawShape[0], shape[0][0]);
        tShape1 = std::min(src0RawShape[1], shape[0][1]);
    }
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dstDtypeStr, src0DtypeStr});
    paramList.insert(paramList.end(), {std::to_string(tShape0), std::to_string(tShape1)});

    std::string templateParam = JoinString(paramList, ", ");
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dstParam, srcParam});
    std::string tileCallParam = JoinString(paramList, ", ");
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintExtractDynamicUnaligned() const {
    SymbolManager::AllocRecord src0, dst;
    auto kS0 = sm->CreateAllocKey(operandWithMagic[ID1]);
    auto kDst = sm->CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string dVar = sm->QueryVariableName(kDst);
    std::vector src0RawShape = this->rawShape[1];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    AppendLocalBufferVarOffset(std::vector{&dVar, &s0Var});

    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dstDtypeStr, src0DtypeStr});
    std::vector dstShape = this->rawShape[0];
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM4);
    paramList.insert(paramList.end(), {std::to_string(ds[ID1]), std::to_string(ds[ID2]), std::to_string(ds[ID3])});

    std::string templateParam = JoinString(paramList, ", ");
    templateParam += GenOpAttr();
    paramList.clear();

    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dstParam, srcParam});
    auto dynSrcShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);
    auto dynDstShape = dynamicValidShape[0];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynDstShape, SHAPE_DIM4 - dynamicValidShape[0].size(), 1);
    for (int i = 0; i < SHAPE_DIM3; ++i) {
        auto tShape = dynSrcShape[i].Min(dynDstShape[i]);
        paramList.emplace_back(tShape.Dump());
    }
    std::string tileCallParam = JoinString(paramList, ", ");

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenExtractOp() const {
    if (isSupportDynamicUnaligned) {
        return PrintExtractDynamicUnaligned();
    }
    return PrintExtractStatic();
}

std::string CodeGenOpCloudNPU::GenVectorScalarOp() const {
    return GenVectorScalarOpByMode(VecScalMode::VEC_MODE);
}

std::string CodeGenOpCloudNPU::GenVectorScalarOpScalarMode() const {
    return GenVectorScalarOpByMode(VecScalMode::SCALAR_MODE);
}

std::string CodeGenOpCloudNPU::PrintBinaryScalarStatic(const PrintBinaryScalarParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    std::vector dstShape = this->rawShape[0];
    std::vector src0Shape = this->rawShape[1];

    std::vector<int64_t> os0 = NormalizeShape(originShape[1], SHAPE_DIM3);
    std::vector<int64_t> ss = NormalizeShape(src0Shape, SHAPE_DIM3);
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM3);

    std::ostringstream os;
    std::vector<std::string> binScalParmList;
    binScalParmList.emplace_back(dstDtypeStr);
    int dimScalar = static_cast<int>(param.dim);
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        binScalParmList.emplace_back(std::to_string(os0[i]));
    }
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        binScalParmList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        binScalParmList.emplace_back(std::to_string(ss[i]));
    }
    std::string templateParam = JoinString(binScalParmList, ", ");
    templateParam += GenOpAttr();
    binScalParmList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    char scalarTmpBuffer[BUFFER_SIZE_256] = "CG_ERROR";
    int ret = sprintf_s(scalarTmpBuffer, sizeof(scalarTmpBuffer), "%.9g", extOperandVal.Cast<float>());
    ASSERT(ret >= 0) << "GenVectorScalarOp sprintf_s failed ";
    std::string tmpBuffer = "(__ubuf__ " + dstDtypeStr + "*)" + scalarTmpBuffer;
    binScalParmList.emplace_back(dst);
    binScalParmList.emplace_back(src0);
    binScalParmList.emplace_back(scalarTmpBuffer);
    std::string tiloOpCallParam = JoinString(binScalParmList, ", ");
    os << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryScalarDynamicUnaligned(const PrintBinaryScalarParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    std::vector dstShape = this->rawShape[0];
    std::vector src0Shape = this->rawShape[1];

    std::vector<int64_t> ss = NormalizeShape(src0Shape, SHAPE_DIM3);
    std::vector<int64_t> ds = NormalizeShape(dstShape, SHAPE_DIM3);

    auto dynSrcShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM3 - dynamicValidShape[1].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    int dimScalar = static_cast<int>(param.dim);
    paramList.emplace_back("/*DstRawShape*/");
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*Src0RawShape*/");
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; ++i) {
        paramList.emplace_back(std::to_string(ss[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    char scalarTmpBuffer[BUFFER_SIZE_256] = "CG_ERROR";
    int ret = sprintf_s(scalarTmpBuffer, sizeof(scalarTmpBuffer), "%.9g", extOperandVal.Cast<float>());
    ASSERT(ret >= 0) << "GenVectorScalarOp sprintf_s failed ";
    std::string tmpBuffer = "(__ubuf__ " + dstDtypeStr + "*)" + scalarTmpBuffer;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(scalarTmpBuffer);
    for (int i = SHAPE_DIM3 - dimScalar; i < SHAPE_DIM3; i++) {
        paramList.emplace_back(dynSrcShape[i].Dump());
    }
    std::string tiloOpCallParam = JoinString(paramList, ", ");

    os << tileOpName.c_str() << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryScalar(const PrintBinaryScalarParam &param) const {
    if (isSupportDynamicUnaligned) {
        return PrintBinaryScalarDynamicUnaligned(param);
    }
    return PrintBinaryScalarStatic(param);
}

std::string CodeGenOpCloudNPU::PrintVectorScalarOpDynamicUnalign(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    auto newDynSrcValidShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynSrcValidShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);
    char scalarTmp[BUFFER_SIZE_256] = "CG_ERROR";
    int ret = sprintf_s(scalarTmp, sizeof(scalarTmp), "%.9g", extOperandVal.Cast<float>());
    ASSERT(ret >= 0) << "GenVectorScalarOpByMode sprintf_s failed ";

    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0S*/");
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    std::string tmp = "(" + dstDtypeStr + ")" + scalarTmp;
    paramList.insert(paramList.end(), {dst, src, tmp});
    for (auto dynShape : newDynSrcValidShape) {
        paramList.emplace_back(dynShape.Dump());
    }

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenVectorScalarOpByMode(VecScalMode mode) const {
    auto kS0 = sm->CreateAllocKey(operandWithMagic[ID1]);
    auto kDst = sm->CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string dVar = sm->QueryVariableName(kDst);
    ASSERT(shape[0] == shape[1]) << " shape between dst " << IntVecToStr(shape[ID0]) << " and src "
                                 << IntVecToStr(shape[ID1]) << " is different";

    char buffer[BUFFER_SIZE_512] = "CG_ERROR";
    char scalarTmpBuffer[BUFFER_SIZE_256] = "CG_ERROR";
    int ret = sprintf_s(scalarTmpBuffer, sizeof(scalarTmpBuffer), "%.9g", extOperandVal.Cast<float>());
    ASSERT(ret >= 0) << "GenVectorScalarOpByMode sprintf_s failed ";
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufferVarOffset(std::vector{&dVar, &s0Var});

    std::vector src0RawShape = this->rawShape[1];
    std::vector dstRawShape = this->rawShape[0];
    std::vector<int64_t> os0 = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    if (mode == VecScalMode::SCALAR_MODE) {
        // Scalar op
        return PrintBinaryScalar({s0Var, dVar, dstDtypeStr, dstDtypeStr, shape[0].size()});
    }

    if (opAttrs.count(npu::tile_fwk::OP_EMUOP_PREFIX + "opc")) {
        // Hack: should be optimized to memory copy in pass
        int emuopc = AnyCast<int64_t>(opAttrs.find(npu::tile_fwk::OP_EMUOP_PREFIX + "opc")->second);
        if (emuopc == npu::tile_fwk::EMUOP_TENSOR_EXTRACT) {
            ret = sprintf_s(buffer, sizeof(buffer),
                "RUNTIME_TensorExtract(/*type=*/%s, /*mem=*/__ubuf__, /*dst*/%s, /*src*/%s);\n", dstDtypeStr.c_str(),
                dVar.c_str(), s0Var.c_str());
            ASSERT(ret >= 0) << "GenVectorScalarOpByMode " << OpcodeManager::Inst().GetOpcodeStr(opCode) << " failed "
                             << ret;
            return buffer;
        }
    }

    if (isSupportDynamicUnaligned) {
        return PrintVectorScalarOpDynamicUnalign({s0Var, dVar, dstDtypeStr, dstDtypeStr});
    }

    ret = sprintf_s(buffer, sizeof(buffer),
        "%s_<%s, %d, %d, %d, %d, /*DS*/ %d, %d, %d, /*S0S*/ %d, %d, %d>"
        "((__ubuf__ %s*)%s, (__ubuf__ %s*)%s, (%s)%s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), os0[ID0], os0[ID1], os0[ID2], os0[ID3], ds[ID1], ds[ID2], ds[ID3],
        s0[ID1], s0[ID2], s0[ID3], dstDtypeStr.c_str(), dVar.c_str(), dstDtypeStr.c_str(), s0Var.c_str(),
        dstDtypeStr.c_str(), scalarTmpBuffer);
    ASSERT(ret >= 0) << "GenVectorScalarOpByMode" << OpcodeManager::Inst().GetOpcodeStr(opCode) << " sprintf_s failed "
                     << ret;
    return buffer;
}

std::string CodeGenOpCloudNPU::GenPoolOp() const {
    const int poolParamsSize = 8;
    ASSERT(poolParams.size() == poolParamsSize);

    auto kSrc = sm->CreateAllocKey(operandWithMagic[ID1]);
    auto kDst = sm->CreateAllocKey(operandWithMagic[ID0]);
    std::string sVar = sm->QueryVariableName(kSrc);
    std::string dVar = sm->QueryVariableName(kDst);

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    AppendLocalBufferVarOffset(std::vector{&dVar, &sVar});

    int paddingLeft = poolParams[0];
    int paddingTop = poolParams[1];
    int paddingRight = poolParams[2];
    int paddingBottom = poolParams[3];
    int strideH = poolParams[4];
    int strideW = poolParams[5];
    int poolH = poolParams[6];
    int poolW = poolParams[7];

    std::vector dstRawShape = this->rawShape[0];
    int outputHeight = dstRawShape[2];
    int outputWidth = dstRawShape[3];
    std::vector srcRawShape = this->rawShape[1];
    int inputHeight = srcRawShape[2];
    int inputWidth = srcRawShape[3];

    std::string templateParamStr;
    templateParamStr += "/*outputHW*/";
    templateParamStr += std::to_string(outputHeight) + ", ";
    templateParamStr += std::to_string(outputWidth) + ", ";

    templateParamStr += "/*inputHW*/";
    templateParamStr += std::to_string(inputHeight) + ", ";
    templateParamStr += std::to_string(inputWidth) + ", ";

    templateParamStr += "/*stride*/";
    templateParamStr += std::to_string(strideH) + ", ";
    templateParamStr += std::to_string(strideW) + ", ";

    templateParamStr += "/*poolHpoolW*/";
    templateParamStr += std::to_string(poolH) + ", ";
    templateParamStr += std::to_string(poolW);

    bool hasPad = paddingLeft || paddingTop || paddingRight || paddingBottom;
    if (hasPad) {
        int poolHSizeHeadInit = poolH - paddingTop;
        int poolHSizeTailInit = inputHeight + paddingTop - (outputHeight - 1) * strideH;
        int poolWSizeHeadInit = poolW - paddingLeft;
        int poolWSizeTailInit = inputWidth + paddingLeft - (outputWidth - 1) * strideW;
        templateParamStr += ", ";
        templateParamStr += "/*actPoolInfo*/";
        templateParamStr += std::to_string(poolHSizeHeadInit) + ", ";
        templateParamStr += std::to_string(poolHSizeTailInit) + ", ";
        templateParamStr += std::to_string(poolWSizeHeadInit) + ", ";
        templateParamStr += std::to_string(poolWSizeTailInit) + ", ";

        int ltInputAddrOffset = 0;
        int ltOutputAddrOffset = 0;
        int rtInputAddrOffset = (inputWidth - poolWSizeTailInit) * BLOCK_SIZE;
        int rtOutputAddrOffset = (outputWidth - 1) * BLOCK_SIZE;
        int ldInputAddrOffset = (inputHeight - poolHSizeTailInit) * inputWidth * BLOCK_SIZE;
        int ldOutputAddrOffset = (outputHeight - 1) * outputWidth * BLOCK_SIZE;
        int rdInputAddrOffset =
            (inputWidth * (inputHeight - poolHSizeTailInit) + inputWidth - poolWSizeTailInit) * BLOCK_SIZE;
        int rdOutputAddrOffset = (outputWidth * (outputHeight - 1) + outputWidth - 1) * BLOCK_SIZE;

        templateParamStr += "/*cornerInfo*/";
        templateParamStr += std::to_string(ltInputAddrOffset) + ", ";
        templateParamStr += std::to_string(ltOutputAddrOffset) + ", ";
        templateParamStr += std::to_string(rtInputAddrOffset) + ", ";
        templateParamStr += std::to_string(rtOutputAddrOffset) + ", ";
        templateParamStr += std::to_string(ldInputAddrOffset) + ", ";
        templateParamStr += std::to_string(ldOutputAddrOffset) + ", ";
        templateParamStr += std::to_string(rdInputAddrOffset) + ", ";
        templateParamStr += std::to_string(rdOutputAddrOffset) + ", ";

        int outputWWithPadHead = (paddingLeft + strideW - 1) / strideW;
        int outputHWithPadHead = (paddingTop + strideH - 1) / strideH;
        int inputWBodyBeginWIdx = outputWWithPadHead * strideW - paddingLeft;
        int inputHBodyBeginHIdx = outputHWithPadHead * strideH - paddingTop;
        int tInputAddrOffset = inputWBodyBeginWIdx * BLOCK_SIZE;
        int tOutputAddrOffset = outputWWithPadHead * BLOCK_SIZE;
        int lInputAddrOffset = inputWidth * inputHBodyBeginHIdx * BLOCK_SIZE;
        int lOutputAddrOffset = outputWidth * outputHWithPadHead * BLOCK_SIZE;
        int rInputAddrOffset = (inputWidth * inputHBodyBeginHIdx + inputWidth - poolWSizeTailInit) * BLOCK_SIZE;
        int rOutputAddrOffset = (outputWidth * outputHWithPadHead + outputWidth - 1) * BLOCK_SIZE;
        int dInputAddrOffset = (inputWidth * (inputHeight - poolHSizeTailInit) + inputWBodyBeginWIdx) * BLOCK_SIZE;
        int dOutputAddrOffset = (outputWidth * (outputHeight - 1) + outputWWithPadHead) * BLOCK_SIZE;

        templateParamStr += "/*borderInfo*/";
        templateParamStr += std::to_string(tInputAddrOffset) + ", ";
        templateParamStr += std::to_string(tOutputAddrOffset) + ", ";
        templateParamStr += std::to_string(lInputAddrOffset) + ", ";
        templateParamStr += std::to_string(lOutputAddrOffset) + ", ";
        templateParamStr += std::to_string(rInputAddrOffset) + ", ";
        templateParamStr += std::to_string(rOutputAddrOffset) + ", ";
        templateParamStr += std::to_string(dInputAddrOffset) + ", ";
        templateParamStr += std::to_string(dOutputAddrOffset) + ", ";

        int outputHBody = (inputHeight + paddingTop - poolH) / strideH + 1 - outputHWithPadHead;
        int outputWBody = (inputWidth + paddingLeft - poolW) / strideW + 1 - outputWWithPadHead;
        int bInputAddrOffset = (inputWidth * inputHBodyBeginHIdx + inputWBodyBeginWIdx) * BLOCK_SIZE;
        int bOutputAddrOffset = (outputWidth * outputHWithPadHead + outputWWithPadHead) * BLOCK_SIZE;
        templateParamStr += "/*bodyInfo*/";
        templateParamStr += std::to_string(outputHBody) + ", ";
        templateParamStr += std::to_string(outputWBody) + ", ";
        templateParamStr += std::to_string(bInputAddrOffset) + ", ";
        templateParamStr += std::to_string(bOutputAddrOffset);
    }
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back(templateParamStr);
    std::string templateStr = JoinString(paramList, ", ");
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + sVar;
    paramList.insert(paramList.end(), {dst, src});
    std::string tileOpParam = JoinString(paramList, ", ");
    oss << tileOpName << "<" << templateStr << ">" << "(" << tileOpParam << ");\n";
    return oss.str();
}
} // namespace npu::tile_fwk
