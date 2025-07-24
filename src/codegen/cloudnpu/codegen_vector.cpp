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
#include "codegen/codegen_utils.h"

namespace npu::tile_fwk {
std::string CodeGenOpCloudNPU::GenCastOp() const {
    auto kS0 = CreateAllocKey(operandWithMagic[ID1]);
    auto kDst = CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string dVar = sm->QueryVariableName(kDst);

    std::vector srcShape = this->rawShape[ID1];
    ALOG_INFO_F("genCastOp %s, srcShape is %s", tileOpName.c_str(), IntVecToStr(srcShape).c_str());

    std::vector dstShape = this->rawShape[ID0];
    ALOG_INFO_F("genCastOp %s, dstShape is %s", tileOpName.c_str(), IntVecToStr(dstShape).c_str());

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufferVarOffset({&dVar, &s0Var}, {ID0, ID1});
    std::vector<int> os = NormalizeShape(originShape[0], SHAPE_DIM4);
    std::vector<int> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int ret = 0;
    if (isSupportDynamicUnaligned) {
        auto dynDstShape = dynamicValidShape[0];
        std::vector<SymbolicScalar> newDynDstShape = dynDstShape;
        FillIntVecWithDummyInHead<SymbolicScalar>(newDynDstShape, SHAPE_DIM4 - dynDstShape.size(), 1);
        ret = sprintf_s(buffer, sizeof(buffer),
            "%s_<%s, %s, %u, %u, %u, %u, %u, %u %s>((__ubuf__ %s *)%s,  (__ubuf__ %s *)%s, %s, %s, %s, %s);\n",
            tileOpName.c_str(), dstDtypeStr.c_str(), srcDtypeStr.c_str(), ds[1], ds[2], ds[3], ss[1], ss[2], ss[3],
            GenOpAttr().c_str(), dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str(),
            newDynDstShape[0].Dump().c_str(), newDynDstShape[1].Dump().c_str(), newDynDstShape[2].Dump().c_str(),
            newDynDstShape[3].Dump().c_str());
        ASSERT(ret >= 0) << "GenCastOp sprintf_s failed " << ret;
        return buffer;
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
    std::vector<int> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

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
    std::vector<int> dos = NormalizeShape(originShape[0], SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

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
    auto kDst = CreateAllocKey(operandWithMagic[ID0]);
    std::string dVar = sm->QueryVariableName(kDst);
    AppendLocalBufferVarOffset({&dVar}, {0});
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    std::string dupV;
    if (dstDtypeStr == "float") {
        auto scalar = opAttrs.at(OpAttributeKey::scalar);
        ASSERT(scalar.HasValue() && (scalar.Type() == typeid(Element)))
            << npu::tile_fwk::AnyCast<Element>(scalar).IsFloat() << "SCALAR attribute has to have float value.";
        dupV = std::to_string(npu::tile_fwk::AnyCast<Element>(scalar).Cast<float>());
    } else if (dstDtypeStr == "int32_t") {
        auto scalar = opAttrs.at(OpAttributeKey::scalar);
        ASSERT(scalar.HasValue() && (scalar.Type() == typeid(Element)))
            << npu::tile_fwk::AnyCast<Element>(scalar).IsSigned() << "SCALAR attribute has to have int value.";
        dupV = std::to_string(npu::tile_fwk::AnyCast<Element>(scalar).Cast<int>());
    } else {
        ASSERT(false) << "unsupported type";
    }
    return PrintDupOp({dVar, dstDtypeStr, dupV});
}

std::string CodeGenOpCloudNPU::PrintRowSumlineStatic(const PrintRowSumLineParam &param) const {
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.HasValue()) {
        reduceAxis = npu::tile_fwk::AnyCast<int>(axis);
    }
    ASSERT(((reduceAxis >= 0) && (reduceAxis < (int(shape[1].size()) - 1)))) << "unsupported reduce axis";
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    reduceAxis += SHAPE_DIM4 - rawShape[0].size();
    std::vector<int> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int> dstShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::vector<int> os = NormalizeShape(originShape[1], SHAPE_DIM4);
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

std::string CodeGenOpCloudNPU::PrintRowSumlineDynamicUnaligned(const PrintRowSumLineParam &param) const {
    int reduceAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "AXIS");
    if (axis.HasValue()) {
        reduceAxis = npu::tile_fwk::AnyCast<int>(axis);
    }
    ASSERT(((reduceAxis >= 0) && (reduceAxis < (int(shape[1].size()) - 1)))) << "unsupported reduce axis";
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    auto dynSrcShape = dynamicValidShape[1];
    // adjust reduceAxis for dim4
    reduceAxis += SHAPE_DIM4 - rawShape[0].size();
    std::vector<int> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int> dstShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
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

std::string CodeGenOpCloudNPU::PrintRowSumline(const PrintRowSumLineParam &param) const {
    if (isSupportDynamicUnaligned) {
        return PrintRowSumlineDynamicUnaligned(param);
    }
    return PrintRowSumlineStatic(param);
}

std::string CodeGenOpCloudNPU::PrintUnaryDynamicUnaligned(const PrintUnaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;

    std::vector<int> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/ " + std::to_string(ds[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*SS*/ " + std::to_string(ss[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
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
    std::vector<int> os0 = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*OS*/ " + std::to_string(os0[0]));
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os0[i]));
    }
    paramList.emplace_back("/*DS*/ " + std::to_string(ds[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*SS*/ " + std::to_string(ss[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
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
    auto kS0 = CreateAllocKey(operandWithMagic[ID1]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string dVar = GenGmParamVar(ID0);

    std::vector<int> srcShape = this->rawShape[1];
    ALOG_INFO_F("GenUnaryOp: srcShape is %s", IntVecToStr(srcShape).c_str());
    std::vector<int> dstShape = this->rawShape[0];
    ALOG_INFO_F("GenUnaryOp: dstShape is %s", IntVecToStr(dstShape).c_str());

    AppendLocalBufferVarOffset({&dVar, &s0Var}, {0, 1});

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    return PrintTransposeDataMove({s0Var, dstShape, srcDtypeStr, dstDtypeStr});
}

std::string CodeGenOpCloudNPU::GenUnaryOp() const {
    auto kS0 = CreateAllocKey(operandWithMagic[ID1]);
    std::string s0Var = sm->QueryVariableName(kS0);

    auto kDst = CreateAllocKey(operandWithMagic[ID0]);
    std::string dVar = sm->QueryVariableName(kDst);

    std::vector<int> srcShape = this->rawShape[1];
    ALOG_INFO_F("GenUnaryOp: srcShape is %s", IntVecToStr(srcShape).c_str());

    std::vector<int> dstShape = this->rawShape[0];
    ALOG_INFO_F("GenUnaryOp: dstShape is %s", IntVecToStr(dstShape).c_str());

    unsigned tShape0 = std::min(dstShape[0], shape[0][0]);
    unsigned tShape1 = std::min(dstShape[1], shape[0][1]);

    unsigned tShape2{1};
    if (shape[0].size() >= SHAPE_DIM3) {
        tShape2 = std::min(dstShape[2], shape[0][2]);
    }
    unsigned tShape3{1};
    if (shape[0].size() >= SHAPE_DIM4) {
        tShape3 = std::min(dstShape[3], shape[0][3]);
    }
    unsigned tShape4{1};
    if (shape[0].size() >= SHAPE_DIM5) {
        tShape4 = std::min(dstShape[4], shape[0][4]);
    }

    AppendLocalBufferVarOffset({&dVar, &s0Var}, {0, 1});

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    if (opCode == Opcode::OP_COPY_UB_TO_UB) {
        srcDtypeStr = GetTypeForB16B32(operandDtype[ID1]);
        dstDtypeStr = GetTypeForB16B32(operandDtype[ID0]);
    }

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int ret = 0;
    if (opCode == Opcode::OP_EXPAND) {
        return PrintExpand(s0Var, dVar, srcDtypeStr, dstDtypeStr);
    } else if (opCode == Opcode::OP_ROWMAX || opCode == Opcode::OP_ROWEXPMAX || opCode == Opcode::OP_ROWEXPSUM) {
        const std::vector<int> &oriSrcShape = originShape[1];
        if (shape[0].size() == SHAPE_DIM2) {
            ret = sprintf_s(buffer, sizeof(buffer), "%s<%s, %u, %u, %d, %d>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s);\n",
                tileOpName.c_str(), dstDtypeStr.c_str(), tShape0, tShape1, oriSrcShape[0], oriSrcShape[1],
                dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str());
        } else if (shape[0].size() == SHAPE_DIM4) {
            ret = sprintf_s(buffer, sizeof(buffer),
                "%s<%s, %u, %u, %u, %u, %d, %d>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s);\n", tileOpName.c_str(),
                dstDtypeStr.c_str(), tShape0, tShape1, tShape2, tShape3, oriSrcShape[2], oriSrcShape[3],
                dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str());
        }
    } else if (opCode == Opcode::OP_ROWSUMLINE) {
        return PrintRowSumline({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    } else if (opCode == Opcode::OP_EXP || opCode == Opcode::OP_SQRT || opCode == Opcode::OP_ABS ||
               opCode == Opcode::OP_RECIPROCAL) {
        return PrintUnary({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    } else {
        if (shape[0].size() == SHAPE_DIM2) {
            ret = sprintf_s(buffer, sizeof(buffer),
                "%s<%s, %u, %u, %d, %d, %d, %d>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s);\n", tileOpName.c_str(),
                dstDtypeStr.c_str(), tShape0, tShape1, dstShape[0], dstShape[1], srcShape[0], srcShape[1],
                dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str());
        } else if (shape[0].size() == SHAPE_DIM3) {
            ret = sprintf_s(buffer, sizeof(buffer),
                "%s<%s, %u, %u, %u, %d, %d, %d, %d>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s);\n", tileOpName.c_str(),
                dstDtypeStr.c_str(), tShape0, tShape1, tShape2, dstShape[1], dstShape[2], srcShape[1], srcShape[2],
                dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str());
        } else if (shape[0].size() == SHAPE_DIM4) {
            ret = sprintf_s(buffer, sizeof(buffer),
                "%s<%s, %u, %u, %u, %u, %d, %d, %d, %d>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s);\n", tileOpName.c_str(),
                dstDtypeStr.c_str(), tShape0, tShape1, tShape2, tShape3, dstShape[2], dstShape[3], srcShape[2],
                srcShape[3], dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str());
        } else if (shape[0].size() == SHAPE_DIM5) {
            ret = sprintf_s(buffer, sizeof(buffer),
                "%s<%s, %u, %u, %u, %u, %u, %d, %d, %d, %d, %d, %d, %d, %d>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s);\n",
                tileOpName.c_str(), dstDtypeStr.c_str(), tShape0, tShape1, tShape2, tShape3, tShape4, dstShape[1],
                dstShape[2], dstShape[3], dstShape[4], srcShape[1], srcShape[2], srcShape[3], srcShape[4],
                dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str());
        }
    }

    ASSERT(ret >= 0) << "GenUnaryOp" << OpcodeManager::Inst().GetOpcodeStr(opCode) << " sprintf_s failed " << ret;
    return buffer;
}

std::string CodeGenOpCloudNPU::PrintExpand(const std::string &s0Var, const std::string &dVar,
    const std::string &srcDtypeStr, const std::string &dstDtypeStr) const {
    char buffer[256] = "CG_ERROR";
    int ret = 0;
    int expandAxis{-1};
    std::vector<int> dos = NormalizeShape(originShape[0], SHAPE_DIM4);
    std::vector<int> os = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int> ss = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "EXPANDDIM");
    if (axis.HasValue()) {
        expandAxis = AnyCast<int>(axis);
    }
    ASSERT((expandAxis >= 0) && (expandAxis <= (static_cast<int>(shape[1].size() - 1)))) << "unsupported reduce axis";
    // modify expandAxis for SHAPE_DIM4
    expandAxis += SHAPE_DIM4 - shape[1].size();

    if (isSupportDynamicUnaligned) {
        auto dynDstShape = dynamicValidShape[0];
        std::vector<SymbolicScalar> newDynDstShape = dynDstShape;
        FillIntVecWithDummyInHead<SymbolicScalar>(newDynDstShape, SHAPE_DIM4 - dynDstShape.size(), 1);
        auto dynSrcShape = dynamicValidShape[1];
        std::vector<SymbolicScalar> newDynSrcShape = dynSrcShape;
        FillIntVecWithDummyInHead<SymbolicScalar>(newDynSrcShape, SHAPE_DIM4 - dynSrcShape.size(), 1);
        ret = sprintf_s(buffer, sizeof(buffer),
            "%s_<%s, /*DS*/ %d, %d, %d, /*SS*/ %d, %d, %d, %d>((__ubuf__ %s*)%s, (__ubuf__ %s*)%s, %s, %s, %s, %s, %s, "
            "%s, %s, %s);\n",
            tileOpName.c_str(), dstDtypeStr.c_str(), ds[ID1], ds[ID2], ds[ID3], ss[ID1], ss[ID2], ss[ID3], expandAxis,
            dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str(), newDynDstShape[0].Dump().c_str(),
            newDynDstShape[1].Dump().c_str(), newDynDstShape[2].Dump().c_str(), newDynDstShape[3].Dump().c_str(),
            newDynSrcShape[0].Dump().c_str(), newDynSrcShape[1].Dump().c_str(), newDynSrcShape[2].Dump().c_str(),
            newDynSrcShape[3].Dump().c_str());
        ASSERT(ret >= 0) << "GenUnaryOp" << OpcodeManager::Inst().GetOpcodeStr(opCode) << " sprintf_s failed " << ret;
        return buffer;
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
    const std::string &s0Var = param.s0Var;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::string dstVar = GenGmParamVar(ID0);
    std::vector<int> os = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int> dstShape = NormalizeShape(param.dstShape, SHAPE_DIM4);
    std::vector<int> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;

    paramList.emplace_back(dstDtypeStr);
    for (auto oriShape : os) {
        paramList.emplace_back(std::to_string(oriShape));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    std::vector<int> transposeAxis = npu::tile_fwk::AnyCast<std::vector<int>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM4 - originShape[0].size();
    for (auto &axis : transposeAxis) {
        axis += correctionAxis;
        paramList.emplace_back(std::to_string(axis));
    }
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();

    std::string dst = "(__gm__ " + dstDtypeStr + "*)" + dstVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dst, src});
    std::string tiloOpCallParam = JoinString(paramList, ", ");

    oss << tileOpName.c_str() << "_<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintTransposeDataMoveDynamic(const PrintTransposeDataMoveParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::string dstVar = GenGmParamVar(ID0);

    int dim = static_cast<int>(paramIdxForDynShape[ID0].size());
    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(ID0, dim, PREFIX_STR_RAW_SHAPE);
    FillIntVecWithDummyInHead<std::string>(gmShapeExpr, SHAPE_DIM4 - dim, "1");
    ALOG_INFO_F("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(ID0, dim, PREFIX_STR_OFFSET);
    FillIntVecWithDummyInHead<std::string>(gmOffsetExpr, SHAPE_DIM4 - dim, "0");
    ALOG_INFO_F("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());

    std::vector<int> os = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (auto oriShape : os) {
        paramList.emplace_back(std::to_string(oriShape));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    std::vector<int> transposeAxis = npu::tile_fwk::AnyCast<std::vector<int>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM4 - originShape[1].size();
    for (auto &axis : transposeAxis) {
        axis += correctionAxis;
        paramList.emplace_back(std::to_string(axis));
    }
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();

    std::string dst = "(__gm__ " + dstDtypeStr + "*)" + dstVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
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
    const std::string &s0Var = param.s0Var;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::string dstVar = GenGmParamVar(ID0);

    int dim = static_cast<int>(paramIdxForDynShape[ID0].size());
    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(ID0, dim, PREFIX_STR_RAW_SHAPE);
    FillIntVecWithDummyInHead<std::string>(gmShapeExpr, SHAPE_DIM4 - dim, "1");
    ALOG_INFO_F("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(ID0, dim, PREFIX_STR_OFFSET);
    FillIntVecWithDummyInHead<std::string>(gmOffsetExpr, SHAPE_DIM4 - dim, "0");
    ALOG_INFO_F("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());
    auto newDynSrcValidShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynSrcValidShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);

    std::vector<int> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    std::vector<int> transposeAxis = npu::tile_fwk::AnyCast<std::vector<int>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM4 - originShape[1].size();
    for (auto &axis : transposeAxis) {
        axis += correctionAxis;
        paramList.emplace_back(std::to_string(axis));
    }
    std::string templateParam = JoinString(paramList, ", ");
    paramList.clear();

    std::string dst = "(__gm__ " + dstDtypeStr + "*)" + dstVar;
    std::string src = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dst, src});
    for (auto dynShape : newDynSrcValidShape) {
        paramList.emplace_back(dynShape.Dump());
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

std::string CodeGenOpCloudNPU::PrintVnchwconvStatic(const PrintVnchwconvParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::vector<int> os0 = NormalizeShape(originShape[2], SHAPE_DIM5);
    std::vector<int> s0 = NormalizeShape(rawShape[2], SHAPE_DIM5);
    std::vector<int> ds = NormalizeShape(rawShape[0], SHAPE_DIM5);
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

std::string CodeGenOpCloudNPU::PrintVnchwconvDynUnaligned(const PrintVnchwconvParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::vector<int> s0 = NormalizeShape(rawShape[2], SHAPE_DIM5);
    std::vector<int> ds = NormalizeShape(rawShape[0], SHAPE_DIM5);
    auto newDynSrcValidShape = dynamicValidShape[2];
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynSrcValidShape, SHAPE_DIM5 - dynamicValidShape[2].size(), 1);

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

std::string CodeGenOpCloudNPU::PrintVnchwconv(const PrintVnchwconvParam &param) const {
    if (isSupportDynamicUnaligned) {
        return PrintVnchwconvDynUnaligned(param);
    }
    return PrintVnchwconvStatic(param);
}

std::string CodeGenOpCloudNPU::GenUnaryOpWithTmpBuff() const {
    // Output{dst, tmp buffer}, Input{src}
    // In this scenario, frontend set tmp buffer in output to optimize ooo schedule result.
    auto kS0 = CreateAllocKey(operandWithMagic[ID2]);
    auto kTmp = CreateAllocKey(operandWithMagic[ID1]);
    auto kDst = CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string tmpVar = sm->QueryVariableName(kTmp);
    std::string dVar = sm->QueryVariableName(kDst);

    std::vector srcShape = this->rawShape[2];
    ALOG_INFO_F("GenUnaryOpWithTmpBuff %s src raw shape: %s", tileOpName.c_str(), IntVecToStr(srcShape).c_str());

    std::vector dstShape = this->rawShape[0];
    ALOG_INFO_F("GenUnaryOpWithTmpBuff %s dst raw shape: %s", tileOpName.c_str(), IntVecToStr(dstShape).c_str());

    unsigned tShape0 = std::min(dstShape[0], shape[0][0]);
    unsigned tShape1 = std::min(dstShape[1], shape[0][1]);
    unsigned srcShape0 = std::min(srcShape[0], shape[2][0]);
    unsigned srcShape1 = std::min(srcShape[1], shape[2][1]);
    unsigned tShape2{1};
    unsigned srcShape2{1};
    if (shape[0].size() >= SHAPE_DIM3) {
        tShape2 = std::min(dstShape[2], shape[0][2]);
        srcShape2 = std::min(srcShape[2], shape[2][2]);
    }
    unsigned tShape3{1};
    unsigned srcShape3{1};
    if (shape[0].size() >= SHAPE_DIM4) {
        tShape3 = std::min(dstShape[3], shape[0][3]);
        srcShape3 = std::min(srcShape[3], shape[2][3]);
    }
    unsigned srcShape4{1};
    if (shape[0].size() >= SHAPE_DIM5) {
        srcShape4 = std::min(srcShape[4], shape[2][4]);
    }

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string tmpDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufferVarOffset({&dVar, &tmpVar, &s0Var}, {0, 1, 2});

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int ret = 0;
    if (opCode == Opcode::OP_TRANSPOSE_VNCHWCONV) {
        return PrintVnchwconv({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    if (opCode == Opcode::OP_ROWSUM_SINGLE || opCode == Opcode::OP_ROWMAX_SINGLE) {
        return PrintReduceLastAxis({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    if (opCode == Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE || opCode == Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE) {
        std::vector<int> dstOriginShape = NormalizeShape(originShape[0], SHAPE_DIM4);
        std::vector<int> srcOriginShape = NormalizeShape(originShape[2], SHAPE_DIM4);
        std::vector<int> srcRawShape = NormalizeShape(rawShape[2], SHAPE_DIM4);
        std::vector<int> dstRawShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
        std::vector<int> tmpRawShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
        ret = sprintf_s(buffer, sizeof(buffer),
            "%s<%s, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s, (__ubuf__ "
            "%s *)%s);\n",
            tileOpName.c_str(), dstDtypeStr.c_str(), srcOriginShape[ID0], srcOriginShape[ID1], srcOriginShape[ID2],
            srcOriginShape[ID3], dstRawShape[ID1], dstRawShape[ID2], dstRawShape[ID3], srcRawShape[ID1],
            srcRawShape[ID2], srcRawShape[ID3], tmpRawShape[ID3], dstDtypeStr.c_str(), dVar.c_str(),
            srcDtypeStr.c_str(), s0Var.c_str(), tmpDtypeStr.c_str(), tmpVar.c_str());
        ASSERT(ret >= 0) << "genUnaryOpWithTmpBuff sprintf_s failed ";
        return std::string(buffer);
    }

    if (shape[0].size() == SHAPE_DIM2) {
        ret = sprintf_s(buffer, sizeof(buffer),
            "%s<%s, %u, %u, %u, %u %s>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s, (__ubuf__ %s *)%s);\n", tileOpName.c_str(),
            dstDtypeStr.c_str(), tShape0, tShape1, srcShape0, srcShape1, GenOpAttr().c_str(), dstDtypeStr.c_str(),
            dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str(), tmpDtypeStr.c_str(), tmpVar.c_str());
    } else if (shape[0].size() == SHAPE_DIM3) {
        ret = sprintf_s(buffer, sizeof(buffer),
            "%s<%s, %u, %u, %u, %u, %u, %u %s>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s, (__ubuf__ %s *)%s);\n",
            tileOpName.c_str(), dstDtypeStr.c_str(), tShape0, tShape1, tShape2, srcShape0, srcShape1, srcShape2,
            GenOpAttr().c_str(), dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str(),
            tmpDtypeStr.c_str(), tmpVar.c_str());
    } else if (shape[0].size() == SHAPE_DIM4) {
        ret = sprintf_s(buffer, sizeof(buffer),
            "%s<%s, %u, %u, %u, %u, %u, %u, %u, %u %s>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s, (__ubuf__ %s *)%s);\n",
            tileOpName.c_str(), dstDtypeStr.c_str(), tShape0, tShape1, tShape2, tShape3, srcShape0, srcShape1,
            srcShape2, srcShape3, GenOpAttr().c_str(), dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(),
            s0Var.c_str(), tmpDtypeStr.c_str(), tmpVar.c_str());
    } else if (shape[0].size() == SHAPE_DIM5) {
        ret = sprintf_s(buffer, sizeof(buffer),
            "%s<%s, %u, %u, %u, %u, %u, %d, %d, %d, %d, %d %s>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s, "
            "(__ubuf__ %s *)%s);\n",
            tileOpName.c_str(), dstDtypeStr.c_str(), srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape[0],
            dstShape[1], dstShape[2], dstShape[3], dstShape[4], GenOpAttr().c_str(), dstDtypeStr.c_str(), dVar.c_str(),
            srcDtypeStr.c_str(), s0Var.c_str(), tmpDtypeStr.c_str(), tmpVar.c_str());
    }
    ASSERT(ret >= 0) << "genUnaryOpWithTmpBuff sprintf_s failed ";
    std::string ostring(buffer);
    return ostring;
}

std::string CodeGenOpCloudNPU::PrintReduceLastAxis(const PrintReduceLastAxisParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int ret = 0;

    std::vector<int> dstOriginShape = NormalizeShape(originShape[0], SHAPE_DIM4);
    std::vector<int> srcOriginShape = NormalizeShape(originShape[2], SHAPE_DIM4);
    std::vector<int> srcRawShape = NormalizeShape(rawShape[2], SHAPE_DIM4);
    std::vector<int> dstRawShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::vector<int> tmpRawShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    ALOG_INFO_F("rawShape[2] is %s", IntVecToStr(rawShape[2]).c_str());
    ASSERT(dstOriginShape[ID3] == 1) << "Dst last axis length must be 1";

    if (isSupportDynamicUnaligned) {
        auto newDynSrcValidShape = dynamicValidShape[2];
        FillIntVecWithDummyInHead<SymbolicScalar>(newDynSrcValidShape, SHAPE_DIM4 - dynamicValidShape[2].size(), 1);
        ret = sprintf_s(buffer, sizeof(buffer),
            "%s_<%s, %u, %u, %u, %u, %u, %u, %u>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s, (__ubuf__ %s *)%s, %s, %s, "
            "%s, %s);\n",
            tileOpName.c_str(), dstDtypeStr.c_str(), dstRawShape[ID1], dstRawShape[ID2], dstRawShape[ID3],
            srcRawShape[ID1], srcRawShape[ID2], srcRawShape[ID3], tmpRawShape[ID3], dstDtypeStr.c_str(), dVar.c_str(),
            srcDtypeStr.c_str(), s0Var.c_str(), tmpDtypeStr.c_str(), tmpVar.c_str(),
            newDynSrcValidShape[0].Dump().c_str(), newDynSrcValidShape[1].Dump().c_str(),
            newDynSrcValidShape[2].Dump().c_str(), newDynSrcValidShape[3].Dump().c_str());
        ASSERT(ret >= 0) << "PrintReduceLastAxis" << OpcodeManager::Inst().GetOpcodeStr(opCode) << " sprintf_s failed "
                         << ret;
        return buffer;
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

    std::vector<int> os0 = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int> os1 = NormalizeShape(originShape[2], SHAPE_DIM4);
    std::vector<int> s0 = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int> s1 = NormalizeShape(rawShape[2], SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*OS0*/ " + std::to_string(os0[0]));
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os0[i]));
    }
    paramList.emplace_back("/*OS1*/ " + std::to_string(os1[ID3]));
    paramList.emplace_back("/*DS*/ " + std::to_string(ds[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/ " + std::to_string(s0[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/ " + std::to_string(s1[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    bool copyFlag = false;
    if (opCode == Opcode::OP_PAIRMAX || opCode == Opcode::OP_PAIRSUM) {
        copyFlag = true;
    }
    paramList.emplace_back("/*copyFlag*/ " + std::to_string(copyFlag));
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryDynamicUnaligned(const PrintBinaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;

    std::vector<int> s0 = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int> s1 = NormalizeShape(rawShape[2], SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::vector<SymbolicScalar> dynSrcShape0 = dynamicValidShape[1];
    std::vector<SymbolicScalar> dynSrcShape1 = dynamicValidShape[2];

    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape0, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape1, SHAPE_DIM4 - dynamicValidShape[2].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/ " + std::to_string(ds[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/ " + std::to_string(s0[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/ " + std::to_string(s1[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
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
    paramList.emplace_back(dynSrcShape1[ID3].Dump());
    bool copyFlag = false;
    if (opCode == Opcode::OP_PAIRMAX || opCode == Opcode::OP_PAIRSUM) {
        copyFlag = true;
    }
    paramList.emplace_back(std::to_string(copyFlag));
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinary(const PrintBinaryParam &param) const {
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

    std::vector<int> os0 = NormalizeShape(originShape[2], SHAPE_DIM4);
    std::vector<int> s0 = NormalizeShape(rawShape[2], SHAPE_DIM4);
    std::vector<int> s1 = NormalizeShape(rawShape[3], SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*OS0*/ " + std::to_string(os0[0]));
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os0[i]));
    }
    paramList.emplace_back("/*DS*/ " + std::to_string(ds[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/ " + std::to_string(s0[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/ " + std::to_string(s1[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    paramList.emplace_back("/*isCombineAxis*/ " + std::to_string(isInputForceCombineAxis));
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);
    paramList.emplace_back(tmp);

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    ;

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

    std::vector<int> os0 = NormalizeShape(originShape[2], SHAPE_DIM4);
    std::vector<int> s0 = NormalizeShape(rawShape[2], SHAPE_DIM4);
    std::vector<int> s1 = NormalizeShape(rawShape[3], SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    auto dynSrcShape = dynamicValidShape[2];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[2].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/ " + std::to_string(ds[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/ " + std::to_string(s0[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/ " + std::to_string(s1[1]));
    for (int i = 2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    paramList.emplace_back("/*isCombineAxis*/ " + std::to_string(isInputForceCombineAxis));
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    std::string tmp = "(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);
    paramList.emplace_back(tmp);
    for (auto dynShape : dynSrcShape) {
        paramList.emplace_back(dynShape.Dump());
    }

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    ;

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryBrc(const PrintBinaryBrcParam &param) const {
    if (isSupportDynamicUnaligned) {
        return PrintBinaryBrcDynamicUnaligned(param);
    }
    return PrintBinaryBrcStatic(param);
}

std::string CodeGenOpCloudNPU::GenBinaryOp() const {
    auto kS0 = CreateAllocKey(operandWithMagic[ID1]);
    auto kDst = CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string dVar = sm->QueryVariableName(kDst);

    std::vector src0RawShape = this->rawShape[ID1];
    std::vector src1RawShape = this->rawShape[ID2];
    ALOG_INFO_F("genBinaryOp %s, src0RawShape is %s", tileOpName.c_str(), IntVecToStr(src0RawShape).c_str());

    unsigned tShape0 = std::min(src0RawShape[ID0], shape[ID0][ID0]);
    unsigned tShape1 = std::min(src0RawShape[ID1], shape[ID0][ID1]);
    unsigned tShape2{1};
    unsigned tShape3{1};
    if (shape[0].size() == SHAPE_DIM3) {
        tShape2 = std::min(src0RawShape[2], shape[0][2]);
    } else if (shape[0].size() == SHAPE_DIM4) {
        tShape2 = std::min(src0RawShape[2], shape[0][2]);
        tShape3 = std::min(src0RawShape[3], shape[0][3]);
    }

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);

    auto kS1 = CreateAllocKey(operandWithMagic[ID2]);
    std::string s1Var = sm->QueryVariableName(kS1);

    AppendLocalBufferVarOffset({&dVar, &s0Var, &s1Var}, {0, 1, 2});
    int ret = 0;
    if (opCode == Opcode::OP_ADD || opCode == Opcode::OP_SUB || opCode == Opcode::OP_MUL || opCode == Opcode::OP_DIV ||
        opCode == Opcode::OP_MAXIMUM || opCode == Opcode::OP_PAIRMAX || opCode == Opcode::OP_PAIRSUM) {
        return PrintBinary({s0Var, s1Var, dVar, src0DtypeStr, src1DtypeStr, dstDtypeStr});
    } else {
        if (shape[0].size() == SHAPE_DIM2) {
            ret = sprintf_s(buffer, sizeof(buffer),
                "%s<%s, %u, %u>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s, (__ubuf__ %s *)%s);\n", tileOpName.c_str(),
                dstDtypeStr.c_str(), tShape0, tShape1, dstDtypeStr.c_str(), dVar.c_str(), src0DtypeStr.c_str(),
                s0Var.c_str(), src1DtypeStr.c_str(), s1Var.c_str());
        } else if (shape[0].size() == SHAPE_DIM4) {
            ret = sprintf_s(buffer, sizeof(buffer),
                "%s<%s, %u, %u, %u, %u>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s, (__ubuf__ %s *)%s);\n",
                tileOpName.c_str(), dstDtypeStr.c_str(), tShape0, tShape1, tShape2, tShape3, dstDtypeStr.c_str(),
                dVar.c_str(), src0DtypeStr.c_str(), s0Var.c_str(), src1DtypeStr.c_str(), s1Var.c_str());
        }
    }
    ASSERT(ret >= 0) << "GenBinaryOp sprintf_s failed ";
    return buffer;
}

std::string CodeGenOpCloudNPU::GenBinaryWithBrc() const {
    auto kS0 = CreateAllocKey(operandWithMagic[ID2]);
    auto kDst = CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string dVar = sm->QueryVariableName(kDst);

    std::vector src0RawShape = this->rawShape[ID2];
    std::vector src1RawShape = this->rawShape[ID3];
    ALOG_INFO_F("GenBinaryWithBrc %s, src0RawShape is %s", tileOpName.c_str(), IntVecToStr(src0RawShape).c_str());

    char buffer[256] = "CG_ERROR";
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID3]);

    auto kS1 = CreateAllocKey(operandWithMagic[ID3]);
    std::string s1Var = sm->QueryVariableName(kS1);
    SymbolManager::AllocKey kTmp;
    std::string tmpVar;
    std::string tmpDtypeStr;
    kTmp = CreateAllocKey(operandWithMagic[ID1]);
    tmpVar = sm->QueryVariableName(kTmp);
    tmpDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    AppendLocalBufferVarOffset({&dVar, &s0Var, &s1Var, &tmpVar}, {0, 1, 2, 3});
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
    std::vector<std::string> varList;
    for (int i = 0; i < operandsNum; i++) {
        auto kS0 = CreateAllocKey(operandWithMagic[i]);
        std::string s0Var = sm->QueryVariableName(kS0);
        std::string dtypeStr = DataType2CCEStr(operandDtype[i]);
        paramString += "(__ubuf__ " + dtypeStr + "*)" + s0Var;
        if (i < operandsNum - 1) {
            paramString += ", ";
        }
        varList.push_back(s0Var);
    }
    std::vector<std::string *> varPtrList;
    std::vector<unsigned> operandIdxes;
    for (size_t i = 0; i < varList.size(); i++) {
        varPtrList.push_back(&varList[i]);
        operandIdxes.push_back(i);
    }

    char buffer[256] = "CG_ERROR";
    AppendLocalBufferVarOffset(varPtrList, operandIdxes);
    int ret = 0;
    ret = sprintf_s(buffer, sizeof(buffer), "%s(%s);\n", fusedOpName.c_str(), paramString.c_str());
    ASSERT(ret >= 0) << "GenBinaryOp sprintf_s failed ";
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

    std::vector<int> dos = NormalizeShape(originShape[ID0], SHAPE_DIM4);
    std::vector<int> ss = NormalizeShape(src0Shape, SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(dstShape, SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(src0DtypeStr);
    paramList.emplace_back(src1DtypeStr);
    paramList.emplace_back("/*DOS*/ " + std::to_string(dos[ID1]));
    for (int i = ID2; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dos[i]));
    }
    paramList.emplace_back("/*SS*/ " + std::to_string(ss[ID3]));
    paramList.emplace_back("/*DS*/ " + std::to_string(ds[ID3]));
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

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
    std::vector<int> dos = NormalizeShape(originShape[ID0], SHAPE_DIM4);
    std::vector<int> ss = NormalizeShape(src0Shape, SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(dstShape, SHAPE_DIM4);

    auto dynDstShape = dynamicValidShape[ID0];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynDstShape, SHAPE_DIM4 - dynamicValidShape[ID0].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(src0DtypeStr);
    paramList.emplace_back(src1DtypeStr);
    paramList.emplace_back("/*SS*/ " + std::to_string(ss[ID3]));
    paramList.emplace_back("/*DS*/ " + std::to_string(ds[ID3]));
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
    auto kS0 = CreateAllocKey(operandWithMagic[ID1]);
    auto kS1 = CreateAllocKey(operandWithMagic[ID2]);
    auto kDst = CreateAllocKey(operandWithMagic[ID0]);
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

    AppendLocalBufferVarOffset({&dVar, &s0Var, &s1Var}, {0, 1, 2});
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
    std::vector<int> &dstOriginShape = param.dstOriginShape;
    std::vector<int> &dstRawShape = param.dstRawShape;
    std::vector<int> &src0RawShape = param.src0RawShape;
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
    std::vector<int> &dstRawShape = param.dstRawShape;
    std::vector<int> &src0RawShape = param.src0RawShape;
    const std::string *dataTypeExpr = param.dataTypeExpr;
    // template param
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID1], dataTypeExpr[ID2]});
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
    auto dstValidShape = dynamicValidShape[ID0];
    paramList.emplace_back(dstValidShape[ID0].Dump());
    paramList.emplace_back(dstValidShape[ID1].Dump());
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenGatherElementOp() const {
    auto kS0 = CreateAllocKey(operandWithMagic[ID1]);
    auto kS1 = CreateAllocKey(operandWithMagic[ID2]);
    auto kDst = CreateAllocKey(operandWithMagic[ID0]);
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
    ALOG_INFO_F("GenGatherElementOp, src Shape is %s ", IntVecToStr(src0Shape).c_str());

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);

    AppendLocalBufferVarOffset({&dVar, &s0Var, &s1Var}, {0, 1, 2});

    // [case1] src0: [S2,D], src1: [B,S], axis: 0, dst: [B,S,D]
    std::vector<int> dos = NormalizeShape(originShape[0], SHAPE_DIM2);
    std::vector<int> ss = NormalizeShape(src0Shape, SHAPE_DIM2);
    std::vector<int> ds = NormalizeShape(dstShape, SHAPE_DIM2);
    std::string dataTypeExpr[3] = {dstDtypeStr, src0DtypeStr, src1DtypeStr};
    int gatherAxis{-1};
    auto axis = opAttrs.at(OP_ATTR_PREFIX + "axis");
    if (axis.HasValue()) {
        gatherAxis = npu::tile_fwk::AnyCast<int>(axis);
    }
    if (isSupportDynamicUnaligned) {
        return PrintGatherElementDynamicUnaligned({gatherAxis, dVar, s0Var, s1Var, dos, ds, ss, dataTypeExpr});
    }
    return PrintGatherElementStatic({gatherAxis, dVar, s0Var, s1Var, dos, ds, ss, dataTypeExpr});
}

std::string CodeGenOpCloudNPU::GenScatterElementOp() const {
    const DataType dstDtype = operandDtype[ID0];
    const DataType src1Dtype = operandDtype[ID2];
    int dst = operandWithMagic[ID0];
    int src0 = operandWithMagic[ID1];
    int src1 = operandWithMagic[ID2];
    const Element &scala = extOperandVal;

    auto kSrc0 = CreateAllocKey(src0);
    auto kSrc1 = CreateAllocKey(src1);
    auto kDst = CreateAllocKey(dst);

    std::string src0Var = sm->QueryVariableName(kSrc0);
    std::string src1Var = sm->QueryVariableName(kSrc1);
    std::string dstVar = sm->QueryVariableName(kDst);

    // shape: dst, src0, src1
    int dstRank = shape[0].size();
    int src1Rank = shape[2].size();

    ALOG_INFO_F("GenScatterElementOp, dst Shape is %s", IntVecToStr(shape[0]).c_str());
    ALOG_INFO_F("GenScatterElementOp, src0 Shape is %s", IntVecToStr(shape[1]).c_str());
    ALOG_INFO_F("GenScatterElementOp, src1 Shape is %s", IntVecToStr(shape[2]).c_str());

    ASSERT(src1Rank == RANK2) << "GenScatterElementOp: src1 shape rank is not supported!";
    ASSERT(dstRank == RANK2) << "GenScatterElementOp: dst shape rank is not supported!";

    std::vector dstShape = this->rawShape[0];
    std::vector src1RawShape = this->rawShape[2];
    std::vector src1Shape = this->originShape[2];

    char buffer[384] = "CG_ERROR";
    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string src0DtypeStr = DataType2CCEStr(dstDtype);
    std::string src1DtypeStr = DataType2CCEStr(src1Dtype);
    ALOG_INFO_F("GenScatterElementOp, dstDtypeStr%s", dstDtypeStr.c_str());
    ALOG_INFO_F("GenScatterElementOp, src1DtypeStr%s", src1DtypeStr.c_str());

    AppendLocalBufferVarOffset({&dstVar, &src0Var, &src1Var}, {0, 1, 2});

    std::vector<int> s1rs = NormalizeShape(src1RawShape, SHAPE_DIM2);
    std::vector<int> drs = NormalizeShape(dstShape, SHAPE_DIM2);
    std::vector<int> s1os = NormalizeShape(src1Shape, SHAPE_DIM2);

    char scalarTmpBuffer[256] = "CG_ERROR";
    int ret =
        snprintf_s(scalarTmpBuffer, sizeof(scalarTmpBuffer), sizeof(scalarTmpBuffer) - 1, "%.9g", scala.Cast<float>());
    if (ret < 0) {
        ALOG_INFO_F("GenScatterElementOp snprintf_s scalarTmpBuffer failed %d", ret);
    }
    ret = snprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1,
        "%s<%s, %s, %u, %u, %u, %u %s>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s, (__ubuf__ %s *)%s, (%s)%s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), src1DtypeStr.c_str(), s1rs[1], drs[1], s1os[0], s1os[1],
        GenOpAttr().c_str(), dstDtypeStr.c_str(), dstVar.c_str(), src0DtypeStr.c_str(), src0Var.c_str(),
        src1DtypeStr.c_str(), src1Var.c_str(), dstDtypeStr.c_str(), scalarTmpBuffer);
    if (ret < 0) {
        ALOG_INFO_F("GenScatterElementOp snprintf_s buffer failed %d", ret);
    }
    std::string ostring(buffer);
    return ostring;
}

std::string CodeGenOpCloudNPU::PrintBitSortDynamicUnaligned(const PrintSortParam &param) const{
    unsigned dstShape0 = param.dstShape0;
    unsigned dstShape1 = param.dstShape1;
    unsigned src0Shape0 = param.srcShape0;
    unsigned src0Shape1 = param.srcShape1;
    const std::string &s0Var = param.s0Var;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::string orisrcShape0;
    std::string orisrcShape1;
    if (dynamicValidShape[1].size() == 1) {
        orisrcShape0 = "1";
        orisrcShape1 = dynamicValidShape[1][0].Dump();
    } else {
        orisrcShape0 = dynamicValidShape[1][0].Dump();
        orisrcShape1 = dynamicValidShape[1][1].Dump();
    }
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    paramList.insert(paramList.end(), {std::to_string(dstShape0), std::to_string(dstShape1)});
    paramList.insert(paramList.end(), {std::to_string(src0Shape0), std::to_string(src0Shape1)});

    std::string templateParam = JoinString(paramList, ", ");
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dstParam, srcParam});
    paramList.insert(paramList.end(), {orisrcShape0, orisrcShape1});
    std::string tileCallParam = JoinString(paramList, ", ");
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">" << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintBitSortStatic(const PrintSortParam &param) const {
    unsigned dstShape0 = param.dstShape0;
    unsigned dstShape1 = param.dstShape1;
    unsigned src0Shape0 = param.srcShape0;
    unsigned src0Shape1 = param.srcShape1;
    unsigned orisrcShape0 = 0;
    unsigned orisrcShape1 = 0;
    const std::string &s0Var = param.s0Var;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::vector<int> &oriSrc0Shape = originShape[1];
    if (oriSrc0Shape.size() == 1) {
        orisrcShape0 = 1;
        orisrcShape1 = oriSrc0Shape[0];
    } else {
        orisrcShape0 = oriSrc0Shape[0];
        orisrcShape1 = oriSrc0Shape[1];
    }
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    paramList.insert(paramList.end(), {std::to_string(dstShape0), std::to_string(dstShape1)});
    paramList.insert(paramList.end(), {std::to_string(src0Shape0), std::to_string(src0Shape1)});
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

std::string CodeGenOpCloudNPU::GenBitSortOp() const {
    const DataType dstDtype = operandDtype[ID0];
    const DataType src0Dtype = operandDtype[ID1];
    int dst = operandWithMagic[ID0];
    int src0 = operandWithMagic[ID1];

    auto kSrc0 = CreateAllocKey(src0);
    auto kDst = CreateAllocKey(dst);
    std::string src0Var = sm->QueryVariableName(kSrc0);
    std::string dstVar = sm->QueryVariableName(kDst);

    std::vector dstShape = this->rawShape[0];
    std::vector src0Shape = this->rawShape[1];
    auto shapeSize = src0Shape.size();
    unsigned dstShape0 = 0;
    unsigned dstShape1 = 0;
    unsigned src0Shape0 = 0;
    unsigned src0Shape1 = 0;
    if (shapeSize == 1) {
        dstShape0 = 1;
        dstShape1 = std::min(dstShape[0], shape[0][0]);
        src0Shape0 = 1;
        src0Shape1 = std::min(src0Shape[0], shape[1][0]);
    } else {
        dstShape0 = std::min(dstShape[0], shape[0][0]);
        dstShape1 = dstShape[1];
        src0Shape0 = std::min(src0Shape[0], shape[1][0]);
        src0Shape1 = std::min(src0Shape[1], shape[1][1]);
    }
    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string src0DtypeStr = DataType2CCEStr(src0Dtype);
    AppendLocalBufferVarOffset({&dstVar, &src0Var}, {0, 1});

    if (isSupportDynamicUnaligned) {
        return PrintBitSortDynamicUnaligned(
            {dstShape0, dstShape1, src0Shape0, src0Shape1, src0Var, dstVar, src0DtypeStr, dstDtypeStr});
    }
    return PrintBitSortStatic(
        {dstShape0, dstShape1, src0Shape0, src0Shape1, src0Var, dstVar, src0DtypeStr, dstDtypeStr});
}

std::string CodeGenOpCloudNPU::PrintMrgSortDynamicUnaligned(const PrintSortParam &param) const {
    unsigned dstShape0 = param.dstShape0;
    unsigned dstShape1 = param.dstShape1;
    unsigned src0Shape0 = param.srcShape0;
    unsigned src0Shape1 = param.srcShape1;
    const std::string &s0Var = param.s0Var;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::string orisrcShape0;
    std::string orisrcShape1;
    if (dynamicValidShape[1].size() == 1) {
        orisrcShape0 = "1";
        orisrcShape1 = dynamicValidShape[1][0].Dump();
    } else {
        orisrcShape0 = dynamicValidShape[1][0].Dump();
        orisrcShape1 = dynamicValidShape[1][1].Dump();
    }
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    paramList.insert(paramList.end(), {std::to_string(dstShape0), std::to_string(dstShape1)});
    paramList.insert(paramList.end(), {std::to_string(src0Shape0), std::to_string(src0Shape1)});

    std::string templateParam = JoinString(paramList, ", ");
    templateParam += GenOpAttr();
    paramList.clear();
    std::string dstParam = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string srcParam = "(__ubuf__ " + srcDtypeStr + "*)" + s0Var;
    paramList.insert(paramList.end(), {dstParam, srcParam});
    paramList.insert(paramList.end(), {orisrcShape0, orisrcShape1});
    std::string tileCallParam = JoinString(paramList, ", ");
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">" << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintMrgSortStatic(const PrintSortParam &param) const {
    unsigned dstShape0 = param.dstShape0;
    unsigned dstShape1 = param.dstShape1;
    unsigned src0Shape0 = param.srcShape0;
    unsigned src0Shape1 = param.srcShape1;
    unsigned orisrcShape0 = 0;
    unsigned orisrcShape1 = 0;
    const std::string &s0Var = param.s0Var;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::vector<int> &oriSrc0Shape = originShape[1];
    if (oriSrc0Shape.size() == 1) {
        orisrcShape0 = 1;
        orisrcShape1 = oriSrc0Shape[0];
    } else {
        orisrcShape0 = oriSrc0Shape[0];
        orisrcShape1 = oriSrc0Shape[1];
    }
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    paramList.insert(paramList.end(), {std::to_string(dstShape0), std::to_string(dstShape1)});
    paramList.insert(paramList.end(), {std::to_string(src0Shape0), std::to_string(src0Shape1)});
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

std::string CodeGenOpCloudNPU::GenMrgSortOp() const {
    const DataType dstDtype = operandDtype[ID0];
    const DataType src0Dtype = operandDtype[ID1];
    int dst = operandWithMagic[ID0];
    int src0 = operandWithMagic[ID1];

    auto kSrc0 = CreateAllocKey(src0);
    auto kDst = CreateAllocKey(dst);
    std::string src0Var = sm->QueryVariableName(kSrc0);
    std::string dstVar = sm->QueryVariableName(kDst);

    std::vector dstShape = this->rawShape[0];
    std::vector src0Shape = this->rawShape[1];
    auto shapeSize = src0Shape.size();

    unsigned dstShape0 = 0;
    unsigned dstShape1 = 0;
    unsigned src0Shape0 = 0;
    unsigned src0Shape1 = 0;
    if (shapeSize == 1) {
        dstShape0 = 1;
        dstShape1 = std::min(dstShape[0], shape[0][0]);
        src0Shape0 = 1;
        src0Shape1 = std::min(src0Shape[0], shape[1][0]);
    } else {
        dstShape0 = std::min(dstShape[0], shape[0][0]);
        dstShape1 = dstShape[1];
        src0Shape0 = std::min(src0Shape[0], shape[1][0]);
        src0Shape1 = std::min(src0Shape[1], shape[1][1]);
    }
    std::string dstDtypeStr = DataType2CCEStr(dstDtype);
    std::string src0DtypeStr = DataType2CCEStr(src0Dtype);
    AppendLocalBufferVarOffset({&dstVar, &src0Var}, {0, 1});
    if (isSupportDynamicUnaligned) {
        return PrintMrgSortDynamicUnaligned(
            {dstShape0, dstShape1, src0Shape0, src0Shape1, src0Var, dstVar, src0DtypeStr, dstDtypeStr});
    }
    return PrintMrgSortStatic(
        {dstShape0, dstShape1, src0Shape0, src0Shape1, src0Var, dstVar, src0DtypeStr, dstDtypeStr});
}

std::string CodeGenOpCloudNPU::GenExtractOp() const {
    SymbolManager::AllocRecord src0, dst;
    auto kS0 = CreateAllocKey(operandWithMagic[ID1]);
    auto kDst = CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string dVar = sm->QueryVariableName(kDst);
    std::vector src0RawShape = this->rawShape[1];
    unsigned tShape0 = 0;
    unsigned tShape1 = 0;

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::vector src0Shape = this->rawShape[0];
    AppendLocalBufferVarOffset({&dVar, &s0Var}, {0, 1});
    if (this->rawShape[1].size() == 1) {
        tShape1 = std::min(src0RawShape[0], shape[0][0]);
        tShape0 = 1;
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
    oss << tileOpName << "<" << templateParam << ">" << "(" << tileCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenVectorScalarOp() const {
    return GenVectorScalarOpByMode(false);
}

std::string CodeGenOpCloudNPU::GenVectorScalarOpScalarMode() const {
    return GenVectorScalarOpByMode(true);
}

std::string CodeGenOpCloudNPU::PrintBinaryScalarStatic(const PrintBinaryScalarParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const size_t dim = param.dim;

    std::vector dstShape = this->rawShape[0];
    std::vector src0Shape = this->rawShape[1];

    std::vector<int> os0 = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int> ss = NormalizeShape(src0Shape, SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(dstShape, SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    int dimScalar{0};
    if (dim == SHAPE_DIM2) {
        dimScalar = SHAPE_DIM2;
    } else if (dim == SHAPE_DIM3 && opCode == Opcode::OP_S_DIVS) {
        dimScalar = SHAPE_DIM3;
    } else {
        ASSERT(false) << "GenVectorScalarOp ERROR! Unexpect situation!!! \n";
    }
    paramList.emplace_back("/*StatictShape*/ " + std::to_string(os0[SHAPE_DIM4 - dimScalar]));
    for (int i = SHAPE_DIM4 - dimScalar + 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os0[i]));
    }
    paramList.emplace_back("/*DstRawShape*/ " + std::to_string(ds[SHAPE_DIM4 - dimScalar]));
    for (int i = SHAPE_DIM4 - dimScalar + 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*Src0RawShape*/ " + std::to_string(ss[SHAPE_DIM4 - dimScalar]));
    for (int i = SHAPE_DIM4 - dimScalar + 1; i < SHAPE_DIM4 - 1; ++i) {
        paramList.emplace_back(std::to_string(ss[i]));
    }
    paramList.emplace_back(std::to_string(ss[3]) + GenOpAttr());
    std::string templateParam = JoinString(paramList, ", ");

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
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryScalarDynamicUnaligned(const PrintBinaryScalarParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const size_t dim = param.dim;

    std::vector dstShape = this->rawShape[0];
    std::vector src0Shape = this->rawShape[1];

    std::vector<int> ss = NormalizeShape(src0Shape, SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(dstShape, SHAPE_DIM4);

    auto dynSrcShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    int dimScalar{0};
    if (dim == SHAPE_DIM2) {
        dimScalar = SHAPE_DIM2;
    } else if (dim == SHAPE_DIM3 && opCode == Opcode::OP_S_DIVS) {
        dimScalar = SHAPE_DIM3;
    } else {
        ASSERT(false) << "GenVectorScalarOp ERROR! Unexpect situation!!! \n";
    }
    paramList.emplace_back("/*DstRawShape*/ " + std::to_string(ds[SHAPE_DIM4 - dimScalar]));
    for (int i = SHAPE_DIM4 - dimScalar + 1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*Src0RawShape*/ " + std::to_string(ss[SHAPE_DIM4 - dimScalar]));
    for (int i = SHAPE_DIM4 - dimScalar + 1; i < SHAPE_DIM4 - 1; ++i) {
        paramList.emplace_back(std::to_string(ss[i]));
    }
    paramList.emplace_back(std::to_string(ss[3]) + GenOpAttr());
    std::string templateParam = JoinString(paramList, ", ");

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
    for (int i = 0; i < dimScalar; i++) {
        paramList.emplace_back(dynSrcShape[i].Dump());
    }
    std::string tiloOpCallParam = JoinString(paramList, ", ");

    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintBinaryScalar(const PrintBinaryScalarParam &param) const {
    if (isSupportDynamicUnaligned) {
        return PrintBinaryScalarDynamicUnaligned(param);
    }
    return PrintBinaryScalarStatic(param);
}

std::string CodeGenOpCloudNPU::GenVectorScalarOpByMode(bool isUseScalar) const {
    auto kS0 = CreateAllocKey(operandWithMagic[ID1]);
    auto kDst = CreateAllocKey(operandWithMagic[ID0]);
    std::string s0Var = sm->QueryVariableName(kS0);
    std::string dVar = sm->QueryVariableName(kDst);
    ASSERT(shape[0] == shape[1]) << " shape between dst " << IntVecToStr(shape[ID0]) << " and src "
                                 << IntVecToStr(shape[ID1]) << " is different";

    char buffer[BUFFER_SIZE_512] = "CG_ERROR";
    char scalarTmpBuffer[BUFFER_SIZE_256] = "CG_ERROR";
    int ret = sprintf_s(scalarTmpBuffer, sizeof(scalarTmpBuffer), "%.9g", extOperandVal.Cast<float>());
    ASSERT(ret >= 0) << "GenVectorScalarOpByMode sprintf_s failed ";
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufferVarOffset({&dVar, &s0Var}, {0, 1});

    std::vector src0RawShape = this->rawShape[1];
    std::vector dstRawShape = this->rawShape[0];
    std::vector<int> os0 = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int> s0 = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    if (isUseScalar) {
        // Scalar op
        return PrintBinaryScalar({s0Var, dVar, dstDtypeStr, dstDtypeStr, shape[0].size()});
    }

    if (isSupportDynamicUnaligned) {
        auto newDynSrcValidShape = dynamicValidShape[1];
        FillIntVecWithDummyInHead<SymbolicScalar>(newDynSrcValidShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);
        ret = sprintf_s(buffer, sizeof(buffer),
            "%s_<%s, /*DS*/ %d, %d, %d, /*S0S*/ %d, %d, %d>((__ubuf__ %s*)%s, (__ubuf__ %s*)%s, (%s)%s, %s, %s, %s, "
            "%s);\n",
            tileOpName.c_str(), dstDtypeStr.c_str(), ds[ID1], ds[ID2], ds[ID3], s0[ID1], s0[ID2], s0[ID3],
            dstDtypeStr.c_str(), dVar.c_str(), dstDtypeStr.c_str(), s0Var.c_str(), dstDtypeStr.c_str(), scalarTmpBuffer,
            newDynSrcValidShape[ID0].Dump().c_str(), newDynSrcValidShape[ID1].Dump().c_str(),
            newDynSrcValidShape[ID2].Dump().c_str(), newDynSrcValidShape[ID3].Dump().c_str());
        ASSERT(ret >= 0) << "GenVectorScalarOpByMode" << OpcodeManager::Inst().GetOpcodeStr(opCode)
                         << " sprintf_s failed " << ret;
        return buffer;
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

    auto kSrc = CreateAllocKey(operandWithMagic[ID1]);
    auto kDst = CreateAllocKey(operandWithMagic[ID0]);
    std::string sVar = sm->QueryVariableName(kSrc);
    std::string dVar = sm->QueryVariableName(kDst);

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    AppendLocalBufferVarOffset({&dVar, &sVar}, {0, 1});

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

    int ret = sprintf_s(buffer, sizeof(buffer), "%s<%s, %s>((__ubuf__ %s *)%s, (__ubuf__ %s *)%s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), templateParamStr.c_str(), dstDtypeStr.c_str(), dVar.c_str(),
        srcDtypeStr.c_str(), sVar.c_str());
    ASSERT(ret >= 0) << "GenPoolOpS sprintf_s failed ";
    return buffer;
}
} // namespace npu::tile_fwk
