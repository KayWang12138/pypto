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
#include "codegen_op_litenpu.h"
#include "securec.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
std::string GetBrcOprandIdxStrLite(int64_t brcbOperandIdx) {
    CODEGEN_LOGI("input brcbOperandIdx is %ld", static_cast<long>(brcbOperandIdx));
    std::string ret = "TileOp::";
    switch (brcbOperandIdx) {
        case ToUnderlying(BroadcastOperand::NONE): ret.append("BroadcastOperand::NONE"); break;
        case ToUnderlying(BroadcastOperand::LEFT_OPERAND): ret.append("BroadcastOperand::LEFT_OPERAND"); break;
        case ToUnderlying(BroadcastOperand::RIGHT_OPERAND): ret.append("BroadcastOperand::RIGHT_OPERAND"); break;
        default: ret.append("BroadcastOperand::NONE");
    }
    return ret;
}

std::string CodeGenOpLiteNPU::PrintIndexPutLayout(size_t indicesSize, bool accumulate) const {
    std::string gmVarName = GenGmParamVar(ID0);
    std::string dstTensor = sm->QueryTileTensorByBufVarName(gmVarName);
    std::string valuesTensor = QueryTileTensorNameByIdx(ID2);
    std::vector<std::string> paramList = {dstTensor, valuesTensor};
    for (size_t i = 0; i < SHAPE_DIM4; ++i) {
        if (i < indicesSize) {
            std::string indices = QueryTileTensorNameByIdx(ID3 + i);
            paramList.push_back(indices);
        } else {
            paramList.push_back(paramList.back());
        }
    }
    std::ostringstream oss;
    oss << tileOpName << "<" << accumulate << ", " << indicesSize << ">" << WrapParamByParentheses(paramList)
        << STMT_END;
    return oss.str();
}

std::string CodeGenOpLiteNPU::GenIndexPutOp() const {
    ASSERT(opAttrs.count(OpAttributeKey::accumulate)) << "cannot get accumulate attr";
    ASSERT(opAttrs.count(OpAttributeKey::indicesSize)) << "cannot get indicesSize attr";
    bool accumulate = AnyCast<bool>(opAttrs.at(OpAttributeKey::accumulate));
    int64_t indicesSize = AnyCast<int64_t>(opAttrs.at(OpAttributeKey::indicesSize));
    if (isSupportLayout) {
        return PrintIndexPutLayout(indicesSize, accumulate);
    }

    return "";
}

std::string CodeGenOpLiteNPU::PrintVnchwconvStatic(const PrintUnaryTmpBuffParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::vector<int64_t> os0 = NormalizeShape(originShape[2], SHAPE_DIM5);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[2], SHAPE_DIM5);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM5);
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

std::string CodeGenOpLiteNPU::PrintVnchwconvDynUnaligned(const PrintUnaryTmpBuffParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::vector<int64_t> s0 = NormalizeShape(rawShape[2], SHAPE_DIM5);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM5);
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

std::string CodeGenOpLiteNPU::PrintUnaryWithTmpTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    std::string tmpTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::ostringstream oss;
    oss << tileOpName << "(" << dstTensor << ", " << srcTensor << "," << tmpTensor << ");\n";
    return oss.str();
}

std::string CodeGenOpLiteNPU::PrintVnchwconv(const PrintUnaryTmpBuffParam &param) const {
    if (isSupportLayout) {
        return PrintUnaryWithTmpTileTensor();
    }
    if (!isSupportDynamicAligned) {
        return PrintVnchwconvDynUnaligned(param);
    }
    return PrintVnchwconvStatic(param);
}

std::string CodeGenOpLiteNPU::PrintCompactStatic(const PrintUnaryTmpBuffParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[2], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
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

std::string CodeGenOpLiteNPU::PrintCompact(const PrintUnaryTmpBuffParam &param) const {
    return PrintCompactStatic(param);
}

std::string CodeGenOpLiteNPU::GenUnaryOpWithTmpBuff() const {
    // In this scenario, frontend set tmp buffer in output to optimize ooo schedule result.
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector srcShape = this->rawShape[2];
    ALOG_INFO_F("GenUnaryOpWithTmpBuff %s src raw shape: %s", tileOpName.c_str(), IntVecToStr(srcShape).c_str());

    std::vector dstShape = this->rawShape[0];
    ALOG_INFO_F("GenUnaryOpWithTmpBuff %s dst raw shape: %s", tileOpName.c_str(), IntVecToStr(dstShape).c_str());

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string tmpDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    // AppendLocalBufferVarOffset({&dVar, &tmpVar, &s0Var}, {0, 1, 2});
    std::map<unsigned, std::reference_wrapper<std::string>> vars;
    vars.insert({static_cast<unsigned>(0), std::ref(dVar)});
    vars.insert({static_cast<unsigned>(1), std::ref(tmpVar)});
    vars.insert({static_cast<unsigned>(2), std::ref(s0Var)});
    AppendLocalBufferVarOffset(vars);

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int ret = 0;
    if (opCode == Opcode::OP_TRANSPOSE_VNCHWCONV) {
        return PrintVnchwconv({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    if (opCode == Opcode::OP_ROWSUM_SINGLE || opCode == Opcode::OP_ROWMAX_SINGLE || opCode == Opcode::OP_ROWMIN_SINGLE) {
        return PrintReduceLastAxis({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    if (opCode == Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE || opCode == Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE) {
        std::vector<int64_t> dstOriginShape = NormalizeShape(originShape[0], SHAPE_DIM4);
        std::vector<int64_t> srcOriginShape = NormalizeShape(originShape[2], SHAPE_DIM4);
        std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[2], SHAPE_DIM4);
        std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
        std::vector<int64_t> tmpRawShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
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

    if (opCode == Opcode::OP_COMPACT) {
        return PrintCompact({s0Var, tmpVar, dVar, srcDtypeStr, tmpDtypeStr, dstDtypeStr});
    }

    std::string ostring(buffer);
    return ostring;
}

std::string CodeGenOpLiteNPU::PrintReduceLastAxisTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string tmpTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    std::ostringstream oss;
    std::vector<std::string> templateParamList;
    std::string lastUse = GetLastUse();
    oss << tileOpName;
    if (!lastUse.empty()) {
        oss << WrapParamByAngleBrackets({lastUse});
    }
    oss << WrapParamByParentheses({dstTensor, src0Tensor, tmpTensor});
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpLiteNPU::PrintReduceLastAxis(const PrintUnaryTmpBuffParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &tmpVar = param.tmpVar;
    const std::string &dVar = param.dVar;
    const std::string &srcDtypeStr = param.srcDtypeStr;
    const std::string &tmpDtypeStr = param.tmpDtypeStr;
    const std::string &dstDtypeStr = param.dstDtypeStr;

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int ret = 0;

    std::vector<int64_t> dstOriginShape = NormalizeShape(originShape[0], SHAPE_DIM4);
    std::vector<int64_t> srcOriginShape = NormalizeShape(originShape[2], SHAPE_DIM4);
    std::vector<int64_t> srcRawShape = NormalizeShape(rawShape[2], SHAPE_DIM4);
    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[0], SHAPE_DIM4);
    std::vector<int64_t> tmpRawShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    ALOG_INFO_F("rawShape[2] is %s", IntVecToStr(rawShape[2]).c_str());
    ASSERT(dstOriginShape[ID3] == 1) << "Dst last axis length must be 1";
    if (isSupportLayout) {
        return PrintReduceLastAxisTileTensor();
    }
    if (!isSupportDynamicAligned) {
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

std::string CodeGenOpLiteNPU::PrintBinaryStatic(const PrintBinaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;

    std::vector<int64_t> os0 = NormalizeShape(originShape[1], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[2], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

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

std::string CodeGenOpLiteNPU::PrintBinaryDynamicUnaligned(const PrintBinaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;

    std::vector<int64_t> s0 = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[2], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    auto dynSrcShape = dynamicValidShape[1];
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape, SHAPE_DIM4 - dynamicValidShape[1].size(), 1);

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
    for (auto dynShape : dynSrcShape) {
        paramList.emplace_back(dynShape.Dump());
    }
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "_<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpLiteNPU::PrintBinaryTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    std::vector<std::string> tileOpCallParamList = {dstTensor, src0Tensor, src1Tensor};

    std::vector<std::string> templateParamList;
    int64_t brcOperandIdx = 0;
    std::string lastUse = GetLastUse();
    if (!lastUse.empty()) {
        templateParamList.emplace_back(lastUse);
    }
    if (GetAttr(OpAttributeKey::brcbIdx, brcOperandIdx)) {
        templateParamList.emplace_back(GetBrcOprandIdxStrLite(brcOperandIdx));
    }

    std::ostringstream oss;
    oss << tileOpName;
    if (!templateParamList.empty()) {
        oss << WrapParamByAngleBrackets(templateParamList);
    }
    oss << WrapParamByParentheses(tileOpCallParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpLiteNPU::PrintBinary(const PrintBinaryParam &param) const {
    if (isSupportLayout) {
        return PrintBinaryTileTensor();
    }
    if (!isSupportDynamicAligned) {
        return PrintBinaryDynamicUnaligned(param);
    }
    return PrintBinaryStatic(param);
}

std::string CodeGenOpLiteNPU::GenBinaryOp() const {
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector src0RawShape = this->rawShape[ID1];
    CODEGEN_LOGI("genBinaryOp %s, src0RawShape is %s", tileOpName.c_str(), IntVecToStr(src0RawShape).c_str());

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);

    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);

    auto offset0 = GetOperandStartOffsetLite(ID0);
    auto offset1 = GetOperandStartOffsetLite(ID1);
    auto offset2 = GetOperandStartOffsetLite(ID2);
    if (!offset0.ConcreteValid() || offset0.Concrete() != 0) {
        dVar += "+" + GetOperandStartOffsetLite(ID0).Dump();
    }
    if (!offset1.ConcreteValid() || offset1.Concrete() != 0) {
        s0Var += "+" + GetOperandStartOffsetLite(ID1).Dump();
    }
    if (!offset2.ConcreteValid() || offset2.Concrete() != 0) {
        s1Var += "+" + GetOperandStartOffsetLite(ID2).Dump();
    }
    return PrintBinary({s0Var, s1Var, dVar, src0DtypeStr, src1DtypeStr, dstDtypeStr});
}
} // namespace npu::tile_fwk
