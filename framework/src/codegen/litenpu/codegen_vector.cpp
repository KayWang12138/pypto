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

std::string CodeGenOpLiteNPU::PrintUnaryWithTmpTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    std::string tmpTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::ostringstream oss;
    oss << tileOpName << "(" << dstTensor << ", " << srcTensor << "," << tmpTensor << ");\n";
    return oss.str();
}

std::string CodeGenOpLiteNPU::PrintVnchwconv() const {
    if (isSupportLayout) {
        return PrintUnaryWithTmpTileTensor();
    }
    return "";
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
        return PrintVnchwconv();
    }

    if (opCode == Opcode::OP_ROWSUM_SINGLE || opCode == Opcode::OP_ROWMAX_SINGLE || opCode == Opcode::OP_ROWMIN_SINGLE) {
        return PrintReduceLastAxis();
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

std::string CodeGenOpLiteNPU::PrintUnaryTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));

    std::ostringstream oss;
    std::vector<std::string> templateParamList;
    std::string lastUse = GetLastUse();
    oss << tileOpName;
    if (!lastUse.empty()) {
        oss << WrapParamByAngleBrackets({lastUse});
    }
    oss << WrapParamByParentheses({dstTensor, srcTensor});
    oss << ";\n";
    return oss.str();
}


std::string CodeGenOpLiteNPU::PrintUnary() const {
    std::cout<<"********** debug message"<<isSupportLayout << std::endl;
    if (isSupportLayout) {
        return PrintUnaryTileTensor();
    }
    return "";
}

std::string CodeGenOpLiteNPU::GenUnaryOp() const {
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::cout<<"*******debug message********"<<s0Var <<"***********"<<dVar<<std::endl;

    std::map<unsigned, std::reference_wrapper<std::string>> varsMap;
    // AppendLocalBufVarOffsetInOrder(dVar, s0Var);
    varsMap.insert(std::make_pair(0, std::ref(dVar)));
    varsMap.insert(std::make_pair(1, std::ref(s0Var)));
    AppendLocalBufferVarOffset(varsMap);

    std::cout<<"*******debug message********"<<s0Var <<"***********"<<dVar<<std::endl;

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    if (opCode == Opcode::OP_COPY_UB_TO_UB) {
        srcDtypeStr = GetTypeForB16B32(operandDtype[ID1]);
        dstDtypeStr = GetTypeForB16B32(operandDtype[ID0]);
    }

    // if (opCode == Opcode::OP_EXPAND) {
    //     return PrintExpand(s0Var, dVar, srcDtypeStr, dstDtypeStr);
    // }   
    if (opCode == Opcode::OP_EXP || opCode == Opcode::OP_SQRT || opCode == Opcode::OP_ABS || opCode == Opcode::OP_RELU ||
               opCode == Opcode::OP_RECIPROCAL || opCode == Opcode::OP_NEG || opCode == Opcode::OP_RSQRT ||
               opCode == Opcode::OP_LN || opCode == Opcode::OP_LOGICALNOT || opCode == Opcode::OP_BRCB ||
               opCode == Opcode::OP_CEIL|| opCode == Opcode::OP_FLOOR|| opCode == Opcode::OP_TRUNC || opCode == Opcode::OP_ISFINITE) {
        return PrintUnary();
    }  
    CODEGEN_LOGI("unsupported tileop: %s", opCodeStr.c_str());
    return "CG_ERROR";
}


std::string CodeGenOpLiteNPU::PrintCastTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    auto mode = opAttrs.at(OP_ATTR_PREFIX + "mode");
    int64_t modeEnum{0};
    if (mode.HasValue()) {
        modeEnum = AnyCast<int64_t>(mode);
    }
    std::ostringstream oss;
    std::vector<std::string> templateParamList;
    std::string lastUse = GetLastUse();
    oss << tileOpName;
    if (!lastUse.empty()) {
        templateParamList.emplace_back(lastUse);
    }
    templateParamList.emplace_back(std::to_string(modeEnum));
    oss << WrapParamByAngleBrackets(templateParamList);
    oss << WrapParamByParentheses({dstTensor, srcTensor});
    oss << ";\n";
    return oss.str();
}

std::string CodeGenOpLiteNPU::GenCastOp() const {
    if (isSupportLayout) {
        return PrintCastTileTensor();
    }
    
    return "";
}


std::string CodeGenOpLiteNPU::PrintDupTileTensor(const PrintDupOpParam &param) const {
    const std::string &dupV = param.dupV;
    const std::string &dstDtypeStr = param.dstDtypeStr;
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByAngleBrackets({dstDtypeStr});
    oss << WrapParamByParentheses({dstTensor, dupV});
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpLiteNPU::PrintDupOp(const PrintDupOpParam &param) const {
    if (isSupportLayout) {
        return PrintDupTileTensor(param);
    }

    return "";
}

std::string CodeGenOpLiteNPU::GenDupOp() const {
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    std::string dupV;
    if (opAttrs.count(OpAttributeKey::dynScalar)) {
        auto scalar = opAttrs.at(OpAttributeKey::dynScalar);
        ASSERT((scalar.HasValue()) && (scalar.Type() == typeid(SymbolicScalar)))
            << AnyCast<SymbolicScalar>(scalar).IsValid() << "SCALAR attribute has to have symbolic value.";
        auto scalarExpr = AnyCast<SymbolicScalar>(scalar);
        dupV = SymbolicExpressionTable::BuildExpression(scalarExpr);
    } else if (dstDtypeStr == "float" || dstDtypeStr == "half" || dstDtypeStr == "bfloat16_t") {
        auto scalar = opAttrs.at(OpAttributeKey::scalar);
        ASSERT((scalar.HasValue()) && (scalar.Type() == typeid(Element)))
            << AnyCast<Element>(scalar).IsFloat() << "SCALAR attribute has to have float value.";
        dupV = FormatFloat(AnyCast<Element>(scalar).Cast<float>());
    } else if (dstDtypeStr == "int32_t") {
        auto scalar = opAttrs.at(OpAttributeKey::scalar);
        ASSERT((scalar.HasValue()) && (scalar.Type() == typeid(Element)))
            << AnyCast<Element>(scalar).IsSigned() << "SCALAR attribute has to have int value.";
        dupV = std::to_string(AnyCast<Element>(scalar).Cast<int>());
    } else {
        ASSERT(false) << "unsupported type";
    }
    return PrintDupOp({dVar, dstDtypeStr, dupV});
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

std::string CodeGenOpLiteNPU::PrintReduceLastAxis() const {
    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    if (isSupportLayout) {
        return PrintReduceLastAxisTileTensor();
    }
    return buffer;
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
    oss << tileOpName; // e.g TPairMin
    if (!templateParamList.empty()) {
        oss << WrapParamByAngleBrackets(templateParamList);
    }
    oss << WrapParamByParentheses(tileOpCallParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpLiteNPU::PrintBinary() const {
    if (isSupportLayout) {
        return PrintBinaryTileTensor();
    }
    return "";
}

std::string CodeGenOpLiteNPU::GenBinaryOp() const {
    return PrintBinary();
}
} // namespace npu::tile_fwk
