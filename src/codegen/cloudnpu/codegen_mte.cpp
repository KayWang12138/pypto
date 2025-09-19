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
 * \file codegen_mte.cpp
 * \brief
 */

#include "codegen_op_cloudnpu.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/utils/codegen_utils.h"
#include "securec.h"

namespace npu::tile_fwk {
template <typename T>
bool CodeGenOpCloudNPU::GetAttr(const std::string &key, T &value) const {
    static_assert(!std::is_same_v<T, int>);
    static_assert(!std::is_same_v<T, std::vector<int>>);
    auto it = opAttrs.find(key);
    if (it == opAttrs.end()) {
        return false;
    }
    if (it->second.Type() == typeid(T)) {
        value = npu::tile_fwk::AnyCast<T>(it->second);
        return true;
    }
    return false;
}

CodeGenOpCloudNPU::DynamicParamPackMTE CodeGenOpCloudNPU::PrepareDynamicShapeInfoForMTE(
    int dynShapeIdx, int ShapeDim, bool isNeedGmOffset) const {
    CodeGenOpCloudNPU::DynamicParamPackMTE pack;
    int dim = static_cast<int>(paramIdxForDynShape[dynShapeIdx].size());
    pack.gmShapeExpr = GenGetParamMacroPacked(dynShapeIdx, dim, PREFIX_STR_RAW_SHAPE);
    FillIntVecWithDummyInHead<std::string>(pack.gmShapeExpr, ShapeDim - dim, "1");
    ALOG_INFO_F("dynamic gmShape param: %s", IntVecToStr(pack.gmShapeExpr).c_str());

    if (offsetGmSymbolic[dynShapeIdx][ID0].IsValid()) {
        pack.gmOffsetExpr = GenSymbolicArgument(offsetGmSymbolic[dynShapeIdx]);
    } else {
        pack.gmOffsetExpr = GenGetParamMacroPacked(dynShapeIdx, dim, PREFIX_STR_OFFSET);
    }
    FillIntVecWithDummyInHead<std::string>(pack.gmOffsetExpr, ShapeDim - dim, "0");
    ALOG_INFO_F("dynamic gmOffset param: %s", IntVecToStr(pack.gmOffsetExpr).c_str());

    for (const auto &gs : pack.gmShapeExpr) {
        pack.paramList.emplace_back(gs);
    }
    if (isNeedGmOffset) {
        for (const auto &go : pack.gmOffsetExpr) {
            pack.paramList.emplace_back(go);
        }
    }

    return pack;
}

std::string CodeGenOpCloudNPU::GenMemL1CopyIn() const {
    struct OpInfo opInfo(tileOpName, {operandWithMagic[ID0]}, {operandDtype[ID1], operandDtype[ID0]});
    opInfo.bufferId = operand[ID1];
    return GenMemCopyCube(opInfo, false, false, 0);
}

std::string CodeGenOpCloudNPU::GenMemL1CopyOut() const {
    struct OpInfo opInfo(tileOpName, {operandWithMagic[ID1]}, {operandDtype[ID0], operandDtype[ID1]});
    opInfo.bufferId = operand[ID0];
    return GenMemCopyCube(opInfo, false, true, 0);
}

std::string CodeGenOpCloudNPU::GenMemL0CCopyOut() const {
    struct OpInfo opInfo(tileOpName, {operandWithMagic[ID1]}, {operandDtype[ID0], operandDtype[ID1]});
    opInfo.bufferId = operand[ID0];
    return GenMemCopyCube(opInfo, true, false, 0);
}

std::string CodeGenOpCloudNPU::GenMemCopyCube(
    const struct OpInfo &opInfo, bool isCopyL0CToGM, bool isCopyL1ToGM, unsigned uf) const {
    ASSERT(!(isCopyL0CToGM && isCopyL1ToGM)) << "isCopyL0CToGM and isCopyL1ToGM can only set one to true in onetime.";

    if (opInfo.bufferId != SYMBOL_STACK_BASE) {
        // Query rawtensor from tilemanager
        if (isCopyL0CToGM) {
            return GenMemCopyVar(isCopyL0CToGM, OperandType::BUF_L0C, uf);
        }
        return GenMemCopyVar(isCopyL1ToGM, OperandType::BUF_L1, uf);
    }

    return GenMemL1SpillIntoGM(opInfo, isCopyL0CToGM, isCopyL1ToGM, uf);
}

std::string CodeGenOpCloudNPU::GenMemL1SpillIntoGM(
    const OpInfo &opInfo, bool isCopyL0CToGM, bool isCopyL1ToGM, unsigned int uf) const {
    unsigned gmIdx = (isCopyL0CToGM || isCopyL1ToGM) ? 0 : 1;
    unsigned l1Idx = (isCopyL0CToGM || isCopyL1ToGM) ? 1 : 0;
    DataType gmDtype = (isCopyL0CToGM || isCopyL1ToGM) ? opInfo.dataTypes[ID0] : opInfo.dataTypes[ID1];
    DataType l1Dtype = (isCopyL0CToGM || isCopyL1ToGM) ? opInfo.dataTypes[ID1] : opInfo.dataTypes[ID0];

    std::string addrTypeHead[ID2];
    addrTypeHead[gmIdx] = GetAddrTypeByOperandType(BUF_DDR);
    addrTypeHead[l1Idx] = isCopyL0CToGM ? GetAddrTypeByOperandType(BUF_L0C) : GetAddrTypeByOperandType(BUF_L1);

    // Query ub variable name
    auto l1AllocKey = sm->CreateAllocKey(opInfo.operands[ID0]);
    std::vector<int64_t> gmOffset = offset[gmIdx];
    std::vector<int64_t> l1TileOffset = offset[l1Idx];
    unsigned l1Offset = l1TileOffset[ID0] * l1TileOffset[ID1];
    std::string addrExpr[ID2];
    addrExpr[gmIdx] = GenGMAddrExprWithOffset(GM_STACK_BASE, gmIdx);
    addrExpr[l1Idx] = GenAddrExpr(sm->QueryVariableName(l1AllocKey), l1Offset);

    std::vector<int64_t> gmShape = rawShape[gmIdx];
    ALOG_INFO_F("GenMemOpL1 op: %s, gmShape: %s", opInfo.op.c_str(), IntVecToStr(gmShape).c_str());

    std::vector<int64_t> l1Shape = rawShape[l1Idx];
    ALOG_INFO_F("GenMemOpL1 op: %s, l1Shape: %s", opInfo.op.c_str(), IntVecToStr(l1Shape).c_str());

    // Spilling out scene only support 2-dim shape
    ASSERT(l1Shape.size() == SHAPE_DIM2) << "!";
    int tileShape0 = l1Shape[ID0];
    int tileShape1 = l1Shape[ID1];

    std::string typeExpr[ID2];
    typeExpr[gmIdx] = DataType2CCEStr(gmDtype);
    typeExpr[l1Idx] = DataType2CCEStr(l1Dtype);

    int ret{0};
    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    if (isSupportDynamicUnaligned) {
        // The layout of gm stack data is continuous, so just use dynValidShape[ID1] as gm stride to match the
        // implementation of "Copy TileOp" in order to make gm gap value be zero in copy intrinsic.
        auto dynValidShape = dynamicValidShape[l1Idx];
        std::ostringstream oss;
        oss << tileOpName << "<" << typeExpr[gmIdx] << ", " << typeExpr[l1Idx] << ">"
            << "((" << addrTypeHead[ID0] << " " << typeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
            << "(" << addrTypeHead[ID1] << " " << typeExpr[ID1] << "*)" << addrExpr[ID1] << ", "
            << dynValidShape[ID0].Dump() << ", " << dynValidShape[ID1].Dump() << ", " << dynValidShape[ID0].Dump()
            << ", " << dynValidShape[ID1].Dump() << ", " << uf << ");\n";

        return oss.str();
    } else {
        ret = sprintf_s(buffer, BUFFER_SIZE_1024, "%s<%s, %s, %d, %d, %d, %d>((%s %s*)%s, (%s %s*)%s, %u);\n",
            tileOpName.c_str(), typeExpr[gmIdx].c_str(), typeExpr[l1Idx].c_str(), tileShape0, tileShape1, gmShape[ID0],
            gmShape[ID1], addrTypeHead[ID0].c_str(), typeExpr[ID0].c_str(), addrExpr[ID0].c_str(),
            addrTypeHead[ID1].c_str(), typeExpr[ID1].c_str(), addrExpr[ID1].c_str(), uf);
    }
    ASSERT(ret >= 0) << "sprintf_s failed in genMemOp_L1, return value:" << ret;
    return buffer;
}

std::string CodeGenOpCloudNPU::GenMemL1ToL0() const {
    std::string paramStr = GenParamsStr();

    std::vector<int64_t> l1Shape = this->rawShape[ID1];
    ALOG_INFO_F("GenMemL1ToL0 %s, l1Shape is %s", tileOpName.c_str(), IntVecToStr(l1Shape).c_str());

    std::vector<int64_t> l0Shape = this->rawShape[ID0];
    ALOG_INFO_F("GenMemL1ToL0 %s, l0Shape is %s", tileOpName.c_str(), IntVecToStr(l0Shape).c_str());

    std::vector<int64_t> l1Offset = this->offset[ID1];

    unsigned srcOffset0 = l1Offset[ID0];
    unsigned srcOffset1 = l1Offset[ID1];
    unsigned srcShape0 = l1Shape[ID0];
    unsigned srcShape1 = l1Shape[ID1];
    unsigned tileShape0 = l0Shape[ID0];
    unsigned tileShape1 = l0Shape[ID1];

    std::string dtypeStr = DataType2CCEStr(operandDtype[ID0]);
    auto l1ShapeDyn = dynamicValidShape[ID1];
    auto l0ShapeDyn = dynamicValidShape[ID0];

    std::ostringstream oss;
    if (isSupportDynamicUnaligned) {
        oss << tileOpName << "<" << dtypeStr << ", " << srcOffset0 << ", " << srcOffset1 << ">"
            << "(" << paramStr << ", " << l0ShapeDyn[ID0].Dump() << ", " << l0ShapeDyn[ID1].Dump() << ", "
            << l1ShapeDyn[ID0].Dump() << ", " << l1ShapeDyn[ID1].Dump() << ");\n";
    } else {
        oss << tileOpName << "<" << dtypeStr << ", " << tileShape0 << ", " << tileShape1 << ", " << srcOffset0 << ", "
            << srcOffset1 << ", " << srcShape0 << ", " << srcShape1 << ">"
            << "(" << paramStr << ");\n";
    }

    return oss.str();
}

std::string CodeGenOpCloudNPU::GenMemUBSpillIntoGM(bool isCopyUBToGM) const {
    unsigned gmIdx = isCopyUBToGM ? 0 : 1;
    unsigned ubIdx = isCopyUBToGM ? 1 : 0;
    std::vector<int64_t> gmShape = this->rawShape[gmIdx];
    std::vector<int64_t> gmOffset = this->offset[gmIdx];

    std::string addrTypeHead[ID2];
    addrTypeHead[gmIdx] = GetAddrTypeByOperandType(BUF_DDR);
    addrTypeHead[ubIdx] = GetAddrTypeByOperandType(BUF_UB);

    // Query ub variable name
    std::string addrExpr[ID2];
    auto ubAllocKey = sm->CreateAllocKey(operandWithMagic[ubIdx]);
    addrExpr[ubIdx] = sm->QueryVariableName(ubAllocKey);
    // In spilling out scene,  GM offset is added after "GMStackBase" var.
    // "GMStackBase" is a base address of a gm workspace which is using for spilled tensors.
    addrExpr[gmIdx] = GenGMAddrExprWithOffset(GM_STACK_BASE, gmIdx);

    std::string dataTypeExpr[ID2];
    dataTypeExpr[gmIdx] = DataType2CCEStr(operandDtype[gmIdx]);
    dataTypeExpr[ubIdx] = DataType2CCEStr(operandDtype[ubIdx]);

    PrintMemCopyWithUBParam param = {gmIdx, ubIdx, addrTypeHead, addrExpr, dataTypeExpr, true};
    return PrintMemCopyWithUB(param);
}

std::string CodeGenOpCloudNPU::GenMemUBTransfer(bool isCopyUBToGM) const {
    unsigned gmIdx = isCopyUBToGM ? 0 : 1;
    if (operand[gmIdx] != SYMBOL_STACK_BASE) {
        return GenMemCopyVar(isCopyUBToGM, OperandType::BUF_UB, 0);
    }

    return GenMemUBSpillIntoGM(isCopyUBToGM);
}

std::string CodeGenOpCloudNPU::GenUBCopyIn() const {
    return GenMemUBTransfer(false);
}

std::string CodeGenOpCloudNPU::GenUBCopyOut() const {
    return GenMemUBTransfer(true);
}

std::string CodeGenOpCloudNPU::GenIndexOutCastOp() const {
    ASSERT(opAttrs.count(OpAttributeKey::cacheMode)) << "cannot get cacheMode attr";
    ASSERT(opAttrs.count(OpAttributeKey::panzBlockSize)) << "cannot get panzBlockSize attr";
    auto cacheMode = npu::tile_fwk::AnyCast<std::string>(opAttrs.at(OpAttributeKey::cacheMode));
    auto blockSize = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at(OpAttributeKey::panzBlockSize));
    unsigned gmIdx = 0;
    unsigned localIdx = 1;
    std::string addrTypeHead[ID2];
    std::string addrExpr[ID2];
    OperandType localType = OperandType::BUF_UB;
    addrTypeHead[gmIdx] = GetAddrTypeByOperandType(BUF_DDR);
    addrTypeHead[localIdx] = GetAddrTypeByOperandType(localType);

    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);

    std::vector gmShape = this->rawShape[gmIdx];
    ALOG_INFO_F("genIndexOutCastOp gm shape: %s", IntVecToStr(gmShape).c_str());

    addrExpr[gmIdx] = GenGmParamVar(gmIdx);

    std::vector<int64_t> src0OriginShape = this->originShape[ID1];
    std::vector<int64_t> src1OriginShape = this->originShape[ID2];
    std::vector<int64_t> src0RawShape = this->rawShape[ID1];
    std::vector<int64_t> src1RawShape = this->rawShape[ID2];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string dataTypeExpr[ID3] = {dstDtypeStr, src0DtypeStr, src1DtypeStr};

    AppendLocalBufferVarOffset({
        {localIdx, &s0Var}
    });

    std::vector<int64_t> s0os = NormalizeShape(src0OriginShape, SHAPE_DIM4);
    std::vector<int64_t> gms = NormalizeShape(gmShape, SHAPE_DIM4);
    std::vector<int64_t> s0rs = NormalizeShape(src0RawShape, SHAPE_DIM4);
    std::vector<int64_t> s1rs = NormalizeShape(src1RawShape, SHAPE_DIM4);
    std::string blockSizeStr = std::to_string(blockSize);

    return PrintIndexOutCast(
        {s0Var, s1Var, addrExpr, gms, s0os, s0rs, src1OriginShape, s1rs, dataTypeExpr, cacheMode, blockSizeStr});
}

std::string CodeGenOpCloudNPU::PrintIndexOutCast(const PrintIndexOutCastParam &param) const {
    if (isSupportDynamicUnaligned) {
        return PrintIndexOutCastDynamicUnaligned(param);
    } else if (functionType == FunctionType::DYNAMIC_LOOP_PATH) {
        return PrintIndexOutCastDynamic(param);
    }
    return PrintIndexOutCastStatic(param);
}

int CodeGenOpCloudNPU::GetCacheModeFlag(const std::string &cacheMode) const {
    const int PA_BNSD = 0;
    const int PA_NZ = 1;
    const int PA_BSND = 2;
    int cacheModeFlag = PA_BNSD;
    if (cacheMode == "PA_NZ") {
        cacheModeFlag = PA_NZ;
    } else if (cacheMode == "PA_BSND") {
        cacheModeFlag = PA_BSND;
    }
    return cacheModeFlag;
}
// template <typename T,unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned src1Shape0,
// unsigned src1Shape1, unsigned GmShape0, unsigned GmShape1, unsigned GmShape2, unsigned GmShape3>
// TILEOP void TIndexoutcast(__gm__ T* dst, __ubuf__ T* src0, __ubuf__ int32_t* index, unsigned Offset0, unsigned
// Offset1) {
std::string CodeGenOpCloudNPU::PrintIndexOutCastStatic(const PrintIndexOutCastParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    const std::string *addrExpr = param.addrExpr;
    const std::vector<int64_t> &gmShape = param.gmShape;
    std::vector<int64_t> &src0OriginShape = param.src0OriginShape;
    std::vector<int64_t> &src0RawShape = param.src0RawShape;
    // src1OriginShape do not need to normalize in current scene, so it has only 2 dim
    std::vector<int64_t> &src1OriginShape = param.src1OriginShape;
    std::vector<int64_t> &src1RawShape = param.src1RawShape;
    const std::string *dataTypeExpr = param.dataTypeExpr;
    int cacheModeFlag = GetCacheModeFlag(param.cacheMode);
    // template param
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID0], dataTypeExpr[ID2]});
    paramList.insert(paramList.end(), {std::to_string(src0OriginShape[ID0]), std::to_string(src0OriginShape[ID1]),
                                          std::to_string(src0OriginShape[ID3])});
    paramList.insert(paramList.end(),
        {std::to_string(src0RawShape[ID1]), std::to_string(src0RawShape[ID2]), std::to_string(src0RawShape[ID3])});
    paramList.emplace_back(std::to_string(src1OriginShape[ID0]));
    paramList.emplace_back(std::to_string(src1OriginShape[ID1]));
    paramList.emplace_back(std::to_string(src1RawShape[ID3]));
    paramList.insert(paramList.end(), {std::to_string(gmShape[ID2]), std::to_string(gmShape[ID3])});
    paramList.emplace_back(std::to_string(cacheModeFlag));
    paramList.emplace_back(param.blockSize);
    std::string templateParam = JoinString(paramList, ", ");
    // func actual param
    paramList.clear();
    std::string dst = "(__gm__ " + dataTypeExpr[ID0] + "*)" + addrExpr[ID0];
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";

    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintIndexOutCastDynamic(const PrintIndexOutCastParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    const std::string *addrExpr = param.addrExpr;
    std::vector<int64_t> &src0OriginShape = param.src0OriginShape;
    std::vector<int64_t> &src0RawShape = param.src0RawShape;
    // src1OriginShape do not need to normalize in current scene, so it has only 2 dim
    std::vector<int64_t> &src1OriginShape = param.src1OriginShape;
    std::vector<int64_t> &src1RawShape = param.src1RawShape;
    const std::string *dataTypeExpr = param.dataTypeExpr;
    int cacheModeFlag = GetCacheModeFlag(param.cacheMode);

    auto paramPack = PrepareDynamicShapeInfoForMTE(ID0);
    std::vector<std::string> gmShapeExpr = paramPack.gmOffsetExpr;
    std::vector<std::string> gmOffsetExpr = paramPack.gmOffsetExpr;

    std::ostringstream os;
    std::vector<std::string> paramList;
    // template param
    paramList.insert(paramList.end(), {dataTypeExpr[ID0], dataTypeExpr[ID2]});
    paramList.insert(paramList.end(), {std::to_string(src0OriginShape[ID0]), std::to_string(src0OriginShape[ID1]),
                                          std::to_string(src0OriginShape[ID3])});
    paramList.insert(paramList.end(),
        {std::to_string(src0RawShape[ID1]), std::to_string(src0RawShape[ID2]), std::to_string(src0RawShape[ID3])});
    paramList.emplace_back(std::to_string(src1OriginShape[ID0]));
    paramList.emplace_back(std::to_string(src1OriginShape[ID1]));
    paramList.emplace_back(std::to_string(src1RawShape[ID3]));
    paramList.emplace_back(std::to_string(cacheModeFlag));
    paramList.emplace_back(param.blockSize);

    std::string templateParam = JoinString(paramList, ", ");

    // func actual param
    paramList.clear();
    std::string dst = "(__gm__ " + dataTypeExpr[ID0] + "*)" + addrExpr[ID0];
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    paramList.insert(paramList.end(), paramPack.paramList.begin(), paramPack.paramList.end());

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintIndexOutCastDynamicUnaligned(const PrintIndexOutCastParam &param) const {
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    const std::string *addrExpr = param.addrExpr;
    std::vector<int64_t> &src0RawShape = param.src0RawShape;
    // src1OriginShape do not need to normalize in current scene, so it has only 2 dim
    std::vector<int64_t> &src1RawShape = param.src1RawShape;
    const std::string *dataTypeExpr = param.dataTypeExpr;
    int cacheModeFlag = GetCacheModeFlag(param.cacheMode);

    auto paramPack = PrepareDynamicShapeInfoForMTE(ID0);
    std::vector<std::string> gmShapeExpr = paramPack.gmOffsetExpr;
    std::vector<std::string> gmOffsetExpr = paramPack.gmOffsetExpr;

    auto src0ValidShape = dynamicValidShape[ID1];
    FillIntVecWithDummyInHead<SymbolicScalar>(src0ValidShape, SHAPE_DIM4 - dynamicValidShape[ID1].size(), 1);
    auto src1ValidShape = dynamicValidShape[ID2];

    std::ostringstream os;
    std::vector<std::string> paramList;
    // template param
    paramList.insert(paramList.end(), {dataTypeExpr[ID0], dataTypeExpr[ID2]});
    paramList.insert(paramList.end(),
        {std::to_string(src0RawShape[ID1]), std::to_string(src0RawShape[ID2]), std::to_string(src0RawShape[ID3])});
    paramList.emplace_back(std::to_string(src1RawShape[ID3]));
    paramList.emplace_back(std::to_string(cacheModeFlag));
    paramList.emplace_back(param.blockSize);
    std::string templateParam = JoinString(paramList, ", ");

    // func actual param
    paramList.clear();
    std::string dst = "(__gm__ " + dataTypeExpr[ID0] + "*)" + addrExpr[ID0];
    std::string src0 = "(__ubuf__ " + dataTypeExpr[ID1] + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    paramList.insert(
        paramList.end(), {src0ValidShape[ID0].Dump(), src0ValidShape[ID1].Dump(), src0ValidShape[ID3].Dump()});
    paramList.emplace_back(src1ValidShape[ID0].Dump());
    paramList.emplace_back(src1ValidShape[ID1].Dump());
    paramList.insert(paramList.end(), paramPack.paramList.begin(), paramPack.paramList.end());

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::vector<int64_t> CodeGenOpCloudNPU::GetTileShapeForMemTransfer(
    OperandType localType, std::vector<int64_t> gmShape, unsigned localIdx) const {
    std::vector<int64_t> tileShapeForMT;
    if (localIdx == 0 && localType == BUF_UB) { // copy gm to local, use shape[ID1]
        ASSERT(gmShape.size() == shape[ID1].size())
            << "gmShape size: " << gmShape.size() << ",shape[ID1] size: " << shape[ID1].size() << ", is not equal !!";
        for (size_t i = 0; i < shape[ID1].size(); ++i) {
            tileShapeForMT.emplace_back(std::min(gmShape[i], shape[ID1][i]));
        }
    } else if (localType == BUF_L1 || localType == BUF_L0C) {
        std::vector l1Shape = this->rawShape[localIdx];
        ALOG_INFO_F("getTileShape src1Shape is [%d,%d]", l1Shape[ID0], l1Shape[ID1]);
        for (size_t i = 0; i < this->rawShape[localIdx].size(); ++i) {
            tileShapeForMT.emplace_back(rawShape[localIdx][i]);
        }
    } else {
        // NEXTNEXT: verify if this branch could be merged with "localIdx == 0 && localType == BUF_UB" branch before
        ASSERT(gmShape.size() == shape[ID0].size())
            << "gmShape size: " << gmShape.size() << ",shape[ID0] size: " << shape[ID0].size() << ", is not equal !!";
        for (size_t i = 0; i < shape[ID0].size(); ++i) {
            tileShapeForMT.emplace_back(std::min(gmShape[i], shape[ID0][i]));
        }
    }

    return tileShapeForMT;
}

// In static shape scene, GM Offset is already calculated and added to GM Addr in host side, so TileOp do not need
// GM offset
std::string CodeGenOpCloudNPU::GenMemCopyVar(bool isCopyLocalToGM, OperandType localType, unsigned uf) const {
    unsigned gmIdx = isCopyLocalToGM ? 0 : 1;
    unsigned localIdx = isCopyLocalToGM ? 1 : 0;
    std::string addrTypeHead[ID2];
    addrTypeHead[gmIdx] = GetAddrTypeByOperandType(BUF_DDR);
    addrTypeHead[localIdx] = GetAddrTypeByOperandType(localType);

    std::vector<int64_t> gmShape = this->rawShape[gmIdx];
    ALOG_INFO_F("gmShape is %s", IntVecToStr(gmShape).c_str());
    std::vector<int64_t> tileShapeForMT = GetTileShapeForMemTransfer(localType, gmShape, localIdx);
    ALOG_INFO_F("========dst shape is %s", IntVecToStr(shape[ID0]).c_str());
    ALOG_INFO_F("========tileShapeForMT is %s", IntVecToStr(tileShapeForMT).c_str());

    auto localAllocKey = sm->CreateAllocKey(operandWithMagic[localIdx]);
    std::string addrExpr[ID2];
    addrExpr[localIdx] = sm->QueryVariableName(localAllocKey);
    addrExpr[gmIdx] = GenGmParamVar(gmIdx);

    std::string dataTypeExpr[ID2];
    dataTypeExpr[gmIdx] = DataType2CCEStr(operandDtype[gmIdx]);
    dataTypeExpr[localIdx] = DataType2CCEStr(operandDtype[localIdx]);

    if (localType == BUF_L0C) {
        return PrintMemCopyWithL0C(
            {uf, gmIdx, localIdx, addrTypeHead, addrExpr, gmShape, tileShapeForMT, dataTypeExpr});
    } else if (localType == BUF_L1) {
        return PrintMemCopyWithL1({uf, gmIdx, localIdx, addrTypeHead, addrExpr, gmShape, tileShapeForMT, dataTypeExpr});
    } else if (localType == BUF_UB) {
        PrintMemCopyWithUBParam param = {gmIdx, localIdx, addrTypeHead, addrExpr, dataTypeExpr, false};
        return PrintMemCopyWithUB(param);
    }

    ASSERT(0) << "GenMemCopyVar: cannot support current localType!!!" << localType;
    return {};
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithL0C(const PrintMemCopyWithL0CParam &param) const {
    if (functionType == FunctionType::DYNAMIC_LOOP_PATH) {
        return PrintMemCopyWithL0CDynamic(param);
    }
    return PrintMemCopyWithL0CStatic(param);
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithL0CStatic(const PrintMemCopyWithL0CParam &param) const {
    unsigned uf = param.uf;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::string *addrTypeHead = param.addrTypeHead;
    const std::string *addrExpr = param.addrExpr;
    const std::vector<int64_t> &gmShape = param.gmShape;
    const std::vector<int64_t> &tileShapeForMT = param.tileShapeForMT;
    const std::vector<SymbolicScalar> &outputOffset = offsetGmSymbolic[gmIdx];
    const std::string *dataTypeExpr = param.dataTypeExpr;

    int oriTileShape0 = std::min(originShape[localIdx][ID0], tileShapeForMT[ID0]);
    int oriTileShape1 = std::min(originShape[localIdx][ID1], tileShapeForMT[ID1]);

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int printRet = sprintf_s(buffer, BUFFER_SIZE_1024,
        "%s<%s, %s, %u, %u, %d, %d, %s, %s, %d, %d>((%s %s*)%s, (%s %s*)%s, %u);\n", tileOpName.c_str(),
        dataTypeExpr[gmIdx].c_str(), dataTypeExpr[localIdx].c_str(), tileShapeForMT[ID0], tileShapeForMT[ID1],
        gmShape[ID0], gmShape[ID1], outputOffset[ID0].Dump().c_str(), outputOffset[ID1].Dump().c_str(), oriTileShape0,
        oriTileShape1, addrTypeHead[ID0].c_str(), dataTypeExpr[ID0].c_str(), addrExpr[ID0].c_str(),
        addrTypeHead[ID1].c_str(), dataTypeExpr[ID1].c_str(), addrExpr[ID1].c_str(), uf);
    ASSERT(printRet >= 0) << "sprintf_s failed in genMemCopyVar(BUF_L0C), return value:" << printRet;
    return buffer;
}

std::string CodeGenOpCloudNPU::PrintL0CCopyOutDynamicUnalign(const PrintMemCopyWithL0CParam &param,
    std::vector<std::string> &gmShapeExpr, std::vector<std::string> &gmOffsetExpr) const {
    std::ostringstream os;
    std::vector<std::string> paramList;
    const std::string *dataTypeExpr = param.dataTypeExpr;
    const std::string *addrTypeHead = param.addrTypeHead;
    const std::string *addrExpr = param.addrExpr;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    paramList.emplace_back(dataTypeExpr[gmIdx]);
    paramList.emplace_back(dataTypeExpr[localIdx]);
    int64_t nzValue = 0;
    int64_t isAcc = 0;
    auto ret = GetAttr(OP_ATTR_PREFIX + "atomic_add", isAcc);
    if (ret) {
        paramList.emplace_back(std::to_string(isAcc));
    }
    ret = GetAttr("op_attr_is_nz", nzValue);
    if (ret && nzValue == 1) {
        paramList.emplace_back("false");
    } else {
        paramList.emplace_back("true");
    }
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(" + addrTypeHead[0] + " " + dataTypeExpr[0] + "*)" + addrExpr[0];
    std::string src = "(" + addrTypeHead[1] + " " + dataTypeExpr[1] + "*)" + addrExpr[1];
    paramList.insert(paramList.end(), {dst, src});
    auto dynValidShape = dynamicValidShape[localIdx];
    for (int i = 0; i < SHAPE_DIM2; i++) {
        paramList.emplace_back(dynValidShape[i].Dump());
    }
    paramList.emplace_back(gmShapeExpr[0]);
    paramList.emplace_back(gmOffsetExpr[0]);
    int64_t outerValue = 0;
    int64_t innerValue = 0;
    ret = GetAttr("op_attr_curH", outerValue);
    ret = GetAttr("op_attr_curW", innerValue);
    auto gmShapeExprByIndex = GenParamIdxExprByIndex(gmIdx, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);
    std::string outerValueStr = outerValue == 0 ? gmShapeExprByIndex[0] : std::to_string(outerValue);
    std::string innerValueStr = innerValue == 0 ? gmShapeExprByIndex[1] : std::to_string(innerValue);
    paramList.emplace_back(outerValueStr);
    paramList.emplace_back(innerValueStr);
    paramList.emplace_back(std::to_string(param.uf));
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName << "<" << templateParam << ">" << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithL0CDynamic(const PrintMemCopyWithL0CParam &param) const {
    unsigned uf = param.uf;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::string *addrTypeHead = param.addrTypeHead;
    const std::string *addrExpr = param.addrExpr;
    const std::vector<int64_t> &tileShapeForMT = param.tileShapeForMT;
    const std::string *dataTypeExpr = param.dataTypeExpr;

    int oriTileShape0 = std::min(originShape[localIdx][ID0], tileShapeForMT[ID0]);
    int oriTileShape1 = std::min(originShape[localIdx][ID1], tileShapeForMT[ID1]);

    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(param.gmIdx, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);
    ALOG_INFO_F("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(param.gmIdx, SHAPE_DIM2, PREFIX_STR_OFFSET);
    ALOG_INFO_F("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";

    int printRet{0};

    if (isSupportDynamicUnaligned) {
        return PrintL0CCopyOutDynamicUnalign(param, gmShapeExpr, gmOffsetExpr);
    }

    printRet = sprintf_s(buffer, BUFFER_SIZE_1024, "%s<%s, %s, %d, %d, %d, %d>((%s %s*)%s, (%s %s*)%s, %s, %s, %u);\n",
        tileOpName.c_str(), dataTypeExpr[gmIdx].c_str(), dataTypeExpr[localIdx].c_str(), tileShapeForMT[ID0],
        tileShapeForMT[ID1], oriTileShape0, oriTileShape1, addrTypeHead[ID0].c_str(), dataTypeExpr[ID0].c_str(),
        addrExpr[ID0].c_str(), addrTypeHead[ID1].c_str(), dataTypeExpr[ID1].c_str(), addrExpr[ID1].c_str(),
        gmShapeExpr[ID0].c_str(), gmOffsetExpr[ID0].c_str(), uf);
    ASSERT(printRet >= 0) << "sprintf_s failed in genMemCopyVar(BUF_L0C), return value:" << printRet;
    return buffer;
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithL1(const PrintMemCopyWithL1Param &param) const {
    if (functionType == FunctionType::DYNAMIC_LOOP_PATH) {
        return PrintMemCopyWithL1Dynamic(param);
    }
    return PrintMemCopyWithL1Static(param);
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithL1Static(const PrintMemCopyWithL1Param &param) const {
    unsigned uf = param.uf;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::string *addrTypeHead = param.addrTypeHead;
    const std::string *addrExpr = param.addrExpr;
    const std::vector<int64_t> &gmShape = param.gmShape;
    const std::vector<int64_t> &tileShapeForMT = param.tileShapeForMT;
    const std::string *dataTypeExpr = param.dataTypeExpr;

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";

    std::shared_ptr<LogicalTensor> tensor = sm->GetTensorByMagic(operandWithMagic[ID1]);
    std::string opName = tileOpName;
    char addrBuffer[BUFFER_SIZE_1024] = "";
    char oriAddrBuffer[BUFFER_SIZE_1024] = "";

    int printRet = sprintf_s(addrBuffer, BUFFER_SIZE_1024, "%s", addrExpr[ID1].c_str());
    ASSERT(printRet >= 0) << "sprintf_s failed in PrintMemCopyWithL1Static, return value:" << printRet;
    int64_t nzValue = 0, outerValue = 0, innerValue = 0;
    auto ret = GetAttr("op_attr_is_nz", nzValue);
    if (ret && nzValue == 1) {
        opName = "TileOp::L1CopyInNZ2NZ";
        ret = GetAttr("op_attr_outer_value", outerValue);
        ret = GetAttr("op_attr_inner_value", innerValue);
        std::string curAddrBuffer =
            "((__gm__ GMTensorInfo*)(oriAddrParam) + " + std::to_string(paramLocation[gmIdx]) + ")->Addr";
        printRet = sprintf_s(
            oriAddrBuffer, BUFFER_SIZE_1024, "(__gm__ %s*)%s", dataTypeExpr[ID1].c_str(), curAddrBuffer.c_str());
        ASSERT(printRet >= 0) << "sprintf_s failed in PrintMemCopyWithL1Static, return value:" << printRet;
        printRet = sprintf_s(addrBuffer, BUFFER_SIZE_1024, "%s", addrExpr[ID1].c_str());
        outerValue = outerValue == 0 ? gmShape[ID0] : outerValue;
        innerValue = innerValue == 0 ? gmShape[ID1] : innerValue;
        printRet = sprintf_s(buffer, BUFFER_SIZE_1024,
            "%s<%s, %s, %u, %u, %d, %d, %d, %d>((%s %s*)%s, (%s %s*)%s, %s, %u);\n", opName.c_str(),
            dataTypeExpr[gmIdx].c_str(), dataTypeExpr[localIdx].c_str(), tileShapeForMT[ID0], tileShapeForMT[ID1],
            gmShape[ID0], gmShape[ID1], outerValue, innerValue, addrTypeHead[ID0].c_str(), dataTypeExpr[ID0].c_str(),
            addrExpr[ID0].c_str(), addrTypeHead[ID1].c_str(), dataTypeExpr[ID1].c_str(), addrBuffer, oriAddrBuffer, uf);
        ASSERT(printRet >= 0) << "sprintf_s failed in genMemCopyVar, return value:" << printRet;
    } else {
        std::vector<SymbolicScalar> gmOffset = this->offsetGmSymbolic[gmIdx];
        printRet = sprintf_s(addrBuffer, BUFFER_SIZE_1024, "%s", addrExpr[ID1].c_str());
        ASSERT(printRet >= 0) << "sprintf_s failed in PrintMemCopyWithL1Static, return value:" << printRet;
        printRet =
            sprintf_s(buffer, BUFFER_SIZE_1024, "%s<%s, %s, %u, %u, %s, %s, %d, %d>((%s %s*)%s, (%s %s*)%s, %u);\n",
                opName.c_str(), dataTypeExpr[gmIdx].c_str(), dataTypeExpr[localIdx].c_str(), tileShapeForMT[ID0],
                tileShapeForMT[ID1], gmOffset[ID0].Dump().c_str(), gmOffset[ID1].Dump().c_str(), gmShape[ID0],
                gmShape[ID1], addrTypeHead[ID0].c_str(), dataTypeExpr[ID0].c_str(), addrExpr[ID0].c_str(),
                addrTypeHead[ID1].c_str(), dataTypeExpr[ID1].c_str(), addrBuffer, uf);
    }
    ASSERT(printRet >= 0) << "sprintf_s failed in PrintMemCopyWithL1Static, return value:" << printRet;
    return buffer;
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithL1Dynamic(const PrintMemCopyWithL1Param &param) const {
    std::ostringstream oss;

    unsigned uf = param.uf;
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::string *addrTypeHead = param.addrTypeHead;
    const std::string *addrExpr = param.addrExpr;
    const std::vector<int64_t> &tileShapeForMT = param.tileShapeForMT;
    const std::string *dataTypeExpr = param.dataTypeExpr;

    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(param.gmIdx, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);
    ALOG_INFO_F("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(param.gmIdx, SHAPE_DIM2, PREFIX_STR_OFFSET);
    ALOG_INFO_F("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());

    std::shared_ptr<LogicalTensor> tensor = sm->GetTensorByMagic(operandWithMagic[ID1]);
    std::string opName = tileOpName;
    std::string addrBuffer = addrExpr[ID1];

    int64_t nzValue = 0;
    auto ret = GetAttr(OP_ATTR_PREFIX + "is_nz", nzValue);
    if (ret && nzValue == 1) {
        opName = tileOpName + "NZ2NZ";
        int64_t outerValue = 0;
        int64_t innerValue = 0;
        ret = GetAttr("op_attr_outer_value", outerValue);
        ret = GetAttr("op_attr_inner_value", innerValue);

        auto gmShapeExprByIndex = GenParamIdxExprByIndex(param.gmIdx, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);
        std::string outerValueStr = outerValue == 0 ? gmShapeExprByIndex[ID0] : std::to_string(outerValue);
        std::string innerValueStr = innerValue == 0 ? gmShapeExprByIndex[ID1] : std::to_string(innerValue);

        if (isSupportDynamicUnaligned) {
            auto dynValidShape = dynamicValidShape[localIdx];
            oss << opName << "<" << dataTypeExpr[gmIdx] << ", " << dataTypeExpr[localIdx] << ">"
                << "((" << addrTypeHead[ID0] << " " << dataTypeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
                << "(" << addrTypeHead[ID1] << " " << dataTypeExpr[ID1] << "*)" << addrBuffer << ", "
                << dynValidShape[ID0].Dump() << ", " << dynValidShape[ID1].Dump() << ", " << gmShapeExpr[ID0] << ", "
                << gmOffsetExpr[ID0] << ", " << outerValueStr << ", " << innerValueStr << ", " << uf << ");\n";
        } else {
            oss << opName << "<" << dataTypeExpr[gmIdx] << ", " << dataTypeExpr[localIdx] << ", " << tileShapeForMT[ID0]
                << ", " << tileShapeForMT[ID1] << ">"
                << "((" << addrTypeHead[ID0] << " " << dataTypeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
                << "(" << addrTypeHead[ID1] << " " << dataTypeExpr[ID1] << "*)" << addrBuffer << ", "
                << gmShapeExpr[ID0] << ", " << gmOffsetExpr[ID0] << ", " << outerValueStr << ", " << innerValueStr
                << ", " << uf << ");\n";
        }
    } else {
        if (isSupportDynamicUnaligned) {
            auto dynValidShape = dynamicValidShape[localIdx];
            oss << opName << "<" << dataTypeExpr[gmIdx] << ", " << dataTypeExpr[localIdx] << ">"
                << "((" << addrTypeHead[ID0] << " " << dataTypeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
                << "(" << addrTypeHead[ID1] << " " << dataTypeExpr[ID1] << "*)" << addrBuffer << ", "
                << dynValidShape[ID0].Dump() << ", " << dynValidShape[ID1].Dump() << ", " << gmShapeExpr[ID0] << ", "
                << gmOffsetExpr[ID0] << ", " << uf << ");\n";
        } else {
            oss << opName << "<" << dataTypeExpr[gmIdx] << ", " << dataTypeExpr[localIdx] << ", " << tileShapeForMT[ID0]
                << ", " << tileShapeForMT[ID1] << ">"
                << "((" << addrTypeHead[ID0] << " " << dataTypeExpr[ID0] << "*)" << addrExpr[ID0] << ", "
                << "(" << addrTypeHead[ID1] << " " << dataTypeExpr[ID1] << "*)" << addrBuffer << ", "
                << gmShapeExpr[ID0] << ", " << gmOffsetExpr[ID0] << ", " << uf << ");\n";
        }
    }

    return oss.str();
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithUB(PrintMemCopyWithUBParam &param) const {
    unsigned localIdx = param.localIdx;
    std::string *addrExpr = param.addrExpr;
    // When ub tensor spilling to GM occurred, the spilling unit is entire raw shape of ub tensor.
    // So ub offset is always zero under this scene, do not need to calculate anymore.
    if (!param.isSpillIntoGM) {
        AppendLocalBufferVarOffset({
            {localIdx, &addrExpr[localIdx]}
        });
    }
    if (isSupportLayout) {
        return PrintMemCopyWithUBTileTensor();
    }
    if (isSupportDynamicUnaligned) {
        return PrintMemCopyWithUBDynamicSupportUnaligned(param);
    }
    if (functionType == FunctionType::DYNAMIC_LOOP_PATH) {
        return PrintMemCopyWithUBDynamic(param);
    }
    return PrintMemCopyWithUBStatic(param);
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithUBStatic(const PrintMemCopyWithUBParam &param) const {
    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";

    unsigned localIdx = param.localIdx;
    const std::string *addrTypeHead = param.addrTypeHead;
    const std::string *addrExpr = param.addrExpr;
    std::string *dataTypeExpr = param.dataTypeExpr;

    std::vector<int64_t> os = NormalizeShape(originShape[localIdx], SHAPE_DIM5);
    std::vector<int64_t> dstStride = NormalizeShape(rawShape[ID0], SHAPE_DIM5);
    std::vector<int64_t> srcStride = NormalizeShape(rawShape[ID1], SHAPE_DIM5);

    std::vector<std::vector<int64_t> *> container = {&os, &dstStride, &srcStride};
    if (!CombineAxis(container)) {
        CombineAxis(container, true);
    }

    // Support int64
    if (dataTypeExpr[localIdx] == "int64_t") {
        constexpr int numAlign = 2;
        dataTypeExpr[localIdx] = "int32_t";
        os[SHAPE_DIM5 - 1] *= numAlign;
        srcStride[SHAPE_DIM5 - 1] *= numAlign;
        dstStride[SHAPE_DIM5 - 1] *= numAlign;
    }
    int printRet = sprintf_s(buffer, BUFFER_SIZE_1024,
        "%s<%s, %u, %u, %u, %u, %u, /*dst stride*/ %u, %u, %u, %u,"
        "/*src stride*/ %u, %u, %u, %u %s>((%s %s*)%s, (%s %s*)%s);\n",
        tileOpName.c_str(), dataTypeExpr[localIdx].c_str(), os[ID0], os[ID1], os[ID2], os[ID3], os[4], dstStride[ID1],
        dstStride[ID2], dstStride[ID3], dstStride[4], srcStride[ID1], srcStride[ID2], srcStride[ID3], srcStride[4],
        GenOpAttr().c_str(), addrTypeHead[ID0].c_str(), dataTypeExpr[localIdx].c_str(), addrExpr[ID0].c_str(),
        addrTypeHead[ID1].c_str(), dataTypeExpr[localIdx].c_str(), addrExpr[ID1].c_str());
    ASSERT(printRet >= 0) << "sprintf_s failed in genMemCopyVar(BUF_UB), return value:" << printRet;
    return buffer;
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithUBDynamic(const PrintMemCopyWithUBParam &param) const {
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::string *addrTypeHead = param.addrTypeHead;
    const std::string *addrExpr = param.addrExpr;
    const std::string *dataTypeExpr = param.dataTypeExpr;

    std::vector<int64_t> newOriginShape = originShape[localIdx];
    FillIntVecWithDummyInHead<int64_t>(newOriginShape, MAX_DIM - originShape[localIdx].size(), 1);
    const std::vector<int64_t> &localRawShape = NormalizeShape(rawShape[localIdx], SHAPE_DIM5);

    auto paramPack = PrepareDynamicShapeInfoForMTE(gmIdx, MAX_DIM, !param.isSpillIntoGM);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dataTypeExpr[localIdx].c_str());
    for (auto ts : newOriginShape) {
        paramList.emplace_back(std::to_string(ts));
    }
    for (int i = 1; i < MAX_DIM; ++i) {
        paramList.emplace_back(std::to_string(localRawShape[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(" + addrTypeHead[ID0] + " " + dataTypeExpr[localIdx] + "*)" + addrExpr[ID0];
    std::string src = "(" + addrTypeHead[ID1] + " " + dataTypeExpr[localIdx] + "*)" + addrExpr[ID1];
    paramList.emplace_back(dst);
    paramList.emplace_back(src);
    paramList.insert(paramList.end(), paramPack.paramList.begin(), paramPack.paramList.end());

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithUBDynamicSupportUnaligned(const PrintMemCopyWithUBParam &param) const {
    unsigned gmIdx = param.gmIdx;
    unsigned localIdx = param.localIdx;
    const std::string *addrTypeHead = param.addrTypeHead;
    const std::string *addrExpr = param.addrExpr;
    const std::string *dataTypeExpr = param.dataTypeExpr;

    std::vector<SymbolicScalar> newDynamicShape = dynamicValidShape[localIdx];
    FillIntVecWithDummyInHead<SymbolicScalar>(newDynamicShape, MAX_DIM - dynamicValidShape[localIdx].size(), 1);
    const std::vector<int64_t> &localRawShape = NormalizeShape(rawShape[localIdx], SHAPE_DIM5);

    auto paramPack = PrepareDynamicShapeInfoForMTE(gmIdx, MAX_DIM, !param.isSpillIntoGM);
    std::vector<std::string> gmShapeExpr = paramPack.gmOffsetExpr;
    std::vector<std::string> gmOffsetExpr = paramPack.gmOffsetExpr;

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dataTypeExpr[localIdx].c_str());
    for (int i = 1; i < MAX_DIM; ++i) {
        paramList.emplace_back(std::to_string(localRawShape[i]));
    }
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(" + addrTypeHead[ID0] + " " + dataTypeExpr[localIdx] + "*)" + addrExpr[ID0];
    std::string src = "(" + addrTypeHead[ID1] + " " + dataTypeExpr[localIdx] + "*)" + addrExpr[ID1];
    paramList.emplace_back(dst);
    paramList.emplace_back(src);
    for (auto ts : newDynamicShape) {
        paramList.emplace_back(ts.Dump());
    }
    paramList.insert(paramList.end(), paramPack.paramList.begin(), paramPack.paramList.end());

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpCloudNPU::PrintMemCopyWithUBTileTensor() const {
    std::string dstTensor = sm->QueryTileTensorByMagic(operandWithMagic[ToUnderlying(SISOIdx::DST_IDX)]);
    std::string srcTensor = sm->QueryTileTensorByMagic(operandWithMagic[ToUnderlying(SISOIdx::SRC_IDX)]);
    std::ostringstream oss;
    oss << tileOpName << "(" << dstTensor << ", " << srcTensor << ");\n";
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenGMAddrExprWithOffset(const std::string &addrExpr, unsigned gmIdx) const {
    // gm offset of spilling workspace is calculated by pass, the value is saved in dim 0.
    SymbolicScalar gmOffset = this->offsetGmSymbolic[gmIdx][ID0];
    bool isZero = gmOffset.IsValid() && gmOffset.ConcreteValid() && static_cast<int>(gmOffset.Concrete()) == 0;

    std::ostringstream oss;
    if (isZero) {
        oss << addrExpr;
    } else {
        oss << "((__gm__ uint8_t*)" << addrExpr << " + " << gmOffset.Dump() << ")";
    }

    return oss.str();
}

std::string CodeGenOpCloudNPU::GenAddrExpr(const std::string &addrExpr, unsigned offsetParam) const {
    std::ostringstream oss;
    if (offsetParam != 0) {
        oss << addrExpr << " + 0x" << std::hex << offsetParam;
    } else {
        oss << addrExpr;
    }

    return oss.str();
}
} // namespace npu::tile_fwk
