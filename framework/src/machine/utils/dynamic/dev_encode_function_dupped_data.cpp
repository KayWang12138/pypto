/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dev_encode_function_dupped_data.cpp
 * \brief
 */

#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"

namespace npu::tile_fwk::dynamic {
namespace {
constexpr int ARG_ATTR_TYPE = 4;
const uint32_t RAW_TENSOR_DESC_PRE_SIZE = 8;
}

std::string DevAscendFunctionDuppedData::Dump(int indent) const {
    if (GetSource()->GetOperationSize() != GetOperationSize()) {
        DEV_ERROR("GetOperationSize mismatch: source=%zu, self=%u", GetSource()->GetOperationSize(), GetOperationSize());
    }
    DEV_ASSERT(GetSource()->GetOperationSize() == GetOperationSize());
    std::string INDENT(indent, ' ');
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');

    std::ostringstream oss;
    oss << INDENT << "DevFunctionDupped " << GetSource()->GetFuncKey() << " {\n";
    for (size_t incastIndex = 0; incastIndex < GetIncastSize(); incastIndex++) {
        oss << INDENTINNER << "#incast:" << incastIndex << " = " << GetIncastAddress(incastIndex).Dump() << "\n";
    }
    for (size_t outcastIndex = 0; outcastIndex < GetOutcastSize(); outcastIndex++) {
        oss << INDENTINNER << "#outcast:" << outcastIndex << " = " << GetOutcastAddress(outcastIndex).Dump() << "\n";
    }
    for (size_t operationIndex = 0; operationIndex < GetOperationSize(); operationIndex++) {
        oss << INDENTINNER << "!" << operationIndex;
        oss << " #pred:" << GetSource()->GetOperationDepGraphPredCount(operationIndex);
        oss << " #succ:[";
        size_t succSize;
        auto succList = GetSource()->GetOperationDepGraphSuccAddr(operationIndex, succSize);
        for (size_t j = 0; j < succSize; j++) {
            oss << Delim(j != 0, ",") << "[" << j << "]=!" << succList[j];
        }
        oss << "]";
        oss << " #dynpred:" << GetOperationCurrPredCount(operationIndex);
        oss << " #dynsucc:" << GetOperationStitch(operationIndex).Dump();
        oss << "\n";
    }
    oss << INDENTINNER << "#expr:[";
    for (size_t exprIndex = 0; exprIndex < GetExpressionSize(); exprIndex++) {
        oss << Delim(exprIndex != 0, ",") << "[" << exprIndex << "]=" << GetExpression(exprIndex);
    }
    oss << "]";
    oss << INDENT << "}\n";
    return oss.str();
}

std::string DevAscendFunctionDupped::DumpDyn(int funcIdx, int operIdx, const DevCceBinary *cceBinary) const {
    std::stringstream oss;
    auto func = GetSource();

    auto attrBase = reinterpret_cast<SymInt *>(&funcData->opAttrs[funcData->opAtrrOffsets[operIdx]]);
    auto funcIndex = attrBase[0].Value();
    oss << std::hex << " #funcKey " << func->funcKey << " #operIndex " << operIdx
        << " #funcHash: " << std::to_string(cceBinary[funcIndex].funcHash)
        << " #coreType: " << cceBinary[funcIndex].coreType
        << " #taskID:" << MakeTaskID(funcIdx, operIdx) << "\n";

    auto dumpAttr = [this, &oss](auto attrs, auto &info) {
        int attrIndex = info.staticOffsetAttrBeginIndex;
        auto rawIndex = attrs[attrIndex - 1].Value();
        oss << rawIndex << "@" << GetRawTensorAddrEx(rawIndex) << ", ";
        int dim = info.GetDim();
        for (int i = 0; i < dim * ARG_ATTR_TYPE; i++) {
            oss << GetValue(attrs, attrIndex + i) << ", ";
        }
    };

    int offset = 0;
    for (size_t idx = 0; idx < func->GetOperationIOperandSize(operIdx); idx++) {
        auto &opInfo = func->GetOperationIOperandInfo(operIdx, idx);
        offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
        dumpAttr(attrBase, opInfo);
    }
    for (size_t idx = 0; idx < func->GetOperationOOperandSize(operIdx); idx++) {
        auto &opInfo = func->GetOperationOOperandInfo(operIdx, idx);
        offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
        dumpAttr(attrBase, opInfo);
    }
    for (size_t idx = static_cast<size_t>(offset); idx < func->GetOperationAttrSize(operIdx); idx++) {
        oss << GetValue(attrBase, idx) << ", ";
    }
    return oss.str();
}

void DevAscendFunctionDupped::DumpTopo(std::ofstream &os, int seqNo, int funcIdx, const DevCceBinary *cceBinary) const {
    auto func = GetSource();
    for (size_t opIdx = 0; opIdx < DupData()->GetSource()->GetOperationSize(); opIdx++) {
        auto &cceInfo = cceBinary[func->GetOperationAttrCalleeIndex(opIdx)];
        os << seqNo << "," << MakeTaskID(funcIdx, opIdx) << "," << func->funcKey << "," << func->rootHash << ","
            <<func->GetOperationDebugOpmagic(opIdx) << "," << func->GetOperationAttrCalleeIndex(opIdx) << ","
            << cceInfo.funcHash << "," << cceInfo.coreType << "," << cceInfo.psgId << ",";
        auto &succList = func->GetOperationDepGraphSuccList(opIdx);
        for (size_t j = 0; j < succList.size(); j++) {
            os << "," << MakeTaskID(funcIdx, func->At(succList, j));
        }
        auto &stitch = GetOperationStitch(opIdx);
        stitch.ForEach([&os](uint32_t id) {
            os << "," << id;
        });
        os << "\n";
    }
}

#if DEBUG_INFINITE_LIFETIME
void DevAscendFunctionDupped::DumpTensorAddrInfo(std::vector<std::string> &infos, uint32_t seqNo, uint32_t funcIdx) {
    // seqNo,taskId,rawMagic,address,dtype,bytesOfDtype,(shapes,)
    auto *srcFunc = GetSource();

    auto dumpOperand = [&](const DevAscendOperationOperandInfo &operandInfo, size_t opIdx) {
        std::stringstream os;
        uint64_t rawIdx = srcFunc->GetTensor(operandInfo.tensorIndex)->rawIndex;
        auto *rawTensor = srcFunc->GetRawTensor(rawIdx);
        os << seqNo << "," << MakeTaskID(funcIdx, opIdx) << "," <<
            rawTensor->rawMagic << "," <<
            GetRawTensorAddrEx(rawIdx) << "," <<
            BriefDataType2String(rawTensor->dataType) << "," <<
            BytesOf(rawTensor->dataType);

        uint32_t dimSize = rawTensor->GetDim();
        os << ",(";
        bool isFirstDim = true;
        for (uint32_t i = 0; i < dimSize; i++) {
            if (isFirstDim) {
                isFirstDim = false;
            } else {
                os << ",";
            }
            os << rawTensor->shape.At(i, GetExpressionAddr());
        }
        os << ")";

        os << "\n";
        infos.emplace_back(std::move(os).str());
    };

    for (size_t opIdx = 0; opIdx < srcFunc->GetOperationSize(); opIdx++) {
        for (size_t iopIdx = 0; iopIdx < srcFunc->GetOperationIOperandSize(opIdx); iopIdx++) {
            auto &iopInfo = srcFunc->GetOperationIOperandInfo(opIdx, iopIdx);
            dumpOperand(iopInfo, opIdx);
        }
        for (size_t oopIdx = 0; oopIdx < srcFunc->GetOperationOOperandSize(opIdx); oopIdx++) {
            auto &oopInfo = srcFunc->GetOperationOOperandInfo(opIdx, oopIdx);
            dumpOperand(oopInfo, opIdx);
        }
    }
}
#endif // DEBUG_INFINITE_LIFETIME

static void FlushStream(std::vector<std::string> &lines, std::stringstream &oss) {
    lines.push_back(std::move(oss).str());
    oss.clear();
    oss.str("");
}

void DevAscendFunctionDupped::DumpRawShape(const DevAscendRawTensor *rawTensor, uint32_t dimSize,
                                           std::vector<std::string> &lines, std::stringstream &oss) const {
    oss << "        rawShape=[";
    bool isFirstDim = true;
    for (uint32_t i = 0; i < dimSize; i++) {
        if (isFirstDim) {
            isFirstDim = false;
        } else {
            oss << ", ";
        }
        oss << rawTensor->shape.At(i, GetExpressionAddr());
    }
    oss << "]";
    FlushStream(lines, oss);
}

void DevAscendFunctionDupped::DumpOperandShape(uint32_t dimSize, size_t opIdx, size_t operandIdx, bool isIn,
                                               std::vector<std::string> &lines, std::stringstream &oss) const {
    uint64_t offset[DEV_SHAPE_DIM_MAX];
    uint64_t shape[DEV_SHAPE_DIM_MAX];
    GetFuncTensorOffsetAndShape(offset, shape, dimSize, opIdx, operandIdx, isIn);

    oss << "          offset=[";
    bool isFirstDim = true;
    for (uint32_t i = 0; i < dimSize; i++) {
        if (isFirstDim) {
            isFirstDim = false;
        } else {
            oss << ", ";
        }
        oss << offset[i];
    }
    oss << "]";
    FlushStream(lines, oss);

    oss << "           shape=[";
    isFirstDim = true;
    for (uint32_t i = 0; i < dimSize; i++) {
        if (isFirstDim) {
            isFirstDim = false;
        } else {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    FlushStream(lines, oss);
}

// Return result lines
std::vector<std::string> DevAscendFunctionDupped::DumpLeafs(uint32_t seqNo, uint32_t funcIdx) const {
    std::vector<std::string> lines;
    std::stringstream oss;

    auto *srcFunc = GetSource();

    oss << "seqNo=" << seqNo << ", rootHash=" << srcFunc->rootHash;
    FlushStream(lines, oss);

    for (size_t opIdx = 0; opIdx < srcFunc->GetOperationSize(); opIdx++) {
        size_t iopNum = srcFunc->GetOperationIOperandSize(opIdx);
        size_t oopNum = srcFunc->GetOperationOOperandSize(opIdx);
        oss << "> taskId = " << MakeTaskID(funcIdx, opIdx) << ", opIdx=" << opIdx << ", #iop=" << iopNum << ", #oop=" << oopNum;
        FlushStream(lines, oss);

        for (size_t iopIdx = 0; iopIdx < iopNum; iopIdx++) {
            uint64_t rawIdx = srcFunc->GetOperationIOperand(opIdx, iopIdx)->rawIndex;
            auto *rawTensor = srcFunc->GetRawTensor(rawIdx);

            oss << "    iop [" << std::setw(IDENT_SIZE_THREE) << iopIdx << "]: rawMagic=" << rawTensor->rawMagic
                << ", addr=0x" << std::hex << GetRawTensorAddrEx(rawIdx) << std::dec;
            FlushStream(lines, oss);

            uint32_t dimSize = rawTensor->GetDim();
            DumpOperandShape(dimSize, opIdx, iopIdx, true, lines, oss);
            DumpRawShape(rawTensor, dimSize, lines, oss);
        }

        for (size_t oopIdx = 0; oopIdx < oopNum; oopIdx++) {
            uint64_t rawIdx = srcFunc->GetOperationOOperand(opIdx, oopIdx)->rawIndex;
            auto *rawTensor = srcFunc->GetRawTensor(rawIdx);

            oss << "    oop [" << std::setw(IDENT_SIZE_THREE) << oopIdx << "]: rawMagic=" << rawTensor->rawMagic
                << ", addr=0x" << std::hex << GetRawTensorAddrEx(rawIdx) << std::dec;
            FlushStream(lines, oss);

            uint32_t dimSize = rawTensor->GetDim();
            DumpOperandShape(dimSize, opIdx, oopIdx, false, lines, oss);
            DumpRawShape(rawTensor, dimSize, lines, oss);
        }
    }

    return lines;
}

void DevAscendFunctionDupped::DumpAttr(const DevAscendFunction *func, const SymInt *attrs,
                                       const DevAscendOperationOperandInfo &info, std::stringstream &oss) const {
    int attrOffset = info.staticOffsetAttrBeginIndex;
    auto rawIndex = attrs[attrOffset - 1].Value();
    oss << "@(rawidx:" << rawIndex << " attridx:" << (attrOffset - 1) << ")" << ", ";

    int dim = info.GetDim();
    auto rawTensor = func->GetRawTensor(rawIndex);
    if (rawIndex >= func->GetRawTensorSize()) {
        DEV_ERROR("Invalid rawIndex=%lu, exceeds raw tensor size=%lu", rawIndex, func->GetRawTensorSize());
    }
    if (dim != rawTensor->GetDim()) {
        DEV_ERROR("Dimension mismatch: info.dim=%d, rawTensor->dim=%d", dim, rawTensor->GetDim());
    }
    DEV_ASSERT(rawIndex < func->GetRawTensorSize());
    DEV_ASSERT(dim == rawTensor->GetDim());

    for (int d = 0; d < rawTensor->GetDim(); d++) {
        auto shapeIdx = attrOffset + d + rawTensor->GetDim() * 2;
        auto shape = static_cast<int64_t>(rawTensor->shape.At(d, funcData->exprTbl));
        auto actualShape = GetValue(attrs, shapeIdx);
        if (actualShape != shape) {
            DEV_ERROR("Shape mismatch at dim %d: expacted=%ld, got=%ld", d, shape, actualShape);
        }
        DEV_ASSERT(actualShape == shape);
    }
    if (dim != rawTensor->GetDim()) {
        DEV_ERROR("Final dimension mismatch after shape validation: info.dim=%d, rawTensor->dim=%d", dim, rawTensor->GetDim());
    }
    DEV_ASSERT(dim == rawTensor->GetDim());
    for (int i = 0; i < dim * ARG_ATTR_TYPE; i++) {
        oss << GetValue(attrs, attrOffset + i) << ", ";
    }
}

void DevAscendFunctionDupped::DumpFuncData(const DevAscendFunction *func, int funcIdx, const DevCceBinary *cceBinary,
                                           std::stringstream &oss) const {
    oss << "#funcData: [\n" << std::dec;
    for (size_t operIdx = 0; operIdx < func->GetOperationSize(); operIdx++) {
        auto attrBase = &func->GetOperationAttr(operIdx, 0);
        auto funcIndex = attrBase[0].Value();
        oss << "  [" << operIdx << "]  #funcHash: " << std::to_string(cceBinary[funcIndex].funcHash)
            << " #funcIndex: " << funcIndex << " #taskID:" << MakeTaskID(funcIdx, operIdx) 
            << " #opMagic: " << func->GetOperationDebugOpmagic(operIdx) << "\n";
        oss << "  #invokeAttrs : ";
        int offset = 0;
        for (size_t idx = 0; idx < func->GetOperationIOperandSize(operIdx); idx++) {
            auto &opInfo = func->GetOperationIOperandInfo(operIdx, idx);
            offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
            oss << " in:";
            DumpAttr(func, attrBase, opInfo, oss);
        }
        for (size_t idx = 0; idx < func->GetOperationOOperandSize(operIdx); idx++) {
            auto &opInfo = func->GetOperationOOperandInfo(operIdx, idx);
            offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
            oss << " out:";
            DumpAttr(func, attrBase, opInfo, oss);
        }
        oss << "\n other attr:";
        for (size_t idx = offset; idx < func->GetOperationAttrSize(operIdx); idx++) {
            oss << GetValue(attrBase, idx) << ", ";
        }
        oss << "\n";
    }
}

std::string DevAscendFunctionDupped::DumpDyn(int funcIdx, const DevCceBinary *cceBinary) const {
    std::stringstream oss;
    auto func = GetSource();
    for (size_t opIdx = 0; opIdx < DupData()->GetSource()->GetOperationSize(); opIdx++) {
        oss << std::hex << "[" << opIdx << "] #predCnt:" << GetOperationCurrPredCount(opIdx);
        auto &succList = func->GetOperationDepGraphSuccList(opIdx);
        oss << " #succList: [";
        for (size_t j = 0; j < succList.size(); j++) {
            if (j != 0)
                oss << ", ";
            oss << func->At(succList, j);
        }
        oss << ']';
        auto &stitch = GetOperationStitch(opIdx);
        if (!stitch.IsNull())
            oss << std::hex << " #stitch:" << stitch.Dump();
        oss << "\n";
    }

    oss << " #funcKey: " << func->funcKey << " #gmStackBase: " << funcData->stackWorkSpaceAddr
        << " #stackSize: " << funcData->stackWorkSpaceSize << " #workspace: " << funcData->workspaceAddr << "\n";

    DumpFuncData(func, funcIdx, cceBinary, oss);

    oss << std::hex << "  #rawTensorAddrs: ";
    for (uint64_t i = 0; i < func->GetRawTensorDescSize(); i++) {
        if (i % RAW_TENSOR_DESC_PRE_SIZE == 0)
            oss << "\n   ";
        if (GetRawTensorAddrEx(i) != GetRawTensorAddr(i)) {
            DEV_ERROR("Tensor address mismatch at index %lu: addr=%lu, addrEx=%lu.", i, GetRawTensorAddr(i), GetRawTensorAddrEx(i));
        }
        DEV_ASSERT(GetRawTensorAddrEx(i) == GetRawTensorAddr(i));
        auto desc = funcData->rawTensorDesc[i];
        oss << GetRawTensorAddrEx(i) << "(location:" << desc.location << " offsetOrIdex: " << desc.offsetOrIndex << ")" << ", ";
    }
    oss << "\n]";
    return oss.str();
}
}
