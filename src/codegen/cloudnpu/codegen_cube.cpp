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
 * \file codegen_cube.cpp
 * \brief
 */

#include "codegen_op_cloudnpu.h"
#include "codegen/codegen_utils.h"
#include "codegen/codegen_symbol.h"
#include "securec.h"

namespace npu::tile_fwk {
std::string CodeGenOpCloudNPU::GenCubeOp(bool zeroC) const {
    // shape: dst, src0, src1
    bool isOffsetValid =
        (offset[ID1][ID0] == 0) && (offset[ID1][ID1] == 0) && (offset[ID2][ID0] == 0) && (offset[ID2][ID1] == 0);
    ASSERT(isOffsetValid) << "CUBE: haven't consider l0A/B offset yet.";
    bool isShapeValid = (shape[ID0][ID0] == shape[ID1][ID0]) &&
                        (shape[ID0][ID1] == shape[ID2][ID1] || shape[ID0][ID1] == shape[ID2][ID0]) &&
                        (shape[ID1][ID1] == shape[ID2][ID1] || shape[ID1][ID1] == shape[ID2][ID0]);
    ASSERT(isShapeValid) << "CUBE: m k n is invalid.";
    int m = shape[ID0][ID0];
    int k = shape[ID1][ID1]; // NEXTNEXT assume A is not transposed for now
    int n = shape[ID0][ID1];
    unsigned uf = 0;

    auto kL0C = sm->CreateAllocKey(operandWithMagic[ID0]);
    auto kL0A = sm->CreateAllocKey(operandWithMagic[ID1]);
    auto kL0B = sm->CreateAllocKey(operandWithMagic[ID2]);

    std::string aVar = sm->QueryVariableName(kL0A);
    std::string bVar = sm->QueryVariableName(kL0B);
    std::string cVar = sm->QueryVariableName(kL0C);

    std::string aDtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string bDtypeStr = DataType2CCEStr(operandDtype[ID2]);
    std::string cDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    std::ostringstream oss;

    if (isSupportDynamicUnaligned) {
        auto l0cShapeDyn = dynamicValidShape[ID0];
        auto l0aShapeDyn = dynamicValidShape[ID1];
        auto l0bShapeDyn = dynamicValidShape[ID2];
        auto mSymbol = l0cShapeDyn[ID0];
        auto kSymbol = l0aShapeDyn[ID1];
        auto nSymbol = l0cShapeDyn[ID1];

        oss << tileOpName << "<" << cDtypeStr << ", " << aDtypeStr << ", " << bDtypeStr << ", " << offset[ID0][ID0]
            << ", " << offset[ID0][ID1] << ">"
            << "((" << GetAddrTypeByOperandType(operandType[ID0]) << " " << cDtypeStr << "*)" << cVar << ", "
            << "(" << GetAddrTypeByOperandType(operandType[ID1]) << " " << aDtypeStr << "*)" << aVar << ", "
            << "(" << GetAddrTypeByOperandType(operandType[ID2]) << " " << bDtypeStr << "*)" << bVar << ", "
            << mSymbol.Dump() << ", " << kSymbol.Dump() << ", " << nSymbol.Dump() << ", " << (zeroC ? "true" : "false")
            << ", " << uf << ", " << l0cShapeDyn[ID0].Dump() << ", " << l0cShapeDyn[ID1].Dump() << ");\n";
    } else {
        oss << tileOpName << "<" << cDtypeStr << ", " << aDtypeStr << ", " << bDtypeStr << ", " << offset[ID0][ID0]
            << ", " << offset[ID0][ID1] << ", " << shape[ID0][ID0] << ", " << shape[ID0][ID1] << ">"
            << "((" << GetAddrTypeByOperandType(operandType[ID0]) << " " << cDtypeStr << "*)" << cVar << ", "
            << "(" << GetAddrTypeByOperandType(operandType[ID1]) << " " << aDtypeStr << "*)" << aVar << ", "
            << "(" << GetAddrTypeByOperandType(operandType[ID2]) << " " << bDtypeStr << "*)" << bVar << ", " << m
            << ", " << k << ", " << n << ", " << (zeroC ? "true" : "false") << ", " << uf << ");\n";
    }

    return oss.str();
}

std::string CodeGenOpCloudNPU::GenCubeOpMatmul() const{
    return GenCubeOp(true);
}

std::string CodeGenOpCloudNPU::GenCubeOpMatmulAcc() const{
    return GenCubeOp(false);
}

std::string CodeGenOpCloudNPU::GenParamsStr() const {
    std::vector<std::string> params;
    for (int i = 0; i < MAX_OPERANDS; i++) {
        if (operand[i] == NULL_OPERAND) {
            continue;
        }

        std::string dtypeStr = DataType2CCEStr(operandDtype[i]);
        std::string prefix = GetAddrTypeByOperandType(operandType[i]);

        if (operandType[i] == BUF_DDR) {
            std::string var = GenGmParamVar(i);
            std::ostringstream oss;
            oss << "(" << prefix << " " << dtypeStr << "*)" << var;
            params.emplace_back(oss.str());
        } else {
            auto localAllocKey = sm->CreateAllocKey(operandWithMagic[i]);
            std::string var = sm->QueryVariableName(localAllocKey);

            if (opCode != Opcode::OP_L1_TO_L0A && opCode != Opcode::OP_L1_TO_L0B && opCode != Opcode::OP_L1_TO_L0_BT) {
                // 大包搬运场景下，L1搬运至L0不需要计算L1地址偏移
                // 非大包搬运场景下，L1与L0数据大小一致，也不需要地址偏移
                // 偏移计算仅用于L1_Copy_In 和 L1_Copy_Out
                AppendLocalBufferVarOffset({&var}, {static_cast<unsigned>(i)});
            }

            std::ostringstream oss;
            if (this->addrOffset[i] == 0) {
                ALOG_DEBUG_F("GenParamsStr var: %s", var.c_str());
                oss << "(" << prefix << " " << dtypeStr << "*)" << var;
            } else {
                oss << "(" << prefix << " " << dtypeStr << "*)"
                    << "((" << prefix << " uint8_t*)" << var << " + 0x" << std::hex << this->addrOffset[i] << ")";
            }

            params.emplace_back(oss.str());
        }
    }
    return JoinString(params, ", ");
}

} // namespace npu::tile_fwk
