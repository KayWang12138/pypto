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

#include "codegen_op_litenpu.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "securec.h"

namespace npu::tile_fwk {
// std::string CodeGenOpLiteNPU::GenCubeOp(bool zeroC) const {
//     // shape: dst, src0, src1
//     bool isOffsetValid =
//         (offset[ID1][ID0] == 0) && (offset[ID1][ID1] == 0) && (offset[ID2][ID0] == 0) && (offset[ID2][ID1] == 0);
//     ASSERT(isOffsetValid) << "CUBE: haven't consider l0A/B offset yet.";
//     bool isShapeValid = (shape[ID0][ID0] == shape[ID1][ID0]) &&
//                         (shape[ID0][ID1] == shape[ID2][ID1] || shape[ID0][ID1] == shape[ID2][ID0]) &&
//                         (shape[ID1][ID1] == shape[ID2][ID1] || shape[ID1][ID1] == shape[ID2][ID0]);
//     ASSERT(isShapeValid) << "CUBE: m k n is invalid.";
//     int m = shape[ID0][ID0];
//     int k = shape[ID1][ID1]; // NEXTNEXT assume A is not transposed for now
//     int n = shape[ID0][ID1];
//     unsigned uf = 0;

//     auto kL0C = sm->CreateAllocKey(operandWithMagic[ID0]);
//     auto kL0A = sm->CreateAllocKey(operandWithMagic[ID1]);
//     auto kL0B = sm->CreateAllocKey(operandWithMagic[ID2]);

//     std::string aVar = sm->QueryVariableName(kL0A);
//     std::string bVar = sm->QueryVariableName(kL0B);
//     std::string cVar = sm->QueryVariableName(kL0C);

//     char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
//     std::string aDtypeStr = DataType2CCEStr(operandDtype[ID1]);
//     std::string bDtypeStr = DataType2CCEStr(operandDtype[ID2]);
//     std::string cDtypeStr = DataType2CCEStr(operandDtype[ID0]);

//     int ret{0};
//     if (isSupportDynamicUnaligned) {
//         auto l0cShapeDyn = dynamicValidShape[0];
//         auto l0aShapeDyn = dynamicValidShape[1];
//         auto l0bShapeDyn = dynamicValidShape[2];
//         auto mSymbol = l0cShapeDyn[0];
//         auto kSymbol = l0aShapeDyn[1];
//         auto nSymbol = l0cShapeDyn[1];
//         ret = sprintf_s(buffer, sizeof(buffer),
//             "%s<%s, %s, %s,  %u, %u>((%s %s*)%s, (%s %s*)%s, (%s %s*)%s, %s, %s, %s, %s, %u, %s, %s);\n",
//             tileOpName.c_str(), cDtypeStr.c_str(), aDtypeStr.c_str(), bDtypeStr.c_str(), offset[ID0][ID0],
//             offset[ID0][ID1], GetAddrTypeByOperandType(operandType[0]).c_str(), cDtypeStr.c_str(), cVar.c_str(),
//             GetAddrTypeByOperandType(operandType[ID1]).c_str(), aDtypeStr.c_str(), aVar.c_str(),
//             GetAddrTypeByOperandType(operandType[ID2]).c_str(), bDtypeStr.c_str(), bVar.c_str(), mSymbol.Dump().c_str(),
//             kSymbol.Dump().c_str(), nSymbol.Dump().c_str(), zeroC ? "true" : "false", uf, l0cShapeDyn[0].Dump().c_str(),
//             l0cShapeDyn[1].Dump().c_str());
//         ASSERT(ret >= 0) << "GenCubeOp sprintf_s failed ";
//     } else {
//         ret = sprintf_s(buffer, sizeof(buffer),
//             "%s<%s, %s, %s, %u, %u, %u, %u>((%s %s*)%s, (%s %s*)%s, (%s %s*)%s, %d, %d, %d, %s, %u);\n",
//             tileOpName.c_str(), cDtypeStr.c_str(), aDtypeStr.c_str(), bDtypeStr.c_str(), offset[ID0][ID0],
//             offset[ID0][ID1], shape[ID0][ID0], shape[ID0][ID1], GetAddrTypeByOperandType(operandType[0]).c_str(),
//             cDtypeStr.c_str(), cVar.c_str(), GetAddrTypeByOperandType(operandType[1]).c_str(), aDtypeStr.c_str(),
//             aVar.c_str(), GetAddrTypeByOperandType(operandType[ID2]).c_str(), bDtypeStr.c_str(), bVar.c_str(), m, k, n,
//             zeroC ? "true" : "false", uf);
//         ASSERT(ret >= 0) << "GenCubeOp sprintf_s failed ";
//     }

//     return buffer;
// }

// std::string CodeGenOpLiteNPU::GenCubeOpMatmul() const{
//     return GenCubeOp(true);
// }

// std::string CodeGenOpLiteNPU::GenCubeOpMatmulAcc() const{
//     return GenCubeOp(false);
// }

std::string CodeGenOpLiteNPU::GenParamsStr() const {
    std::vector<std::string> params;
    for (int i = 0; i < MAX_OPERANDS; i++) {
        if (operand[i] == NULL_OPERAND) {
            continue;
        }

        std::string dtypeStr = DataType2CCEStr(operandDtype[i]);
        std::string prefix = GetAddrTypeByOperandType(operandType[i]);

        // if (skipOperands.find(i) != skipOperands.end()) {
        //     continue;
        // }

        if (operandType[i] == BUF_DDR) {
            std::string var = GenGmParamVar(i);
            std::ostringstream oss;
            oss << "(" << prefix << " " << dtypeStr << "*)" << var;
            params.emplace_back(oss.str());
        } else {
            std::string var = sm->QueryVarNameByTensorMagic(operandWithMagic[i]);

            if (opCode != Opcode::OP_L1_TO_L0A && opCode != Opcode::OP_L1_TO_L0B && opCode != Opcode::OP_L1_TO_L0_BT &&
                opCode != Opcode::OP_L1_TO_L0_AT) {
                // 大包搬运场景下，L1搬运至L0不需要计算L1地址偏移
                // 非大包搬运场景下，L1与L0数据大小一致，也不需要地址偏移
                // 偏移计算仅用于L1_Copy_In 和 L1_Copy_Out
                AppendLocalBufferVarOffset({
                    {static_cast<unsigned>(i), std::ref(var)}
                });
            }

            std::ostringstream oss;
            CODEGEN_LOGD("GenParamsStr var: %s", var.c_str());
            oss << "(" << prefix << " " << dtypeStr << "*)" << var;
            params.emplace_back(oss.str());
        }
    }
    return JoinString(params, ", ");
}

} // namespace npu::tile_fwk
