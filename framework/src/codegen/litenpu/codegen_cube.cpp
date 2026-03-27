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
std::string CodeGenOpLiteNPU::PrintMatmulTileTensor(
    bool isAcc, std::unordered_map<OperandType, std::string> &tensorWithMemType) const {
    std::ostringstream oss;
    bool hasBias = tensorWithMemType.count(OperandType::BUF_BT);
    int64_t transModeNum = 0;
    GetAttr(OpAttributeKey::transMode, transModeNum);
    TransMode transMode = static_cast<TransMode>(transModeNum);
    std::string transModeStr = "TransMode::CAST_NONE";
    if (transMode == TransMode::CAST_RINT) {
        transModeStr = "TransMode::CAST_RINT";
    } else if (transMode == TransMode::CAST_ROUND) {
        transModeStr = "TransMode::CAST_ROUND";
    }
    std::vector<std::string> paramList = {tensorWithMemType[OperandType::BUF_L0C],
        tensorWithMemType[OperandType::BUF_L0A], tensorWithMemType[OperandType::BUF_L0B]};
    oss << tileOpName;
    if (hasBias) {
        paramList.emplace_back(tensorWithMemType[OperandType::BUF_BT]);
        oss << WrapParamByAngleBrackets({transModeStr});
        oss << WrapParamByParentheses(paramList) << ";\n";
        return oss.str();
    }
    oss << WrapParamByAngleBrackets({std::to_string(isAcc), transModeStr});
    oss << WrapParamByParentheses(paramList) << ";\n";
    return oss.str();
}

std::string CodeGenOpLiteNPU::PrintMatmulTileTensor(bool isAcc) const {
    std::unordered_map<OperandType, std::string> tensorWithMemType;
    for (int i = 0; i < operandCnt; i++) {
        tensorWithMemType.emplace(operandType[i], QueryTileTensorNameByIdx(i));
    }
    return PrintMatmulTileTensor(isAcc, tensorWithMemType);
}


std::string CodeGenOpLiteNPU::GenCubeOp(bool zeroC) const {
    if (isSupportLayout) {
        return PrintMatmulTileTensor(!zeroC);
    }
    return "";
}

std::string CodeGenOpLiteNPU::GenCubeOpMatmul() const{
    return GenCubeOp(true);
}

std::string CodeGenOpLiteNPU::GenCubeOpMatmulAcc() const{
    return GenCubeOp(false);
}

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
