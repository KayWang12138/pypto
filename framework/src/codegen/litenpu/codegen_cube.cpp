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

std::string CodeGenOpLiteNPU::GenCubeOpMatmul() const {
    return GenCubeOp(true);
}

std::string CodeGenOpLiteNPU::GenCubeOpMatmulAcc() const {
    return GenCubeOp(false);
}

} // namespace npu::tile_fwk
