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
 * \file codegen_distributed.cpp
 * \brief
 */

#include "codegen_op_cloudnpu.h"
#include "interface/utils/log.h"
#include "securec.h"

namespace npu::tile_fwk {

std::string CodeGenOpCloudNPU::GetTemplateDType() const {
    if (tileOpName.find("FFNBatching") != std::string::npos) {
        return DataType2CCEStr(operandDtype[0]);
    } else if (tileOpName.find("AttnCombine") != std::string::npos) {
        return DataType2CCEStr(operandDtype[0]);
    } else if (tileOpName.find("DispatchSetFlag") != std::string::npos) {
        return DataType2CCEStr(operandDtype[4]); // 从 operand 4 获取 T
    }
    return DataType2CCEStr(operandDtype[1]);
}

std::string CodeGenOpCloudNPU::GenDistOp() const {
    // 如果有outcast，paramStr的起始位置是outcast的buffer
    // 如果没有outcast，paramStr的起始位置是incast[0]的buffer
    std::string paramStr = GenParamsStr();

    std::string dtypeStr = GetTemplateDType();

    std::string extraTemplateParam;
    if (opAttrs.count("extraTemplateParam") != 0) {
        extraTemplateParam = ", " + npu::tile_fwk::AnyCast<std::string>(opAttrs.at("extraTemplateParam"));
    }

    char buffer[BUFFER_SIZE_1024];
    int ret = sprintf_s(buffer, sizeof(buffer), "%s<%s%s>(%s, %s);\n", tileOpName.c_str(), dtypeStr.c_str(),
        extraTemplateParam.c_str(), paramStr.c_str(), "hcclContext");
    ASSERT(ret >= 0) << "genDistOp sprintf_s failed ";
    return std::string(buffer);
}

} // namespace npu::tile_fwk
