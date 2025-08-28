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

#include <sstream>
#include <string>
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
    } else if (tileOpName.find("ShmemSignal") != std::string::npos) {
        return DataType2CCEStr(operandDtype[2]); // 从 operand 2 获取 T
    }
    return DataType2CCEStr(operandDtype[1]);
}

std::string CodeGenOpCloudNPU::GenTemplateParams() const {
    std::ostringstream oss;
    oss << GetTemplateDType();
    if ((tileOpName.find("ShmemPut") != std::string::npos) || (tileOpName.find("ShmemGet") != std::string::npos)) {
        std::vector<int32_t> inShape = rawShape[3]; // operand 3 是 shmemData
        int rowShape = inShape[inShape.size() - 2]; // 倒数第 2 轴是 row
        ASSERT(rowShape > 0 && rowShape <= std::numeric_limits<uint16_t>::max()) << "rowShape is not valid";
        int colShape = inShape[inShape.size() - 1];
        ASSERT(colShape > 0 && colShape <= std::numeric_limits<uint16_t>::max()) << "colShape is not valid";
        oss << ", " << rowShape << ", " << colShape;
    } else if (tileOpName.find("ShmemSignal") != std::string::npos) {
        std::string value = npu::tile_fwk::AnyCast<std::string>(opAttrs.at("Value"));
        std::string atomicType = npu::tile_fwk::AnyCast<std::string>(opAttrs.at("AtomicType"));
        oss << ", " << value << ", " << atomicType;
    } else if (opAttrs.count("extraTemplateParam") != 0) {
        std::string extraTemplateParam = npu::tile_fwk::AnyCast<std::string>(opAttrs.at("extraTemplateParam"));
        oss << ", " << extraTemplateParam;
    }
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenDistOp() const {
    std::ostringstream oss;
    oss << tileOpName << "<" << GenTemplateParams() << ">(" << GenParamsStr() << ", hcclContext);\n";
    return oss.str();
}

} // namespace npu::tile_fwk
