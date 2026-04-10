/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file codegen_vector_quant.cpp
 * \brief
 */

#include "codegen_op_cloudnpu.h"

namespace npu::tile_fwk {
std::string CodeGenOpCloudNPU::GenQuantMXOp() const
{
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, isSupportLayout) << "QuantMX only supports tile tensor codegen.";

    const std::string dstTensor = QueryTileTensorNameByIdx(ID0);
    const std::string expTensor = QueryTileTensorNameByIdx(ID1);
    const std::string maxTensor = QueryTileTensorNameByIdx(ID2);
    const std::string scalingTensor = QueryTileTensorNameByIdx(ID3);
    const std::string srcTensor = QueryTileTensorNameByIdx(ID4);

    std::ostringstream oss;
    oss << tileOpName << "(" << JoinString({dstTensor, expTensor, maxTensor, scalingTensor, srcTensor}, CONN_COMMA)
        << ");\n";
    return oss.str();
}
} // namespace npu::tile_fwk
