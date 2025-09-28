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
#include "codegen/codegen_common.h"
#include "codegen_op_cloudnpu.h"
#include "interface/utils/log.h"
#include "securec.h"
#include "interface/operation/distributed/distributed_common.h"

namespace npu::tile_fwk {

std::string CodeGenOpCloudNPU::GetTemplateDType() const
{
    if (opCode == Opcode::OP_FFN_BATCHING) {
        return DataType2CCEStr(operandDtype[0]);
    } else if (opCode == Opcode::OP_MOE_ATTN_COMBINE) {
        return DataType2CCEStr(operandDtype[0]);
    } else if (opCode == Opcode::OP_DISPATCH_SET_FLAG) {
        return DataType2CCEStr(operandDtype[4]); // 从 operand 4 获取 T
    }
    return DataType2CCEStr(operandDtype[1]);
}

std::string CodeGenOpCloudNPU::GenTemplateParams() const
{
    std::ostringstream oss;
    if ((opCode == Opcode::OP_SHMEM_PUT) || (opCode == Opcode::OP_SHMEM_GET)) {
        std::vector<int64_t> tileShape = originShape[ID3];
        int64_t tileRowShape = tileShape[tileShape.size() - ID2];
        ASSERT(tileRowShape > 0 && tileRowShape <= std::numeric_limits<uint16_t>::max()) << "tileRowShape is not valid";
        int64_t tileColShape = tileShape[tileShape.size() - ID1];
        ASSERT(tileColShape > 0 && tileColShape <= std::numeric_limits<uint16_t>::max()) << "tileColShape is not valid";
        std::vector<int64_t> bufferShape = originShape[ID1];
        int64_t bufferRowShape = bufferShape[0];
        int64_t bufferColShape = bufferShape[1];
        std::vector<int64_t> originTensorShape = rawShape[ID3]; // 切块之前的shape
        int64_t stride = originTensorShape[originTensorShape.size() - ID1];
        oss << "<" << GetTemplateDType() << ", " << tileRowShape << ", " << tileColShape << ", " << bufferRowShape << ", " << bufferColShape << ", " << stride << ", " << stride << ">";
    } else if (opCode == Opcode::OP_SHMEM_SIGNAL) {
        int64_t value = npu::tile_fwk::AnyCast<int64_t>(opAttrs.at("Value")); 
        npu::tile_fwk::Distributed::AtomicType atomicType = npu::tile_fwk::AnyCast<npu::tile_fwk::Distributed::AtomicType>(opAttrs.at("AtomicType"));
        oss << "<" << std::to_string(value) << ", " << npu::tile_fwk::Distributed::AtomicTypeToString(atomicType) << ">";
    } else if (opCode == Opcode::OP_SHMEM_REDUCE) {
        std::string extraTemplateParam = npu::tile_fwk::AnyCast<std::string>(opAttrs.at("extraTemplateParam"));
        std::vector<int64_t> outShape = rawShape[ID0];
        oss << "<" << GetTemplateDType() << ", " << extraTemplateParam << ", " << outShape[0] << ", " << outShape[1] << ">";
    } else if (opAttrs.count("extraTemplateParam") != 0) {
        std::string extraTemplateParam = npu::tile_fwk::AnyCast<std::string>(opAttrs.at("extraTemplateParam"));
        oss << "<" << GetTemplateDType() << ", " << extraTemplateParam << ">";
    } else if (opCode != Opcode::OP_SHMEM_CLEAR_SIGNAL) {
        oss << "<" << GetTemplateDType() << ">";
    }
    return oss.str();
}

std::pair<std::string, std::string> CodeGenOpCloudNPU::GenOffsetsAndRawShapes(int32_t operandIndex, int32_t dim) const
{
    std::string offsets = GenGetParamMacroPacked(operandIndex, dim, PREFIX_STR_OFFSET)[0];
    std::string rawShapes = GenGetParamMacroPacked(operandIndex, dim, PREFIX_STR_RAW_SHAPE)[0];
    return {offsets, rawShapes};
}

std::string CodeGenOpCloudNPU::GenOffsetsAndRawShapes() const
{
    std::ostringstream oss;
    if ((opCode == Opcode::OP_SHMEM_PUT) || (opCode == Opcode::OP_SHMEM_GET)) {
        int32_t nonShmemDataIndex{0};
        int32_t shmemDataIndex{0};
        if (opCode == Opcode::OP_SHMEM_PUT) {
            nonShmemDataIndex = 2; // nonShmemData 是 operand 2
            shmemDataIndex = 3; // shmemData 是 operand 3
        } else {
            nonShmemDataIndex = 0; // nonShmemData 是 operand 0
            shmemDataIndex = 3; // shmemData 是 operand 3
        }
        constexpr int32_t nonShmemDataDim = 2;
        constexpr int32_t shmemDataDim = 4;
        auto [nonShmemDataOffsets, nonShmemDataRawShapes] = GenOffsetsAndRawShapes(nonShmemDataIndex, nonShmemDataDim);
        auto [shmemDataOffsets, shmemDataRawShapes] = GenOffsetsAndRawShapes(shmemDataIndex, shmemDataDim);
        oss << ", " << nonShmemDataOffsets << ", " << nonShmemDataRawShapes
            << ", " << shmemDataOffsets << ", " << shmemDataRawShapes;
    } else if (opCode == Opcode::OP_SHMEM_CLEAR_SIGNAL || opCode == Opcode::OP_SHMEM_SIGNAL) {
        int32_t shmemSignalIndex = (opCode == Opcode::OP_SHMEM_SIGNAL) ? 3 : 2; // 2、3指shmemSignal是第几个operand
        constexpr int32_t shmemSignalDim = 4;
        auto [shmemSignalOffsets, shmemSignalRawShapes] = GenOffsetsAndRawShapes(shmemSignalIndex, shmemSignalDim);
        oss << ", " << shmemSignalOffsets << ", " << shmemSignalRawShapes;
    } else if (opCode == Opcode::OP_SHMEM_REDUCE) {
        constexpr int32_t outIndex = 0;
        constexpr int32_t outDim = 2;
        auto [outOffsets, outRawShape] = GenOffsetsAndRawShapes(outIndex, outDim);
        oss << ", " << outOffsets << ", " << outRawShape;
    }
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenDistOp() const
{
    std::ostringstream oss;
    oss << tileOpName << GenTemplateParams() << "(" << GenParamsStr() << GenOffsetsAndRawShapes() <<
        ", hcclContext);\n";
    return oss.str();
}

} // namespace npu::tile_fwk
