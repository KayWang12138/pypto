/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file codegen_op_litenpu.cpp
 * \brief
 */

#include "codegen_op_litenpu.h"

namespace npu::tile_fwk {

// ensure funcType is static, and isUnderDynamicFunc is false
CodeGenOpLiteNPU::CodeGenOpLiteNPU(const CodeGenOpNPUCtx& ctx) : CodeGenOpNPU(ctx)
{
    InitOpsGenMap();
    forBlkMgr_ = ctx.forBlockManager;
    CodeGenOp::Init(ctx.operation);
    UpdateTileTensorInfo();
    UpdateLoopInfo();
}

TileTensor CodeGenOpLiteNPU::QueryTileTensorByIdx(int paramIdx) const
{
    const int tensorMagic = operandWithMagic[paramIdx];
    const int opMagic = originalOp.GetOpMagic();
    const TileTensor* tileTensor = nullptr;

    tileTensor = sm->QueryTileTensorByMagic(tensorMagic, opMagic);

    if (tileTensor != nullptr) {
        CODEGEN_LOGI("QueryTileTensorByIdx found: %s", tileTensor->ToString().c_str());
        return *tileTensor;
    }

    ASSERT(GenCodeErr::TENSOR_NOT_FOUND, false) << "TileTensor: paramIdx " << paramIdx << ", tensor magic "
                                                << tensorMagic << ", op magic " << opMagic << " is not found !!!";
    static TileTensor emptyTileTensor;
    return emptyTileTensor;
}

std::string CodeGenOpLiteNPU::GenGmParamVar(unsigned gmParamIdx) const
{
    return std::string("RealizedGM") + std::to_string(paramLocation[gmParamIdx]) + ".Addr";
}

std::string CodeGenOpLiteNPU::GenGMAddrExprWithOffset(const std::string& /* addrExpr */) const
{
    // gm offset of spilling workspace is calculated by pass, the value is saved in dim 0.
    int64_t gmOffset = 0;
    // gmOffset Default to 0 when the attribute is not set
    GetAttr(OpAttributeKey::workspaceBaseOffset, gmOffset);
    std::ostringstream oss;
    if (gmOffset == 0) {
        oss << "workspace";
    } else {
        oss << "((__gm__ uint8_t*)"
            << "workspace"
            << " + " << gmOffset << ")";
    }

    return oss.str();
}

TileTensor CodeGenOpLiteNPU::BuildTileTensor(int paramIdx, const std::string& usingType, const ShapeInLoop& shapeInLoop)
{
    bool isSpillToGm = operand[paramIdx] == SYMBOL_STACK_BASE;

    TileTensor tileTensor;
    tileTensor.isConstant = functionType == FunctionType::STATIC || isMainBlock;
    tileTensor.magic = operandWithMagic[paramIdx];
    tileTensor.shapeInLoop = shapeInLoop;

    if (tileTensor.isConstant) {
        tileTensor.dim = originShape[paramIdx].size();
    } else {
        tileTensor.dim = dynamicValidShape[paramIdx].size();
    }

    tileTensor.dtype = operandDtype[paramIdx];
    tileTensor.bufType = operandType[paramIdx];

    if (tileTensor.bufType == OperandType::BUF_DDR) {
        tileTensor.bufVar = isSpillToGm ? GenGMAddrExprWithOffset(GM_STACK_BASE) : GenGmParamVar(paramIdx);
    } else {
        tileTensor.bufVar = sm->QueryVarNameByTensorMagic(tileTensor.magic, true);
    }

    tileTensor.usingType = usingType;

    tileTensor.tensorName = sm->GenTensorName(tileTensor.bufType);
    UpdateTileTensorShapeAndStride(paramIdx, tileTensor, isSpillToGm, shapeInLoop);

    tileTensor.localBufOffset = offset[paramIdx];

    return tileTensor;
}

void CodeGenOpLiteNPU::UpdateTileTensorShapeAndStride(
    int paramIdx, TileTensor& tileTensor, [[maybe_unused]] bool isSpillToGm,
    [[maybe_unused]] const ShapeInLoop& shapeInLoop)
{
    auto newOriginShape = originShape[paramIdx];
    auto newRawShape = shapeInLoop.loopDepth > 0 ? shapeInLoop.rawShape : rawShape[paramIdx];
    auto newDynValidShape = shapeInLoop.loopDepth > 0 ? shapeInLoop.dynamicValidShape : dynamicValidShape[paramIdx];
    CODEGEN_LOGI(
        "newOriginShape is %s, newRawShape is %s, newDynValidShape is %s", IntVecToStr(newOriginShape).c_str(),
        IntVecToStr(newRawShape).c_str(), IntVecToStr(newDynValidShape).c_str());

    tileTensor.rawShape = newRawShape;

    // ---- static ----
    if (functionType == FunctionType::STATIC) {
        for (auto s : newOriginShape) {
            tileTensor.shape.emplace_back(std::to_string(s));
        }
        tileTensor.stride = BuildStride(newRawShape);
        return;
    }
}

std::vector<std::string> CodeGenOpLiteNPU::GetGmOffsetForTileTensor(unsigned gmIdx, bool isSpillingToGM) const
{
    int dim = static_cast<int>(rawShape[gmIdx].size());
    std::vector<std::string> gmOffsetExpr;
    if (isSpillingToGM) {
        return std::vector<std::string>(dim, "0");
    }

    if (offsetFromAttr[gmIdx][ID0].IsValid()) {
        return GenSymbolicArgument(offsetFromAttr[gmIdx]);
    }

    return GenGetParamMacroPacked(gmIdx, dim, PREFIX_STR_OFFSET);
}

} // namespace npu::tile_fwk
