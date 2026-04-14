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
 * \file codegen_op.cpp
 * \brief
 */

#include "codegen_op_npu.h"

#include <algorithm>

#include "codegen/codegen_common.h"
#include "codegen/utils/codegen_utils.h"
#include "securec.h"

namespace npu::tile_fwk {

std::unordered_map<Opcode, std::set<int>> SKIP_PROC_PRARAM_IDX_IN_LOOP = {
    // scene: reduce for last axis
    // Parameter at 1st index (after numbering by CodeGenOp::Init) is used as temp buffer which is reused in loop body.
    {Opcode::OP_ROWSUM_SINGLE, {ID1}},
    {Opcode::OP_ROWMAX_SINGLE, {ID1}},
    {Opcode::OP_ROWMIN_SINGLE, {ID1}},
};

CodeGenOpNPU::CodeGenOpNPU(const CodeGenOpNPUCtx& ctx) : CodeGenOp(ctx) {}

void CodeGenOpNPU::InitOpsGenMap()
{
    InitScalaOpsMap();
    InitMTEOpsMap();
    InitVecOpsMap();
    InitCubeOpsMap();
    InitDistOpsMap();
    InitPerfOpsMap();
    InitAICPUOpsMap();
}

void CodeGenOpNPU::InitScalaOpsMap() { opsGenMap_.insert(syncOps_.cbegin(), syncOps_.cend()); }

void CodeGenOpNPU::InitMTEOpsMap() { opsGenMap_.insert(mteFixPipeOps_.cbegin(), mteFixPipeOps_.cend()); }

void CodeGenOpNPU::InitVecOpsMap()
{
    opsGenMap_.insert(unaryOps_.cbegin(), unaryOps_.cend());
    opsGenMap_.insert(binaryOps_.cbegin(), binaryOps_.cend());
    opsGenMap_.insert(compositeOps_.cbegin(), compositeOps_.cend());
    opsGenMap_.insert(sortOps_.cbegin(), sortOps_.cend());
    opsGenMap_.insert(gatherScatterOps_.cbegin(), gatherScatterOps_.cend());
    opsGenMap_.insert(normalVecOps_.cbegin(), normalVecOps_.cend());
}

void CodeGenOpNPU::InitCubeOpsMap() { opsGenMap_.insert(cubeOps_.cbegin(), cubeOps_.cend()); }

void CodeGenOpNPU::InitDistOpsMap() { opsGenMap_.insert(distributeOps_.cbegin(), distributeOps_.cend()); }

void CodeGenOpNPU::InitPerfOpsMap() { opsGenMap_.insert(perfOps_.cbegin(), perfOps_.cend()); }

void CodeGenOpNPU::InitAICPUOpsMap() { opsGenMap_.insert(aicpuOps_.cbegin(), aicpuOps_.cend()); }

void CodeGenOpNPU::AppendLocalBufferVarOffset(const std::map<unsigned, std::reference_wrapper<std::string>>& vars) const
{
    for (auto& kv : vars) {
        auto operandIdx = kv.first;
        int64_t resOffset{0};

        std::vector<int64_t> varOffset = offset[operandIdx];
        if (varOffset.empty()) {
            continue;
        }

        std::vector<int64_t> varRawShape = rawShape[operandIdx];
        ASSERT(GenCodeErr::TENSOR_SHAPE_INVALID, !varRawShape.empty())
            << "varRawShape is empty!! operandIdx: " << operandIdx;
        ASSERT(GenCodeErr::TENSOR_SHAPE_MISMATCHED, varOffset.size() == varRawShape.size())
            << "varOffset " << IntVecToStr(varOffset) << ", size " << varOffset.size() << " vs varRawShape "
            << IntVecToStr(varRawShape) << ", size " << varRawShape.size()
            << " is not equal!! operandIdx: " << operandIdx;

        resOffset = CalcLinearOffset(varRawShape, varOffset);
        if (resOffset == 0) {
            continue;
        }

        std::string& var = kv.second.get();

        ASSERT(GenCodeErr::SYMBOL_NOT_FOUND, !var.empty()) << "operandIdx: " << operandIdx << ", var is empty !!";
        CODEGEN_LOGI(
            "var: %s, varRawShape: %s, varOffset: %s, resOffset: %ld", var.c_str(), IntVecToStr(varRawShape).c_str(),
            IntVecToStr(varOffset).c_str(), static_cast<long>(resOffset));

        var.append(" + ").append(std::to_string(resOffset));
    }
}

SymbolicScalar CodeGenOpNPU::GetOperandStartOffset(int operandIdx) const
{
    std::vector varOffset = offset[operandIdx];
    if (varOffset.empty()) {
        return 0;
    }

    const auto& dynOffset = dynamicOffset[operandIdx];
    if (!dynOffset.empty()) {
        std::vector varRawShape = rawShape[operandIdx]; // 内部应该不能出现dynRawShape，所以这里用立即数即可
        ASSERT(GenCodeErr::TENSOR_SHAPE_INVALID, !varRawShape.empty())
            << "varRawShape is empty!! operandIdx: " << operandIdx;
        ASSERT(GenCodeErr::TENSOR_SHAPE_MISMATCHED, dynOffset.size() == varRawShape.size())
            << "dynOffset " << SymbolicVecToStr(dynOffset) << ", size " << dynOffset.size() << " vs varRawShape "
            << IntVecToStr(varRawShape) << ", size " << varRawShape.size()
            << " is not equal!! operandIdx: " << operandIdx;

        SymbolicScalar resOffset = 0;
        for (size_t i = 0; i < dynOffset.size(); i++) {
            resOffset = resOffset * varRawShape[i];
            resOffset = resOffset + dynOffset[i];
        }

        ASSERT(OperErr::OPERAND_COUNT_EXCEEDED, operandIdx < operandCnt)
            << "operandIdx: " << operandIdx << ", operandCnt: " << operandCnt;
        CODEGEN_LOGD(" varRawShape: %s", IntVecToStr(varRawShape).c_str());
        CODEGEN_LOGD(" varOffset: %s", SymbolicVecToStr(dynOffset).c_str());
        CODEGEN_LOGD(" resOffset: %s", resOffset.Dump().c_str());
        if (resOffset.ConcreteValid()) {
            return resOffset.Concrete();
        }
        return SymbolicExpressionTable::BuildExpression(resOffset);
    }

    std::vector varRawShape = rawShape[operandIdx];
    ASSERT(GenCodeErr::TENSOR_SHAPE_INVALID, !varRawShape.empty())
        << "varRawShape is empty!! operandIdx: " << operandIdx;
    ASSERT(GenCodeErr::TENSOR_SHAPE_MISMATCHED, varOffset.size() == varRawShape.size())
        << "varOffset " << IntVecToStr(varOffset) << ", size " << varOffset.size() << " vs varRawShape "
        << IntVecToStr(varRawShape) << ", size " << varRawShape.size() << " is not equal!! operandIdx: " << operandIdx;

    int64_t resOffset = CalcLinearOffset(varRawShape, varOffset);
    if (resOffset == 0) {
        return 0;
    }

    ASSERT(OperErr::OPERAND_COUNT_EXCEEDED, operandIdx < operandCnt)
        << "operandIdx: " << operandIdx << ", operandCnt: " << operandCnt;
    CODEGEN_LOGD(" varRawShape: %s", IntVecToStr(varRawShape).c_str());
    CODEGEN_LOGD(" varOffset: %s", IntVecToStr(varOffset).c_str());
    CODEGEN_LOGD(" resOffset: %ld", static_cast<long>(resOffset));
    return resOffset;
}

std::string CodeGenOpNPU::GenGmParamVar(unsigned gmParamIdx) const
{
    if (isUnderDynamicFunction) {
        std::ostringstream os;
        os << "GET_PARAM_ADDR(" << GM_TENSOR_PARAM_STR << ", " << GmTensorParamIdxInCallFunc << ", "
           << paramLocation[gmParamIdx] << ")";
        return os.str();
    }

    auto paramLoc = paramLocation[gmParamIdx];
    auto iter = paramLocToParamListOffset.find(paramLoc);
    ASSERT(GenCodeErr::PARAM_IDX_INVALID, iter != paramLocToParamListOffset.end())
        << "paramLoc " << paramLoc << " can not be found in paramLocToParamListOffset!! gmParamIdx: " << gmParamIdx;
    std::string gmVar = "((" + GM_PARAM_TYPE_FOR_STATIC + "*)(param) + " + std::to_string(iter->second) + ")->Addr";
    return gmVar;
}

// Used for parameter of GM shape and offset, e.g.
// GET_PARAM_RAWSHAPE_2(param, 19, 9), GET_PARAM_OFFSET_2(param, 19, 9)
// If dim is 2, the macro would be expanded into "shape0, shape1" which is implemented in aicore_runtime.h
std::vector<std::string> CodeGenOpNPU::GenGetParamMacroPacked(
    unsigned gmParamIdx, int dim, const std::string& prefix) const
{
    std::vector<std::string> paramExpr;
    std::ostringstream os;
    os << "GET_PARAM_" << prefix << "_" << dim << "(" << GM_TENSOR_PARAM_STR << ", " << GmTensorParamIdxInCallFunc
       << ", " << paramLocation[gmParamIdx] << ")";
    paramExpr.emplace_back(os.str());
    return paramExpr;
};

std::vector<std::string> CodeGenOpNPU::GenParamIdxExprByIndex(
    unsigned gmParamIdx, int dim, const std::string& prefix) const
{
    std::vector<std::string> paramExpr;
    std::ostringstream os;
    for (int index = 0; index < dim; ++index) {
        os << "GET_PARAM_" << prefix << "_BY_IDX(" << GM_TENSOR_PARAM_STR << ", " << GmTensorParamIdxInCallFunc << ", "
           << paramLocation[gmParamIdx] << ", " << dim << ", " << index << ")";
        paramExpr.emplace_back(os.str());
        os.str("");
    }
    return paramExpr;
}

std::vector<std::string> CodeGenOpNPU::GenSymbolicArgument(const std::vector<SymbolicScalar>& exprList) const
{
    std::vector<std::string> argList;
    for (auto& expr : exprList) {
        std::string exprStr = SymbolicExpressionTable::BuildExpression(expr);
        argList.push_back(exprStr);
    }
    return argList;
}

std::vector<std::string> CodeGenOpNPU::BuildStride(const std::vector<int64_t>& input)
{
    if (input.empty()) {
        return {};
    }

    std::vector<std::string> res(input.size(), "1");
    int64_t base = 1;
    for (int i = input.size() - 2; i >= 0; --i) {
        base *= input[i + 1];
        res[i] = std::to_string(base);
    }

    return res;
}

void CodeGenOpNPU::UpdateTileTensorShapeAndStride(
    int paramIdx, TileTensor& tileTensor, bool isSpillToGm, const ShapeInLoop& shapeInLoop)
{
    auto newOriginShape = shapeInLoop.loopDepth > 0 ? shapeInLoop.originShape : originShape[paramIdx];
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

    // ---- dynamic ----
    // gm tensor
    if (tileTensor.bufType == OperandType::BUF_DDR) {
        if (isSpillToGm) {
            for (auto s : shapeFromAttr[paramIdx]) {
                tileTensor.shape.emplace_back(std::to_string(s));
            }
            tileTensor.stride = BuildStride(shapeFromAttr[paramIdx]);
        } else {
            tileTensor.shape = GenGetParamMacroPacked(paramIdx, tileTensor.dim, PREFIX_STR_RAW_SHAPE);
            tileTensor.stride = GenGetParamMacroPacked(paramIdx, tileTensor.dim, PREFIX_STR_STRIDE);
        }
        return;
    }

    // local tensor

    for (const auto& s : newDynValidShape) {
        tileTensor.shape.emplace_back(SymbolicExpressionTable::BuildExpression(s));
    }
    tileTensor.stride = BuildStride(newRawShape);
}

TileTensor CodeGenOpNPU::BuildTileTensor(int paramIdx, const std::string& usingType, const ShapeInLoop& shapeInLoop)
{
    bool isSpillToGm = operand[paramIdx] == SYMBOL_STACK_BASE;

    TileTensor tileTensor;
    tileTensor.isConstant = functionType == FunctionType::STATIC || isMainBlock;
    tileTensor.magic = operandWithMagic[paramIdx];
    tileTensor.shapeInLoop = shapeInLoop;

    if (tileTensor.isConstant) {
        tileTensor.dim = shapeInLoop.loopDepth > 0 ? shapeInLoop.originShape.size() : originShape[paramIdx].size();
    } else {
        tileTensor.dim =
            shapeInLoop.loopDepth > 0 ? shapeInLoop.dynamicValidShape.size() : dynamicValidShape[paramIdx].size();
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
    if (shapeInLoop.loopDepth != 0) {
        std::string tensorName = tensorNames_[paramIdx];
        if (!tensorName.empty()) {
            tensorName.append("_low").append(std::to_string(tileTensor.dim)).append("DimInLoop");
            tileTensor.tensorName = tensorName;
        }
    }
    UpdateTileTensorShapeAndStride(paramIdx, tileTensor, isSpillToGm, shapeInLoop);

    tileTensor.localBufOffset = offset[paramIdx];

    return tileTensor;
}

void CodeGenOpNPU::UpdateTileTensorInfo()
{
    if (!isSupportLayout) {
        return;
    }

    auto iter = SUPPORT_TILETENSOR_OPS.find(opCode);
    if (iter == SUPPORT_TILETENSOR_OPS.end()) {
        ASSERT(GenCodeErr::OP_CODE_UNSUPPORTED, iter != SUPPORT_TILETENSOR_OPS.end())
            << "opCode: " << opCodeStr << " not support tile tensor!";
        return;
    }

    tileOpName = iter->second; // update tileOpName from SUPPORT_TILETENSOR_OPS

    for (int i = 0; i < operandCnt; ++i) {
        TileTensorUsing tileTensorUsing{
            functionType == FunctionType::STATIC || isMainBlock,
            operandDtype[i],
            operandType[i],
            static_cast<int>(rawShape[i].size()),
            originShape[i],
            rawShape[i]};
        std::string usingType = sm->AddTileTensorUsing(tileTensorUsing);
        TileTensor tileTensor = BuildTileTensor(i, usingType);
        std::string tensorName = sm->AddTileTensor(originalOp.GetOpMagic(), tileTensor);
        tensorNames_[i] = tensorName;
        CODEGEN_LOGI(
            "AddTileTensor op idx: %d, result usingType: %s, tensorName: %s", i, usingType.c_str(), tensorName.c_str());
    }
}

bool CodeGenOpNPU::ShouldSkipProcInLoop(int paramIdx)
{
    auto iter = SKIP_PROC_PRARAM_IDX_IN_LOOP.find(opCode);
    if (iter != SKIP_PROC_PRARAM_IDX_IN_LOOP.end() && iter->second.find(paramIdx) != iter->second.end()) {
        return true;
    }
    // cast with tempbuf which index is 1
    if (opCode == Opcode::OP_CAST && originalOp.oOperand.size() == NUM2 && paramIdx == 1) {
        return true;
    }
    return false;
}

std::vector<SymbolicScalar> CodeGenOpNPU::GetLoopAxes()
{
    std::vector<SymbolicScalar> loopAxes;
    GetAttr(OpAttributeKey::loopAxes, loopAxes);

    if (!isMainBlock) {
        return loopAxes;
    }
    // use dst shape as loop axes in main block
    std::vector<SymbolicScalar> newLoopAxes;
    for (size_t i = 0; i < loopAxes.size(); ++i) {
        SymbolicScalar axis = isDynamicFunction ? dynamicValidShape[0][i] : SymbolicScalar(originShape[0][i]);
        newLoopAxes.emplace_back(axis);
    }

    return newLoopAxes;
}

void CodeGenOpNPU::UpdateLoopInfo()
{
    if (SUPPORT_VF_FUSE_OPS.find(opCode) == SUPPORT_VF_FUSE_OPS.end()) {
        return;
    }

    std::vector<SymbolicScalar> loopAxes = GetLoopAxes();
    if (loopAxes.empty()) {
        return;
    }

    bool isLoopStart{false};
    if (GetAttr(OpAttributeKey::loopGroupStart, isLoopStart) && isLoopStart) {
        forBlkMgr_->LoopStart();
        forBlkMgr_->UpdateAxesList(loopAxes);
    }

    // Add TileTensor info in loop
    CODEGEN_LOGI("opCode %s has loopAxes: %s", opCodeStr.c_str(), IntVecToStr(loopAxes).c_str());
    size_t loopDepth = loopAxes.size();
    for (int i = 0; i < operandCnt; ++i) {
        if (ShouldSkipProcInLoop(i)) {
            continue;
        }
        ShapeInLoop shapeInLoop = BuildShapeInLoop(i, loopDepth);
        CODEGEN_LOGI(
            "shapeInLoop: loopDepth is %zu newOriginShape is %s, newRawShape is %s, newDynValidShape is %s", loopDepth,
            IntVecToStr(shapeInLoop.originShape).c_str(), IntVecToStr(shapeInLoop.rawShape).c_str(),
            IntVecToStr(shapeInLoop.dynamicValidShape).c_str());
        TileTensorUsing tileTensorUsing{
            functionType == FunctionType::STATIC || isMainBlock, operandDtype[i],         operandType[i],
            static_cast<int>(shapeInLoop.rawShape.size()),       shapeInLoop.originShape, shapeInLoop.rawShape};
        std::string usingType = sm->AddTileTensorUsing(tileTensorUsing);
        TileTensor tileTensor = BuildTileTensor(i, usingType, shapeInLoop);
        forBlkMgr_->AddTensorInLoopBody(tensorNames_[i], tileTensor, originalOp.GetOpMagic(), opCode);
    }
}

// Get last 2 dim of shape
ShapeInLoop CodeGenOpNPU::BuildShapeInLoop(int paramIdx, size_t loopDepth)
{
    auto newOriginShape = GetShapeInLoop(originShape[paramIdx]);
    auto newRawShape = GetShapeInLoop(rawShape[paramIdx]);
    auto newDynValidShape = GetShapeInLoop<SymbolicScalar>(dynamicValidShape[paramIdx]);
    return {loopDepth, newOriginShape, newRawShape, newDynValidShape};
}

std::string CodeGenOpNPU::PrintCoord(size_t dim, const std::string& coord) const
{
    std::string ret = COORD;
    ret.append(std::to_string(dim)).append(DIM).append(coord);
    return ret;
}

std::pair<std::string, std::string> CodeGenOpNPU::PrintDstSrcCoordFromAttr() const
{
    std::vector<std::string> dstOffset;
    for (const auto& tmpOffset : offsetFromAttr[ToUnderlying(MISOIdx::DST_IDX)]) {
        dstOffset.emplace_back(SymbolicExpressionTable::BuildExpression(tmpOffset));
    }
    std::vector<std::string> srcOffset;
    for (const auto& tmpOffset : offsetFromAttr[ToUnderlying(MISOIdx::SRC0_IDX)]) {
        srcOffset.emplace_back(SymbolicExpressionTable::BuildExpression(tmpOffset));
    }
    std::string coordCpDst = WrapParamByParentheses(dstOffset);
    std::string coordDst = PrintCoord(rawShape[ToUnderlying(MISOIdx::DST_IDX)].size(), coordCpDst);
    std::string coordCpSrc = WrapParamByParentheses(srcOffset);
    std::string coordSrc = PrintCoord(rawShape[ToUnderlying(MISOIdx::SRC0_IDX)].size(), coordCpSrc);
    return {coordDst, coordSrc};
}

TileTensor CodeGenOpNPU::QueryTileTensorByIdx(int paramIdx) const
{
    const int tensorMagic = operandWithMagic[paramIdx];
    const int opMagic = originalOp.GetOpMagic();
    const TileTensor* tileTensor = nullptr;
    bool isInLoop = forBlkMgr_ != nullptr && forBlkMgr_->IsInLoop();
    if (isInLoop) {
        tileTensor = sm->QueryTileTensorInLoopByMagic(tensorMagic, opMagic);
        // some tensor in loop is reused same tensor out of loop
        if (tileTensor == nullptr) {
            tileTensor = sm->QueryTileTensorByMagic(tensorMagic, opMagic);
        }
    } else {
        tileTensor = sm->QueryTileTensorByMagic(tensorMagic, opMagic);
    }

    if (tileTensor != nullptr) {
        CODEGEN_LOGI("QueryTileTensorByIdx found: %s", tileTensor->ToString().c_str());
        return *tileTensor;
    }

    ASSERT(GenCodeErr::TENSOR_NOT_FOUND, false)
        << "TileTensor: paramIdx " << paramIdx << ", tensor magic " << tensorMagic << ", op magic " << opMagic
        << ", isInLoop " << isInLoop << " is not found !!!";
    static TileTensor emptyTileTensor;
    return emptyTileTensor;
}

std::string CodeGenOpNPU::InsertOpComment(std::string& tileOpSourceCode) const
{
    std::ostringstream os;

    if (config::GetDebugOption<int64_t>(CFG_COMPILE_DBEUG_MODE) == CFG_DEBUG_ALL) {
        tileOpSourceCode.erase(tileOpSourceCode.find_last_not_of(" \n\r\t") + 1);
        // Add comment after op. e.g. [opmagic:10016]
        os << " // [opMagic:" << originalOp.GetOpMagic() << "]\n";
        tileOpSourceCode.append(os.str());
        os.str("");
    }

    // Add comment before op
    for (auto& c : originalOp.GetCommentList()) {
        os << "/*" << c << "*/\n";
    }
    os << tileOpSourceCode;
    return os.str();
}

std::string CodeGenOpNPU::QueryTileTensorNameByIdx(int paramIdx) const
{
    const TileTensor& tileTensor = QueryTileTensorByIdx(paramIdx);
    return tileTensor.tensorName;
}

std::string CodeGenOpNPU::QueryTileTensorTypeByIdx(int paramIdx) const
{
    const TileTensor& tileTensor = QueryTileTensorByIdx(paramIdx);
    return tileTensor.usingType;
}

std::string CodeGenOpNPU::GenOpCode() const
{
    std::string tileOpSourceCode;
    auto iter = opsGenMap_.find(opCode);
    if (iter != opsGenMap_.end()) {
        tileOpSourceCode = iter->second();
    } else {
        // To aid in testing, do not use ASSERT.
        return std::string{"CAN NOT HANDLE OP: " + opCodeStr};
    }

    std::string ret = InsertOpComment(tileOpSourceCode);

    if (forBlkMgr_ == nullptr || !forBlkMgr_->IsInLoop()) {
        CODEGEN_LOGI_FULL("op codegen result: \n, %s", ret.c_str());
        return ret;
    }

    forBlkMgr_->AddOpInLoopBody(ret);

    bool isLoopEnd{false};
    GetAttr(OpAttributeKey::loopGroupEnd, isLoopEnd);
    if (!isLoopEnd) {
        return "";
    }

    ret = forBlkMgr_->Print();
    forBlkMgr_->OutLoop();
    CODEGEN_LOGI_FULL("op codegen result: \n, %s", ret.c_str());
    return ret;
}

std::string CodeGenOpNPU::GetLastUse() const
{
    if (!opAttrs.count(OpAttributeKey::lastUse)) {
        return "";
    }
    std::vector<int64_t> val = GetVectorIntAttribute(OpAttributeKey::lastUse);
    int valSize = val.size();
    ASSERT(OperErr::ATTRIBUTE_INVALID, valSize != 0) << "GetLastUse error!!!";
    std::ostringstream oss;
    oss << "LastUse" << valSize << "Dim";
    oss << WrapParamByAngleBrackets(val);
    return oss.str();
}

} // namespace npu::tile_fwk
