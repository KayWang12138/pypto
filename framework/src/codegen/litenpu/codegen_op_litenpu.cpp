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
 * \file codegen_op.cpp
 * \brief
 */

#include "codegen_op_litenpu.h"

#include <algorithm>

#include "codegen/codegen_common.h"
#include "securec.h"

namespace npu::tile_fwk {
CodeGenOpLiteNPU::CodeGenOpLiteNPU(const std::shared_ptr<SymbolManager> &symbolManager, FunctionType funcType,
    const std::map<int, int> &locToOffset, bool isUnderDynamicFunc, bool isMainBlk)
    : CodeGenOp(symbolManager, funcType, locToOffset, isUnderDynamicFunc, isMainBlk) {}

CodeGenOpLiteNPU::CodeGenOpLiteNPU(const CodeGenOpLiteNPUCtx &ctx)
    : CodeGenOpLiteNPU(ctx.symbolManager, ctx.topFunc.GetFunctionType(), ctx.locToOffset,
          ctx.topFunc.IsUnderDynamicFunction(), ctx.isMainBlock) {
    CodeGenOp::Init(ctx.operation);
    UpdateTileTensorInfo();
}

void CodeGenOpLiteNPU::AppendLocalBufferVarOffset(
    const std::map<unsigned, std::reference_wrapper<std::string>> &vars) const {
    for (auto &kv : vars) {
        auto operandIdx = kv.first;
        int64_t resOffset{0};

        std::vector<int64_t> varOffset = offset[operandIdx];
        if (varOffset.empty()) {
            continue;
        }

        std::vector<int64_t> varRawShape = rawShape[operandIdx];
        ASSERT(!varRawShape.empty()) << "varRawShape is empty!! operandIdx: " << operandIdx;
        ASSERT(varOffset.size() == varRawShape.size())
            << "varOffset " << IntVecToStr(varOffset) << ", size " << varOffset.size() << " vs varRawShape "
            << IntVecToStr(varRawShape) << ", size " << varRawShape.size()
            << " is not equal!! operandIdx: " << operandIdx;

        resOffset = CalcLinearOffset(varRawShape, varOffset);
        if (resOffset == 0) {
            continue;
        }

        std::string &var = kv.second.get();

        ASSERT(!var.empty()) << "operandIdx: " << operandIdx << ", var is empty !!";
        CODEGEN_LOGI("var: %s, varRawShape: %s, varOffset: %s, resOffset: %ld", var.c_str(),
            IntVecToStr(varRawShape).c_str(), IntVecToStr(varOffset).c_str(), static_cast<long>(resOffset));

        var.append(" + ").append(std::to_string(resOffset));
    }
}

std::string CodeGenOpLiteNPU::GenGmParamVar(unsigned gmParamIdx) const {
    return std::string("RealizedGM") + std::to_string(paramLocation[gmParamIdx]) + ".Addr";
}

// Used for parameter of GM shape and offset, e.g.
// GET_PARAM_RAWSHAPE_2(param, 19, 9), GET_PARAM_OFFSET_2(param, 19, 9)
// If dim is 2, the macro would be expanded into "shape0, shape1" which is implemented in aicore_runtime.h
std::vector<std::string> CodeGenOpLiteNPU::GenGetParamMacroPacked(
    unsigned gmParamIdx, int dim, const std::string &prefix) const {
    std::vector<std::string> paramExpr;
    std::ostringstream os;
    os << "GET_PARAM_" << prefix << "_" << dim << "(" << GM_TENSOR_PARAM_STR << ", " << GmTensorParamIdxInCallFunc
       << ", " << paramLocation[gmParamIdx] << ")";
    paramExpr.emplace_back(os.str());
    return paramExpr;
};

std::vector<std::string> CodeGenOpLiteNPU::GenParamIdxExprByIndex(
    unsigned gmParamIdx, int dim, const std::string &prefix) const {
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

std::vector<std::string> CodeGenOpLiteNPU::GenSymbolicArgument(const std::vector<SymbolicScalar> &exprList) const {
    std::vector<std::string> argList;
    for (auto &expr : exprList) {
        std::string exprStr = SymbolicExpressionTable::BuildExpression(expr);
        argList.push_back(exprStr);
    }
    return argList;
}

bool CodeGenOpLiteNPU::CombineAxis(std::vector<std::vector<int64_t> *> &shapes, bool secondLastAxis) const {
    size_t num;
    {
        auto iter = shapes.begin();
        num = (*iter)->size();
        for (; iter != shapes.end(); ++iter) {
            ASSERT(num == (*iter)->size()) << "shapes have to be the same!";
        }
    }

    if (secondLastAxis) {
        num--;
    }

    int64_t i = num - 1;
    for (; i >= 0; i--) {
        bool match = true;
        auto iter = shapes.begin();
        int s = (*iter)->at(i);
        for (; iter != shapes.end(); ++iter) {
            if (s != (*iter)->at(i)) {
                match = false;
                break;
            }
        }
        if (!match) {
            if (i >= static_cast<int64_t>(num - NUM2)) {
                return false;
            }
            break;
        }
    }

    i = std::max(i, static_cast<int64_t>(0));
    size_t numVec = shapes.size();
    std::vector<int> acc(numVec, 1);
    for (size_t j = i; j < num - 1; j++) {
        for (size_t k = 0; k < numVec; k++) {
            acc[k] *= shapes[k]->at(j);
            shapes[k]->at(j) = 1;
        }
    }
    for (size_t k = 0; k < numVec; k++) {
        shapes[k]->at(num - 1) = acc[k] * shapes[k]->at(num - 1);
    }

    for (size_t k = 0; k < numVec; k++) {
        // remove redundant ones so better chance for repeat
        shapes[k]->erase(shapes[k]->begin() + i, shapes[k]->begin() + (num - 1));
        // pad ones to preserve shape length
        shapes[k]->insert(shapes[k]->begin(), (num - 1) - i, 1);
    }

    return true;
}

TileTensor CodeGenOpLiteNPU::BuildTileTensor(int paramIdx, const std::string &usingType) {
    TileTensor tileTensor;
    tileTensor.magic = operandWithMagic[paramIdx];
    tileTensor.dim = dynamicValidShape[paramIdx].size();
    tileTensor.dtype = operandDtype[paramIdx];
    tileTensor.bufType = operandType[paramIdx];
    if (tileTensor.bufType == OperandType::BUF_DDR) {
        tileTensor.bufVar = GenGmParamVar(paramIdx);
    } else {
        tileTensor.bufVar = sm->QueryVarNameByTensorMagic(tileTensor.magic);
    }
    tileTensor.usingType = usingType;
    tileTensor.tensorName = BUFFER_TYPE_TO_PREFIX_LC.at(tileTensor.bufType) + "Tensor_" + std::to_string(tileTensor.magic);

    if (tileTensor.bufType == OperandType::BUF_DDR) {
        tileTensor.shape = GenGetParamMacroPacked(paramIdx, tileTensor.dim, PREFIX_STR_RAW_SHAPE);
        tileTensor.stride = GenGetParamMacroPacked(paramIdx, tileTensor.dim, PREFIX_STR_STRIDE);
    } else {
        for (const auto &s : dynamicValidShape[paramIdx]) {
            tileTensor.shape.emplace_back(s.Dump());
        }
        for (int i = 1; i < tileTensor.dim; ++i) {
            tileTensor.stride.emplace_back(std::to_string(rawShape[paramIdx][i]));
        }
    }

    // default last axis stride is 1, which means data is consecutive in memory
    tileTensor.stride.emplace_back("1");

    return tileTensor;
}

void CodeGenOpLiteNPU::UpdateTileTensorInfo() {
    // if (!isSupportLayout) {
    //     return;
    // }

    std::cout<<"UpdateTileTensorInfo for: "<<opCodeStr<<"\n";

    auto iter = SUPPORT_TILETENSOR_OPS.find(opCode);
    if (iter == SUPPORT_TILETENSOR_OPS.end()) {
        // ASSERT(iter != SUPPORT_TILETENSOR_OPS.end()) << "opCode: " << opCodeStr << " not support tile tensor!";
        std::cout<<"fail here: "<<opCodeStr<<"..\n";
        return;
    }

    tileOpName = iter->second; // update tileOpName from SUPPORT_TILETENSOR_OPS

    for (int i = 0; i < operandCnt; ++i) {
        TileTensorUsing tileTensorUsing{functionType == FunctionType::STATIC || isMainBlock, operandDtype[i],
            operandType[i], static_cast<int>(rawShape[i].size()), originShape[i], rawShape[i]};
        std::string usingType = sm->AddTileTensorUsing(tileTensorUsing);
        TileTensor tileTensor = BuildTileTensor(i, usingType);
        std::string tensorName = sm->AddTileTensor(tileTensor);
        std::cout<<"added tensorName: "<<tensorName<<"\n";
        tensorNames_[i] = tensorName;
        CODEGEN_LOGI(
            "AddTileTensor op idx: %d, result usingType: %s, tensorName: %s", i, usingType.c_str(), tensorName.c_str());
    }
}

std::string CodeGenOpLiteNPU::QueryTileTensorNameByIdx(int paramIdx) const {
    std::vector<TileTensor> res;
    // bool isInLoop = forBlkMgr_ != nullptr && forBlkMgr_->IsInLoop();
    // if (isInLoop) {
    //     res = sm->QueryTileTensorInLoopByMagic(operandWithMagic[paramIdx]);
    //     // some tensor in loop is reused same tensor out of loop
    //     if (res.empty()) {
    //         res = sm->QueryTileTensorByMagic(operandWithMagic[paramIdx]);
    //     }
    // } else {
        res = sm->QueryTileTensorByMagic(operandWithMagic[paramIdx]);
    // }

    // if (res.size() == 1) { //TODO: figure out why this case has 2...
        CODEGEN_LOGI("QueryTileTensorNameByIdx found: %s", res[0].tensorName.c_str());
        return res[0].tensorName;
    // }
    // CODEGEN_LOGI("isInLoop: %d, paramIdx is %d, tensor magic is %d, res size is %zu", isInLoop, paramIdx,
    //     operandWithMagic[paramIdx], res.size());
    CODEGEN_LOGI("paramIdx is %d, tensor magic is %d, res size is %zu", paramIdx,
        operandWithMagic[paramIdx], res.size());

    // auto targetRawShape =
    //     isInLoop ? std::vector{*(rawShape[paramIdx].rbegin() + 1), rawShape[paramIdx].back()} : rawShape[paramIdx];
    // CODEGEN_LOGI("isInLoop: %d,rawShape is %s, targetRawShape is %s", isInLoop, IntVecToStr(rawShape[paramIdx]).c_str(),
    //     IntVecToStr(targetRawShape).c_str());
    auto targetRawShape = rawShape[paramIdx];
    CODEGEN_LOGI("rawShape is %s, targetRawShape is %s", IntVecToStr(rawShape[paramIdx]).c_str(),
        IntVecToStr(targetRawShape).c_str());

    for (const auto &tileTensor : res) {
        // CODEGEN_LOGI("isInLoop: %d, tileTensor.shapeInLoop.rawShape is %s, tileTensor.rawShape is %s", isInLoop,
        //     IntVecToStr(tileTensor.shapeInLoop.rawShape).c_str(), IntVecToStr(tileTensor.rawShape).c_str());
        CODEGEN_LOGI("tileTensor.shapeInLoop.rawShape is %s, tileTensor.rawShape is %s",
            IntVecToStr(tileTensor.shapeInLoop.rawShape).c_str(), IntVecToStr(tileTensor.rawShape).c_str());
        // Currently only support additional comparison of rawShape
        if (tileTensor.rawShape == targetRawShape) {
            CODEGEN_LOGI("QueryTileTensorNameByIdx found: %s", tileTensor.tensorName.c_str());
            return tileTensor.tensorName;
        }
    }

    ASSERT(false) << "paramIdx " << paramIdx << ", tensor magic " << operandWithMagic[paramIdx]
                  << " is not found !!! res size is " << res.size();
    return "";
}

} // namespace npu::tile_fwk
