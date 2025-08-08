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

#include "codegen_op_cloudnpu.h"

#include <algorithm>

#include "codegen/codegen_common.h"
#include "securec.h"

namespace npu::tile_fwk {
void CodeGenOpCloudNPU::AppendLocalBufferVarOffset(
    const std::vector<std::string *> &vars, const std::vector<unsigned> &operandIdxes) const {
    ASSERT(vars.size() == operandIdxes.size())
        << "vars size vs operandIdxes is not equal" << vars.size() << " vs. " << operandIdxes.size();

    for (size_t i = 0; i < vars.size(); ++i) {
        int resOffset{0};

        std::vector varOffset = offset[operandIdxes[i]];
        if (varOffset.empty()) {
            continue;
        }

        std::vector varRawShape = rawShape[operandIdxes[i]];
        ASSERT(!varRawShape.empty()) << "varRawShape is empty!!";
        ASSERT(varOffset.size() == varRawShape.size())
            << "varOffset " << IntVecToStr(varOffset) << ", size " << varOffset.size() << " vs varRawShape "
            << IntVecToStr(varRawShape) << ", size " << varRawShape.size() << " is not equal!!";

        int base = 1;
        for (int j = varOffset.size() - 1; j >= 0; j--) {
            resOffset += varOffset[j] * base;
            base *= varRawShape[j];
        }

        if (resOffset == 0) {
            continue;
        }

        ALOG_DEBUG_F(" vars[%d]: %s", i, (*vars[i]).c_str());
        ALOG_DEBUG_F(" varRawShape: %s", IntVecToStr(varRawShape).c_str());
        ALOG_DEBUG_F(" varOffset: %s", IntVecToStr(varOffset).c_str());
        ALOG_DEBUG_F(" resOffset: %d", resOffset);
        ASSERT(vars[i]) << "var[" << i << "] is null!!";
        *vars[i] += " + " + std::to_string(resOffset);
    }
}

std::string CodeGenOpCloudNPU::GenGmParamVar(unsigned gmParamIdx) const {
    if (isUnderDynamicFunction) {
        std::ostringstream os;
        os << "GET_PARAM_ADDR(" << GM_TENSOR_PARAM_STR << ", " << GmTensorParamIdxInCallFunc << ", "
           << paramLocation[gmParamIdx] << ")";
        return os.str();
    }

    auto paramLoc = paramLocation[gmParamIdx];
    auto iter = paramLocToParamListOffset.find(paramLoc);
    ASSERT(iter != paramLocToParamListOffset.end())
        << "paramLoc " << paramLoc << " can not be found in paramLocToParamListOffset";
    std::string gmVar = "((" + GM_PARAM_TYPE_FOR_STATIC + "*)(param) + " + std::to_string(iter->second) + ")->Addr";
    return gmVar;
}

// Used for parameter of GM shape and offset, e.g.
// GET_PARAM_RAWSHAPE_2(param, 19, 9), GET_PARAM_OFFSET_2(param, 19, 9)
// If dim is 2, the macro would be expanded into "shape0, shape1" which is implemented in aicore_runtime.h
std::vector<std::string> CodeGenOpCloudNPU::GenGetParamMacroPacked(
    unsigned gmParamIdx, int dim, const std::string &prefix) const {
    std::vector<std::string> paramExpr;
    std::ostringstream os;
    os << "GET_PARAM_" << prefix << "_" << dim << "(" << GM_TENSOR_PARAM_STR << ", " << GmTensorParamIdxInCallFunc
       << ", " << paramLocation[gmParamIdx] << ")";
    paramExpr.emplace_back(os.str());
    return paramExpr;
};

std::vector<std::string> CodeGenOpCloudNPU::GenParamIdxExprByIndex(
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

std::vector<std::string> CodeGenOpCloudNPU::GenSymbolicArgument(const std::vector<SymbolicScalar> &exprList) const {
    std::vector<std::string> argList;
    for (auto &expr : exprList) {
        std::string exprStr = SymbolicExpressionTable::BuildExpression(expr);
        argList.push_back(exprStr);
    }
    return argList;
}

bool CodeGenOpCloudNPU::CombineAxis(std::vector<std::vector<int> *> &shapes, bool secondLastAxis) const {
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

} // namespace npu::tile_fwk
