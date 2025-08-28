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
 * \file symbolic_scalar_evaluate.h
 * \brief
 */
/*for flow Verify Tool */

#pragma once

#include "interface/operation/attribute.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {

constexpr int SIZE_TWO = 2;
constexpr int SIZE_THREE = 3;
constexpr int SIZE_FOUR = 4;

struct FunctionFrame;
class EvaluateSymbol {
public:
    EvaluateSymbol() {}

    void EvaluateDynParam(
        const std::map<std::string, DynParamInfo> &dynParamTable, const std::vector<SymbolicScalar> &linearArgList) {
        for (auto &paramInfo : dynParamTable) {
            std::string symbolName = paramInfo.first;
            int n = paramInfo.second.tensorIndex;
            (void)n;
            int base = paramInfo.second.tensorBaseAddrCoaIndex;
            int dim = paramInfo.second.dimSize;
            int idx = paramInfo.second.dimIndex;
            int argIndex = ((base) + 1) + 3 * (dim) + idx;

            symbolDict_[symbolName] = EvaluateSymbolicScalar(linearArgList[argIndex]);
        }
    }

    std::vector<int64_t> EvaluateValidShape(const std::vector<SymbolicScalar> &dynValidShape) {
        std::vector<int64_t> result;
        for (auto &shape : dynValidShape) {
            result.push_back(EvaluateSymbolicScalar(shape));
        }
        return result;
    }

    std::vector<int64_t> EvaluateOffset(const std::vector<int64_t> &offset, const std::vector<SymbolicScalar> &dynOffset) {
        std::vector<int64_t> resultOffset;
        if (dynOffset.size() != 0) {
            for (auto &off : dynOffset) {
                resultOffset.push_back(EvaluateSymbolicScalar(off));
            }
        } else {
            for (auto &off : offset) {
                resultOffset.push_back(off);
            }
        }
        return resultOffset;
    }

    bool RuntimeIsLoopBegin(ScalarImmediateType idx, ScalarImmediateType begin) { return idx == begin; }
    bool RuntimeIsLoopEnd(ScalarImmediateType idx, ScalarImmediateType end) { return idx >= end; }

    ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputShapeDimSize(
        const std::vector<ScalarImmediateType> &dataList) {
        ASSERT(dataList.size() == 1);
        auto inputIndex = dataList[0];
        auto input = GetInputDataViewList()[inputIndex];

        auto ret = input->GetShape().size();
        return ret;
    }

    ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputShapeDim(const std::vector<ScalarImmediateType> &dataList) {
        ASSERT(dataList.size() == SIZE_TWO);
        auto inputIndex = dataList[0];
        auto input = GetInputDataViewList()[inputIndex];
        auto n = dataList[1];

        auto ret = input->GetShape()[n];
        return ret;
    }

    ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputDataInt32Dim1(
        const std::vector<ScalarImmediateType> &dataList) {
        ASSERT(dataList.size() == SIZE_TWO);
        auto inputIndex = dataList[0];
        auto input = GetInputDataViewList()[inputIndex];
        auto off0 = dataList[1];
        ASSERT(input->GetShape().size() == 1);

        int index = off0;
        auto elt = input->GetElement(index);
        auto ret = static_cast<ScalarImmediateType>(elt.Cast<int64_t>());
        return ret;
    }

    ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputDataInt32Dim2(
        const std::vector<ScalarImmediateType> &dataList) {
        ASSERT(dataList.size() == SIZE_THREE);
        auto inputIndex = dataList[0];
        auto input = GetInputDataViewList()[inputIndex];
        auto off0 = dataList[1];
        auto off1 = dataList[2];
        ASSERT(input->GetShape().size() == SIZE_TWO);

        int index = off0 * input->GetShape()[1] + off1;
        auto elt = input->GetElement(index);
        auto ret = static_cast<ScalarImmediateType>(elt.Cast<int64_t>());
        return ret;
    }

    ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputDataInt32Dim3(
        const std::vector<ScalarImmediateType> &dataList) {
        ASSERT(dataList.size() == SIZE_FOUR);
        auto inputIndex = dataList[0];
        auto input = GetInputDataViewList()[inputIndex];
        auto off0 = dataList[1];
        auto off1 = dataList[2];
        auto off2 = dataList[3];
        ASSERT(input->GetShape().size() == SIZE_THREE);

        int index = off0 * input->GetShape()[1] * input->GetShape()[2] + off1 * input->GetShape()[2] + off2;
        auto elt = input->GetElement(index);
        auto ret = static_cast<ScalarImmediateType>(elt.Cast<int64_t>());
        return ret;
    }

    ScalarImmediateType EvaluateSymbolicCallRuntimeIsLoopBegin(const std::vector<ScalarImmediateType> &dataList) {
        ASSERT(dataList.size() == SIZE_TWO);
        auto ret = RuntimeIsLoopBegin(dataList[0], dataList[1]);
        return ret;
    }

    ScalarImmediateType EvaluateSymbolicCallRuntimeIsLoopEnd(const std::vector<ScalarImmediateType> &dataList) {
        ASSERT(dataList.size() == SIZE_TWO);
        auto ret = RuntimeIsLoopEnd(dataList[0], dataList[1]);
        return ret;
    }

    ScalarImmediateType EvaluateSymbolicCallRuntimeGetViewValidShapeDim(
        const std::vector<ScalarImmediateType> &dataList) {
        ASSERT(dataList.size() == SIZE_THREE);
        auto validshape = dataList[0];
        auto viewOffset = dataList[1];
        auto viewshape = dataList[2];
        validshape -= viewOffset;
        if (validshape > viewshape)
            validshape = viewshape;
        else if (validshape < 0)
            validshape = 0;
        return validshape;
    }

    ScalarImmediateType EvaluateSymbolicCall(
        const std::string &name, const std::vector<ScalarImmediateType> &dataList) {
        using CallEntry = ScalarImmediateType (EvaluateSymbol::*)(const std::vector<ScalarImmediateType> &dataList);
        static std::unordered_map<std::string, CallEntry> callEntryDict = {
            { "RUNTIME_GetInputShapeDimSize",  &EvaluateSymbol::EvaluateSymbolicCallRuntimeGetInputShapeDimSize},
            {     "RUNTIME_GetInputShapeDim",      &EvaluateSymbol::EvaluateSymbolicCallRuntimeGetInputShapeDim},
            {"RUNTIME_GetInputDataInt32Dim1", &EvaluateSymbol::EvaluateSymbolicCallRuntimeGetInputDataInt32Dim1},
            {"RUNTIME_GetInputDataInt32Dim2", &EvaluateSymbol::EvaluateSymbolicCallRuntimeGetInputDataInt32Dim2},
            {"RUNTIME_GetInputDataInt32Dim3", &EvaluateSymbol::EvaluateSymbolicCallRuntimeGetInputDataInt32Dim3},
            {          "RUNTIME_IsLoopBegin",           &EvaluateSymbol::EvaluateSymbolicCallRuntimeIsLoopBegin},
            {            "RUNTIME_IsLoopEnd",             &EvaluateSymbol::EvaluateSymbolicCallRuntimeIsLoopEnd},
            { "RUNTIME_GetViewValidShapeDim",  &EvaluateSymbol::EvaluateSymbolicCallRuntimeGetViewValidShapeDim},
        };
        ASSERT(callEntryDict.count(name)) << "Symbolic call not found: " << name;
        auto callEntry = callEntryDict[name];
        auto ret = (this->*callEntry)(dataList);
        return ret;
    }

    ScalarImmediateType EvaluateSymbolicScalar(const RawSymbolicScalarPtr &ss);
    ScalarImmediateType EvaluateSymbolicScalar(const SymbolicScalar &ss) { return EvaluateSymbolicScalar(ss.Raw()); }

    const std::unordered_map<std::string, ScalarImmediateType> &GetSymbolDict() const { return symbolDict_; }
    void UpdateSymbolDict(const std::string key, const ScalarImmediateType value) { symbolDict_[key] = value; }
    void SetSymbolDict(const std::unordered_map<std::string, ScalarImmediateType> &symbolDict) { symbolDict_ = symbolDict; }

    std::vector<std::shared_ptr<LogicalTensorData>> &GetInputDataViewList() { return inputDataViewList_; }
    void UpdateInputDataViewList(size_t index, const std::shared_ptr<LogicalTensorData> &inputDataView) {
        inputDataViewList_[index] = inputDataView;
    }
    void InitInputDataViewList(const std::vector<std::shared_ptr<LogicalTensorData>> &inputDataViewList) {
        inputDataViewList_ = inputDataViewList;
    }

private:
    std::unordered_map<std::string, ScalarImmediateType> symbolDict_;
    std::vector<std::shared_ptr<LogicalTensorData>> inputDataViewList_;
};

} // namespace npu::tile_fwk
