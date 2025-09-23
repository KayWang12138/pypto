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
 * \file dyn_attr_to_static.h
 * \brief
 */

#ifndef PASS_DYNATTR_TO_STATIC_H_
#define PASS_DYNATTR_TO_STATIC_H_

#include <vector>
#include <unordered_map>
#include <regex>
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation_impl.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/utils/log.h"

namespace npu {
namespace tile_fwk {

enum class CoaType {
    PARAM_OFFSET,
    PARAM_VALID_SHAPE,
    PARAM,
    INVALID
};

const std::string COA_PREFIX = "RUNTIME_COA_GET_PARAM";

static const SymbolicScalar MAYBE_CONST_COA_GetOffset = AddRuntimeCoaPrefix("GET_PARAM_OFFSET_MAYBE_CONST");
static const SymbolicScalar MAYBE_CONST_COA_GetValidShape = AddRuntimeCoaPrefix("GET_PARAM_VALID_SHAPE_MAYBE_CONST");
static const SymbolicScalar MAYBE_CONST_COA_GetParam = AddRuntimeCoaPrefix("GET_PARAM_MAYBE_CONST");

Status SToIWrapper(const std::string str, int& result);

static const std::regex paramOffsetPattern("RUNTIME_COA_GET_PARAM_OFFSET\\((\\d+), (\\d+), (\\d+)\\)");
static const std::regex paramShapePattern("RUNTIME_COA_GET_PARAM_VALID_SHAPE\\((\\d+), (\\d+), (\\d+)\\)");
static const std::regex paramPattern("RUNTIME_COA_GET_PARAM\\((\\d+)\\)");

constexpr int OFFSET_INDEX_ORDER = 0;
constexpr int SHAPE_INDEX_ORDER = 1;
constexpr int RAWSHAPE_INDEX_ORDER = 2;
constexpr int VALID_SHAPE_INDEX_ORDER = 3;
constexpr int INPUT_PARAM_POS_ONE = 1;
constexpr int INPUT_PARAM_POS_TWO = 2;
constexpr int INPUT_PARAM_POS_THREE = 3;

struct CoaInfo {
    CoaType macroType = CoaType::INVALID;
    int dim = -1;
    int base = -1;
    int idx = -1;

    Status SToIParamShapeAndOffset(const std::smatch &match) {
        if (SToIWrapper(match[INPUT_PARAM_POS_ONE].str(), dim) != SUCCESS) {
            ALOG_ERROR_F("Failed to convert dim.");
            return FAILED;
        }
        if (SToIWrapper(match[INPUT_PARAM_POS_TWO].str(), base) != SUCCESS) {
            ALOG_ERROR_F("Failed to convert base.");
            return FAILED;
        }
        if (SToIWrapper(match[INPUT_PARAM_POS_THREE].str(), idx) != SUCCESS) {
            ALOG_ERROR_F("Failed to convert idx.");
            return FAILED;
        }
        return SUCCESS;
    }
    
    Status ParseCoaString(const std::string &coaExpr) {
        std::smatch match;
        if (std::regex_search(coaExpr, match, paramOffsetPattern)) {
            macroType = CoaType::PARAM_OFFSET;
            if (SToIParamShapeAndOffset(match) != SUCCESS) {
                ALOG_ERROR_F("ParseCoaString failed to convert indices,"
                    "CoaType::PARAM_OFFSET, input coaExpr %s.", coaExpr.c_str());
                return FAILED;
            }
        } else if (std::regex_search(coaExpr, match, paramShapePattern)) {
            macroType = CoaType::PARAM_VALID_SHAPE;
            if (SToIParamShapeAndOffset(match) != SUCCESS) {
                ALOG_ERROR_F("ParseCoaString failed to convert indices,"
                    "CoaType::PARAM_VALID_SHAPE, input coaExpr %s.", coaExpr.c_str());
                return FAILED;
            }
        } else if (std::regex_search(coaExpr, match, paramPattern)) {
            macroType = CoaType::PARAM;
            if (SToIWrapper(match[INPUT_PARAM_POS_ONE].str(), idx) != SUCCESS) {
                ALOG_ERROR_F("ParseCoaString failed to convert indices,"
                    "CoaType::PARAM, input coaExpr %s.", coaExpr.c_str());
                return FAILED;
            }
        } else {
            ALOG_ERROR_F("ParseCoaString input coaExpr %s is not recognized.", coaExpr.c_str());
            return FAILED;
        }
        return SUCCESS;
    }

    int CalculateCoaIndex() {
        if (macroType == CoaType::PARAM_OFFSET) {
            return ((base) + 1) + OFFSET_INDEX_ORDER * (dim) + idx;
        } else if (macroType == CoaType::PARAM_VALID_SHAPE) {
            return ((base) + 1) + VALID_SHAPE_INDEX_ORDER * (dim) + idx;
        } else if (macroType == CoaType::PARAM) {
            return idx;
        }
        ALOG_ERROR_F("GetCoaFinalIdx Coa type is invalid.");
        return 0;
    }

    SymbolicScalar BuildMaybeConstCoa(int isConst, int attrValue) {
        if (macroType == CoaType::PARAM_OFFSET) {
            return MAYBE_CONST_COA_GetOffset(isConst, attrValue, dim, base, idx);
        } else if (macroType == CoaType::PARAM_VALID_SHAPE) {
            return MAYBE_CONST_COA_GetValidShape(isConst, attrValue, dim, base, idx);
        } else if (macroType == CoaType::PARAM) {
            return MAYBE_CONST_COA_GetParam(isConst, attrValue, idx);
        }
        ALOG_ERROR_F("BuildMaybeConstCoa Coa type is invalid.");
        return 0;
    }
};

struct IsConstMetric {
    int isConst = 1;
    int attrValue = -1;

    void MarkNotConst() {isConst = 0;}
    int GetIsConst() {return isConst;}
    int GetAttrValue() {return attrValue;}
    void UpdateValue(int newValue) {
        if (attrValue == -1) {
            attrValue = newValue;
            return;
        }

        if (newValue < 0 || newValue != attrValue) {
            isConst = 0;
        }
    }
};

class DynAttrToStatic : public Pass {
public:
    DynAttrToStatic() : Pass("DynAttrToStatic") {}
    ~DynAttrToStatic() override = default;
private:
    std::unordered_map<Function*, std::vector<Operation*>> leaf2Caller;
    
    Status RunOnFunction(Function &function) override;
    std::vector<std::reference_wrapper<SymbolicScalar>> GetOpDynamicAttributeList(Operation &op);
    Status GetCallee(const Operation *callop, Function *&callFunc);
    void RefSpecifiedValue(std::vector<SymbolicScalar> &oriList,
        std::vector<std::reference_wrapper<SymbolicScalar>> &newList) const;
    void FilterSpecifiedValue(std::vector<OpImmediate> &oriList,
        std::vector<std::reference_wrapper<SymbolicScalar>> &newList) const;
    Status BuildLeafToCaller(Function *func);
    Status BuildNewCoa(
        std::reference_wrapper<SymbolicScalar>& dynScalar,
        std::vector<std::vector<SymbolicScalar>>& callopArglistOneDim);
    Status TryRemoveDynAttr(Function* leafFunc, std::vector<Operation*> callList);
};
} // namespace tile_fwk
} // namespace npu
#endif // PASS_DYNATTR_TO_STATIC_H_