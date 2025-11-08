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
 * \file indexing.h
 * \brief
 */

#pragma once
#include <string>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

template <typename T, DataType dataType>
Element GetCurStartElement(Element start, Element step, int id) {
    T startValue;
    T stepValue;
    if (dataType == DT_INT32 || dataType == DT_INT64) {
        startValue = start.GetSignedData();
        stepValue = step.GetSignedData();
    } else if (dataType == DT_FP32) {
        startValue = (float)start.GetFloatData();
        stepValue = (float)step.GetFloatData();
    }
    T curStartValue = startValue + id * stepValue;
    Element curStart(dataType, curStartValue);
    return curStart;
}

const float EPSILON = (float)1e-8;
template <typename T, DataType dataType>
int64_t GetRangeResSize(Element &start, Element &end, Element &step) {
    int64_t resultSize;
    if (dataType == DT_INT32 || dataType == DT_INT64) {
        int64_t startValue = start.GetSignedData();
        int64_t endValue = end.GetSignedData();
        int64_t stepValue = step.GetSignedData();
        if (abs(stepValue) <= 0) {
            ASSERT(false && "stepValue must not be 0");
        }
        resultSize = (endValue - startValue) % stepValue ? (endValue - startValue) / stepValue + 1 :
                                                           (endValue - startValue) / stepValue;
    } else if (dataType == DT_FP32) {
        T startValue = (float)start.GetFloatData();
        T endValue = (float)end.GetFloatData();
        T stepValue = (float)step.GetFloatData();
        if (abs(stepValue) <= EPSILON) {
            ASSERT(false && "stepValue must not be 0");
        }
        resultSize = static_cast<int64_t>(std::ceil((endValue - startValue) / stepValue));
    }
    return resultSize;
}

} // namespace npu::tile_fwk
