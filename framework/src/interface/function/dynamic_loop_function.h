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
 * \file dynamic_loop_function.h
 * \brief
 */

#pragma once

#include "interface/function/control_flow_function.h"

namespace npu::tile_fwk {

// DynamicLoopFunction is the dedicated subtype for DYNAMIC_LOOP function type.
// It contains members and methods specific to dynamic loop functions.
class DynamicLoopFunction : public ControlFlowFunction {
public:
    DynamicLoopFunction(const Program &belongTo, const std::string &funcMagicName,
        const std::string &funcRawName, Function *parentFunc);

    virtual ~DynamicLoopFunction() = default;
    DynamicLoopFunction(const DynamicLoopFunction &other) = delete;
    DynamicLoopFunction(DynamicLoopFunction &&other) = delete;
    DynamicLoopFunction &operator=(const DynamicLoopFunction &other) = delete;
    DynamicLoopFunction &operator=(DynamicLoopFunction &&other) = delete;

    void SetDynloopAttribute(const std::shared_ptr<DynloopFunctionAttribute> &attr) override { dynloopAttr_ = attr; }
    const std::shared_ptr<DynloopFunctionAttribute> &GetDynloopAttribute() const override { return dynloopAttr_; }
    std::shared_ptr<DynloopFunctionAttribute> &GetDynloopAttribute() override { return dynloopAttr_; }

private:
    std::shared_ptr<DynloopFunctionAttribute> dynloopAttr_; // Dynamic loop function attributes
};

} // namespace npu::tile_fwk

