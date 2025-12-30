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
 * \file control_flow_function.h
 * \brief
 */

#pragma once

#include "interface/function/function.h"

namespace npu::tile_fwk {

// ControlFlowFunction is the dedicated subtype for control flow graphs.
// It contains members and methods specific to control flow functions.
class ControlFlowFunction : public Function {
public:
    ControlFlowFunction(const Program &belongTo, const std::string &funcMagicName,
        const std::string &funcRawName, Function *parentFunc);

    virtual ~ControlFlowFunction() = default;
    ControlFlowFunction(const ControlFlowFunction &other) = delete;
    ControlFlowFunction(ControlFlowFunction &&other) = delete;
    ControlFlowFunction &operator=(const ControlFlowFunction &other) = delete;
    ControlFlowFunction &operator=(ControlFlowFunction &&other) = delete;

    void SetDyndevAttribute(const std::shared_ptr<DyndevFunctionAttribute> &attr) override { dyndevAttr_ = attr; }
    const std::shared_ptr<DyndevFunctionAttribute> &GetDyndevAttribute() const override { return dyndevAttr_; }
    std::shared_ptr<DyndevFunctionAttribute> &GetDyndevAttribute() override { return dyndevAttr_; }

    bool IsDyndev() const override { return dyndevAttr_ != nullptr; }

    bool IsDynloop() const override;

    void AddLoopCallToOrderGroup(Operation * callOp) override {
        loopCallOrderGroup_.push_back(callOp);
    }

    void ApplyLoopCallOrderGroup() override {
        if (!loopCallOrderGroup_.empty()) {
            AddOperationGroup(loopCallOrderGroup_);
        }
    }

    DyndevFunctionAttribute::ValueDependDesc LookupValueDepend() override;

private:
    std::shared_ptr<DyndevFunctionAttribute> dyndevAttr_; // Dynamic device function attributes
    std::vector<Operation *> loopCallOrderGroup_; // Loop call operations order group
};

} // namespace npu::tile_fwk

