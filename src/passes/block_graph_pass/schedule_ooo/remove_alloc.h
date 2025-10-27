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
 * \file remove_alloc.h
 * \brief
 */

#ifndef PASS_REMOVE_ALLOC_H
#define PASS_REMOVE_ALLOC_H
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {
class RemoveAlloc : public Pass {
public:
    RemoveAlloc() : Pass("RemoveAlloc") {}
    ~RemoveAlloc() override = default;

private:
    Status RunOnFunction(Function &function) override {
        APASS_LOG_INFO_F(GetName().c_str(), "Operation", "===> Start RemoveAlloc.");
        RemoveAllocCall(function);
        APASS_LOG_INFO_F(GetName().c_str(), "Operation", "===> End RemoveAlloc.");
        return SUCCESS;
    }
    void RemoveAllocCall(Function &function) const;
};


} // namespace npu::tile_fwk

#endif // PASS_REMOVE_ALLOC_H