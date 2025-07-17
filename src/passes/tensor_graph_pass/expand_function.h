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
 * \file expand_function.h
 * \brief
 */

#ifndef PASS_EXPAND_FUNCTION_H_
#define PASS_EXPAND_FUNCTION_H_

#include "passes/pass_interface/pass.h"
namespace npu::tile_fwk {
class ExpandFunction : public Pass {
public:
    ExpandFunction() : Pass("ExpandFunction") {}
    ~ExpandFunction() override = default;
private:
    Status RunOnFunction(Function &function) override;
    Status Expandfunction(Function &function) const;
};
}
#endif // PASS_EXPAND_FUNCTION_H_