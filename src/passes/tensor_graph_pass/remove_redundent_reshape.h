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
 * \file remove_redundent_reshape.h
 * \brief
 */

#ifndef PASS_REMOVE_REDUNDEN_RESHAPE_H_
#define PASS_REMOVE_REDUNDEN_RESHAPE_H_

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {
class RemoveRedundentReshape : public Pass {
public:
    RemoveRedundentReshape() : Pass("RemoveRedundentReshape") {}
    ~RemoveRedundentReshape() override = default;
private:
    Status RunOnFunction(Function &function) override;
    Status RemoveReshape(Function &function) const;
};
}
#endif // PASS_REMOVE_REDUNDEN_RESHAPE_H_