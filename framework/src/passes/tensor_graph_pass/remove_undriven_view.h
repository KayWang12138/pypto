/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file remove_undriven_view.h
 * \brief
 */

#pragma once

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {

/*
    为AssembleSSA删除inplaceIdx来源的undriven的View
*/
class RemoveUndrivenView : public Pass {
public:
    RemoveUndrivenView() : Pass("RemoveUndrivenView") {}
    ~RemoveUndrivenView() override = default;

private:
    Status RunOnFunction(Function &function) override;
};
} // namespace npu::tile_fwk