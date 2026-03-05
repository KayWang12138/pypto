/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to to License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file permute.h
 * \brief
 */

#pragma once
#include <string>
#include <vector>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
LogicalTensorPtr TensorPermuteOperation(Function &function, const LogicalTensorPtr &self,
    const std::vector<int> &dims);
Tensor Permute(const Tensor &self, const std::vector<int> &dims);
} // namespace npu::tile_fwk
