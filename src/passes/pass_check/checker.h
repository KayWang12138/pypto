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
 * \file checker.h
 * \brief
 */

#ifndef CHECKER_H
#define CHECKER_H

#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
class Checker {
public:
    virtual ~Checker() = default;
    virtual Status DoPreCheck(Function &function);
    virtual Status DoPostCheck(Function &function);
protected:
    Status CheckValidOp(Function &function);
    Status CheckOpIOValid(Function &function);
};
} // namespace tile_fwk
} // namespace npu
#endif  // CHECKER_H