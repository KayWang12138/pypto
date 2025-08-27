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
 * \file remove_redundant_cast.h
 * \brief
 */

#ifndef PASS_REMOVE_REDUNDANT_CAST_H_
#define PASS_REMOVE_REDUNDANT_CAST_H_

#include "passes/pass_interface/pass.h"
#include "passes/pass_check/remove_redundant_cast_checker.h"
#include "interface/function/function.h"

namespace npu {
namespace tile_fwk {
class RemoveRedundantCast : public Pass {
public:
    RemoveRedundantCast() : Pass("RemoveRedundantCast") {}
    ~RemoveRedundantCast() override = default;
    Status RunOnFunction(Function &function) override;
    bool SupportBF16(Operation *op);
    Status InsertCast(Function &function);
    bool IsLegalCast(DataType ds, DataType dt);
    std::vector<Operation *> GetCastChain(Operation *tailOp);
    Status ShortenChain(Function &function, const std::vector<Operation *> &castChain, Operation *tailOp);
    Status RemoveRedundantCastChain(Function &function);
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;

    Status GetInOutConnectedTensor(Function &function);
    std::unordered_set<int> inCastConnectedTensors_;
    std::unordered_set<int> outCastConnectedTensors_;
};
}
}
#endif // PASS_REMOVE_REDUNDANT_CAST_H_