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
 * \file dep_manager.h
 * \brief Dependency manager for operation scheduling
 */

#ifndef PASS_DEP_MANAGER_H_
#define PASS_DEP_MANAGER_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>
#include <string>
#include "interface/operation/operation.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {

inline bool IsViewOp(const Operation &op) {
    const auto opc = op.GetOpcode();
    return opc == Opcode::OP_VIEW || opc == Opcode::OP_VIEW_TYPE;
}

class DependencyManager {
public:
    void RegisterOp(Operation *op);

    void ClearDependencies();

    static bool IsOpAlloc(Operation *op);

    void AddDependency(Operation *preOp, Operation *postOp);

    void AddAllocDependency(Operation *preOp, Operation *postOp);

    bool RemoveDependency(Operation *preOp, Operation *postOp);

    std::unordered_set<Operation *> &GetSuccessors(Operation *op);
    std::unordered_set<Operation *> &GetPredecessors(Operation *op);
    bool HasOp(Operation *op) const;

    Status TransferSuccessorsByMemId(Operation *opA, Operation *opB, int memId,
        const std::function<bool(Operation *)> &isRetired,
        const std::function<const std::vector<int> &(Operation *)> &getReqMemIds);

    void ReplaceAllocPredecessor(const std::vector<Operation *> &ops, int memId, Operation *newPre,
        const std::function<bool(Operation *)> &isRetired,
        const std::function<const std::vector<int> &(Operation *)> &getReqMemIds);

    void Print(const std::vector<Operation *> &ops, const std::function<std::string(Operation *)> &getInfoFn) const;

    Operation *SkipViewChain(Operation *start, bool followProducers);

    void FindDependencies(Operation *op);

    Status InitDependencies(const std::vector<Operation *> &ops);

    void PrintDependencies(const std::vector<Operation *> &ops);

private:
    void Clear();

    std::unordered_map<Operation *, std::unordered_set<Operation *>> inGraph_;
    std::unordered_map<Operation *, std::unordered_set<Operation *>> outGraph_;
};

} // namespace npu::tile_fwk

#endif