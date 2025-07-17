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
 * \file merge_compile_function.h
 * \brief
 */

#ifndef PASS_MERGE_COMPILE_FUNCTION_H_
#define PASS_MERGE_COMPILE_FUNCTION_H_

#include <vector>
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "allocator.h"
#include "passes/pass_interface/pass.h"
namespace npu::tile_fwk {
class MergeCompileFunction : public Pass {
public:
    MergeCompileFunction() : Pass("MergeCompileFunction") {}
    ~MergeCompileFunction() override = default;

private:
    Status RunOnFunction(Function &function) override;
    void CheckLoop(Function &function) const;
    void Init(Function &function);
    void CutIslands(Function &function);
    bool TryAllocate(Function &function, Operation &src, int subGraphID, TiFWKJointAllocator &ddrAllocator);

    bool CheckNewOperationsValid(Function &function, const std::set<int> &newOperations, int subGraphID);
    bool TryAllocateForOneSolution(Function &function, const std::set<int> &solution, int subGraphID) const;
    std::unordered_map<std::shared_ptr<LogicalTensor>, int> GetTensorOutDegreeByOperations(Function &function, const std::set<int> &solution) const;
    std::set<int> FindOperationsToAppend(Function &function, Operation &src);

    std::unordered_map<Operation *, int> operationIndex_;
    std::unordered_map<std::shared_ptr<LogicalTensor>, int> ddrTensorOutDegree_;
    int subGraphID_;
};
}
#endif // PASS_MERGE_COMPILE_FUNCTION_H_