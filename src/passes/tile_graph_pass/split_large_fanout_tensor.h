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
 * \file split_large_fanout_tensor.h
 * \brief
 */

#ifndef PASS_SPLIT_LARGE_FANOUT_TENSOR_H_
#define PASS_SPLIT_LARGE_FANOUT_TENSOR_H_

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/pass_common_defs.h"
#include "passes/tile_graph_pass/dead_operation_eliminate.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_config/pass_config_manager.h"

namespace npu::tile_fwk {
/*
 * SplitLargeFanoutTensor:
 本pass主要用于处理assemble被多个消费者消费时，如果每人都消费的是部分数据，因为assemble导致需要等待全部数据产生，影响并行度的问题
 * 处理原则：
    1. 仅解决OP_ASSEMBLE作为copyOut后接OP_VIEW作为CopyIn的场景
    2. 比较assemble输入与view输出之前的重合关系
    2.1 完全匹配：一到一情况
    2.2 被覆盖：多到一情况
    2.3 覆盖所有：一到多情况
*/
class SplitLargeFanoutTensor : public Pass {
public:
    SplitLargeFanoutTensor() : Pass("SplitLargeFanoutTensor") {}
    ~SplitLargeFanoutTensor() override = default;

private:
    Status RunOnFunction(Function &function) override;
    void CollectCopyOut(Function &function);
    void CompareWithCopyIn(Function &function);
    void RecordMatched(Function &function, Operation &op, const std::shared_ptr<LogicalTensor> &targetTensor,
        const std::vector<std::shared_ptr<LogicalTensor>> &matchedTensors,
        const std::vector<std::shared_ptr<LogicalTensor>> &overlaps);
    void EraseRedundantCopyOut(Function &function);
    void EraseRedundantCopyIn(Function &function);
    void RemoveOps(Function &function, std::vector<Operation *> &opList) const;
    void UpdateForRedundantAssemble(Operation &op);
    void UpdateForRedundantView(Operation &op, Operation &consumer);

    /*
    key: Assemble输出LogicalTensor所指向的raw tensor Id
    value: vector, 每个元素代表了Assemble输出指向了key对应的raw tensor的Assemble Op 2个信息
        1. Assemble输入的AscendTensor的指针
        2. Assemble的toOffset信息，类型为std::vector<int>
    */
    std::unordered_map<int, std::vector<std::pair<std::shared_ptr<LogicalTensor>, std::vector<int64_t>>>>
        copyOutSources;
    std::vector<AssembleOp> assembles;
};
} // namespace npu::tile_fwk
#endif // PASS_SPLIT_LARGE_FANOUT_TENSOR_H_