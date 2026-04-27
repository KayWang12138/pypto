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
 * \file assign_memory_type.h
 * \brief Memory type assignment pass - refactored version
 */

#ifndef TILE_FWK_ASSIGN_MEMORY_TYPE_H
#define TILE_FWK_ASSIGN_MEMORY_TYPE_H

#include <queue>
#include <unordered_set>
#include "passes/pass_interface/pass.h"
#include "interface/operation/opcode.h"
#include "passes/tile_graph_pass/data_path/convert_op_inserter.h"
#include "tilefwk/platform.h"
#include "tilefwk/data_type.h"
#include "passes/pass_check/assign_memory_type_checker.h"

namespace npu::tile_fwk {

class AssignMemoryType : public Pass {
public:
    AssignMemoryType() : Pass("AssignMemoryType") {}
    void SpecialCallInterfaceToBeDeleted(Function& function) { RunOnFunction(function); }

private:
    // ============================================================
    // 主入口
    // ============================================================
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;

    // ============================================================
    // Phase 1: 确定性 Memtype 设置
    // ============================================================
    void SetIncastOutcastMemtype(Function& function);
    void RunOnOperation(Operation& operation);
    void SetInputTensorsMemtype(Operation& operation, const std::vector<MemoryType>& inputsMemType);
    void SetOutputTensorsMemtype(Operation& operation, const std::vector<MemoryType>& outputsMemType);
    void ProcessViewWithSpecificMem(Operation& operation);
    void ProcessAssembleWithSpecificMem(Operation& operation);
    void AssignOpShmemWaitUntilMemtype(Operation& op);

    // ============================================================
    // Phase 2: 不确定性推导 - 连接Op类型传递
    // ============================================================
    void AssignMoveOp(Operation& operation);
    void AssignMoveOpForAssemble(Operation& operation);
    bool CheckAssembleProducerConsistency(const LogicalTensorPtr& outputTensor, MemoryType& fromType);
    bool CheckAssembleAlignment(const LogicalTensorPtr& outputTensor);
    void DeriveAssembleOutputOriginal(Operation& operation, LogicalTensorPtr& outputTensor);

    void AssignMoveOpForView(Operation& operation);
    MemoryType DeriveViewOutputOriginalFromTobeMap(LogicalTensorPtr& outputTensor, ViewOpAttribute* viewOpAttribute);
    void DeriveViewOutputOriginalFromInput(LogicalTensorPtr& outputTensor, LogicalTensorPtr& inputTensor, ViewOpAttribute* viewOpAttribute);
    void DeriveViewOutputOriginal(Operation& operation, ViewOpAttribute* viewOpAttribute, bool unaligned);
    void DeriveViewInputTobe(Operation& operation, ViewOpAttribute* viewOpAttribute, bool unaligned);
    bool TryMemoryReuse(LogicalTensorPtr& outputTensor, LogicalTensorPtr& inputTensor,
                        MemoryType outputOriginal, MemoryType inputOriginal, bool unaligned, ViewOpAttribute* viewOpAttribute);
    bool TryL0C2L1Pathway(LogicalTensorPtr& inputTensor, MemoryType outputOriginal, MemoryType inputOriginal,
                          Operation& operation, ViewOpAttribute* viewOpAttribute);

    // ============================================================
    // Phase 3: 不确定性推导 - UNKNOWN张量处理
    // ============================================================
    void AssignMemUnknown(Function& function);
    MemoryType DeriveUnknownTensorOriginal(LogicalTensorPtr& tensor);
    void ProcessUnknownInputTensor(Operation& op, LogicalTensorPtr& tensor, std::unordered_set<LogicalTensorPtr>& visited);
    void ProcessUnknownOutputTensor(LogicalTensorPtr& tensor, std::unordered_set<LogicalTensorPtr>& visited);

    // ============================================================
    // Phase 4: 不确定性推导 - 特殊Op修正
    // ============================================================
    void AssignSpecialOpMemtype(Operation& op, bool& infoBufferSize);
    void AssignOpReshapeMemtype(Operation& op);
    void AssignOpViewTypeMemtype(Operation& op);
    void AssignOpNopMemtype(Operation& op);
    void AssignOpAssembleFinalMemtype(Operation& op, bool& infoBufferSize);
    void UpdateOverSizedLocalBuffer(Operation& operation);

    // ============================================================
    // Phase 5: 不确定性推导 - Tile维度约束修正
    // ============================================================
    bool CheckTileTobeConsistency(LogicalTensorPtr& output);
    void ProcessSmallTileToLargeTile(Function& function);
    void ProcessLargeTileToSmallTile(Function& function);
    bool IsDimMultiple(const Shape& shape1, const Shape& shape2);

    // ============================================================
    // 辅助函数
    // ============================================================
    int64_t CalcLineOffset(const Shape& shape, const Offset& offset);
    std::string PrintTensorMem(std::shared_ptr<LogicalTensor>& tensor) const;

    // ============================================================
    // 成员变量
    // ============================================================
    ConvertInserter inserter;
    AssignMemoryTypeChecker checker;
};

static constexpr double UB_THRESHOLD = 0.35;
static constexpr double L1_THRESHOLD = 0.5;

} // namespace npu::tile_fwk

#endif // TILE_FWK_ASSIGN_MEMORY_TYPE_H