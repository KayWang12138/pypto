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
 * \file update_memory_map.cpp
 * \brief
 */

#include "passes/tile_graph_pass/update_memory_map.h"
#include <vector>
#include "interface/operation/opcode.h"
#include "common/data_type.h"
namespace npu{
namespace tile_fwk {

Status UpdateMemoryMap::RunOnFunction(Function &function)
{
    ALOG_INFO_F("===> Start UpdateMemoryMapPass.");
    inserter.RefreshTensorTobeMap(function);
    for (auto &op : function.Operations()) {
        auto subGraphId = op.GetSubgraphID();
        ALOG_DEBUG_F("self op magic: %d, subgraph id: %d", op.opmagic, subGraphId);
        for (auto &inputTensor : op.GetIOperands()) {
            ALOG_DEBUG_F("input tensor magic %d", inputTensor->magic);
            if (inputTensor->GetProducers().size() == 0 ||
                inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            for (auto &producer : inputTensor->GetProducers()) {
                auto parentOpGraphId = producer->GetSubgraphID();
                ALOG_DEBUG_F("parent op magic: %d, subgraph id: %d", producer->opmagic, parentOpGraphId);
                if (subGraphId != parentOpGraphId) {
                    ALOG_DEBUG_F("************* self op magic: %d, parent op magic: %d, has diff color\n", subGraphId, parentOpGraphId);
                    if (inputTensor->memorymap.count(subGraphId) == 0) {
                        inputTensor->memorymap.insert(std::make_pair(subGraphId, TileRange()));
                    }
                    inputTensor->memorymap.insert(std::make_pair(parentOpGraphId, TileRange()));
                }
            }
        }
        if (op.GetOpcode() == Opcode::OP_NOP) {
            if (op.GetOOperands().size() == 1 && op.GetOOperands()[0]->tensor->actualRawmagic != -1) {
                op.SetAsDeleted();
            }
        }
    }
    function.EraseOperations(false);
    UpdateMemMapForCrossSubgraphAccess(function);
    ALOG_INFO_F("===> End UpdateMemoryMapPass.");
    return SUCCESS;
}
void  UpdateMemoryMap::UpdateMemMapForCrossSubgraphAccess (Function &function) {
    bool needInsertConvert = false;
    for (auto &ele : function.GetTensorMap().tensorMap_) {
        for (auto &singleLogicalTensor : ele.second) {
            if (singleLogicalTensor->memorymap.size() <= 1 || singleLogicalTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR){
                continue;
            }
            
            std::set<int> producerColorSet;
            // std::set<int> consumerColorSet;
            for (auto &producerOp: singleLogicalTensor->GetProducers()) {
                producerColorSet.insert(producerOp->GetSubgraphID());
            }

            bool isAllChildView = true;
            for (auto &consumerOp: singleLogicalTensor->GetConsumers()) {
                if (consumerOp->GetOpcode() != Opcode::OP_VIEW) {
                    isAllChildView = false;
                    break;
                }
            }
            if (producerColorSet.size() > 1) {
                singleLogicalTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
                needInsertConvert = true;
                ASLOGE("@@@@@@@@@@@@@@@ Force setting tensor memtype ori %d to MEM_DEVICE_DDR", singleLogicalTensor->magic);
            }
            if (producerColorSet.size() > 1 && isAllChildView) {
                for (auto &consumerOp : singleLogicalTensor->GetConsumers()) {
                    inserter.UpdateTensorTobeMap(*singleLogicalTensor, *consumerOp, MemoryType::MEM_DEVICE_DDR);
                }
                needInsertConvert = true;
                ASLOGE("@@@@@@@@@@@@@@@ Force setting tensor memtype tobe %d to MEM_DEVICE_DDR", singleLogicalTensor->magic);
            }    
        }
    }
    if (needInsertConvert) {
        // 插入convert op
        inserter.DoInsertion(function);
    }
}
}
}  // namespace npu::tile_fwk