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
 * \file insert_copy_op.cpp
 * \brief
 */

#include "insert_copy_op.h"
#include "passes/pass_utils/graph_utils.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
Status InsertInterGraphCopy::RunOnFunction(Function &function) {
    copysToCreate.clear();
    SplitTensor(function);
    CreateCopyOp(function);
    return SUCCESS;
}

void InsertInterGraphCopy::SplitTensor(Function &function) {
    std::unordered_map<int, std::unordered_map<int, std::shared_ptr<LogicalTensor>>> tensorSplits;
    for (auto &op : function.Operations()) {
        for (auto &input : op.GetIOperands()) {
            if (input->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR && tensorSplits.count(input->magic) != 0) {
                auto newInput = tensorSplits[input->magic][op.GetSubgraphID()];
                if (newInput == input) {
                    continue;
                }

                auto ddrTensor = std::make_shared<LogicalTensor>(function, input->Datatype(), input->shape);
                ddrTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
                // Add a global ddr allocator to allocate memory of ddr tensor
            copysToCreate.emplace_back(CopyOp{{input}, {ddrTensor}, {newInput}, &op});
            }
        }

        for (auto &output : op.GetOOperands()) {
            if (output->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR && output->memorymap.size() > 1) {
                tensorSplits.insert({output->magic, {}});
                std::pair<int, TileRange> reallyRange;
                for (auto &[subgraphId, range] : output->memorymap) {
                    if (subgraphId == op.GetSubgraphID()) {
                        reallyRange = {subgraphId, range};
                        tensorSplits[output->magic].insert({subgraphId, output});
                        continue;
                    }
                    auto splitTensor = std::make_shared<LogicalTensor>(function, output->Datatype(), output->shape);
                    splitTensor->SetMemoryTypeBoth(output->GetMemoryTypeOriginal());
                    splitTensor->subGraphID = subgraphId;
                    splitTensor->memorymap.insert({subgraphId, range});
                    tensorSplits[output->magic].insert({subgraphId, splitTensor});
                }
                output->memorymap.clear();
                output->memorymap.insert(reallyRange);
                output->subGraphID = op.GetSubgraphID();
            }
        }
    }
}
void InsertInterGraphCopy::CreateCopyOp(Function &function) {
    for (auto &copy : copysToCreate) {
        CopyInOutOp copyOutOp = {copy.input->GetMemoryTypeOriginal(), OpImmediate::Specified(std::vector<int64_t>(copy.input->shape.size(), 0)),
                                 OpImmediate::Specified(copy.input->shape), OpImmediate::Specified(copy.output->tensor->GetDynRawShape()), {}, copy.input, copy.ddr};
        GraphUtils::AddCopyOutOperation(function, copyOutOp);
        CopyInOutOp copyInOp = {copy.output->GetMemoryTypeOriginal(), OpImmediate::Specified(std::vector<int64_t>(copy.output->shape.size(), 0)), 
                          OpImmediate::Specified(copy.output->shape), OpImmediate::Specified(copy.input->tensor->GetDynRawShape()),
                          OpImmediate::Specified(copy.output->GetDynValidShape()), copy.ddr, copy.output};
        GraphUtils::AddCopyInOperation(function, copyInOp);
        if (copy.usedOp) {
            copy.usedOp->ReplaceInput(copy.output, copy.input);
        }
    }
}
} // namespace npu::tile_fwk
