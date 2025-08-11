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
 * \file infer_dyn_shape.cpp
 * \brief
 */

#include <queue>
#include "interface/function/function.h"
#include "infer_dyn_shape.h"
#include "passes/pass_utils/parallel_tool.h"
namespace npu {
namespace tile_fwk {
Status InferDynShape::PostCheck(Function &function) {
    for (auto& op : function.Operations()) {
        if (OpcodeManager::Inst().IsCopyIn(op.GetOpcode())) {
            const std::shared_ptr<OpAttribute> &attr = op.GetOpAttribute();
            ASSERT(attr != nullptr) << "Copy In attr is null";
            std::shared_ptr<CopyOpAttribute> copyAttr = std::static_pointer_cast<CopyOpAttribute>(attr);
            if (copyAttr->GetToDynValidShape().empty()) {
                ALOG_ERROR_F("Op %s[%d] has no dyn to shape attr.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
        }
        for (auto opOut : op.GetOOperands()) {
            if (opOut->GetDynValidShape().empty()) {
                ALOG_ERROR_F("Op %s[%d] output [%d] has no dynamic valid shape.", op.GetOpcodeStr().c_str(), op.GetOpMagic(), opOut->GetMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status InferDynShape::InferShape(Function& function){
    size_t i = 0U;
    std::map<int, size_t> opMagic2Idx;
    std::vector<Operation*> opList = function.Operations().DuplicatedOpList();
    for (auto op : opList) {
        opMagic2Idx[op->GetOpMagic()] = i;
        i++;
    }
    std::vector<std::vector<size_t>> opInGraph(opList.size());
    std::vector<std::vector<size_t>> opOutGraph(opList.size());
    ParallelTool::Instance().Parallel_for(0, opList.size(),1,[&](int st,int et,int tid) {
        (void) tid;
        for (int opIdx = st; opIdx < et; opIdx++) {
            auto& op = opList[opIdx];
            for (auto producer : op->ProducerOpsOrdered()) {
                opInGraph[opMagic2Idx[op->GetOpMagic()]].push_back(opMagic2Idx[producer->GetOpMagic()]);
            }
            for (auto consumer : op->ConsumerOpsOrdered()) {
                opOutGraph[opMagic2Idx[op->GetOpMagic()]].push_back(opMagic2Idx[consumer->GetOpMagic()]);
            }
        }
    });
    bool isInferIndex = false;
    TopoProgramUtils::TopoProgram(opList, opInGraph, opOutGraph, isInferIndex);
    return SUCCESS;
}

Status InferDynShape::RunOnFunction(Function &function)
{
    // 遍历每一个op，调用对应的infershape函数
    // 遍历顺序，按照入度解依赖
    ALOG_INFO_F("===> Start InferDynShape.");
    if (InferShape(function) != SUCCESS) {
        return FAILED;
    }
    ALOG_DEBUG(function.Dump());
    ALOG_INFO_F("===> End InferDynShape.");
    return SUCCESS;
}
} 
} // namespace npu::tile_fwk