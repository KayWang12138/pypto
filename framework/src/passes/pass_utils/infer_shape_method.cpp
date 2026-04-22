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
 * \file infer_shape_method.cpp
 * \brief
 */

#include "infer_shape_method.h"
#include "topo_program.h"

namespace npu {
namespace tile_fwk {
Status InferShapeMethod::BuildGraph(Function& function, std::vector<Operation*>& opList,
                                  std::vector<std::vector<size_t>>& opInGraph, std::vector<std::vector<size_t>>& opOutGraph)
{
    size_t i = 0U;
    std::map<int, size_t> opMagic2Idx;
    opList = function.Operations().DuplicatedOpList();
    for (const auto op : opList) {
        opMagic2Idx[op->GetOpMagic()] = i;
        i++;
    }
    opInGraph.resize(opList.size());
    opOutGraph.resize(opList.size());
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        const auto& op = opList[opIdx];
        for (const auto producer : op->ProducerOpsOrdered()) {
            opInGraph[opMagic2Idx[op->GetOpMagic()]].push_back(opMagic2Idx[producer->GetOpMagic()]);
        }
        for (const auto consumer : op->ConsumerOpsOrdered()) {
            opOutGraph[opMagic2Idx[op->GetOpMagic()]].push_back(opMagic2Idx[consumer->GetOpMagic()]);
        }
    }
    return SUCCESS;
}

Status InferShapeMethod::InferShape(Function& function)
{
    std::vector<Operation*> opList;
    std::vector<std::vector<size_t>> opInGraph(opList.size());
    std::vector<std::vector<size_t>> opOutGraph(opList.size());
    if (BuildGraph(function, opList, opInGraph, opOutGraph) != SUCCESS) {
        return FAILED;
    }
    bool isInferIndex = false;
    TopoProgramUtils::TopoProgram(opList, opInGraph, opOutGraph, isInferIndex);
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu