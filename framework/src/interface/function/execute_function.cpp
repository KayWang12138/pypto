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
 * \file execute_function.cpp
 * \brief
 */

#include "interface/function/execute_function.h"

#include "interface/operation/operation.h"
#include "interface/program/program.h"
 
namespace npu::tile_fwk {
ExecuteFunction::ExecuteFunction(const Program &belongTo, const std::string &funcMagicName,
const std::string &funcRawName, Function *parentFunc)
: Function(belongTo, funcMagicName, funcRawName, parentFunc) {
// Make the intent explicit to avoid forgetting it at creation sites.
    SetGraphType(GraphType::EXECUTE_GRAPH);
}

void ExecuteFunction::DumpTopoFile(const std::string &fileName) const
{
    Json totalTopoJson;
    for (const auto &topo : topoInfo_.GetTopology()) {
        Json sJson;
        sJson["taskId"] = topo.esgId;
        sJson["successors"] = Json::array();
        for (const auto &successor : topo.outGraph) {
            sJson["successors"].push_back(successor);
        }
        int id = operations_[topo.esgId]->GetProgramId();
        if (static_cast<size_t>(id) >= calleeMagicNameList_.size()) {
            continue;
        }
        sJson["funcName"] = calleeMagicNameList_[id];
        sJson["semanticLabel"] = operations_[topo.esgId]->GetSemanticLabelStr();
        totalTopoJson.push_back(sJson);
    }
    std::ofstream ofs(fileName);
    ofs << totalTopoJson.dump(1) << std::endl;
    ofs.close();
}

} // namespace npu::tile_fwk
 