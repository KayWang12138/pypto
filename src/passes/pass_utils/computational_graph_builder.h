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
 * \file computational_graph_builder.h
 * \brief
 */

#ifndef COMPUTATIONAL_GRAPH_BUILDER_H
#define COMPUTATIONAL_GRAPH_BUILDER_H

#include <vector>
#include <unordered_map>
#include "operation/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"

namespace npu {
namespace tile_fwk {

class ComputationalGraphBuilder {
public:
    explicit ComputationalGraphBuilder() : function(Program::GetInstance().GetCurrentFunction())
    {
        function->SetFunctionType(FunctionType::STATIC);
    }
    bool AddTensor(DataType dataType, const std::vector<int>& tileShape, const std::string& name);
    bool AddTensors(DataType dataType, const std::vector<int>& tileShape, const std::vector<std::string>& names);
    bool AddOp(Opcode opcode, const std::vector<std::string>& ioperands, const std::vector<std::string>& ooperands,
               const std::string& name, bool updateFunctionMap = true);
    bool AddOps(const std::vector<Opcode>& opcodes, const std::vector<std::vector<std::string>>& ioperandss,
                const std::vector<std::vector<std::string>>& ooperandss, const std::vector<std::string>& names,
                bool updateFunctionMap = true);
    bool SetInCast(std::vector<std::string> ioperands);
    bool SetOutCast(std::vector<std::string> ooperands);
    Function *GetFunction();
    Operation *GetOp(const std::string& name);
    std::shared_ptr<LogicalTensor> GetTensor(const std::string& name);
    Function *function;
    std::unordered_map<std::string, std::shared_ptr<LogicalTensor>> tensors_;
    std::unordered_map<std::string, Operation*> operations_;
};
} // namespace tile_fwk
} // namespace npu
#endif