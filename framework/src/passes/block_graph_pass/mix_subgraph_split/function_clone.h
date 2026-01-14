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
 * \file schedule_ooo.h
 * \brief
 */

#ifndef PASS_SCHEDULE_OOO_H
#define PASS_SCHEDULE_OOO_H

#include "passes/block_graph_pass/schedule_ooo/buffer_pool.h"
#include "passes/block_graph_pass/schedule_ooo/scheduler.h"
#include "passes/statistics/ooo_schedule_statistic.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/utils/id_gen.h"

namespace npu::tile_fwk {
class MixSubgraphFunctionClone { //类名
public:
    MixSubgraphFunctionClone(Function& rootFunc, Function& originalMixFunc) : rootFunc(rootFunc), originalMixFunc(originalMixFunc){}
    ~MixSubgraphFunctionClone() override {}

    Function* CloneFunction(const InternalComponentInfo& component,
                            uint64_t newProgramID,
                            SubgraphToFunction& subgraphToFunction);
    Operation* CloneOperation(Operation& originalOp, Function& targetFunc);

    std::unordered_map<int, int> magicMap; // 原始magic -> 新magic
    std::vector<std::shared_ptr<Operation>> programOps;
    Function& rootFunc;
    Function& originalMixFunc;
}
}