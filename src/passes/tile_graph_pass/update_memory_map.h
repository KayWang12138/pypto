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
 * \file update_memory_map.h
 * \brief
 */

#ifndef PASSES_UPDATER_MEMORY_MAP_H_
#define PASSES_UPDATER_MEMORY_MAP_H_

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"
#include "passes/tile_graph_pass/convert_op_inserter.h"
namespace npu::tile_fwk {
class UpdateMemoryMap : public Pass {
public:
    UpdateMemoryMap() : Pass("UpdateMemoryMap") {}
    ~UpdateMemoryMap() override = default;

private:
    Status RunOnFunction(Function &function) override;
    void  UpdateMemMapForCrossSubgraphAccess (Function &function);
    ConvertInserter inserter;
};
}  // namespace npu::tile_fwk
#endif  // PASSES_UPDATER_MEMORY_MAP_H_