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
 * \file Scheduler.h
 * \brief
 */

#pragma once

#include <string>
#include <unordered_map>
#include "simulation/base/Machine.h"
#include "simulation/common/ISA.h"

namespace CostModel {
class Scheduler {
public:
    std::shared_ptr<SimSys> sim = nullptr;
    std::string sortTileAllocPolicy = "DOM_COUNT";
    std::vector<std::map<int, int>> pipeIssueOrders;
    std::vector<int> issueSequencePtr;
    void MergeCopyOutGroup(int srcCopyOutIdx, int curCopyOutIdx, std::map<int, int> &copyOutSeq);
    void TileInsertQueue(TilePtr tile, std::vector<std::vector<int>> &tileAllocSequence);
    void TileOpInsertQueue(TileOpPtr tileOp);
    void SortTile(std::unordered_map<int, TilePtr> &tiles, std::unordered_map<int, TileOpPtr> &tileOps,
                  std::vector<std::vector<int>> &tileAllocSequence);
    std::shared_ptr<SimSys> GetSim();
};
}  // namespace CostModel